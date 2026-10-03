"""Stage D1 per-endpoint DVFS and runtime-feedback pair-routing isolation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.adaptive_exploration import (
    AdaptiveExplorationConfig,
    AdaptivePoolController,
    evidence_from_window,
)
from xpyd.d1_control import (
    ENDPOINT_PHASE,
    EndpointFastSafetyRuntime,
    active_candidate,
    exploration_permissions,
)
from xpyd.phase3b_energy import _read_jsonl
from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    load_config as load_runtime_config,
)
from xpyd.phase3d_control import NvidiaSmiClockBackend, PerEndpointClockActuator
from xpyd.stagec0_capacity import percentile
from xpyd.stagec1_stationary import ControllerContext, _read_csv, _trace_signature, _write_csv
from xpyd.stagec2_dynamic import (
    C2AccountingWindowHarness,
    ClockSegmentLedger,
    FastSafetyRuntime,
    StageC2Error,
    _bool,
    derived_context_signature,
)
from xpyd.stagec2a_fast_safety import StageC2AFastSafetyHarness, segment_residency
from xpyd.workload_harness import build_trace, load_config as load_workload_config


POLICIES = (
    "STATIC-HIGH-RR",
    "SHARED-FAST-RR",
    "PER-ENDPOINT-FAST-RR",
    "JOINT-ROUTING-PER-ENDPOINT",
)
ENDPOINTS = ("P0", "P1", "D0", "D1")
PAIRS = (("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"))


def policy_order(repeat: int, load_index: int) -> tuple[str, ...]:
    offset = (int(repeat) - 1 + int(load_index) * 3) % len(POLICIES)
    return POLICIES[offset:] + POLICIES[:offset]


def load_configs(
    runtime_path: Path, experiment_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_runtime_config(runtime_path)
    experiment = load_workload_config(experiment_path)
    settings = experiment.get("stage_d1")
    if not isinstance(settings, Mapping):
        raise StageC2Error("stage_d1 settings are required")
    if tuple(settings.get("policies", ())) != POLICIES:
        raise StageC2Error("D1 requires the frozen four-policy matrix")
    if settings.get("slo") != {
        "ttft_ms": 600.0, "tpot_ms": 65.0, "attainment": 0.95,
    }:
        raise StageC2Error("D1 must retain the TIGHT 600/65 SLO")
    if int(settings.get("max_concurrency", 0)) != 4:
        raise StageC2Error("D1 requires concurrency 4")
    if settings.get("frequency_state") != "independent per endpoint":
        raise StageC2Error("D1 requires independent endpoint frequency state")
    forbidden = ("predictor", "offline_profile", "rl", "bandit", "tp2")
    if any(bool(settings.get(key)) for key in forbidden):
        raise StageC2Error("D1 forbids predictors/profiles/RL/bandits/TP2")
    expected = {
        "P0": {"LOW": 1260, "MID": 1890, "HIGH": 2520},
        "P1": {"LOW": 1260, "MID": 1890, "HIGH": 2520},
        "D0": {"LOW": 750, "MID": 1125, "HIGH": 1500},
        "D1": {"LOW": 750, "MID": 1125, "HIGH": 1500},
    }
    if settings.get("frequency_levels_mhz") != expected:
        raise StageC2Error("D1 frequency action space changed")
    nodes = {str(item["endpoint_id"]): str(item["node"]) for item in runtime["endpoints"]}
    if nodes != {"P0": "neptune", "P1": "neptune", "D0": "europa", "D1": "europa"}:
        raise StageC2Error("D1 topology must be Neptune L40S + Europa L4")
    if runtime.get("routing_policy") != "d1_runtime_feedback":
        raise StageC2Error("D1 runtime config must activate request-level routing")
    if not str(runtime.get("routing_control_file", "")):
        raise StageC2Error("D1 routing control file is required")
    runtime["phase3d"] = {
        "minimum_dwell_s": float(settings.get("minimum_dwell_s", 1.0)),
        "feedback": {
            "ttft_slo_ms": 600.0,
            "tpot_slo_ms": 65.0,
            "telemetry_max_age_s": float(
                settings["fast_safety"]["telemetry_max_age_s"]
            ),
        },
    }
    return runtime, experiment


class D1AccountingWindowHarness(C2AccountingWindowHarness):
    def _metadata_overrides(self) -> dict[str, Any]:
        value = super()._metadata_overrides()
        value.update({
            "phase": "Stage_D1_per_endpoint_routing_isolation",
            "frequency_state": "actual per-endpoint clock segments",
            "routing_context": "admission-time input/output budget/class",
        })
        return value

    def _scrape(self, endpoint: Any, scheduled: float, label: str = "periodic") -> Any:
        # Bypass C2's pool-scoped failure callback; D1 stale/invalid safety is
        # endpoint-scoped for per-endpoint policies.
        attempt = Phase3CSubstrateHarness._scrape(self, endpoint, scheduled, label)
        if (
            self.safety is not None
            and self.safety.started_wall_s is not None
            and not self.safety.stop_event.is_set()
            and attempt.error is not None
        ):
            target = (
                endpoint.endpoint_id
                if isinstance(self.safety, EndpointFastSafetyRuntime)
                else ENDPOINT_PHASE[endpoint.endpoint_id]
            )
            self.safety.process_internal_failure(
                target, "INVALID_SAFETY_TELEMETRY", time.time()
            )
        return attempt

    def _claim_boundary(self) -> str:
        return (
            "D1 stationary per-endpoint DVFS and request-level pair-routing "
            "isolation; no model, TP diversity, scaling, or physical "
            "request-energy attribution claim"
        )


class StageD1Harness(StageC2AFastSafetyHarness):
    def __init__(
        self,
        runtime: Mapping[str, Any],
        experiment: Mapping[str, Any],
        *,
        run_id: Optional[str] = None,
        backend: Optional[NvidiaSmiClockBackend] = None,
        stage: str = "preflight",
    ) -> None:
        if stage not in {"preflight", "full"}:
            raise StageC2Error("D1 stage must be preflight or full")
        wrapped = copy.deepcopy(dict(experiment))
        wrapped["stagec2a"] = copy.deepcopy(wrapped["stage_d1"])
        wrapped["stagec2"] = copy.deepcopy(wrapped["stage_d1"])
        # Parent validation is intentionally bypassed: D1 has different policy
        # and frequency-state semantics but reuses its physical substrate.
        super(StageC2AFastSafetyHarness, self).__init__(
            runtime, wrapped, run_id=run_id, backend=backend, stage=stage
        )
        self.stage = stage
        self.settings = wrapped["stage_d1"]
        self.experiment = wrapped
        self.run_dir = Path(os.path.expandvars(str(wrapped["output_root"]))) / self.run_id
        self.routing_control_file = Path(str(runtime["routing_control_file"]))
        self.endpoint_energy: dict[str, Optional[float]] = {item: None for item in ENDPOINTS}
        self.endpoint_latency: dict[str, Optional[float]] = {item: None for item in ENDPOINTS}
        self.pair_feedback: dict[str, dict[str, dict[str, Any]]] = {}

    def _controllers_endpoint(self) -> dict[str, AdaptivePoolController]:
        config = AdaptiveExplorationConfig.from_mapping(
            self.settings["adaptive_exploration"],
            ttft_slo_ms=600.0,
            tpot_slo_ms=65.0,
        )
        return {
            endpoint_id: AdaptivePoolController(
                ENDPOINT_PHASE[endpoint_id],
                "prefill" if endpoint_id.startswith("P") else "decode",
                ([1260, 1890, 2520] if endpoint_id.startswith("P") else [750, 1125, 1500]),
                config,
                slo_regime="TIGHT",
            )
            for endpoint_id in ENDPOINTS
        }

    @staticmethod
    def _endpoint_status(
        endpoint_id: str,
        targets: Mapping[str, int],
        controllers: Optional[Mapping[str, AdaptivePoolController]],
    ) -> str:
        if controllers is None:
            return "VALIDATED_SAFE"
        controller = controllers[endpoint_id]
        if active_candidate(controller):
            return "CANDIDATE_PROBE"
        if (
            controller.state.unsafe_bound is not None
            and int(controller.state.unsafe_bound) == int(targets[endpoint_id])
        ):
            return "UNSAFE"
        if int(controller.state.known_safe_frequency) == int(targets[endpoint_id]):
            return "VALIDATED_SAFE"
        return "UNKNOWN"

    def _write_routing_state(
        self,
        *,
        block_id: str,
        routing_mode: str,
        context: ControllerContext,
        endpoint_controllers: Optional[Mapping[str, AdaptivePoolController]],
        generation_suffix: str,
    ) -> None:
        endpoints = {}
        for endpoint_id in ENDPOINTS:
            controller = endpoint_controllers.get(endpoint_id) if endpoint_controllers else None
            endpoints[endpoint_id] = {
                "healthy": True,
                "telemetry_fresh": True,
                "safety_status": self._endpoint_status(
                    endpoint_id, context.targets, endpoint_controllers
                ),
                "requested_frequency_mhz": int(context.targets[endpoint_id]),
                "observed_frequency_mhz": int(context.targets[endpoint_id]),
                "validated_safe_frequency_mhz": (
                    int(controller.state.known_safe_frequency)
                    if controller is not None else int(context.targets[endpoint_id])
                ),
                "candidate_frequency_mhz": (
                    controller.state.candidate_frequency if controller is not None else None
                ),
                "recent_latency_ms": self.endpoint_latency[endpoint_id],
                "recent_energy_j_per_assigned_request": self.endpoint_energy[endpoint_id],
            }
        value = {
            "schema_version": 1,
            "updated_unix_s": time.time(),
            "generation": block_id,
            "control_epoch": generation_suffix,
            "routing_mode": routing_mode,
            "endpoints": endpoints,
            "class_pair_feedback": self.pair_feedback,
            "energy_semantics": (
                "recent endpoint gross board energy divided by requests assigned "
                "to that endpoint; window-amortized routing evidence, not physical "
                "per-request attribution"
            ),
        }
        self.routing_control_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.routing_control_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.routing_control_file)

    def _update_runtime_feedback(
        self, metrics: Mapping[str, Any], requests: Sequence[Mapping[str, Any]],
    ) -> None:
        for endpoint_id in ENDPOINTS:
            key = (
                "selected_prefill_endpoint_id"
                if endpoint_id.startswith("P") else "selected_decode_endpoint_id"
            )
            selected = [row for row in requests if row.get(key) == endpoint_id]
            energy = metrics.get("%s_energy_j" % endpoint_id)
            self.endpoint_energy[endpoint_id] = (
                float(energy) / len(selected) if energy is not None and selected else None
            )
            latency_key = "ttft_ms" if endpoint_id.startswith("P") else "tpot_ms"
            values = [float(row[latency_key]) for row in selected]
            self.endpoint_latency[endpoint_id] = percentile(values, 0.95) if values else None
        feedback: dict[str, dict[str, dict[str, Any]]] = {}
        for workload_class in ("SS", "SL", "LS", "LL"):
            feedback[workload_class] = {}
            for pair in PAIRS:
                selected = [
                    row for row in requests
                    if row.get("workload_class") == workload_class
                    and row.get("selected_prefill_endpoint_id") == pair[0]
                    and row.get("selected_decode_endpoint_id") == pair[1]
                ]
                if not selected:
                    continue
                ttft = [float(row["ttft_ms"]) for row in selected]
                tpot = [float(row["tpot_ms"]) for row in selected]
                feedback[workload_class]["%s->%s" % pair] = {
                    "observations": len(selected),
                    "recent_ttft_ms": percentile(ttft, 0.95),
                    "recent_tpot_ms": percentile(tpot, 0.95),
                    "slo_safe": all(item <= 600.0 for item in ttft)
                    and all(item <= 65.0 for item in tpot),
                }
        self.pair_feedback = feedback

    def _recover_endpoint(
        self,
        endpoint_id: str,
        recommendation: Mapping[str, Any],
        *,
        actuator: PerEndpointClockActuator,
        context: ControllerContext,
        controllers: Mapping[str, AdaptivePoolController],
        policy: str,
        block_id: str,
    ) -> dict[str, Any]:
        target = int(recommendation["target_mhz"])
        actions = self._actuate_targets(
            actuator, context, {endpoint_id: target},
            reason="stage_d1_%s" % str(recommendation["trigger"]).lower(),
            policy=policy,
            block_id=block_id,
            decision_timestamp_s=float(recommendation["decision_timestamp_s"]),
            recover_on_failure=False,
        )
        if endpoint_id not in actions:
            raise StageC2Error("D1 fast recovery did not actuate selected endpoint")
        row = actions[endpoint_id]
        readback = float(row["readback_timestamp_unix_s"])
        controllers[endpoint_id].record_actuation(
            target, int(row["observed_freq_after_mhz"]),
            readback_valid=bool(row["readback_valid"]), timestamp_s=readback,
        )
        # Publish the recovered endpoint state before the next admission. The
        # proxy otherwise still sees the pre-recovery candidate snapshot for
        # the remainder of the accounting window.
        self._write_routing_state(
            block_id=block_id,
            routing_mode=(
                "RUNTIME_FEEDBACK"
                if policy == "JOINT-ROUTING-PER-ENDPOINT" else "FIXED_RR"
            ),
            context=context,
            endpoint_controllers=controllers,
            generation_suffix="fast-recovery-%s" % endpoint_id,
        )
        return {
            "action_timestamp_s": row["timestamp_unix_s"],
            "readback_timestamp_s": readback,
            "readback_frequency_mhz": row["observed_freq_after_mhz"],
            "readback_valid": row["readback_valid"],
        }

    def _endpoint_efficiency_actuation(
        self,
        endpoint_id: str,
        target: int,
        *,
        actuator: PerEndpointClockActuator,
        context: ControllerContext,
        controller: AdaptivePoolController,
        policy: str,
        block_id: str,
        decision_timestamp_s: float,
    ) -> dict[str, dict[str, Any]]:
        try:
            return self._actuate_targets(
                actuator, context, {endpoint_id: int(target)},
                reason="stage_d1_efficiency_epoch",
                policy=policy, block_id=block_id,
                decision_timestamp_s=decision_timestamp_s,
                recover_on_failure=False,
            )
        except StageC2Error as exc:
            fallback = int(controller.state.known_safe_frequency)
            actions = self._actuate_targets(
                actuator, context, {endpoint_id: fallback},
                reason="stage_d1_actuator_readback_failure_recovery",
                policy=policy, block_id=block_id,
                decision_timestamp_s=time.time(),
                recover_on_failure=False,
            )
            row = actions.get(endpoint_id)
            if row is not None:
                readback = float(row["readback_timestamp_unix_s"])
                controller.record_actuation(
                    fallback, int(row["observed_freq_after_mhz"]),
                    readback_valid=bool(row["readback_valid"]),
                    timestamp_s=readback,
                )
                self.safety_rows.append({
                    "policy": policy, "block_id": block_id,
                    "endpoint_id": endpoint_id,
                    "pool": ENDPOINT_PHASE[endpoint_id],
                    "trigger": "ACTUATOR_READBACK_FAILURE",
                    "trigger_timestamp_s": decision_timestamp_s,
                    "action": "FAST_RECOVER",
                    "prior_mhz": target,
                    "target_mhz": fallback,
                    "action_timestamp_s": row["timestamp_unix_s"],
                    "readback_timestamp_s": readback,
                    "readback_frequency_mhz": row["observed_freq_after_mhz"],
                    "readback_valid": row["readback_valid"],
                    "fast_path_enabled": True,
                    "violation_recorded": False,
                    "error": str(exc),
                })
            return actions

    def _endpoint_efficiency_control(
        self,
        context: ControllerContext,
        actuator: PerEndpointClockActuator,
        controllers: Mapping[str, AdaptivePoolController],
        metrics: Mapping[str, Any],
        requests: Sequence[Mapping[str, Any]],
        policy: str,
        block_id: str,
        before_window_index: int,
        *,
        allow_exploration: bool = True,
    ) -> None:
        now = time.time()
        permissions = {
            phase: exploration_permissions(controllers, phase)
            for phase in ("P", "D")
        }
        for endpoint_id in ENDPOINTS:
            phase = ENDPOINT_PHASE[endpoint_id]
            controller = controllers[endpoint_id]
            controller.set_current_frequency(int(context.targets[endpoint_id]))
            key = (
                "selected_prefill_endpoint_id"
                if phase == "P" else "selected_decode_endpoint_id"
            )
            selected = [row for row in requests if row.get(key) == endpoint_id]
            if not selected:
                self.decision_rows.append({
                    "policy": policy, "block_id": block_id,
                    "endpoint_id": endpoint_id, "pool": phase,
                    "control_path": "EFFICIENCY", "decision": "HOLD",
                    "action": "HOLD", "trigger": "INSUFFICIENT_ENDPOINT_EVIDENCE",
                    "reason": "no requests assigned to endpoint in evidence window",
                    "decision_timestamp_s": now,
                    "before_window_index": before_window_index,
                })
                continue
            evidence_metrics = dict(metrics)
            evidence_metrics.update({
                "measurement_valid": True,
                "telemetry_fresh": True,
                "readback_valid": True,
            })
            signature = derived_context_signature(phase, selected, self.settings)
            evidence = evidence_from_window(
                phase,
                evidence_metrics,
                selected,
                observed_frequency_mhz=int(context.targets[endpoint_id]),
                slo_regime="TIGHT",
                ttft_slo_ms=600.0,
                tpot_slo_ms=65.0,
                context_signature=signature,
                captured_at_s=now,
                endpoint_ids=(endpoint_id,),
            )
            decision = controller.decide(
                evidence,
                now_s=now,
                allow_exploration=(
                    allow_exploration and permissions[phase][endpoint_id]
                ),
            )
            actions = self._endpoint_efficiency_actuation(
                endpoint_id,
                int(decision.requested_frequency_mhz),
                actuator=actuator,
                context=context,
                controller=controller,
                policy=policy,
                block_id=block_id,
                decision_timestamp_s=now,
            )
            target = int(context.targets[endpoint_id])
            if target != int(decision.current_frequency_mhz):
                row = actions[endpoint_id]
                controller.record_actuation(
                    target,
                    int(row["observed_freq_after_mhz"]),
                    readback_valid=bool(row["readback_valid"]),
                    timestamp_s=float(row["readback_timestamp_unix_s"]),
                )
            log = decision.as_log()
            log.update({
                "policy": policy,
                "block_id": block_id,
                "endpoint_id": endpoint_id,
                "pool": phase,
                "control_path": "EFFICIENCY",
                "before_window_index": before_window_index,
                "evidence_window_index": before_window_index - 1,
                "context_signature": signature,
                "finalization": not allow_exploration,
            })
            if actions:
                row = actions[endpoint_id]
                log["action_timestamp_s"] = row["timestamp_unix_s"]
                log["readback_timestamp_s"] = row["readback_timestamp_unix_s"]
            self.decision_rows.append(log)

    def _window_config(
        self, requested: Mapping[str, int], windows_root: Path,
    ) -> dict[str, Any]:
        config = super()._window_config(requested, windows_root)
        # Endpoint coverage remains hard. Pair coverage is audited over the D1
        # block because the joint policy is intentionally not per-window RR.
        config["coverage_policy"]["required_pairs"] = []
        return config

    def _diagnostics(self, request_ids: set[str]) -> list[dict[str, Any]]:
        path = Path(str(self.config["proxy_diagnostics_log"]))
        if not path.is_file():
            return []
        return [
            row for row in _read_jsonl(path)
            if row.get("logical_request_id") in request_ids
        ]

    def _run_experiment(self) -> Path:
        if self.stage == "full":
            self._check_accepted("XPYD_STAGED1_ACCEPTED_PREFLIGHT", "preflight_pass")
        if self.run_dir.exists():
            raise StageC2Error("D1 run directory already exists")
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        expected = self.settings["frequency_levels_mhz"]
        if {key: self._levels(value) for key, value in capabilities.items()} != expected:
            raise StageC2Error("discovered clocks differ from frozen D1 action space")
        _write_json(
            self.run_dir / "capabilities.json",
            {"endpoints": {key: asdict(value) for key, value in capabilities.items()}},
        )
        actuator = PerEndpointClockActuator(
            self.backend, capabilities, float(self.config["phase3d"]["minimum_dwell_s"])
        )
        high = {key: value.selected_high_mhz for key, value in capabilities.items()}
        repetitions = 1 if self.stage == "preflight" else int(self.settings["repetitions"])
        windows = int(
            self.settings["preflight_windows"]
            if self.stage == "preflight" else self.settings["windows_per_block"]
        )
        selected_loads = [
            next(
                item for item in self.settings["load_points"]
                if item["id"] == self.settings["preflight_load_point"]
            )
        ] if self.stage == "preflight" else list(self.settings["load_points"])
        rows: list[dict[str, Any]] = []
        requests_all: list[dict[str, Any]] = []
        run_error: Optional[str] = None
        restoration_valid = False
        try:
            for load_index, point in enumerate(selected_loads):
                self.settings["offered_rate_rps"] = float(point["offered_rate_rps"])
                for repeat in range(1, repetitions + 1):
                    policies = (
                        ("JOINT-ROUTING-PER-ENDPOINT",)
                        if self.stage == "preflight"
                        else policy_order(repeat, load_index)
                    )
                    for policy in policies:
                        block_id = "r%02d-%s-%s" % (
                            repeat, str(point["id"]).lower(), policy.lower()
                        )
                        self.endpoint_energy = {item: None for item in ENDPOINTS}
                        self.endpoint_latency = {item: None for item in ENDPOINTS}
                        self.pair_feedback = {}
                        registry, telemetry, scheduler = self._build_scheduler(capabilities)
                        shared = (
                            self._controllers(capabilities)
                            if policy == "SHARED-FAST-RR" else None
                        )
                        endpoint_controllers = (
                            self._controllers_endpoint()
                            if policy in {
                                "PER-ENDPOINT-FAST-RR",
                                "JOINT-ROUTING-PER-ENDPOINT",
                            } else None
                        )
                        context = ControllerContext(
                            registry, telemetry, scheduler, dict(high),
                            shared if shared is not None else endpoint_controllers,
                        )
                        self._actuate_targets(
                            actuator, context, high,
                            reason="stage_d1_cold_start_high",
                            policy=policy, block_id=block_id,
                        )
                        if self.stage == "preflight":
                            assert endpoint_controllers is not None
                            mid = capabilities["P0"].selected_mid_mhz
                            self._actuate_targets(
                                actuator, context, {"P0": mid},
                                reason="stage_d1_preflight_fixed_asymmetry",
                                policy=policy, block_id=block_id,
                            )
                            endpoint_controllers["P0"].state.candidate_frequency = mid
                            endpoint_controllers["P0"].set_current_frequency(mid)
                        previous = None
                        previous_requests: list[dict[str, Any]] = []
                        for window_index in range(1, windows + 1):
                            if previous is not None:
                                self._update_runtime_feedback(previous, previous_requests)
                                if shared is not None:
                                    self._efficiency_control(
                                        context, capabilities, actuator, previous,
                                        previous_requests, policy, block_id, window_index,
                                    )
                                elif endpoint_controllers is not None:
                                    self._endpoint_efficiency_control(
                                        context, actuator, endpoint_controllers,
                                        previous, previous_requests, policy, block_id,
                                        window_index,
                                    )
                            routing_mode = (
                                "RUNTIME_FEEDBACK"
                                if policy == "JOINT-ROUTING-PER-ENDPOINT"
                                else "FIXED_RR"
                            )
                            self._write_routing_state(
                                block_id=block_id,
                                routing_mode=routing_mode,
                                context=context,
                                endpoint_controllers=endpoint_controllers,
                                generation_suffix="w%02d" % window_index,
                            )
                            trace = build_trace(
                                self.experiment,
                                str(point["scenario"]),
                                "d1-%s-r%02d-w%02d" % (
                                    str(point["id"]).lower(), repeat, window_index
                                ),
                            )
                            signature = _trace_signature(trace)
                            window_id = "%s-w%02d" % (block_id, window_index)
                            ledger = ClockSegmentLedger(window_id)
                            self.active_ledger = ledger
                            safety = None
                            if shared is not None:
                                safety = FastSafetyRuntime(
                                    shared,
                                    lambda pool, recommendation, a=actuator, c=context, p=policy, b=block_id: self._fast_recover(
                                        pool, recommendation, actuator=a, context=c,
                                        policy=p, block_id=b,
                                    ),
                                    self.settings, policy=policy, block_id=block_id,
                                    window_index=window_index, enabled=True,
                                )
                            elif endpoint_controllers is not None:
                                safety = EndpointFastSafetyRuntime(
                                    endpoint_controllers,
                                    lambda endpoint, recommendation, a=actuator, c=context, cs=endpoint_controllers, p=policy, b=block_id: self._recover_endpoint(
                                        endpoint, recommendation, actuator=a, context=c,
                                        controllers=cs, policy=p, block_id=b,
                                    ),
                                    self.settings, policy=policy, block_id=block_id,
                                    window_index=window_index, enabled=True,
                                )
                            config = self._window_config(context.targets, windows_root)
                            window_dir = D1AccountingWindowHarness(
                                config, window_id, trace, ledger, safety, expected
                            ).run()
                            self.active_ledger = None
                            if safety is not None:
                                self.safety_rows.extend(safety.event_rows)
                            for segment in ledger.rows:
                                segment.update({
                                    "policy": policy, "block_id": block_id,
                                    "repeat": repeat, "load_point": point["id"],
                                    "window_index": window_index,
                                })
                            self.segment_rows.extend(ledger.rows)
                            row, request_rows = self._measure_window(
                                window_dir, block_id=block_id, policy=policy,
                                repeat=repeat, window_index=window_index,
                                context_id=str(point["id"]),
                                trace_signature=signature, ledger=ledger,
                            )
                            row["load_point"] = point["id"]
                            row["offered_rate_rps"] = point["offered_rate_rps"]
                            rows.append(row)
                            for item in request_rows:
                                item["load_point"] = point["id"]
                            requests_all.extend(request_rows)
                            previous, previous_requests = row, request_rows
                        if previous is not None:
                            self._update_runtime_feedback(previous, previous_requests)
                            if shared is not None:
                                self._efficiency_control(
                                    context, capabilities, actuator, previous,
                                    previous_requests, policy, block_id, windows + 1,
                                    allow_exploration=False,
                                )
                            elif endpoint_controllers is not None:
                                self._endpoint_efficiency_control(
                                    context, actuator, endpoint_controllers,
                                    previous, previous_requests, policy, block_id,
                                    windows + 1, allow_exploration=False,
                                )
                                for endpoint_id, controller in endpoint_controllers.items():
                                    state = controller.state
                                    self.search_rows.append({
                                        "policy": policy, "block_id": block_id,
                                        "repeat": repeat, "load_point": point["id"],
                                        "endpoint_id": endpoint_id,
                                        "current_frequency_mhz": state.current_frequency,
                                        "final_validated_safe_frequency_mhz": state.known_safe_frequency,
                                        "unsafe_bound_mhz": state.unsafe_bound,
                                        "best_measured_safe_frequency_mhz": state.best_measured_safe_frequency,
                                        "best_measured_safe_energy_j": state.best_measured_safe_energy_j,
                                        "windows_observed": state.windows_observed,
                                        "requests_observed": state.requests_observed,
                                        "converged": state.converged,
                                    })
        except BaseException as exc:
            run_error = "%s: %s" % (type(exc).__name__, exc)
            raise
        finally:
            self.active_ledger = None
            if "actuator" in locals() and "context" in locals():
                self._actuate_targets(
                    actuator, context, high,
                    reason="stage_d1_final_restore_high",
                    policy="RESTORE", block_id="final",
                )
                restoration_valid = all(
                    actuator.read(key).graphics_clock_mhz == value
                    for key, value in high.items()
                )
            if self.run_dir.exists():
                self._write_d1_reports(
                    rows, requests_all, repetitions, windows,
                    restoration_valid, run_error,
                )
        print(self.run_dir)
        return self.run_dir

    def _write_d1_reports(
        self,
        rows: Sequence[Mapping[str, Any]],
        requests: Sequence[Mapping[str, Any]],
        repetitions: int,
        windows: int,
        restoration_valid: bool,
        run_error: Optional[str],
    ) -> None:
        mutable_rows = [dict(row) for row in rows]
        for block_id in sorted({str(row["block_id"]) for row in mutable_rows}):
            cumulative_energy = 0.0
            cumulative_requests = 0
            for row in sorted(
                (item for item in mutable_rows if item["block_id"] == block_id),
                key=lambda item: int(item["window_index"]),
            ):
                cumulative_energy += float(row["total_gross_energy_j"])
                cumulative_requests += int(row["request_count"])
                row["cumulative_energy_j"] = cumulative_energy
                row["cumulative_requests"] = cumulative_requests
                row["cumulative_joules_per_request"] = (
                    cumulative_energy / cumulative_requests
                )
        rows = mutable_rows
        _write_csv(self.run_dir / "stage_d1_measurements.csv", rows)
        _write_csv(self.run_dir / "stage_d1_requests.csv", requests)
        _write_csv(self.run_dir / "controller_decisions.csv", self.decision_rows)
        _write_csv(self.run_dir / "safety_events.csv", self.safety_rows)
        _write_csv(self.run_dir / "clock_actions.csv", self.action_rows)
        _write_csv(self.run_dir / "clock_segments.csv", self.segment_rows)
        _write_csv(self.run_dir / "endpoint_search_state.csv", self.search_rows)
        request_ids = {str(row["trace_request_id"]) for row in requests}
        diagnostics = self._diagnostics(request_ids)
        routing_rows = []
        for item in diagnostics:
            decision = item.get("routing_decision") or {}
            context = item.get("routing_context") or {}
            routing_rows.append({
                "request_id": item.get("logical_request_id"),
                "request_class": context.get("request_class"),
                "input_tokens": context.get("input_tokens"),
                "requested_output_tokens": context.get("requested_output_tokens"),
                "selected_prefill_endpoint_id": item.get("selected_prefill_endpoint_id"),
                "selected_decode_endpoint_id": item.get("selected_decode_endpoint_id"),
                "routing_reason": decision.get("reason"),
                "candidate_states_json": json.dumps(
                    decision.get("candidate_states"), sort_keys=True
                ),
                "selected_state_json": json.dumps(
                    decision.get("selected_state"), sort_keys=True
                ),
            })
        _write_csv(self.run_dir / "routing_decisions.csv", routing_rows)
        endpoint_rows = []
        route_class_rows = []
        for load in sorted({str(row["load_point"]) for row in rows}):
            for policy in POLICIES:
                policy_windows = [
                    row for row in rows
                    if row["load_point"] == load and row["policy"] == policy
                ]
                policy_requests = [
                    row for row in requests
                    if row["load_point"] == load and row["policy"] == policy
                ]
                if not policy_windows:
                    continue
                for endpoint_id in ENDPOINTS:
                    selection_key = (
                        "selected_prefill_endpoint_id"
                        if endpoint_id.startswith("P")
                        else "selected_decode_endpoint_id"
                    )
                    assigned = [
                        row for row in policy_requests
                        if row.get(selection_key) == endpoint_id
                    ]
                    endpoint_rows.append({
                        "load_point": load, "policy": policy,
                        "endpoint_id": endpoint_id,
                        "request_count": len(assigned),
                        "input_tokens": sum(int(row["input_len"]) for row in assigned),
                        "requested_output_tokens": sum(
                            int(row["output_len"]) for row in assigned
                        ),
                        "gross_energy_j": sum(
                            float(row["%s_energy_j" % endpoint_id])
                            for row in policy_windows
                        ),
                        "frequency_residency_json": json.dumps(
                            segment_residency(
                                [
                                    row for row in self.segment_rows
                                    if row.get("policy") == policy
                                    and row.get("load_point") == load
                                ],
                                endpoint_id,
                            ),
                            sort_keys=True,
                        ),
                    })
                for workload_class in ("SS", "SL", "LS", "LL"):
                    class_requests = [
                        row for row in policy_requests
                        if row.get("workload_class") == workload_class
                    ]
                    for pair in PAIRS:
                        route_class_rows.append({
                            "load_point": load, "policy": policy,
                            "workload_class": workload_class,
                            "prefill_endpoint_id": pair[0],
                            "decode_endpoint_id": pair[1],
                            "request_count": sum(
                                row.get("selected_prefill_endpoint_id") == pair[0]
                                and row.get("selected_decode_endpoint_id") == pair[1]
                                for row in class_requests
                            ),
                        })
        _write_csv(self.run_dir / "stage_d1_endpoint_summary.csv", endpoint_rows)
        _write_csv(self.run_dir / "stage_d1_route_by_class.csv", route_class_rows)
        reason_rows = []
        for reason in sorted({str(row["routing_reason"]) for row in routing_rows}):
            reason_rows.append({
                "routing_reason": reason,
                "request_count": sum(
                    row["routing_reason"] == reason for row in routing_rows
                ),
            })
        _write_csv(self.run_dir / "stage_d1_routing_reason_summary.csv", reason_rows)
        repeat_rows = []
        for load in sorted({str(row["load_point"]) for row in rows}):
            for repeat in range(1, repetitions + 1):
                for policy in POLICIES:
                    selected = [
                        row for row in rows
                        if row["load_point"] == load
                        and int(row["repeat"]) == repeat
                        and row["policy"] == policy
                    ]
                    selected_requests = [
                        row for row in requests
                        if row["load_point"] == load
                        and int(row["repeat"]) == repeat
                        and row["policy"] == policy
                    ]
                    if not selected:
                        continue
                    energy = sum(
                        float(row["total_gross_energy_j"]) for row in selected
                    )
                    ttft = [float(row["ttft_ms"]) for row in selected_requests]
                    tpot = [float(row["tpot_ms"]) for row in selected_requests]
                    repeat_rows.append({
                        "load_point": load, "repeat": repeat, "policy": policy,
                        "request_count": len(selected_requests),
                        "total_gross_energy_j": energy,
                        "joules_per_request": energy / len(selected_requests),
                        "ttft_attainment": sum(value <= 600 for value in ttft) / len(ttft),
                        "tpot_attainment": sum(value <= 65 for value in tpot) / len(tpot),
                        "slo_contract_pass": (
                            sum(value <= 600 for value in ttft) / len(ttft) >= 0.95
                            and sum(value <= 65 for value in tpot) / len(tpot) >= 0.95
                        ),
                    })
        static_by_pair = {
            (row["load_point"], int(row["repeat"])): row
            for row in repeat_rows if row["policy"] == "STATIC-HIGH-RR"
        }
        for row in repeat_rows:
            baseline = static_by_pair.get((row["load_point"], int(row["repeat"])))
            if baseline is not None and row["policy"] != "STATIC-HIGH-RR":
                raw = (
                    float(baseline["joules_per_request"])
                    - float(row["joules_per_request"])
                ) / float(baseline["joules_per_request"])
                row["paired_energy_saving_fraction_raw"] = raw
                row["paired_energy_saving_fraction_slo_valid"] = (
                    raw if row["slo_contract_pass"] else None
                )
        _write_csv(self.run_dir / "stage_d1_repeat_summary.csv", repeat_rows)
        signatures: dict[tuple[str, int, int], set[str]] = {}
        for row in rows:
            signatures.setdefault((
                str(row["load_point"]), int(row["repeat"]),
                int(row["window_index"]),
            ), set()).add(str(row["trace_signature"]))
        matched = bool(signatures) and all(len(value) == 1 for value in signatures.values())
        hard = bool(rows) and all(_bool(row["window_hard_gate_valid"]) for row in rows)
        context_valid = len(routing_rows) == len(requests) and all(
            row["request_class"] in {"SS", "SL", "LS", "LL"}
            and int(row["input_tokens"] or 0) > 0
            and int(row["requested_output_tokens"] or 0) > 0
            for row in routing_rows
        )
        observed_pairs = {
            (str(row["selected_prefill_endpoint_id"]), str(row["selected_decode_endpoint_id"]))
            for row in requests
        }
        all_pairs = observed_pairs == set(PAIRS)
        energy_safe = all(
            row["routing_reason"] != "ENERGY_PREFERENCE"
            or json.loads(row["selected_state_json"] or "{}").get("known_safe") is True
            for row in routing_rows
        )
        pressure_spill = any(
            row["routing_reason"] == "LOWER_PRESSURE" for row in routing_rows
        )
        energy_preference = any(
            row["routing_reason"] == "ENERGY_PREFERENCE" for row in routing_rows
        )
        preflight_pass = (
            self.stage == "preflight" and run_error is None and hard
            and context_valid and all_pairs and restoration_valid
            and energy_safe and energy_preference and pressure_spill
        )
        summaries = []
        for load in {str(row["load_point"]) for row in rows}:
            policies = {str(row["policy"]) for row in rows if row["load_point"] == load}
            for policy in policies:
                selected = [
                    row for row in rows
                    if row["load_point"] == load and row["policy"] == policy
                ]
                selected_requests = [
                    row for row in requests
                    if row["load_point"] == load and row["policy"] == policy
                ]
                energy = sum(float(row["total_gross_energy_j"]) for row in selected)
                ttft = [float(row["ttft_ms"]) for row in selected_requests]
                tpot = [float(row["tpot_ms"]) for row in selected_requests]
                summaries.append({
                    "load_point": load, "policy": policy,
                    "request_count": len(selected_requests),
                    "total_gross_energy_j": energy,
                    "joules_per_request": energy / len(selected_requests),
                    "ttft_p95_ms": percentile(ttft, 0.95),
                    "ttft_p99_ms": percentile(ttft, 0.99),
                    "ttft_attainment": sum(value <= 600 for value in ttft) / len(ttft),
                    "tpot_p95_ms": percentile(tpot, 0.95),
                    "tpot_p99_ms": percentile(tpot, 0.99),
                    "tpot_attainment": sum(value <= 65 for value in tpot) / len(tpot),
                    "slo_contract_pass": (
                        sum(value <= 600 for value in ttft) / len(ttft) >= 0.95
                        and sum(value <= 65 for value in tpot) / len(tpot) >= 0.95
                    ),
                    "throughput_rps_mean": statistics.fmean(
                        float(row["completed_rate_rps"]) for row in selected
                    ),
                    "fast_recovery_count": sum(
                        row.get("policy") == policy
                        and row.get("action") == "FAST_RECOVER"
                        for row in self.safety_rows
                    ),
                    "route_distribution_json": json.dumps({
                        "%s->%s" % pair: sum(
                            row.get("selected_prefill_endpoint_id") == pair[0]
                            and row.get("selected_decode_endpoint_id") == pair[1]
                            for row in selected_requests
                        ) for pair in PAIRS
                    }, sort_keys=True),
                    "frequency_residency_json": json.dumps({
                        endpoint_id: segment_residency(
                            [row for row in self.segment_rows if row.get("policy") == policy],
                            endpoint_id,
                        ) for endpoint_id in ENDPOINTS
                    }, sort_keys=True),
                })
        _write_csv(self.run_dir / "stage_d1_policy_summary.csv", summaries)
        summary = {
            "schema_version": 1,
            "phase": "Stage_D1_per_endpoint_routing_isolation",
            "stage": self.stage,
            "valid": (
                run_error is None and hard and matched and context_valid
                and energy_safe and restoration_valid
            ),
            "error": run_error,
            "preflight_pass": preflight_pass,
            "matched_trace_audit": matched,
            "window_hard_gate_audit": hard,
            "request_context_before_routing_audit": context_valid,
            "all_compatible_pairs_observed_audit": all_pairs,
            "energy_preference_known_safe_only_audit": energy_safe,
            "energy_preference_observed_audit": energy_preference,
            "pressure_spillover_observed_audit": pressure_spill,
            "final_high_restoration_audit": restoration_valid,
            "policies": list(POLICIES),
            "repetitions": repetitions,
            "windows_per_block": windows,
            "policy_results": summaries,
            "claim_boundary": (
                "runtime-feedback stationary D1 only; endpoint energy is "
                "window-amortized routing evidence, not physical request attribution"
            ),
        }
        _write_json(self.run_dir / "stage_d1_summary.json", summary)
        lines = [
            "# XPYD Stage D1", "",
            "Structural verdict: **%s**" % ("PASS" if summary["valid"] else "FAIL"),
            "", "| Load | Policy | J/request | TTFT attainment | TPOT attainment |",
            "|---|---|---:|---:|---:|",
        ]
        for row in summaries:
            lines.append("| %s | %s | %.2f | %.1f%% | %.1f%% |" % (
                row["load_point"], row["policy"], row["joules_per_request"],
                100 * row["ttft_attainment"], 100 * row["tpot_attainment"],
            ))
        (self.run_dir / "stage_d1_summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        if self.stage == "preflight" and run_error is None and not preflight_pass:
            raise StageC2Error("D1 preflight routing/asymmetry audit failed")

    def run(self) -> Path:
        return self._run_experiment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("preflight", "full"), required=True)
    args = parser.parse_args()
    runtime, experiment = load_configs(args.runtime_config, args.experiment_config)
    StageD1Harness(
        runtime, experiment, run_id=args.run_id, stage=args.stage
    ).run()


if __name__ == "__main__":
    main()
