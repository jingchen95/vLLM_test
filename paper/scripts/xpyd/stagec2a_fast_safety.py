"""Stage C2a stationary isolation of event-driven DVFS safety recovery."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.phase3c_substrate import (
    _read_json,
    _write_json,
    load_config as load_runtime_config,
)
from xpyd.phase3d_control import (
    NvidiaSmiClockBackend,
    PerEndpointClockActuator,
)
from xpyd.stagec0_capacity import percentile
from xpyd.stagec1_stationary import (
    ControllerContext,
    _read_csv,
    _trace_signature,
    _write_csv,
)
from xpyd.stagec2_dynamic import (
    C2AccountingWindowHarness,
    ClockSegmentLedger,
    FastSafetyRuntime,
    StageC2Error,
    StageC2Harness,
    _bool,
)
from xpyd.workload_harness import build_trace, load_config as load_workload_config


POLICIES = (
    "STATIC-HIGH",
    "WINDOW-BOUNDED-FEEDBACK",
    "FAST-SAFETY-FEEDBACK",
)


class C2AAccountingWindowHarness(C2AccountingWindowHarness):
    def _metadata_overrides(self) -> dict[str, Any]:
        value = super()._metadata_overrides()
        value.update({
            "phase": "Stage_C2a_stationary_fast_safety_isolation",
            "safety_recovery": "event-driven for FAST-SAFETY-FEEDBACK only",
            "efficiency_control": "evidence-window driven and identical to C1c",
        })
        return value

    def _claim_boundary(self) -> str:
        return "C2a stationary safety-recovery granularity only; no context transition or routing claim"


def policy_order(repeat: int) -> tuple[str, ...]:
    offset = (int(repeat) - 1) % len(POLICIES)
    return POLICIES[offset:] + POLICIES[:offset]


def segment_residency(
    segments: Sequence[Mapping[str, Any]], endpoint_id: str,
) -> dict[str, float]:
    selected = [
        row for row in segments
        if row.get("endpoint_id") == endpoint_id
        and row.get("segment_kind") == "STABLE"
    ]
    total = sum(float(row["duration_s"]) for row in selected)
    frequencies = sorted({int(row["observed_readback_frequency_mhz"]) for row in selected})
    return {
        str(frequency): (
            sum(float(row["duration_s"]) for row in selected
                if int(row["observed_readback_frequency_mhz"]) == frequency) / total
            if total else 0.0
        )
        for frequency in frequencies
    }


def load_configs(runtime_path: Path, experiment_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_runtime_config(runtime_path)
    experiment = load_workload_config(experiment_path)
    settings = experiment.get("stagec2a")
    if not isinstance(settings, Mapping):
        raise StageC2Error("stagec2a settings are required")
    if settings.get("policies") != list(POLICIES):
        raise StageC2Error("C2a compares exactly STATIC, WINDOW-BOUNDED, and FAST-SAFETY")
    if settings.get("routing") != "fixed four-pair round-robin":
        raise StageC2Error("C2a routing must remain fixed four-pair round robin")
    if int(settings.get("max_concurrency", 0)) != 4:
        raise StageC2Error("C2a requires concurrency 4")
    if settings.get("pool_frequency_state") != {"P": ["P0", "P1"], "D": ["D0", "D1"]}:
        raise StageC2Error("C2a requires shared P and D frequency states")
    if settings.get("slo") != {"ttft_ms": 600.0, "tpot_ms": 65.0, "attainment": 0.95}:
        raise StageC2Error("C2a must retain the TIGHT 600/65 contract")
    if any(bool(settings.get(key, False)) for key in (
        "request_aware_routing", "per_endpoint_dvfs", "predictor", "oracle",
    )):
        raise StageC2Error("C2a isolation forbids routing, asymmetric DVFS, models, and oracle input")
    expected_loads = [
        {"id": "MEDIUM", "scenario": "stationary_medium", "offered_rate_rps": 0.10},
        {"id": "HIGH-SUSTAINABLE", "scenario": "stationary_high_sustainable", "offered_rate_rps": 0.20},
    ]
    if settings.get("load_points") != expected_loads:
        raise StageC2Error("C2a must reuse the exact C1 MEDIUM/HIGH load points")
    endpoints = {str(item["endpoint_id"]): str(item["node"]) for item in runtime["endpoints"]}
    if endpoints != {"P0": "neptune", "P1": "neptune", "D0": "europa", "D1": "europa"}:
        raise StageC2Error("C2a topology must remain Neptune L40S + Europa L4")
    runtime["phase3d"] = {
        "minimum_dwell_s": float(settings.get("minimum_dwell_s", 1.0)),
        "feedback": {
            "ttft_slo_ms": 600.0,
            "tpot_slo_ms": 65.0,
            "telemetry_max_age_s": float(settings["fast_safety"]["telemetry_max_age_s"]),
        },
    }
    return runtime, experiment


class StageC2AFastSafetyHarness(StageC2Harness):
    def __init__(
        self, runtime: Mapping[str, Any], experiment: Mapping[str, Any],
        run_id: Optional[str] = None, backend: Optional[NvidiaSmiClockBackend] = None,
        stage: str = "preflight",
    ) -> None:
        if stage not in {"preflight", "full"}:
            raise StageC2Error("C2a stage must be preflight or full")
        wrapped = copy.deepcopy(dict(experiment))
        wrapped["stagec2"] = copy.deepcopy(wrapped["stagec2a"])
        super().__init__(runtime, wrapped, run_id=run_id, backend=backend, stage=stage)
        self.stage = stage
        self.settings = wrapped["stagec2a"]
        self.experiment = wrapped
        self.run_dir = Path(os.path.expandvars(str(wrapped["output_root"]))) / self.run_id

    def _selected_loads(self) -> list[dict[str, Any]]:
        if self.stage == "full":
            return [dict(item) for item in self.settings["load_points"]]
        wanted = str(self.settings["preflight_load_point"])
        return [dict(next(item for item in self.settings["load_points"] if item["id"] == wanted))]

    def _run_experiment(self) -> Path:
        if self.stage == "full":
            self._check_accepted("XPYD_STAGEC2_ACCEPTED_PREFLIGHT", "preflight_pass")
        if self.run_dir.exists():
            raise StageC2Error("run directory already exists")
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        expected = self.settings["frequency_levels_mhz"]
        if {key: self._levels(value) for key, value in capabilities.items()} != expected:
            raise StageC2Error("discovered clocks differ from the frozen C1/C2a action space")
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
        rows: list[dict[str, Any]] = []
        requests_all: list[dict[str, Any]] = []
        restoration_valid = False
        run_error: Optional[str] = None
        try:
            for point in self._selected_loads():
                self.settings["offered_rate_rps"] = float(point["offered_rate_rps"])
                for repeat in range(1, repetitions + 1):
                    for policy in policy_order(repeat):
                        block_id = "r%02d-%s-%s" % (
                            repeat, str(point["id"]).lower(), policy.lower()
                        )
                        registry, telemetry, scheduler = self._build_scheduler(capabilities)
                        controllers = None if policy == "STATIC-HIGH" else self._controllers(capabilities)
                        context = ControllerContext(
                            registry, telemetry, scheduler, dict(high), controllers
                        )
                        self._actuate_targets(
                            actuator, context, high, reason="stagec2a_cold_start_high",
                            policy=policy, block_id=block_id,
                        )
                        previous = None
                        previous_requests: list[dict[str, Any]] = []
                        for window_index in range(1, windows + 1):
                            if controllers is not None and previous is not None:
                                self._efficiency_control(
                                    context, capabilities, actuator, previous,
                                    previous_requests, policy, block_id, window_index,
                                )
                            trace = build_trace(
                                self.experiment, str(point["scenario"]),
                                "c2a-%s-r%02d-w%02d" % (
                                    str(point["id"]).lower(), repeat, window_index
                                ),
                            )
                            signature = _trace_signature(trace)
                            window_id = "%s-w%02d" % (block_id, window_index)
                            ledger = ClockSegmentLedger(window_id)
                            self.active_ledger = ledger
                            safety = None
                            if controllers is not None:
                                safety = FastSafetyRuntime(
                                    controllers,
                                    lambda pool, recommendation, a=actuator, c=context, p=policy, b=block_id: self._fast_recover(
                                        pool, recommendation, actuator=a, context=c,
                                        policy=p, block_id=b,
                                    ),
                                    self.settings, policy=policy, block_id=block_id,
                                    window_index=window_index,
                                    enabled=policy == "FAST-SAFETY-FEEDBACK",
                                )
                            config = self._window_config(context.targets, windows_root)
                            window_dir = C2AAccountingWindowHarness(
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
                            row["window_start_wall_s"] = min(
                                float(item["send_unix_s"]) for item in request_rows
                            )
                            row["window_end_wall_s"] = max(
                                float(item["complete_unix_s"]) for item in request_rows
                            )
                            for item in request_rows:
                                item["load_point"] = point["id"]
                            rows.append(row)
                            requests_all.extend(request_rows)
                            previous, previous_requests = row, request_rows
                        if controllers is not None and previous is not None:
                            self._efficiency_control(
                                context, capabilities, actuator, previous,
                                previous_requests, policy, block_id, windows + 1,
                                allow_exploration=False,
                            )
                            for pool, controller in controllers.items():
                                state = controller.state
                                self.search_rows.append({
                                    "policy": policy, "block_id": block_id,
                                    "repeat": repeat, "load_point": point["id"],
                                    "pool": pool,
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
                    actuator, context, high, reason="stagec2a_final_restore_high",
                    policy="RESTORE", block_id="final",
                )
                restoration_valid = all(
                    actuator.read(key).graphics_clock_mhz == value
                    for key, value in high.items()
                )
            if self.run_dir.exists():
                self._write_c2a_reports(
                    rows, requests_all, restoration_valid, repetitions,
                    windows, run_error,
                )
        print(self.run_dir)
        return self.run_dir

    def _unsafe_exposure_rows(
        self, rows: Sequence[Mapping[str, Any]],
        requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        output = []
        for probe in self.decision_rows:
            if probe.get("decision") != "EXPLORE_DOWN":
                continue
            policy = str(probe["policy"])
            block_id = str(probe["block_id"])
            pool = str(probe["pool"])
            window_index = int(probe["before_window_index"])
            target = int(probe["requested_freq_mhz"])
            window = next((
                row for row in rows
                if row["block_id"] == block_id
                and int(row["window_index"]) == window_index
            ), None)
            if window is None:
                continue
            request_rows = [
                row for row in requests
                if row["block_id"] == block_id
                and int(row["window_index"]) == window_index
            ]
            hard_trigger = "HARD_TTFT_VIOLATION" if pool == "P" else "HARD_TPOT_VIOLATION"
            fast = next((
                row for row in self.safety_rows
                if row.get("block_id") == block_id
                and int(row.get("window_index", 0)) == window_index
                and row.get("pool") == pool
                and row.get("trigger") == hard_trigger
                and row.get("action") == "FAST_RECOVER"
            ), None)
            boundary = next((
                row for row in self.decision_rows
                if row.get("block_id") == block_id
                and int(row.get("evidence_window_index", 0)) == window_index
                and row.get("pool") == pool
                and row.get("evidence_classification") == "UNSAFE"
                and row.get("action") in {"RECOVER_HIGH", "STEP_UP"}
            ), None)
            recovery = fast if fast is not None else boundary
            if recovery is None:
                continue
            first_violation = min((
                float(row["first_token_unix_s"] if pool == "P" else row["complete_unix_s"])
                for row in request_rows
                if float(row["ttft_ms"] if pool == "P" else row["tpot_ms"])
                > (600.0 if pool == "P" else 65.0)
            ), default=None)
            recovery_readback = float(recovery["readback_timestamp_s"])
            endpoint = "P0" if pool == "P" else "D0"
            candidate_start = min((
                float(row["segment_start_wall_s"])
                for row in self.segment_rows
                if row.get("block_id") == block_id
                and int(row.get("window_index", 0)) == window_index
                and row.get("endpoint_id") == endpoint
                and row.get("segment_kind") == "STABLE"
                and int(row.get("requested_frequency_mhz")) == target
            ), default=float(window["window_start_wall_s"]))
            output.append({
                "policy": policy, "block_id": block_id,
                "repeat": window["repeat"], "load_point": window["load_point"],
                "pool": pool, "window_index": window_index,
                "candidate_frequency_mhz": target,
                "recovery_frequency_mhz": recovery.get("target_mhz", recovery.get("requested_freq_mhz")),
                "recovery_granularity": "EVENT_DRIVEN" if fast is not None else "EVIDENCE_WINDOW_BOUNDARY",
                "trigger": recovery.get("trigger"),
                "trigger_request_id": recovery.get("request_id"),
                "candidate_start_wall_s": candidate_start,
                "first_hard_violation_wall_s": first_violation,
                "recovery_readback_wall_s": recovery_readback,
                "unsafe_probe_duration_s": recovery_readback - candidate_start,
                "first_violation_to_readback_s": (
                    recovery_readback - first_violation
                    if first_violation is not None else None
                ),
                "requests_exposed_before_readback": sum(
                    float(row["send_unix_s"]) <= recovery_readback
                    for row in request_rows
                ),
                "requests_sent_after_first_violation_before_readback": sum(
                    first_violation is not None
                    and first_violation < float(row["send_unix_s"]) <= recovery_readback
                    for row in request_rows
                ),
                "hard_violation_count_in_probe_window": sum(
                    float(row["ttft_ms"] if pool == "P" else row["tpot_ms"])
                    > (600.0 if pool == "P" else 65.0)
                    for row in request_rows
                ),
            })
        return output

    def _transition_timeline(self) -> list[dict[str, Any]]:
        output = []
        for action in self.action_rows:
            endpoint = str(action["endpoint_id"])
            pool = "P" if endpoint.startswith("P") else "D"
            event = min((
                row for row in self.safety_rows
                if row.get("policy") == action.get("policy")
                and row.get("block_id") == action.get("block_id")
                and row.get("pool") == pool
                and row.get("action") == "FAST_RECOVER"
                and abs(float(row.get("action_timestamp_s", 0)) - float(action["timestamp_unix_s"])) < 5.0
            ), key=lambda row: abs(float(row["action_timestamp_s"]) - float(action["timestamp_unix_s"])), default=None)
            output.append({
                "policy": action.get("policy"), "block_id": action.get("block_id"),
                "window_id": action.get("window_id"), "endpoint_id": endpoint,
                "pool": pool,
                "causing_event": event.get("trigger") if event else "EVIDENCE_EPOCH_OR_LIFECYCLE",
                "causing_request_id": event.get("request_id") if event else None,
                "decision_timestamp_s": action.get("decision_timestamp_s"),
                "action_timestamp_s": action.get("timestamp_unix_s"),
                "readback_timestamp_s": action.get("readback_timestamp_unix_s"),
                "previous_frequency_mhz": action.get("previous_requested_freq_mhz"),
                "requested_frequency_mhz": action.get("requested_freq_mhz"),
                "observed_frequency_mhz": action.get("observed_freq_after_mhz"),
                "reason": action.get("reason"),
                "readback_valid": action.get("readback_valid"),
            })
        return output

    def _write_c2a_reports(
        self, rows: list[dict[str, Any]], requests: list[dict[str, Any]],
        restoration_valid: bool, repetitions: int, windows: int,
        run_error: Optional[str],
    ) -> None:
        for block_id in sorted({str(row["block_id"]) for row in rows}):
            energy = 0.0
            request_count = 0
            for row in sorted(
                (item for item in rows if item["block_id"] == block_id),
                key=lambda item: int(item["window_index"]),
            ):
                energy += float(row["total_gross_energy_j"])
                request_count += int(row["request_count"])
                row["cumulative_energy_j"] = energy
                row["cumulative_requests"] = request_count
                row["cumulative_joules_per_request"] = energy / request_count
        exposure = self._unsafe_exposure_rows(rows, requests)
        transitions = self._transition_timeline()
        _write_csv(self.run_dir / "stagec2a_measurements.csv", rows)
        _write_csv(self.run_dir / "stagec2a_requests.csv", requests)
        _write_csv(self.run_dir / "controller_decisions.csv", self.decision_rows)
        _write_csv(self.run_dir / "safety_events.csv", self.safety_rows)
        _write_csv(self.run_dir / "clock_actions.csv", self.action_rows)
        _write_csv(self.run_dir / "clock_segments.csv", self.segment_rows)
        _write_csv(self.run_dir / "clock_transition_timeline.csv", transitions)
        _write_csv(self.run_dir / "unsafe_probe_exposure.csv", exposure)
        _write_csv(self.run_dir / "stagec2a_search_state.csv", self.search_rows)

        repeat_rows = []
        for load in self._selected_loads():
            for repeat in range(1, repetitions + 1):
                for policy in POLICIES:
                    selected = [
                        row for row in rows
                        if row["load_point"] == load["id"]
                        and int(row["repeat"]) == repeat and row["policy"] == policy
                    ]
                    selected_requests = [
                        row for row in requests
                        if row["load_point"] == load["id"]
                        and int(row["repeat"]) == repeat and row["policy"] == policy
                    ]
                    if not selected:
                        continue
                    ttft = [float(row["ttft_ms"]) for row in selected_requests]
                    tpot = [float(row["tpot_ms"]) for row in selected_requests]
                    energy = sum(float(row["total_gross_energy_j"]) for row in selected)
                    policy_exposure = [
                        row for row in exposure
                        if row["policy"] == policy and row["load_point"] == load["id"]
                        and int(row["repeat"]) == repeat
                    ]
                    block_ids = {item["block_id"] for item in selected}
                    policy_segments = [
                        row for row in self.segment_rows
                        if row.get("block_id") in block_ids
                    ]
                    repeat_rows.append({
                        "load_point": load["id"], "repeat": repeat, "policy": policy,
                        "request_count": len(selected_requests),
                        "total_gross_energy_j": energy,
                        "joules_per_request": energy / len(selected_requests),
                        "ttft_p95_ms": percentile(ttft, 0.95),
                        "ttft_p99_ms": percentile(ttft, 0.99),
                        "ttft_attainment": sum(value <= 600 for value in ttft) / len(ttft),
                        "ttft_violation_count": sum(value > 600 for value in ttft),
                        "tpot_p95_ms": percentile(tpot, 0.95),
                        "tpot_p99_ms": percentile(tpot, 0.99),
                        "tpot_attainment": sum(value <= 65 for value in tpot) / len(tpot),
                        "tpot_violation_count": sum(value > 65 for value in tpot),
                        "slo_contract_pass": (
                            sum(value <= 600 for value in ttft) / len(ttft) >= 0.95
                            and sum(value <= 65 for value in tpot) / len(tpot) >= 0.95
                        ),
                        "completed_rate_rps_mean": statistics.fmean(float(row["completed_rate_rps"]) for row in selected),
                        "admission_delay_p95_ms": percentile([float(row["client_admission_delay_ms"]) for row in selected_requests], 0.95),
                        "probe_count": sum(
                            row.get("block_id") in block_ids
                            and row.get("decision") == "EXPLORE_DOWN"
                            for row in self.decision_rows
                        ),
                        "recovery_count": len(policy_exposure),
                        "unsafe_probe_duration_s_total": sum(float(row["unsafe_probe_duration_s"]) for row in policy_exposure),
                        "unsafe_probe_requests_exposed": sum(int(row["requests_exposed_before_readback"]) for row in policy_exposure),
                        "hard_violations_in_unsafe_probe_windows": sum(int(row["hard_violation_count_in_probe_window"]) for row in policy_exposure),
                        "first_violation_to_readback_s_mean": (
                            statistics.fmean(float(row["first_violation_to_readback_s"]) for row in policy_exposure)
                            if policy_exposure else None
                        ),
                        "P_frequency_residency_json": json.dumps(
                            segment_residency(policy_segments, "P0"), sort_keys=True
                        ),
                        "D_frequency_residency_json": json.dumps(
                            segment_residency(policy_segments, "D0"), sort_keys=True
                        ),
                    })
        static = {
            (row["load_point"], int(row["repeat"])): row
            for row in repeat_rows if row["policy"] == "STATIC-HIGH"
        }
        for row in repeat_rows:
            baseline = static.get((row["load_point"], int(row["repeat"])))
            if baseline and row["policy"] != "STATIC-HIGH":
                saving = (
                    float(baseline["joules_per_request"]) - float(row["joules_per_request"])
                ) / float(baseline["joules_per_request"])
                row["paired_energy_saving_fraction_raw"] = saving
                row["paired_energy_saving_fraction"] = saving if row["slo_contract_pass"] else None
        _write_csv(self.run_dir / "stagec2a_repeat_summary.csv", repeat_rows)

        aggregate_rows = []
        for load in self._selected_loads():
            for policy in POLICIES:
                selected = [
                    row for row in repeat_rows
                    if row["load_point"] == load["id"] and row["policy"] == policy
                ]
                if not selected:
                    continue
                count = sum(int(row["request_count"]) for row in selected)
                energy = sum(float(row["total_gross_energy_j"]) for row in selected)
                request_rows = [
                    row for row in requests
                    if row["load_point"] == load["id"] and row["policy"] == policy
                ]
                ttft = [float(row["ttft_ms"]) for row in request_rows]
                tpot = [float(row["tpot_ms"]) for row in request_rows]
                policy_segments = [
                    row for row in self.segment_rows
                    if row.get("load_point") == load["id"]
                    and row.get("policy") == policy
                ]
                aggregate_rows.append({
                    "load_point": load["id"], "policy": policy,
                    "request_count": count, "total_gross_energy_j": energy,
                    "joules_per_request": energy / count,
                    "joules_per_request_repeat_std": statistics.stdev(float(row["joules_per_request"]) for row in selected) if len(selected) > 1 else None,
                    "ttft_p95_ms": percentile(ttft, 0.95),
                    "ttft_p99_ms": percentile(ttft, 0.99),
                    "ttft_attainment": sum(value <= 600 for value in ttft) / len(ttft),
                    "ttft_violation_count": sum(value > 600 for value in ttft),
                    "tpot_p95_ms": percentile(tpot, 0.95),
                    "tpot_p99_ms": percentile(tpot, 0.99),
                    "tpot_attainment": sum(value <= 65 for value in tpot) / len(tpot),
                    "tpot_violation_count": sum(value > 65 for value in tpot),
                    "slo_contract_pass": (
                        sum(value <= 600 for value in ttft) / len(ttft) >= 0.95
                        and sum(value <= 65 for value in tpot) / len(tpot) >= 0.95
                    ),
                    "probe_count": sum(int(row["probe_count"]) for row in selected),
                    "recovery_count": sum(int(row["recovery_count"]) for row in selected),
                    "unsafe_probe_duration_s_total": sum(float(row["unsafe_probe_duration_s_total"]) for row in selected),
                    "unsafe_probe_requests_exposed": sum(int(row["unsafe_probe_requests_exposed"]) for row in selected),
                    "hard_violations_in_unsafe_probe_windows": sum(int(row["hard_violations_in_unsafe_probe_windows"]) for row in selected),
                    "P_frequency_residency_json": json.dumps(
                        segment_residency(policy_segments, "P0"), sort_keys=True
                    ),
                    "D_frequency_residency_json": json.dumps(
                        segment_residency(policy_segments, "D0"), sort_keys=True
                    ),
                })
        _write_csv(self.run_dir / "stagec2a_policy_summary.csv", aggregate_rows)

        signatures: dict[tuple[str, int, int], set[str]] = {}
        for row in rows:
            signatures.setdefault((
                str(row["load_point"]), int(row["repeat"]), int(row["window_index"])
            ), set()).add(str(row["trace_signature"]))
        matched = bool(signatures) and all(len(value) == 1 for value in signatures.values())
        window_valid = bool(rows) and all(_bool(row["window_hard_gate_valid"]) for row in rows)
        segment_valid = bool(self.segment_rows) and all(float(row["duration_s"]) > 0 for row in self.segment_rows)
        fast_p = [row for row in exposure if row["policy"] == "FAST-SAFETY-FEEDBACK" and row["pool"] == "P"]
        fast_d = [row for row in exposure if row["policy"] == "FAST-SAFETY-FEEDBACK" and row["pool"] == "D"]
        fast_timing = bool(fast_p and fast_d) and all(
            float(row["first_violation_to_readback_s"]) >= 0
            and int(row["requests_exposed_before_readback"]) < 20
            for row in [*fast_p, *fast_d]
        )
        preflight_pass = (
            run_error is None and self.stage == "preflight" and matched
            and window_valid and segment_valid and restoration_valid and fast_timing
        )
        valid = run_error is None and matched and window_valid and segment_valid and restoration_valid
        summary = {
            "schema_version": 1,
            "phase": "Stage_C2a_stationary_fast_safety_isolation",
            "stage": self.stage, "valid": valid, "error": run_error,
            "preflight_pass": preflight_pass,
            "matched_trace_audit": matched,
            "window_hard_gate_audit": window_valid,
            "clock_segment_audit": segment_valid,
            "fast_p_recovery_audit": bool(fast_p),
            "fast_d_recovery_audit": bool(fast_d),
            "fast_recovery_before_window_boundary_audit": fast_timing,
            "final_high_restoration_audit": restoration_valid,
            "policies": list(POLICIES),
            "loads": self._selected_loads(), "repetitions": repetitions,
            "windows_per_block": windows,
            "policy_results": aggregate_rows,
            "unsafe_probe_exposure": exposure,
            "claim_boundary": "stationary safety-recovery granularity only; unchanged C1 exploration and no routing, asymmetric DVFS, TP, model, oracle, or request-energy attribution",
        }
        _write_json(self.run_dir / "stagec2a_summary.json", summary)
        lines = [
            "# XPYD Stage C2a fast-safety isolation", "",
            "Structural verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
            "| Load | Policy | J/request | TTFT attainment | TPOT attainment | Unsafe exposure requests |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in aggregate_rows:
            lines.append("| %s | %s | %.2f | %.1f%% | %.1f%% | %d |" % (
                row["load_point"], row["policy"], row["joules_per_request"],
                100 * row["ttft_attainment"], 100 * row["tpot_attainment"],
                row["unsafe_probe_requests_exposed"],
            ))
        (self.run_dir / "stagec2a_summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        if self.stage == "preflight" and run_error is None and not preflight_pass:
            raise StageC2Error("C2a preflight structural/fast-recovery audit failed")

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
    StageC2AFastSafetyHarness(
        runtime, experiment, run_id=args.run_id, stage=args.stage
    ).run()


if __name__ == "__main__":
    main()
