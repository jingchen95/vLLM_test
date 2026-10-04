"""Stage D1 per-endpoint DVFS and runtime-feedback pair-routing isolation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import copy
import json
import math
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


NORMALIZED_D1_REQUIRED_FIELDS = {
    "request_id",
    "prefill_endpoint_id",
    "decode_endpoint_id",
    "input_len",
    "requested_output_len",
    "workload_class",
    "ttft_ms",
    "tpot_ms",
    "send_unix_s",
    "first_token_unix_s",
    "complete_unix_s",
    "real_sse",
    "logical_request_id_propagated",
}


@dataclass(frozen=True)
class D1RequestRecord:
    """Canonical internal request schema used by all D1 control/report code."""

    request_id: str
    prefill_endpoint_id: str
    decode_endpoint_id: str
    requested_input_len: int
    requested_output_len: int
    actual_output_len: Optional[int]
    request_class: str
    ttft_ms: float
    tpot_ms: float
    send_unix_s: float
    first_token_unix_s: float
    complete_unix_s: float
    client_admission_delay_ms: Optional[float]
    real_sse: bool
    logical_request_id_propagated: bool

    def __post_init__(self) -> None:
        if not self.request_id:
            raise StageC2Error("D1 canonical request_id must be non-empty")
        if self.prefill_endpoint_id not in {"P0", "P1"}:
            raise StageC2Error("D1 canonical prefill endpoint is invalid")
        if self.decode_endpoint_id not in {"D0", "D1"}:
            raise StageC2Error("D1 canonical decode endpoint is invalid")
        if self.requested_input_len <= 0 or self.requested_output_len <= 0:
            raise StageC2Error("D1 canonical requested token counts must be positive")
        if self.actual_output_len is not None and self.actual_output_len < 0:
            raise StageC2Error("D1 canonical actual output length cannot be negative")
        if self.request_class not in {"SS", "SL", "LS", "LL"}:
            raise StageC2Error("D1 canonical request class is invalid")
        if not all(math.isfinite(value) for value in (
            self.ttft_ms, self.tpot_ms, self.send_unix_s,
            self.first_token_unix_s, self.complete_unix_s,
        )):
            raise StageC2Error("D1 canonical latency/timing values must be finite")
        if not self.send_unix_s <= self.first_token_unix_s <= self.complete_unix_s:
            raise StageC2Error("D1 canonical request timestamps are out of order")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise StageC2Error("D1 normalized request schema missing required field: %s" % key)
    return str(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _required_bool(row: Mapping[str, Any], key: str) -> bool:
    value = _required_text(row, key).strip().lower()
    if value not in {"true", "false"}:
        raise StageC2Error("D1 normalized request schema has invalid %s" % key)
    return value == "true"


def normalize_d1_request(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map one Phase3C normalized CSV row into the sole D1 request schema."""
    missing = sorted(
        key for key in NORMALIZED_D1_REQUIRED_FIELDS
        if row.get(key) is None or str(row.get(key)).strip() == ""
    )
    if missing:
        raise StageC2Error(
            "D1 normalized request schema missing required fields: %s"
            % ", ".join(missing)
        )
    try:
        actual_raw = row.get("observed_output_tokens")
        record = D1RequestRecord(
            request_id=_required_text(row, "request_id"),
            prefill_endpoint_id=_required_text(row, "prefill_endpoint_id"),
            decode_endpoint_id=_required_text(row, "decode_endpoint_id"),
            requested_input_len=int(_required_text(row, "input_len")),
            requested_output_len=int(_required_text(row, "requested_output_len")),
            actual_output_len=(
                int(actual_raw)
                if actual_raw is not None and str(actual_raw).strip() != ""
                else None
            ),
            request_class=_required_text(row, "workload_class"),
            ttft_ms=float(_required_text(row, "ttft_ms")),
            tpot_ms=float(_required_text(row, "tpot_ms")),
            send_unix_s=float(_required_text(row, "send_unix_s")),
            first_token_unix_s=float(_required_text(row, "first_token_unix_s")),
            complete_unix_s=float(_required_text(row, "complete_unix_s")),
            client_admission_delay_ms=_optional_float(
                row.get("client_admission_delay_ms")
            ),
            real_sse=_required_bool(row, "real_sse"),
            logical_request_id_propagated=_required_bool(
                row, "logical_request_id_propagated"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise StageC2Error("D1 normalized request schema contains invalid values") from exc
    value = record.as_dict()
    # Preserve experiment coordinates only after the request contract validates.
    for key in ("block_id", "policy", "repeat", "window_index", "context", "load_point"):
        if key in row:
            value[key] = row[key]
    return value


def normalize_d1_requests(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise StageC2Error("D1 normalized request input is empty")
    return [normalize_d1_request(row) for row in rows]


def endpoint_evidence_summary(
    requests: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build exact endpoint-local evidence from canonical D1 records."""
    output: dict[str, dict[str, Any]] = {}
    for endpoint_id in ENDPOINTS:
        phase = ENDPOINT_PHASE[endpoint_id]
        key = "prefill_endpoint_id" if phase == "P" else "decode_endpoint_id"
        selected = [row for row in requests if row[key] == endpoint_id]
        latency_key = "ttft_ms" if phase == "P" else "tpot_ms"
        latency = [float(row[latency_key]) for row in selected]
        energy = metrics.get("%s_energy_j" % endpoint_id)
        output[endpoint_id] = {
            "request_count": len(selected),
            "class_counts": {
                request_class: sum(
                    row["request_class"] == request_class for row in selected
                ) for request_class in ("SS", "SL", "LS", "LL")
            },
            "requested_input_tokens": sum(
                int(row["requested_input_len"]) for row in selected
            ),
            "requested_output_tokens": sum(
                int(row["requested_output_len"]) for row in selected
            ),
            "actual_output_tokens": sum(
                int(row["actual_output_len"] or 0) for row in selected
            ),
            "latency_metric": "TTFT" if phase == "P" else "TPOT",
            "latency_count": len(latency),
            "latency_mean_ms": statistics.fmean(latency) if latency else None,
            "latency_p95_ms": percentile(latency, 0.95) if latency else None,
            "gross_energy_j": float(energy) if energy is not None else None,
            "gross_energy_j_per_assigned_request": (
                float(energy) / len(selected)
                if energy is not None and selected else None
            ),
        }
    return output


def d1_context_signature(
    phase: str,
    requests: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> str:
    """Derive controller context exclusively from canonical D1 fields."""
    if not requests:
        return "%s:unknown" % phase
    if phase == "P":
        boundary = int(settings["context_inference"]["long_input_tokens"])
        fraction = sum(
            int(row["requested_input_len"]) > boundary for row in requests
        ) / len(requests)
        threshold = float(
            settings["context_inference"]["prefill_heavy_fraction"]
        )
        return "P:long-input-high" if fraction >= threshold else "P:mixed-input"
    boundary = int(settings["context_inference"]["long_output_tokens"])
    fraction = sum(
        int(row["requested_output_len"]) > boundary for row in requests
    ) / len(requests)
    threshold = float(settings["context_inference"]["decode_heavy_fraction"])
    return "D:long-output-high" if fraction >= threshold else "D:mixed-output"


def d1_evidence_from_window(
    phase: str,
    metrics: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    *,
    endpoint_id: str,
    observed_frequency_mhz: int,
    settings: Mapping[str, Any],
    captured_at_s: float,
) -> Any:
    """Adapt canonical D1 records once at the generic C1 evidence boundary."""
    latency_key = "ttft_ms" if phase == "P" else "tpot_ms"
    latency = [float(row[latency_key]) for row in requests]
    evidence_metrics = dict(metrics)
    evidence_metrics.update({
        "measurement_valid": True,
        "telemetry_fresh": True,
        "readback_valid": True,
        "%s_p95_ms" % latency_key.removesuffix("_ms"): percentile(latency, 0.95),
    })
    # evidence_from_window is shared with C1/C2 and consumes their normalized
    # names. This is the sole adapter from the canonical D1 schema.
    adapted = canonical_to_c1_evidence_rows(requests)
    return evidence_from_window(
        phase,
        evidence_metrics,
        adapted,
        observed_frequency_mhz=observed_frequency_mhz,
        slo_regime="TIGHT",
        ttft_slo_ms=600.0,
        tpot_slo_ms=65.0,
        context_signature=d1_context_signature(phase, requests, settings),
        captured_at_s=captured_at_s,
        endpoint_ids=(endpoint_id,),
    )


def canonical_to_c1_evidence_rows(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One adapter for reused shared-pool C1/C2 evidence code."""
    return [{
        "ttft_ms": row["ttft_ms"],
        "tpot_ms": row["tpot_ms"],
        "input_len": row["requested_input_len"],
        "requested_output_len": row["requested_output_len"],
        "observed_output_tokens": (
            row["actual_output_len"]
            if row["actual_output_len"] is not None
            else 0
        ),
        "real_sse": row["real_sse"],
        "logical_request_id_propagated": row["logical_request_id_propagated"],
    } for row in requests]


class D1EvidenceLedger:
    """Separate declared warm-up traffic from controller-eligible evidence."""

    def __init__(self) -> None:
        self.warmup_request_count = 0
        self.control_requests: list[dict[str, Any]] = []

    def record(
        self, requests: Sequence[Mapping[str, Any]], *, evidence_eligible: bool,
    ) -> None:
        if evidence_eligible:
            self.control_requests.extend(dict(row) for row in requests)
        else:
            self.warmup_request_count += len(requests)


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
    if int(settings.get("preflight_warmup_windows", 0)) < 1:
        raise StageC2Error("D1 preflight requires a declared all-HIGH warm-up")
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
        self.routing_diagnostic_rows: list[dict[str, Any]] = []
        self.preflight_warmup_rows: list[dict[str, Any]] = []
        self.evidence_ledger = D1EvidenceLedger()
        self.latest_endpoint_evidence: dict[str, dict[str, Any]] = {}

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
        evidence = endpoint_evidence_summary(requests, metrics)
        self.latest_endpoint_evidence = evidence
        for endpoint_id, values in evidence.items():
            self.endpoint_energy[endpoint_id] = values[
                "gross_energy_j_per_assigned_request"
            ]
            self.endpoint_latency[endpoint_id] = values["latency_p95_ms"]
        feedback: dict[str, dict[str, dict[str, Any]]] = {}
        for workload_class in ("SS", "SL", "LS", "LL"):
            feedback[workload_class] = {}
            for pair in PAIRS:
                selected = [
                    row for row in requests
                    if row["request_class"] == workload_class
                    and row["prefill_endpoint_id"] == pair[0]
                    and row["decode_endpoint_id"] == pair[1]
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
                "prefill_endpoint_id" if phase == "P" else "decode_endpoint_id"
            )
            selected = [row for row in requests if row[key] == endpoint_id]
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
            signature = d1_context_signature(phase, selected, self.settings)
            evidence = d1_evidence_from_window(
                phase,
                metrics,
                selected,
                endpoint_id=endpoint_id,
                observed_frequency_mhz=int(context.targets[endpoint_id]),
                settings=self.settings,
                captured_at_s=now,
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

    def _capture_window_diagnostics(
        self, requests: Sequence[Mapping[str, Any]],
    ) -> None:
        request_ids = {str(row["request_id"]) for row in requests}
        captured = self._diagnostics(request_ids)
        if len(captured) != len(request_ids):
            raise StageC2Error(
                "D1 routing diagnostics mismatch for completed window: "
                "%d diagnostics for %d requests" % (len(captured), len(request_ids))
            )
        self.routing_diagnostic_rows.extend(captured)

    def _run_preflight_warmup(
        self,
        *,
        actuator: PerEndpointClockActuator,
        context: ControllerContext,
        capabilities: Mapping[str, Any],
        point: Mapping[str, Any],
        repeat: int,
        policy: str,
        block_id: str,
        windows_root: Path,
    ) -> None:
        """Run declared all-HIGH warm-up without feeding controller evidence."""
        high = {key: value.selected_high_mhz for key, value in capabilities.items()}
        for warmup_index in range(
            1, int(self.settings["preflight_warmup_windows"]) + 1
        ):
            self._write_routing_state(
                block_id=block_id,
                routing_mode="FIXED_RR",
                context=context,
                endpoint_controllers=None,
                generation_suffix="warmup-%02d" % warmup_index,
            )
            trace = build_trace(
                self.experiment,
                str(point["scenario"]),
                "d1-warmup-%s-r%02d-w%02d" % (
                    str(point["id"]).lower(), repeat, warmup_index,
                ),
            )
            window_id = "%s-warmup-%02d" % (block_id, warmup_index)
            ledger = ClockSegmentLedger(window_id)
            self.active_ledger = ledger
            config = self._window_config(high, windows_root)
            window_dir = D1AccountingWindowHarness(
                config, window_id, trace, ledger, None,
                self.settings["frequency_levels_mhz"],
            ).run()
            self.active_ledger = None
            row, raw_requests = self._measure_window(
                window_dir,
                block_id=block_id,
                policy=policy,
                repeat=repeat,
                window_index=0,
                context_id=str(point["id"]),
                trace_signature=_trace_signature(trace),
                ledger=ledger,
            )
            for raw in raw_requests:
                raw["load_point"] = point["id"]
            canonical = normalize_d1_requests(raw_requests)
            self.evidence_ledger.record(canonical, evidence_eligible=False)
            self.preflight_warmup_rows.append({
                **row,
                "warmup_index": warmup_index,
                "evidence_eligible": False,
                "canonical_request_count": len(canonical),
                "frequency_state": "P0=P1=HIGH,D0=D1=HIGH",
            })

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
        self.routing_diagnostic_rows = []
        self.preflight_warmup_rows = []
        self.evidence_ledger = D1EvidenceLedger()
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
                            self._run_preflight_warmup(
                                actuator=actuator,
                                context=context,
                                capabilities=capabilities,
                                point=point,
                                repeat=repeat,
                                policy=policy,
                                block_id=block_id,
                                windows_root=windows_root,
                            )
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
                                        canonical_to_c1_evidence_rows(previous_requests),
                                        policy, block_id, window_index,
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
                            for item in request_rows:
                                item["load_point"] = point["id"]
                            canonical_requests = normalize_d1_requests(request_rows)
                            self._capture_window_diagnostics(canonical_requests)
                            self.evidence_ledger.record(
                                canonical_requests, evidence_eligible=True
                            )
                            rows.append(row)
                            requests_all.extend(canonical_requests)
                            previous, previous_requests = row, canonical_requests
                        if previous is not None:
                            self._update_runtime_feedback(previous, previous_requests)
                            if shared is not None:
                                self._efficiency_control(
                                    context, capabilities, actuator, previous,
                                    canonical_to_c1_evidence_rows(previous_requests),
                                    policy, block_id, windows + 1,
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
                                        "candidate_frequency_mhz": state.candidate_frequency,
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
        _write_csv(
            self.run_dir / "preflight_warmup.csv", self.preflight_warmup_rows
        )
        diagnostics = list(self.routing_diagnostic_rows)
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
                        "prefill_endpoint_id"
                        if endpoint_id.startswith("P") else "decode_endpoint_id"
                    )
                    assigned = [
                        row for row in policy_requests
                        if row[selection_key] == endpoint_id
                    ]
                    latency_key = (
                        "ttft_ms" if endpoint_id.startswith("P") else "tpot_ms"
                    )
                    latency = [float(row[latency_key]) for row in assigned]
                    gross_energy = sum(
                        float(row["%s_energy_j" % endpoint_id])
                        for row in policy_windows
                    )
                    final_state = next((
                        row for row in reversed(self.search_rows)
                        if row.get("policy") == policy
                        and row.get("load_point") == load
                        and row.get("endpoint_id") == endpoint_id
                    ), {})
                    endpoint_rows.append({
                        "load_point": load, "policy": policy,
                        "endpoint_id": endpoint_id,
                        "request_count": len(assigned),
                        "class_counts_json": json.dumps({
                            request_class: sum(
                                row["request_class"] == request_class
                                for row in assigned
                            ) for request_class in ("SS", "SL", "LS", "LL")
                        }, sort_keys=True),
                        "input_tokens": sum(
                            int(row["requested_input_len"]) for row in assigned
                        ),
                        "requested_output_tokens": sum(
                            int(row["requested_output_len"]) for row in assigned
                        ),
                        "actual_output_tokens": sum(
                            int(row["actual_output_len"] or 0) for row in assigned
                        ),
                        "latency_metric": (
                            "TTFT" if endpoint_id.startswith("P") else "TPOT"
                        ),
                        "latency_count": len(latency),
                        "latency_mean_ms": (
                            statistics.fmean(latency) if latency else None
                        ),
                        "latency_p95_ms": (
                            percentile(latency, 0.95) if latency else None
                        ),
                        "gross_energy_j": gross_energy,
                        "gross_energy_j_per_assigned_request": (
                            gross_energy / len(assigned) if assigned else None
                        ),
                        "current_frequency_mhz": final_state.get(
                            "current_frequency_mhz"
                        ),
                        "validated_safe_frequency_mhz": final_state.get(
                            "final_validated_safe_frequency_mhz"
                        ),
                        "candidate_frequency_mhz": final_state.get(
                            "candidate_frequency_mhz"
                        ),
                        "probe_count": sum(
                            row.get("policy") == policy
                            and row.get("endpoint_id") == endpoint_id
                            and row.get("decision") == "EXPLORE_DOWN"
                            for row in self.decision_rows
                        ),
                        "recovery_count": sum(
                            row.get("policy") == policy
                            and row.get("endpoint_id") == endpoint_id
                            and row.get("action") == "FAST_RECOVER"
                            for row in self.safety_rows
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
                        if row["request_class"] == workload_class
                    ]
                    for pair in PAIRS:
                        route_class_rows.append({
                            "load_point": load, "policy": policy,
                            "workload_class": workload_class,
                            "prefill_endpoint_id": pair[0],
                            "decode_endpoint_id": pair[1],
                            "request_count": sum(
                                row["prefill_endpoint_id"] == pair[0]
                                and row["decode_endpoint_id"] == pair[1]
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
            (str(row["prefill_endpoint_id"]), str(row["decode_endpoint_id"]))
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
        endpoint_evidence_valid = bool(endpoint_rows) and all(
            int(row["request_count"]) > 0
            and int(row["input_tokens"]) > 0
            and int(row["requested_output_tokens"]) > 0
            and int(row["latency_count"]) > 0
            and row["gross_energy_j"] is not None
            for row in endpoint_rows
        )
        final_state_endpoints = {
            str(row["endpoint_id"]) for row in self.search_rows
            if row.get("policy") == "JOINT-ROUTING-PER-ENDPOINT"
        }
        independent_state_valid = final_state_endpoints == set(ENDPOINTS)
        search_progress_valid = independent_state_valid and all(
            int(row.get("windows_observed") or 0) > 0
            and int(row.get("requests_observed") or 0) > 0
            for row in self.search_rows
            if row.get("policy") == "JOINT-ROUTING-PER-ENDPOINT"
        )
        warmup_valid = (
            self.stage != "preflight"
            or (
                bool(self.preflight_warmup_rows)
                and all(
                    _bool(row["window_hard_gate_valid"])
                    and not _bool(row["evidence_eligible"])
                    for row in self.preflight_warmup_rows
                )
                and self.evidence_ledger.warmup_request_count == sum(
                    int(row["canonical_request_count"])
                    for row in self.preflight_warmup_rows
                )
                and len(self.evidence_ledger.control_requests) == len(requests)
            )
        )
        asymmetry_actuated = any(
            row.get("endpoint_id") == "P0"
            and row.get("reason") == "stage_d1_preflight_fixed_asymmetry"
            and int(row.get("observed_freq_after_mhz") or 0) == 1890
            for row in self.action_rows
        )
        preflight_pass = (
            self.stage == "preflight" and run_error is None and hard
            and context_valid and all_pairs and restoration_valid
            and energy_safe and energy_preference and pressure_spill
            and endpoint_evidence_valid and independent_state_valid
            and search_progress_valid and warmup_valid and asymmetry_actuated
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
                            row["prefill_endpoint_id"] == pair[0]
                            and row["decode_endpoint_id"] == pair[1]
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
            "endpoint_evidence_populated_audit": endpoint_evidence_valid,
            "independent_endpoint_search_state_audit": independent_state_valid,
            "endpoint_safe_search_progress_audit": search_progress_valid,
            "preflight_all_high_warmup_audit": warmup_valid,
            "preflight_warmup_request_count": (
                self.evidence_ledger.warmup_request_count
            ),
            "preflight_control_evidence_request_count": len(
                self.evidence_ledger.control_requests
            ),
            "preflight_p0_mid_p1_high_asymmetry_audit": asymmetry_actuated,
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
