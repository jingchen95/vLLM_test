"""Phase 4B controlled feedback-only policy evaluation.

The stationary stage compares four runtime policies against the frozen Phase
4A empirical oracle.  It reuses the Phase 3D feedback scheduler unchanged and
never trains or loads a predictive model.  Dynamic traces are deliberately a
separate stage, gated on an accepted stationary audit.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.feedback_scheduler import DVFSAction
from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    build_registry_and_compatibility,
)
from xpyd.phase3d_control import (
    EndpointClockCapability,
    NvidiaSmiClockBackend,
    PerEndpointClockActuator,
    Phase3DClosedLoopHarness,
    _csv,
    _phase3c_window_config,
    load_config as load_phase3d_config,
)
from xpyd.route_probing import (
    ContextRouteCostStore,
    RouteProbeDecision,
    SafeRouteProber,
    probe_safe_routes,
)


class Phase4BError(RuntimeError):
    """A fail-closed Phase 4B configuration, measurement, or audit error."""


POLICIES = (
    "STATIC",
    "FEEDBACK_ROUTING_ONLY",
    "FEEDBACK_DVFS_ONLY",
    "FULL_FEEDBACK",
)


@dataclass
class ControllerContext:
    registry: Any
    telemetry: Any
    scheduler: Any
    target_frequencies: dict[str, int]
    selected_pairs: list[list[str]]
    initialized: bool = False


def _mean(values: Sequence[Any]) -> Optional[float]:
    clean = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.fmean(clean) if clean else None


def _std(values: Sequence[Any]) -> Optional[float]:
    clean = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def _ci95(values: Sequence[Any]) -> Optional[float]:
    clean = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return 1.96 * statistics.stdev(clean) / math.sqrt(len(clean)) if len(clean) >= 2 else None


def _percentile(values: Sequence[Any], percentile: float) -> Optional[float]:
    clean = sorted(
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return clean[lower]
    return clean[lower] + (clean[upper] - clean[lower]) * (rank - lower)


def classify_oracle_gap(gap: float) -> str:
    """Classify normalized energy without treating below-oracle noise as failure."""
    if gap <= 0.05:
        return "<=5%"
    if gap <= 0.10:
        return "5-10%"
    return ">10%"


def relative_energy_savings(
    baseline_energy_j: Optional[float], candidate_energy_j: Optional[float],
) -> Optional[float]:
    """Return a relative saving only for a positive measured baseline."""
    if baseline_energy_j is None or candidate_energy_j is None:
        return None
    baseline = float(baseline_energy_j)
    if baseline <= 0.0:
        return None
    return (baseline - float(candidate_energy_j)) / baseline


def counterbalanced_blocks(
    policies: Sequence[str], workloads: Sequence[Mapping[str, Any]], seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Return a reproducible randomized block order for thermal/order control."""
    blocks = [(str(policy), dict(workload)) for policy in policies for workload in workloads]
    random.Random(int(seed)).shuffle(blocks)
    return blocks


def load_phase4a_reference(path: Path) -> dict[str, Any]:
    """Load and validate the frozen empirical oracle without recomputing it."""
    raw = path.read_bytes()
    summary = json.loads(raw)
    if not isinstance(summary, dict) or not summary.get("valid"):
        raise Phase4BError("Phase 4A oracle summary is missing or invalid")
    if not summary.get("ready_for_phase4b"):
        raise Phase4BError("Phase 4A summary is not ready for Phase 4B")
    if summary.get("models_trained_or_used"):
        raise Phase4BError("Phase 4A reference unexpectedly records model use")
    oracles = {str(item["workload"]): dict(item) for item in summary.get("oracles", [])}
    if len(oracles) != 4:
        raise Phase4BError("Phase 4A reference must contain exactly four workloads")
    near_signatures: dict[str, list[dict[str, Any]]] = {}
    for workload, oracle in oracles.items():
        best = float(oracle["J_per_request"])
        signatures = []
        for item in summary.get("configuration_aggregates", []):
            if str(item.get("workload")) != workload or not item.get("oracle_eligible"):
                continue
            if float(item["joules_per_request_mean"]) > best * 1.05:
                continue
            signatures.append({
                "config_id": str(item["config_id"]),
                "route": [str(item["prefill_endpoint_id"]), str(item["decode_endpoint_id"])],
                "frequencies": {
                    endpoint_id: int(item["%s_freq_mhz" % endpoint_id])
                    for endpoint_id in ("P0", "P1", "D0", "D1")
                },
            })
        near_signatures[workload] = signatures
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "summary": summary,
        "oracles": oracles,
        "near_signatures": near_signatures,
    }


def exact_near_optimal_match(
    workload: str,
    selected_pairs: Sequence[Sequence[str]],
    frequencies: Mapping[str, int],
    reference: Mapping[str, Any],
) -> Optional[bool]:
    """Map a runtime state only when it is exactly comparable to Phase 4A."""
    if len(selected_pairs) != 1:
        return None
    route = list(map(str, selected_pairs[0]))
    for signature in reference["near_signatures"].get(workload, []):
        if route == signature["route"] and all(
            int(frequencies[key]) == int(signature["frequencies"][key])
            for key in ("P0", "P1", "D0", "D1")
        ):
            return True
    return False


def summarize_stationary(
    rows: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    workloads: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
    reference: Mapping[str, Any],
    repeats: int,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> list[dict[str, Any]]:
    """Aggregate repeated policy windows and preserve run-to-run variability."""
    result = []
    metric_fields = (
        "total_gpu_gross_energy_j", "joules_per_request",
        "joules_per_output_token", "mean_ttft_ms", "mean_tpot_ms",
        "mean_itl_ms", "mean_e2e_latency_ms", "throughput_requests_s",
    )
    for policy in policies:
        for workload_spec in workloads:
            workload = str(workload_spec["id"])
            group = [
                item for item in rows
                if item["policy"] == policy and item["workload"] == workload
            ]
            request_group = [
                item for item in requests
                if item["policy"] == policy and item["workload"] == workload
            ]
            control_group = [
                item for item in controls
                if item["policy"] == policy and item["workload"] == workload
                and item.get("measured_iteration")
            ]
            valid_group = [item for item in group if item.get("measurement_valid")]
            oracle = reference["oracles"][workload]
            mean_jpr = _mean([item.get("joules_per_request") for item in valid_group])
            gap = (
                (mean_jpr - float(oracle["J_per_request"])) / float(oracle["J_per_request"])
                if mean_jpr is not None else None
            )
            ttft_values = [item.get("ttft_ms") for item in request_group]
            tpot_values = [item.get("tpot_ms") for item in request_group]
            route_counts: dict[str, int] = {}
            for item in request_group:
                key = "%s->%s" % (
                    item["prefill_endpoint_id"], item["decode_endpoint_id"]
                )
                route_counts[key] = route_counts.get(key, 0) + 1
            residency: dict[str, dict[str, float]] = {}
            for endpoint_id in ("P0", "P1", "D0", "D1"):
                durations: dict[str, float] = {}
                total_duration = sum(float(item["duration_s"]) for item in valid_group)
                for item in valid_group:
                    level = str(item["%s_requested_level" % endpoint_id])
                    durations[level] = durations.get(level, 0.0) + float(item["duration_s"])
                requested_fraction = {
                    key: value / total_duration for key, value in sorted(durations.items())
                } if total_duration else {}
                match_values = [
                    item.get("%s_frequency_match_fraction" % endpoint_id)
                    for item in valid_group
                ]
                residency[endpoint_id] = {
                    "requested_time_fraction": requested_fraction,
                    "observed_target_match_fraction_mean": _mean(match_values),
                }
            comparable = [
                item["phase4a_near_optimal_match"] for item in valid_group
                if item.get("phase4a_near_optimal_match") is not None
            ]
            row: dict[str, Any] = {
                "policy": policy,
                "workload": workload,
                "repeat_count": len(group),
                "valid_repeat_count": len(valid_group),
                "complete": len(group) == repeats and len(valid_group) == repeats,
                "request_count": len(request_group),
                "slo_pass": bool(request_group) and all(
                    float(item["ttft_ms"]) <= ttft_slo_ms
                    and float(item["tpot_ms"]) <= tpot_slo_ms
                    for item in request_group
                ),
                "ttft_slo_violation_count": sum(
                    float(item["ttft_ms"]) > ttft_slo_ms for item in request_group
                ),
                "tpot_slo_violation_count": sum(
                    float(item["tpot_ms"]) > tpot_slo_ms for item in request_group
                ),
                "any_slo_violation_count": sum(
                    float(item["ttft_ms"]) > ttft_slo_ms
                    or float(item["tpot_ms"]) > tpot_slo_ms
                    for item in request_group
                ),
                "slo_violation_rate": (
                    sum(
                        float(item["ttft_ms"]) > ttft_slo_ms
                        or float(item["tpot_ms"]) > tpot_slo_ms
                        for item in request_group
                    ) / len(request_group) if request_group else None
                ),
                "ttft_p50_ms": _percentile(ttft_values, 50),
                "ttft_p95_ms": _percentile(ttft_values, 95),
                "ttft_p99_ms": _percentile(ttft_values, 99),
                "tpot_p50_ms": _percentile(tpot_values, 50),
                "tpot_p95_ms": _percentile(tpot_values, 95),
                "tpot_p99_ms": _percentile(tpot_values, 99),
                "ttft_headroom_ms": ttft_slo_ms - max(map(float, ttft_values)) if ttft_values else None,
                "tpot_headroom_ms": tpot_slo_ms - max(map(float, tpot_values)) if tpot_values else None,
                "route_distribution_json": json.dumps(route_counts, sort_keys=True),
                "frequency_residency_json": json.dumps(residency, sort_keys=True),
                "dvfs_action_count": sum(int(item.get("dvfs_action_count", 0)) for item in control_group),
                "routing_change_count": sum(bool(item.get("routing_changed")) for item in control_group),
                "fallback_event_count": sum(bool(item.get("fallback_reason")) for item in control_group),
                "stale_telemetry_event_count": sum(
                    not bool(item.get("telemetry_fresh")) for item in control_group
                ),
                "oracle_joules_per_request": float(oracle["J_per_request"]),
                "oracle_gap": gap,
                "oracle_gap_class": classify_oracle_gap(gap) if gap is not None else None,
                "phase4a_near_optimal_comparable_window_count": len(comparable),
                "phase4a_near_optimal_window_fraction": (
                    sum(bool(item) for item in comparable) / len(comparable)
                    if comparable else None
                ),
            }
            for metric in metric_fields:
                values = [item.get(metric) for item in valid_group]
                row[metric + "_mean"] = _mean(values)
                row[metric + "_std"] = _std(values)
                row[metric + "_ci95_half_width"] = _ci95(values)
            result.append(row)
    return result


def summarize_dynamic_adaptation(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    trace_specs: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> list[dict[str, Any]]:
    """Summarize each real workload transition without inventing a reaction."""
    control_by_id = {str(item["control_iteration_id"]): item for item in controls}
    result = []
    for trace_spec in trace_specs:
        trace_id = str(trace_spec["id"])
        for policy in policies:
            repeats = sorted({
                int(item.get("dynamic_repeat", 1)) for item in rows
                if item["trace_id"] == trace_id and item["policy"] == policy
            })
            for dynamic_repeat in repeats:
                trace_rows = sorted(
                    (
                        item for item in rows
                        if item["trace_id"] == trace_id
                        and item["policy"] == policy
                        and int(item.get("dynamic_repeat", 1)) == dynamic_repeat
                    ),
                    key=lambda item: int(item["sequence_index"]),
                )
                trace_requests = [
                    item for item in requests
                    if item["trace_id"] == trace_id
                    and item["policy"] == policy
                    and int(item.get("dynamic_repeat", 1)) == dynamic_repeat
                ]
                cumulative_energy = sum(
                    float(item["total_gpu_gross_energy_j"]) for item in trace_rows
                )
                for state_index in range(1, len(trace_spec["states"])):
                    state_rows = [
                        item for item in trace_rows
                        if int(item["state_index"]) == state_index
                    ]
                    previous_rows = [
                        item for item in trace_rows
                        if int(item["state_index"]) == state_index - 1
                    ]
                    if not state_rows or not previous_rows:
                        continue
                    transition_s = float(
                        state_rows[0]["transition_timestamp_unix_s"]
                    )
                    prior_signature = str(
                        previous_rows[-1]["control_state_signature"]
                    )
                    state_signatures = [
                        str(item["control_state_signature"]) for item in state_rows
                    ]
                    reaction_row = None
                    prior = prior_signature
                    for item in state_rows:
                        changed = str(item["control_state_signature"]) != prior
                        # Window 1 is deliberately blind to the new state. Only
                        # a later change is attributable to new-state feedback.
                        if int(item["window_in_state"]) >= 2 and changed:
                            reaction_row = item
                            break
                        prior = str(item["control_state_signature"])
                    stable_index = 0
                    for index, signature in enumerate(state_signatures):
                        if all(
                            value == signature
                            for value in state_signatures[index:]
                        ):
                            stable_index = index
                            break
                    settling_s = (
                        0.0 if stable_index == 0 else
                        max(
                            0.0,
                            float(state_rows[stable_index]["control_timestamp_unix_s"])
                            - transition_s,
                        )
                    )
                    signature_changes = sum(
                        state_signatures[index] != (
                            prior_signature
                            if index == 0 else state_signatures[index - 1]
                        )
                        for index in range(len(state_signatures))
                    )
                    transition_requests = [
                        item for item in trace_requests
                        if int(item["state_index"]) == state_index
                    ]
                    violations = sum(
                        float(item["ttft_ms"]) > ttft_slo_ms
                        or float(item["tpot_ms"]) > tpot_slo_ms
                        for item in transition_requests
                    )
                    state_controls = [
                        control_by_id[str(item["control_iteration_id"])]
                        for item in state_rows
                    ]
                    result.append({
                        "trace_id": trace_id,
                        "policy": policy,
                        "dynamic_repeat": dynamic_repeat,
                        "transition_index": state_index,
                        "from_workload": str(
                            trace_spec["states"][state_index - 1]
                        ),
                        "to_workload": str(trace_spec["states"][state_index]),
                        "transition_timestamp_unix_s": transition_s,
                        "reaction_observed": reaction_row is not None,
                        "time_to_first_reaction_s": (
                            max(
                                0.0,
                                float(reaction_row["control_timestamp_unix_s"])
                                - transition_s,
                            ) if reaction_row else None
                        ),
                        "settling_time_s": settling_s,
                        "stable_for_remaining_observed_windows": True,
                        "no_action_needed_or_no_observed_reaction": (
                            reaction_row is None
                        ),
                        "transition_window_energy_j": float(
                            state_rows[0]["total_gpu_gross_energy_j"]
                        ),
                        "state_energy_j": sum(
                            float(item["total_gpu_gross_energy_j"])
                            for item in state_rows
                        ),
                        "cumulative_trace_energy_j": cumulative_energy,
                        "slo_violation_count": violations,
                        "slo_violation_rate": (
                            violations / len(transition_requests)
                            if transition_requests else None
                        ),
                        "control_action_count": sum(
                            int(item.get("dvfs_action_count", 0))
                            for item in state_controls
                        ),
                        "routing_change_count": sum(
                            bool(item.get("routing_changed"))
                            for item in state_controls
                        ),
                        "oscillation_or_reversal_count": max(
                            0, signature_changes - 1
                        ),
                        "fallback_event_count": sum(
                            bool(item.get("fallback_reason"))
                            for item in state_controls
                        ),
                        "stale_telemetry_event_count": sum(
                            not bool(item.get("telemetry_fresh"))
                            for item in state_controls
                        ),
                    })
    return result


class Phase4BEvaluationHarness(Phase3DClosedLoopHarness):
    """Physical stationary comparison with isolated feedback state per block."""

    def __init__(
        self, config: Mapping[str, Any], run_id: Optional[str] = None,
        backend: Optional[NvidiaSmiClockBackend] = None, smoke: bool = False,
        stage: str = "stationary",
    ) -> None:
        super().__init__(config, run_id=run_id, backend=backend)
        self.settings = dict(self.config["phase4b"])
        self.smoke = bool(smoke)
        self.stage = str(stage)
        if self.stage not in {"stationary", "dynamic"}:
            raise Phase4BError("Phase 4B stage must be stationary or dynamic")
        root = (
            self.settings["dynamic"].get(
                "output_root", "results/phase4b_dynamic_evaluation"
            )
            if self.stage == "dynamic"
            else self.settings.get("output_root", "results/phase4b_evaluation")
        )
        self.run_dir = Path(os.path.expandvars(str(root))) / self.run_id
        self.control_rows: list[dict[str, Any]] = []
        self.actuator_rows: list[dict[str, Any]] = []

    def _reference(self) -> dict[str, Any]:
        return load_phase4a_reference(Path(str(self.settings["phase4a_summary"])))

    def _accepted_frequencies(
        self, reference: Mapping[str, Any],
    ) -> dict[str, dict[str, int]]:
        values = reference["summary"].get("frequency_levels_mhz", {})
        result = {
            endpoint_id: {
                level: int(values[endpoint_id][level])
                for level in ("LOW", "MID", "HIGH")
            }
            for endpoint_id in ("P0", "P1", "D0", "D1")
        }
        return result

    def _discover_exact(
        self, accepted: Mapping[str, Mapping[str, int]],
    ) -> dict[str, EndpointClockCapability]:
        capabilities = {}
        for endpoint in self.config["endpoints"]:
            endpoint_id = str(endpoint["endpoint_id"])
            capability = self.backend.discover(
                endpoint_id, str(endpoint["node"]), int(endpoint["gpu_ids"][0]),
                int(accepted[endpoint_id]["HIGH"]),
            )
            observed = {
                "LOW": capability.selected_low_mhz,
                "MID": capability.selected_mid_mhz,
                "HIGH": capability.selected_high_mhz,
            }
            if observed != dict(accepted[endpoint_id]):
                raise Phase4BError(
                    "discovered frequency states differ from frozen Phase 4A states"
                )
            capabilities[endpoint_id] = capability
        return capabilities

    def _new_context(
        self, capabilities: Mapping[str, EndpointClockCapability],
    ) -> ControllerContext:
        registry, telemetry, scheduler = self._build_scheduler(capabilities)
        high = {
            endpoint_id: capability.selected_high_mhz
            for endpoint_id, capability in capabilities.items()
        }
        return ControllerContext(registry, telemetry, scheduler, high, [])

    def _actuate(
        self, actuator: PerEndpointClockActuator, context: ControllerContext,
        targets: Mapping[str, int], reason: str, *, controller_action: bool,
        policy: str, workload: str, iteration_id: str,
    ) -> int:
        count = 0
        for endpoint_id in sorted(targets):
            target = int(targets[endpoint_id])
            if actuator.requested.get(endpoint_id) != target:
                row = actuator.actuate(endpoint_id, target, reason)
                row.update({
                    "policy": policy, "workload": workload,
                    "control_iteration_id": iteration_id,
                    "controller_action": controller_action,
                })
                self.actuator_rows.append(row)
                if row["command_status"] != "success" or not row["readback_valid"]:
                    raise Phase4BError("physical DVFS actuation failed for %s" % endpoint_id)
                if controller_action:
                    count += 1
            state = context.registry.get_state(endpoint_id)
            state.freq_mhz = target
            context.registry.update_state(state)
            context.target_frequencies[endpoint_id] = target
        return count

    def _stale_latency_endpoints(
        self, context: ControllerContext, endpoint_ids: Sequence[str], now_s: float,
    ) -> dict[str, Optional[float]]:
        """Return role-specific missing/stale latency ages for diagnostics."""
        stale: dict[str, Optional[float]] = {}
        for endpoint_id in endpoint_ids:
            snapshot = context.telemetry.snapshot(endpoint_id, now_s=now_s)
            role = context.registry.get_spec(endpoint_id).role
            observed_at = (
                snapshot.last_ttft_observation_s
                if role == "prefill" else snapshot.last_tpot_observation_s
            )
            age = now_s - observed_at if observed_at is not None else None
            if age is None or age > context.scheduler.config.telemetry_max_age_s:
                stale[endpoint_id] = age
        return stale

    def _decide(
        self, policy: str, workload: str, iteration_id: str,
        context: ControllerContext, capabilities: Mapping[str, EndpointClockCapability],
        actuator: PerEndpointClockActuator, all_pairs: Sequence[Sequence[str]],
        *, measured_iteration: bool, feedback_context_key: Optional[str] = None,
        route_override: Optional[Sequence[str]] = None,
        freeze_dvfs: bool = False,
        routing_decision_mode: Optional[str] = None,
        routing_score_basis: Optional[str] = None,
        route_costs_json: str = "{}",
    ) -> list[list[str]]:
        self._actuate(
            actuator, context, context.target_frequencies,
            "phase4b_resume_isolated_policy_state", controller_action=False,
            policy=policy, workload=workload, iteration_id=iteration_id,
        )
        high = {
            endpoint_id: capability.selected_high_mhz
            for endpoint_id, capability in capabilities.items()
        }
        now = time.time()
        feedback_routing = policy in {
            "FEEDBACK_ROUTING_ONLY", "FULL_FEEDBACK", "PASSIVE_FULL", "ACTIVE_FULL",
        }
        feedback_dvfs = policy in {
            "FEEDBACK_DVFS_ONLY", "FULL_FEEDBACK", "PASSIVE_FULL", "ACTIVE_FULL",
        }
        count = 0
        required = (
            sorted(capabilities) if feedback_routing else ["P0", "D0"]
        )
        stale_endpoints = (
            self._stale_latency_endpoints(context, required, now)
            if feedback_routing or feedback_dvfs else {}
        )
        telemetry_fresh = not stale_endpoints
        fallback_reason = None
        evaluations: Sequence[Any] = ()
        recommendations: Sequence[Any] = ()
        observed_context = feedback_context_key or workload
        if policy == "STATIC":
            pairs = [list(item) for item in all_pairs]
            self._actuate(
                actuator, context, high, "phase4b_static_all_high",
                controller_action=False, policy=policy, workload=workload,
                iteration_id=iteration_id,
            )
        elif feedback_routing and len(stale_endpoints) == len(required):
            pairs = [list(item) for item in all_pairs]
            fallback_reason = "missing_or_stale_feedback_use_safe_high"
            count = self._actuate(
                actuator, context, high, "phase4b_conservative_fallback_high",
                controller_action=True, policy=policy, workload=workload,
                iteration_id=iteration_id,
            )
        elif policy == "FEEDBACK_DVFS_ONLY" and not telemetry_fresh:
            pairs = [["P0", "D0"]]
            fallback_reason = "missing_or_stale_feedback_use_safe_high"
            count = self._actuate(
                actuator, context, high, "phase4b_conservative_fallback_high",
                controller_action=True, policy=policy, workload=workload,
                iteration_id=iteration_id,
            )
        else:
            if feedback_routing:
                evaluations = context.scheduler.evaluate_routes(
                    now_s=now, workload_context_key=observed_context
                )
                eligible = [item for item in evaluations if item.eligible]
                if eligible:
                    eligible_pairs = {
                        (item.prefill_endpoint_id, item.decode_endpoint_id)
                        for item in eligible
                    }
                    if route_override is not None:
                        requested_pair = tuple(map(str, route_override))
                        if len(requested_pair) != 2 or requested_pair not in eligible_pairs:
                            pairs = [["P0", "D0"]]
                            fallback_reason = "active_route_override_not_measured_safe"
                            count = self._actuate(
                                actuator, context, high,
                                "phase4b_active_route_override_fallback_high",
                                controller_action=True, policy=policy,
                                workload=workload, iteration_id=iteration_id,
                            )
                        else:
                            pairs = [list(requested_pair)]
                    else:
                        route = context.scheduler.choose_route(
                            now_s=now, workload_context_key=observed_context
                        )
                        pairs = [[route.prefill_endpoint_id, route.decode_endpoint_id]]
                else:
                    pairs = [["P0", "D0"]]
                    fallback_reason = "no_fresh_measured_safe_route_use_p0d0_safe_high"
                    count = self._actuate(
                        actuator, context, high, "phase4b_no_safe_route_fallback_high",
                        controller_action=True, policy=policy, workload=workload,
                        iteration_id=iteration_id,
                    )
            else:
                pairs = [["P0", "D0"]]
            if feedback_dvfs and fallback_reason is None and not freeze_dvfs:
                recommendations = [
                    context.scheduler.choose_frequency_adjustment(endpoint_id, now_s=now)
                    for endpoint_id in sorted(capabilities)
                ]
                targets = {
                    item.endpoint_id: int(item.target_freq_mhz)
                    for item in recommendations if item.action != DVFSAction.HOLD
                }
                count += self._actuate(
                    actuator, context, targets, "phase4b_feedback_dvfs",
                    controller_action=True, policy=policy, workload=workload,
                    iteration_id=iteration_id,
                )
                for item in recommendations:
                    if item.action == DVFSAction.HOLD:
                        continue
                    state = context.registry.get_state(item.endpoint_id)
                    context.scheduler.record_dvfs_actuation(
                        item.endpoint_id, observed_freq_mhz=state.freq_mhz,
                        timestamp_s=time.time(),
                    )
            else:
                count = 0
        if policy in {"STATIC", "FEEDBACK_ROUTING_ONLY"}:
            count = 0
        previous_pairs = context.selected_pairs
        routing_changed = bool(previous_pairs) and previous_pairs != pairs
        context.selected_pairs = copy.deepcopy(pairs)
        self._write_routes(pairs, fallback_reason or "phase4b_%s" % policy.lower())
        self.control_rows.append({
            "control_timestamp": now,
            "control_iteration_id": iteration_id,
            "policy": policy,
            "workload": workload,
            "feedback_context_key": observed_context,
            "measured_iteration": measured_iteration,
            "telemetry_fresh": telemetry_fresh,
            "stale_endpoint_ids_json": json.dumps(sorted(stale_endpoints)),
            "stale_endpoint_ages_s_json": json.dumps(stale_endpoints, sort_keys=True),
            "fallback_reason": fallback_reason,
            "selected_pairs_json": json.dumps(pairs, sort_keys=True),
            "requested_frequencies_json": json.dumps(
                context.target_frequencies, sort_keys=True
            ),
            "routing_changed": routing_changed,
            "dvfs_action_count": count,
            "routing_decision_mode": routing_decision_mode or (
                "PASSIVE_ENDPOINT_EWMA" if feedback_routing else "FIXED_ROUTING"
            ),
            "routing_score_basis": routing_score_basis or (
                "endpoint_window_gross_energy_per_request"
                if feedback_routing else "none"
            ),
            "route_probe": bool(freeze_dvfs and route_override is not None),
            "dvfs_frozen_for_probe": bool(freeze_dvfs),
            "route_cost_context": observed_context,
            "route_costs_json": route_costs_json,
            "candidate_evaluations_json": json.dumps(
                [asdict(item) for item in evaluations], sort_keys=True
            ),
            "dvfs_recommendations_json": json.dumps([
                {
                    "endpoint_id": item.endpoint_id,
                    "action": item.action.value,
                    "target_freq_mhz": item.target_freq_mhz,
                    "reason": item.reason,
                } for item in recommendations
            ], sort_keys=True),
        })
        return pairs

    def _run_window(
        self, window_id: str, workload: Mapping[str, Any], count: int,
        requested: Mapping[str, int], pairs: Sequence[Sequence[str]],
        windows_root: Path, *, require_all_pairs: bool = False,
    ) -> Path:
        spec = dict(workload)
        spec["count"] = int(count)
        required_pairs: Sequence[Sequence[str]] = pairs if len(pairs) == 1 or require_all_pairs else []
        window_config = _phase3c_window_config(
            self.config, windows_root, [spec], requested,
            required_pairs=required_pairs,
        )
        window_config["server_logs"] = {}
        window_config["client"]["max_concurrency"] = 1
        window_config["request_id_namespace"] = "phase4b-%s" % window_id
        Phase3CSubstrateHarness(window_config, run_id=window_id).run()
        return windows_root / window_id

    def _measurement_row(
        self, window_id: str, policy: str, workload: Mapping[str, Any], repeat: int,
        context: ControllerContext, accepted: Mapping[str, Mapping[str, int]],
        reference: Mapping[str, Any], window_dir: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        summary = _read_json(window_dir / "summary.json")
        audit = _read_json(window_dir / "audit.json")
        client = _read_json(window_dir / "client" / "summary.json")
        with (window_dir / "requests.csv").open(newline="", encoding="utf-8") as stream:
            request_values = list(csv.DictReader(stream))
        duration = float(client["timing_end_unix_s"]) - float(client["timing_start_unix_s"])
        requested = dict(context.target_frequencies)
        levels = {
            endpoint_id: next(
                level for level, frequency in accepted[endpoint_id].items()
                if int(frequency) == int(requested[endpoint_id])
            )
            for endpoint_id in ("P0", "P1", "D0", "D1")
        }
        oracle = reference["oracles"][str(workload["id"])]
        gap = (
            float(summary["joules_per_request"]) - float(oracle["J_per_request"])
        ) / float(oracle["J_per_request"])
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "window_id": window_id,
            "policy": policy,
            "workload": str(workload["id"]),
            "input_len": int(workload["input_len"]),
            "output_len": int(workload["output_len"]),
            "rate_rps": float(workload["rate_rps"]),
            "repeat": repeat,
            "timing_start_unix_s": float(client["timing_start_unix_s"]),
            "timing_end_unix_s": float(client["timing_end_unix_s"]),
            "duration_s": duration,
            "completed_requests": int(client["successful_requests"]),
            "output_tokens": int(client["output_tokens_total"]),
            "total_gpu_gross_energy_j": float(summary["energy_j"]["total"]),
            "joules_per_request": float(summary["joules_per_request"]),
            "joules_per_output_token": float(summary["joules_per_output_token"]),
            "mean_ttft_ms": float(client["mean_ttft_ms"]),
            "p99_ttft_ms": float(client["p99_ttft_ms"]),
            "mean_tpot_ms": float(client["mean_tpot_ms"]),
            "p99_tpot_ms": float(client["p99_tpot_ms"]),
            "mean_itl_ms": float(client["mean_itl_ms"]),
            "mean_e2e_latency_ms": float(client["mean_e2e_latency_ms"]),
            "throughput_requests_s": int(client["successful_requests"]) / duration,
            "slo_pass": (
                float(client["mean_ttft_ms"]) <= float(self.settings["slo"]["ttft_ms"])
                and float(client["mean_tpot_ms"]) <= float(self.settings["slo"]["tpot_ms"])
            ),
            "measurement_valid": bool(audit.get("valid")),
            "route_distribution_json": json.dumps(summary["route_matrix"], sort_keys=True),
            "selected_pairs_json": json.dumps(context.selected_pairs, sort_keys=True),
            "phase4a_oracle_joules_per_request": float(oracle["J_per_request"]),
            "oracle_gap": gap,
            "oracle_gap_class": classify_oracle_gap(gap),
            "phase4a_near_optimal_match": exact_near_optimal_match(
                str(workload["id"]), context.selected_pairs, requested, reference
            ),
            "hard_gates_json": json.dumps(audit.get("hard_gates", {}), sort_keys=True),
        }
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            clock = summary["endpoint_clocks"][endpoint_id]["graphics"]
            row["%s_requested_freq_mhz" % endpoint_id] = requested[endpoint_id]
            row["%s_requested_level" % endpoint_id] = levels[endpoint_id]
            row["%s_observed_freq_mhz" % endpoint_id] = clock["mean_mhz"]
            row["%s_frequency_match_fraction" % endpoint_id] = clock["target_match_fraction"]
            row["%s_gross_energy_j" % endpoint_id] = summary["endpoint_energy"][endpoint_id]["gross_gpu_energy_j"]
        request_rows = []
        for item in request_values:
            request_rows.append({
                "window_id": window_id,
                "policy": policy,
                "workload": str(workload["id"]),
                "repeat": repeat,
                "request_id": item["request_id"],
                "prefill_endpoint_id": item["prefill_endpoint_id"],
                "decode_endpoint_id": item["decode_endpoint_id"],
                "ttft_ms": float(item["ttft_ms"]),
                "tpot_ms": float(item["tpot_ms"]),
                "mean_itl_ms": float(item["mean_itl_ms"]),
                "e2e_latency_ms": float(item["e2e_latency_ms"]),
            })
        return row, request_rows

    def run_stationary(self) -> Path:
        if self.run_dir.exists():
            raise Phase4BError("run directory already exists: %s" % self.run_dir)
        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        reference = self._reference()
        accepted = self._accepted_frequencies(reference)
        capabilities = self._discover_exact(accepted)
        _write_json(self.run_dir / "raw" / "capabilities.json", {
            key: asdict(value) for key, value in capabilities.items()
        })
        _write_json(self.run_dir / "raw" / "frozen_phase4a_reference.json", {
            "path": reference["path"],
            "sha256": reference["sha256"],
            "oracles": reference["oracles"],
            "near_signatures": reference["near_signatures"],
        })
        actuator = PerEndpointClockActuator(
            self.backend, capabilities,
            float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        high = {
            endpoint_id: values["HIGH"] for endpoint_id, values in accepted.items()
        }
        actuator.requested.update(high)
        _, _, all_pairs = build_registry_and_compatibility(self.config)
        policies = list(self.settings.get("policies", POLICIES))
        workloads = [dict(item) for item in self.config["workloads"]]
        repeats = 1 if self.smoke else int(self.settings["repeats"])
        request_count = int(self.settings["requests_per_repeat"])
        blocks = counterbalanced_blocks(
            policies, workloads, int(self.settings.get("order_seed", 4))
        )
        _write_json(self.run_dir / "raw" / "stationary_plan.json", {
            "smoke": self.smoke,
            "policies": policies,
            "workloads": workloads,
            "repeats": repeats,
            "requests_per_repeat": request_count,
            "block_order": [
                {"policy": policy, "workload": workload["id"]}
                for policy, workload in blocks
            ],
            "oracle_sha256": reference["sha256"],
            "models": [],
        })
        rows: list[dict[str, Any]] = []
        request_rows: list[dict[str, Any]] = []
        warmup_audits: list[dict[str, Any]] = []
        contexts: dict[tuple[str, str], ControllerContext] = {}
        restoration_valid = False
        error: Optional[str] = None
        try:
            global_context = self._new_context(capabilities)
            self._actuate(
                actuator, global_context, high, "phase4b_global_warmup_high",
                controller_action=False, policy="WARMUP", workload="all",
                iteration_id="global_warmup",
            )
            self._write_routes(all_pairs, "phase4b_global_all_route_warmup")
            warmup_dir = self._run_window(
                "global-warmup", {"id": "global_warmup", "input_len": 128,
                "output_len": 16, "rate_rps": 0.5}, 4, high, all_pairs,
                windows_root, require_all_pairs=True,
            )
            warmup_audits.append({
                "window_id": "global-warmup",
                **_read_json(warmup_dir / "audit.json"),
            })
            if not warmup_audits[-1].get("valid"):
                raise Phase4BError("global all-route warmup failed")

            for block_index, (policy, workload) in enumerate(blocks, 1):
                key = (policy, str(workload["id"]))
                context = self._new_context(capabilities)
                contexts[key] = context
                if policy != "STATIC":
                    initial_pairs = (
                        [list(item) for item in all_pairs]
                        if policy in {"FEEDBACK_ROUTING_ONLY", "FULL_FEEDBACK"}
                        else [["P0", "D0"]]
                    )
                    warmup_id = "warm-%02d-%s-%s" % (
                        block_index, policy.lower(), workload["id"]
                    )
                    self._actuate(
                        actuator, context, high, "phase4b_feedback_warmup_high",
                        controller_action=False, policy=policy,
                        workload=str(workload["id"]), iteration_id=warmup_id,
                    )
                    context.selected_pairs = copy.deepcopy(initial_pairs)
                    self._write_routes(initial_pairs, "phase4b_feedback_observation_warmup")
                    warm_count = 4 if len(initial_pairs) > 1 else request_count
                    warmup_dir = self._run_window(
                        warmup_id, workload, warm_count, high, initial_pairs,
                        windows_root, require_all_pairs=len(initial_pairs) > 1,
                    )
                    warmup_audit = _read_json(warmup_dir / "audit.json")
                    warmup_audits.append({"window_id": warmup_id, **warmup_audit})
                    if not warmup_audit.get("valid"):
                        raise Phase4BError("feedback observation warmup failed: %s" % warmup_id)
                    self._observe_window(warmup_id, warmup_dir, context.registry, context.telemetry)
                    context.initialized = True

                for repeat in range(1, repeats + 1):
                    iteration_id = "b%02d-r%02d-%s-%s" % (
                        block_index, repeat, policy.lower(), workload["id"]
                    )
                    pairs = self._decide(
                        policy, str(workload["id"]), iteration_id, context,
                        capabilities, actuator, all_pairs, measured_iteration=True,
                    )
                    window_dir = self._run_window(
                        iteration_id, workload, request_count,
                        context.target_frequencies, pairs, windows_root,
                    )
                    row, observed_requests = self._measurement_row(
                        iteration_id, policy, workload, repeat, context,
                        accepted, reference, window_dir,
                    )
                    rows.append(row)
                    request_rows.extend(observed_requests)
                    if not row["measurement_valid"]:
                        raise Phase4BError("stationary window audit failed: %s" % iteration_id)
                    if policy != "STATIC":
                        self._observe_window(
                            iteration_id, window_dir, context.registry, context.telemetry
                        )
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            restoration_valid = True
            restore_context = self._new_context(capabilities)
            try:
                self._actuate(
                    actuator, restore_context, high, "phase4b_final_safe_high_restoration",
                    controller_action=False, policy="RESTORE", workload="all",
                    iteration_id="final_restore",
                )
                restoration_valid = all(
                    actuator.read(endpoint_id).graphics_clock_mhz == target
                    for endpoint_id, target in high.items()
                )
            except Exception:
                restoration_valid = False

        summaries = summarize_stationary(
            rows, request_rows, self.control_rows, workloads, policies,
            reference, repeats, float(self.settings["slo"]["ttft_ms"]),
            float(self.settings["slo"]["tpot_ms"]),
        )
        stationary_fields = (
            "run_id", "window_id", "policy", "workload", "input_len",
            "output_len", "rate_rps", "repeat", "duration_s",
            "timing_start_unix_s", "timing_end_unix_s",
            "completed_requests", "output_tokens", "measurement_valid", "slo_pass",
            "total_gpu_gross_energy_j", "joules_per_request",
            "joules_per_output_token", "mean_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms",
            "mean_e2e_latency_ms", "throughput_requests_s",
            "route_distribution_json", "selected_pairs_json",
            "phase4a_oracle_joules_per_request", "oracle_gap",
            "oracle_gap_class", "phase4a_near_optimal_match",
            "P0_requested_freq_mhz", "P0_requested_level", "P0_observed_freq_mhz", "P0_frequency_match_fraction", "P0_gross_energy_j",
            "P1_requested_freq_mhz", "P1_requested_level", "P1_observed_freq_mhz", "P1_frequency_match_fraction", "P1_gross_energy_j",
            "D0_requested_freq_mhz", "D0_requested_level", "D0_observed_freq_mhz", "D0_frequency_match_fraction", "D0_gross_energy_j",
            "D1_requested_freq_mhz", "D1_requested_level", "D1_observed_freq_mhz", "D1_frequency_match_fraction", "D1_gross_energy_j",
            "hard_gates_json",
        )
        _csv(self.run_dir / "phase4b_stationary_results.csv", rows, stationary_fields)
        _csv(self.run_dir / "raw" / "stationary_requests.csv", request_rows, (
            "window_id", "policy", "workload", "repeat", "request_id",
            "prefill_endpoint_id", "decode_endpoint_id", "ttft_ms", "tpot_ms",
            "mean_itl_ms", "e2e_latency_ms",
        ))
        _csv(self.run_dir / "phase4b_policy_summary.csv", summaries, tuple(summaries[0]) if summaries else ())
        oracle_gap_rows = [{
            key: item[key] for key in (
                "policy", "workload", "joules_per_request_mean",
                "joules_per_request_std", "joules_per_request_ci95_half_width",
                "oracle_joules_per_request", "oracle_gap", "oracle_gap_class",
                "slo_pass", "slo_violation_rate",
                "phase4a_near_optimal_comparable_window_count",
                "phase4a_near_optimal_window_fraction",
            )
        } for item in summaries]
        _csv(self.run_dir / "phase4b_oracle_gap.csv", oracle_gap_rows, tuple(oracle_gap_rows[0]) if oracle_gap_rows else ())
        _csv(self.run_dir / "phase4b_control_actions.csv", self.control_rows, (
            "control_timestamp", "control_iteration_id", "policy", "workload",
            "feedback_context_key", "measured_iteration", "telemetry_fresh", "fallback_reason",
            "stale_endpoint_ids_json", "stale_endpoint_ages_s_json",
            "selected_pairs_json", "requested_frequencies_json", "routing_changed",
            "dvfs_action_count", "routing_decision_mode", "routing_score_basis",
            "route_probe", "dvfs_frozen_for_probe", "route_cost_context",
            "route_costs_json", "candidate_evaluations_json",
            "dvfs_recommendations_json",
        ))
        _csv(self.run_dir / "raw" / "actuator_actions.csv", self.actuator_rows, (
            "timestamp_unix_s", "policy", "workload", "control_iteration_id",
            "controller_action", "endpoint_id", "node", "gpu_id",
            "previous_requested_freq_mhz", "requested_freq_mhz",
            "observed_freq_before_mhz", "observed_freq_after_mhz",
            "command_status", "readback_valid", "transition_readback_latency_s",
            "reason", "error",
        ))
        _csv(self.run_dir / "raw" / "endpoint_telemetry.csv", self.telemetry_rows, (
            "window_id", "endpoint_id", "timestamp_s", "health", "queue",
            "kv_cache_usage_frac", "ttft_ms", "tpot_ms", "power_w", "energy_j",
            "observed_frequency_mhz", "telemetry_age_s_at_observation",
            "measurement_valid",
        ))
        # Dynamic traces are intentionally not populated by the stationary stage.
        _csv(self.run_dir / "phase4b_dynamic_results.csv", [], (
            "trace_id", "policy", "segment", "workload_state", "window_id",
            "transition_timestamp", "reaction_latency_s", "settling_time_s",
            "total_gpu_gross_energy_j", "slo_violation_count", "measurement_valid",
        ))
        expected = len(policies) * len(workloads) * repeats
        hard_gates = {
            "frozen_phase4a_reference_valid": bool(reference["summary"].get("valid")),
            "exact_phase4a_frequency_states": bool(accepted),
            "complete_stationary_plan": len(rows) == expected,
            "all_stationary_measurements_valid": len(rows) == expected and all(
                item.get("measurement_valid") for item in rows
            ),
            "all_feedback_warmups_valid": bool(warmup_audits) and all(
                item.get("valid") for item in warmup_audits
            ),
            "all_policy_workload_summaries_present": len(summaries) == len(policies) * len(workloads),
            "oracle_gap_for_every_policy_workload": bool(summaries) and all(
                item.get("oracle_gap") is not None for item in summaries
            ),
            "safe_high_restoration": restoration_valid,
            "no_models_or_future_oracle_inputs": True,
            "no_unresolved_error": error is None,
        }
        stationary_valid = all(hard_gates.values())
        full_feedback = [item for item in summaries if item["policy"] == "FULL_FEEDBACK"]
        audit = {
            "phase": "4B_stationary_feedback_policy_comparison",
            "valid": stationary_valid,
            "smoke": self.smoke,
            "hard_gates": hard_gates,
            "error": error,
            "phase4a_reference_path": reference["path"],
            "phase4a_reference_sha256": reference["sha256"],
            "warmup_audits": warmup_audits,
            "models_trained_or_used": [],
            "controller_inputs": "latest valid measured runtime feedback within explicit freshness limits",
            "oracle_usage": "post-measurement comparison only; never supplied to feedback policies",
            "dynamic_status": "not_run_until_stationary_passes",
        }
        _write_json(self.run_dir / "phase4b_audit.json", audit)
        summary = {
            "phase": "4B_controlled_feedback_only_evaluation",
            "stationary_valid": stationary_valid,
            "smoke": self.smoke,
            "hard_gates": hard_gates,
            "error": error,
            "stationary_window_count": len(rows),
            "policy_summaries": summaries,
            "full_feedback_oracle_gaps": {
                item["workload"]: item["oracle_gap"] for item in full_feedback
            },
            "full_feedback_slo_safe": bool(full_feedback) and all(
                item["slo_pass"] for item in full_feedback
            ),
            "ready_for_dynamic_trace_evaluation": stationary_valid and not self.smoke,
            "decision_gate": "PENDING_DYNAMIC_TRACE_EVALUATION",
            "models_trained_or_used": [],
            "claim_boundary": "controlled stationary comparison against the frozen Phase 4A empirical oracle; not a theoretical optimum",
        }
        _write_json(self.run_dir / "phase4b_summary.json", summary)
        lines = [
            "# XpYd Phase 4B stationary feedback-policy evaluation", "",
            "Verdict: **%s**" % ("PASS" if stationary_valid else "FAIL"), "",
            "The empirical oracle is the frozen Phase 4A measured reference and was never exposed to feedback decisions.", "",
            "| Policy | Workload | J/request | Oracle gap | Gap class | SLO safe | Violations |",
            "|---|---|---:|---:|---|---|---:|",
        ]
        for item in summaries:
            lines.append(
                "| %s | %s | %.3f | %.2f%% | %s | %s | %d |" % (
                    item["policy"], item["workload"],
                    float(item["joules_per_request_mean"] or math.nan),
                    100.0 * float(item["oracle_gap"] or 0.0),
                    item["oracle_gap_class"], item["slo_pass"],
                    int(item["any_slo_violation_count"]),
                )
            )
        lines.extend([
            "", "Dynamic trace status: `not_run_until_stationary_passes`.",
            "", "Ready for dynamic trace evaluation: **%s**." % summary["ready_for_dynamic_trace_evaluation"],
            "", "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), "",
        ])
        if error:
            lines.extend(["Failure: `%s`" % error, ""])
        (self.run_dir / "phase4b_summary.md").write_text("\n".join(lines), encoding="utf-8")
        if not stationary_valid:
            raise Phase4BError(error or "Phase 4B stationary hard-gate audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir

    def run_dynamic(self) -> Path:
        """Run the gated formal STATIC/PASSIVE/ACTIVE dynamic comparison."""
        if self.run_dir.exists():
            raise Phase4BError("run directory already exists: %s" % self.run_dir)
        dynamic = dict(self.settings["dynamic"])
        stationary_path = Path(os.path.expandvars(str(dynamic["accepted_stationary_audit"])))
        if not stationary_path.is_file():
            raise Phase4BError("missing accepted Phase 4B stationary audit")
        stationary_raw = stationary_path.read_bytes()
        stationary_audit = json.loads(stationary_raw)
        if not stationary_audit.get("valid") or stationary_audit.get("smoke"):
            raise Phase4BError("dynamic stage requires a valid non-smoke stationary audit")
        active_smoke_path = Path(os.path.expandvars(str(
            dynamic["accepted_active_smoke_audit"]
        )))
        if not active_smoke_path.is_file():
            raise Phase4BError("missing accepted Phase 4B.1 active smoke audit")
        active_smoke_raw = active_smoke_path.read_bytes()
        active_smoke_audit = json.loads(active_smoke_raw)
        if not active_smoke_audit.get("valid") or not active_smoke_audit.get("smoke"):
            raise Phase4BError("dynamic stage requires a valid Phase 4B.1 smoke audit")
        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        reference = self._reference()
        accepted = self._accepted_frequencies(reference)
        capabilities = self._discover_exact(accepted)
        _write_json(self.run_dir / "raw" / "capabilities.json", {
            key: asdict(value) for key, value in capabilities.items()
        })
        _write_json(self.run_dir / "raw" / "accepted_stationary_reference.json", {
            "path": stationary_path.as_posix(),
            "sha256": hashlib.sha256(stationary_raw).hexdigest(),
            "valid": True,
        })
        _write_json(self.run_dir / "raw" / "accepted_active_smoke_reference.json", {
            "path": active_smoke_path.as_posix(),
            "sha256": hashlib.sha256(active_smoke_raw).hexdigest(),
            "valid": True,
        })
        _write_json(self.run_dir / "raw" / "frozen_phase4a_reference.json", {
            "path": reference["path"], "sha256": reference["sha256"],
            "oracles": reference["oracles"],
        })
        actuator = PerEndpointClockActuator(
            self.backend, capabilities,
            float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        high = {
            endpoint_id: values["HIGH"] for endpoint_id, values in accepted.items()
        }
        actuator.requested.update(high)
        _, _, all_pairs = build_registry_and_compatibility(self.config)
        policies = list(dynamic["policies"])
        trace_specs = [dict(item) for item in dynamic["traces"]]
        if self.smoke:
            trace_specs = trace_specs[:1]
        windows_per_state = 1 if self.smoke else int(dynamic["windows_per_state"])
        request_count = int(dynamic["requests_per_window"])
        dynamic_repeats = 1 if self.smoke else int(dynamic["repeats"])
        workload_by_id = {
            str(item["id"]): dict(item) for item in self.config["workloads"]
        }
        blocks = [
            (dynamic_repeat, policy, trace)
            for dynamic_repeat in range(1, dynamic_repeats + 1)
            for policy in policies
            for trace in trace_specs
        ]
        random.Random(int(dynamic.get("order_seed", 43))).shuffle(blocks)
        _write_json(self.run_dir / "raw" / "dynamic_plan.json", {
            "smoke": self.smoke,
            "policies": policies,
            "traces": trace_specs,
            "windows_per_state": windows_per_state,
            "requests_per_window": request_count,
            "repeats": dynamic_repeats,
            "block_order": [
                {
                    "dynamic_repeat": dynamic_repeat,
                    "policy": policy,
                    "trace_id": trace["id"],
                }
                for dynamic_repeat, policy, trace in blocks
            ],
            "transition_information_policy": (
                "the first decision after a state change receives only the prior "
                "observed workload context; the new context is available only "
                "after its first completed measurement window"
            ),
            "models": [],
        })
        rows: list[dict[str, Any]] = []
        request_rows: list[dict[str, Any]] = []
        route_cost_rows: list[dict[str, Any]] = []
        warmup_audits: list[dict[str, Any]] = []
        restoration_valid = False
        error: Optional[str] = None
        sequence_index = 0
        try:
            global_context = self._new_context(capabilities)
            self._actuate(
                actuator, global_context, high, "phase4b_dynamic_global_warmup_high",
                controller_action=False, policy="WARMUP", workload="all",
                iteration_id="dynamic-global-warmup",
            )
            self._write_routes(all_pairs, "phase4b_dynamic_global_all_route_warmup")
            global_dir = self._run_window(
                "dynamic-global-warmup",
                {"id": "dynamic_global_warmup", "input_len": 128,
                 "output_len": 16, "rate_rps": 0.5},
                4, high, all_pairs, windows_root, require_all_pairs=True,
            )
            global_audit = _read_json(global_dir / "audit.json")
            warmup_audits.append({"window_id": "dynamic-global-warmup", **global_audit})
            if not global_audit.get("valid"):
                raise Phase4BError("dynamic global all-route warmup failed")

            for block_index, (dynamic_repeat, policy, trace) in enumerate(blocks, 1):
                trace_id = str(trace["id"])
                states = [str(item) for item in trace["states"]]
                context = self._new_context(capabilities)
                active_costs = None
                active_prober = None
                if policy == "ACTIVE_FULL":
                    probe_settings = self.config["phase4b1"]
                    active_costs = ContextRouteCostStore(
                        alpha=float(probe_settings["route_cost_ewma_alpha"]),
                        maximum_age_s=float(
                            probe_settings["route_cost_maximum_age_s"]
                        ),
                        maximum_age_windows=int(
                            probe_settings["route_cost_maximum_age_windows"]
                        ),
                    )
                    active_prober = SafeRouteProber(
                        active_costs,
                        minimum_probe_interval_windows=int(
                            probe_settings["minimum_probe_interval_windows"]
                        ),
                    )
                if policy in {"PASSIVE_FULL", "ACTIVE_FULL"}:
                    first_workload = workload_by_id[states[0]]
                    warmup_id = "dyn-r%02d-warm-%02d-%s" % (
                        dynamic_repeat, block_index, trace_id
                    )
                    self._actuate(
                        actuator, context, high, "phase4b_dynamic_feedback_warmup_high",
                        controller_action=False, policy=policy,
                        workload=states[0], iteration_id=warmup_id,
                    )
                    initial_pairs = [list(item) for item in all_pairs]
                    context.selected_pairs = copy.deepcopy(initial_pairs)
                    self._write_routes(
                        initial_pairs, "phase4b_dynamic_feedback_observation_warmup"
                    )
                    warmup_dir = self._run_window(
                        warmup_id, first_workload, 4, high, initial_pairs,
                        windows_root, require_all_pairs=True,
                    )
                    warmup_audit = _read_json(warmup_dir / "audit.json")
                    warmup_audits.append({"window_id": warmup_id, **warmup_audit})
                    if not warmup_audit.get("valid"):
                        raise Phase4BError("dynamic feedback warmup failed: %s" % warmup_id)
                    self._observe_window(
                        warmup_id, warmup_dir, context.registry, context.telemetry
                    )

                previous_observed_context = states[0]
                block_sequence = 0
                cumulative_energy = 0.0
                for state_index, workload_id in enumerate(states):
                    workload = workload_by_id[workload_id]
                    transition_s = time.time()
                    for window_in_state in range(1, windows_per_state + 1):
                        sequence_index += 1
                        block_sequence += 1
                        iteration_id = "r%02d-t%02d-s%02d-w%02d-%s-%s" % (
                            dynamic_repeat, block_index, state_index,
                            window_in_state, policy.lower(), trace_id,
                        )
                        # The first decision at a transition cannot know the
                        # new state. Subsequent decisions may use observations
                        # from the first completed new-state window.
                        feedback_context = (
                            previous_observed_context
                            if state_index > 0 and window_in_state == 1
                            else workload_id
                        )
                        route_decision: Optional[RouteProbeDecision] = None
                        probe_safety: dict[str, list[str]] = {}
                        before_frequencies = dict(context.target_frequencies)
                        if policy == "ACTIVE_FULL":
                            if active_costs is None or active_prober is None:
                                raise Phase4BError("ACTIVE_FULL state was not initialized")
                            now = time.time()
                            evaluations = context.scheduler.evaluate_routes(
                                now_s=now,
                                workload_context_key=feedback_context,
                            )
                            eligible_routes = [
                                [item.prefill_endpoint_id, item.decode_endpoint_id]
                                for item in evaluations if item.eligible
                            ]
                            safe_routes, probe_safety = probe_safe_routes(
                                context, evaluations, now,
                                float(self.config["phase4b1"]["probe_headroom_fraction"]),
                            )
                            route_decision = active_prober.choose(
                                feedback_context,
                                compatible_routes=all_pairs,
                                eligible_routes=eligible_routes,
                                probe_safe_routes=safe_routes,
                                now_s=now,
                                sequence=block_sequence,
                            )
                            pairs = self._decide(
                                policy, workload_id, iteration_id, context,
                                capabilities, actuator, all_pairs,
                                measured_iteration=True,
                                feedback_context_key=feedback_context,
                                route_override=route_decision.route,
                                freeze_dvfs=route_decision.freeze_dvfs,
                                routing_decision_mode=route_decision.mode,
                                routing_score_basis=(
                                    "context_route_total_four_gpu_window_"
                                    "joules_per_request"
                                ),
                                route_costs_json=json.dumps(
                                    active_costs.as_dict(
                                        feedback_context, all_pairs, now,
                                        block_sequence,
                                    ),
                                    sort_keys=True,
                                ),
                            )
                        else:
                            pairs = self._decide(
                                policy, workload_id, iteration_id, context,
                                capabilities, actuator, all_pairs,
                                measured_iteration=True,
                                feedback_context_key=feedback_context,
                            )
                        control = self.control_rows[-1]
                        control["dynamic_repeat"] = dynamic_repeat
                        control["trace_id"] = trace_id
                        control["probe_safety_reasons_json"] = json.dumps(
                            probe_safety, sort_keys=True
                        )
                        control["probe_frequency_state_unchanged"] = (
                            not bool(control["route_probe"])
                            or before_frequencies == context.target_frequencies
                        )
                        window_dir = self._run_window(
                            iteration_id, workload, request_count,
                            context.target_frequencies, pairs, windows_root,
                        )
                        base_row, observed_requests = self._measurement_row(
                            iteration_id, policy, workload, dynamic_repeat,
                            context, accepted, reference, window_dir,
                        )
                        cumulative_energy += float(
                            base_row["total_gpu_gross_energy_j"]
                        )
                        signature = json.dumps({
                            "pairs": context.selected_pairs,
                            "frequencies": context.target_frequencies,
                        }, sort_keys=True)
                        dynamic_row = {
                            **base_row,
                            "trace_id": trace_id,
                            "dynamic_repeat": dynamic_repeat,
                            "sequence_index": sequence_index,
                            "block_sequence": block_sequence,
                            "state_index": state_index,
                            "window_in_state": window_in_state,
                            "workload_changed": state_index > 0 and window_in_state == 1,
                            "transition_timestamp_unix_s": transition_s,
                            "control_iteration_id": iteration_id,
                            "control_timestamp_unix_s": control["control_timestamp"],
                            "feedback_context_key": feedback_context,
                            "route_cost_observation_context": workload_id,
                            "control_state_signature": signature,
                            "dvfs_action_count": control["dvfs_action_count"],
                            "routing_changed": control["routing_changed"],
                            "fallback_reason": control["fallback_reason"],
                            "stale_endpoint_ids_json": control["stale_endpoint_ids_json"],
                            "routing_decision_mode": control["routing_decision_mode"],
                            "routing_score_basis": control["routing_score_basis"],
                            "route_probe": control["route_probe"],
                            "dvfs_frozen_for_probe": control[
                                "dvfs_frozen_for_probe"
                            ],
                            "probe_frequency_state_unchanged": control[
                                "probe_frequency_state_unchanged"
                            ],
                            "cumulative_trace_energy_j": cumulative_energy,
                        }
                        rows.append(dynamic_row)
                        for item in observed_requests:
                            request_rows.append({
                                **item,
                                "trace_id": trace_id,
                                "dynamic_repeat": dynamic_repeat,
                                "state_index": state_index,
                                "window_in_state": window_in_state,
                            })
                        if not base_row["measurement_valid"]:
                            raise Phase4BError(
                                "dynamic window audit failed: %s" % iteration_id
                            )
                        if policy == "ACTIVE_FULL":
                            if active_costs is None or route_decision is None:
                                raise Phase4BError("ACTIVE_FULL cost state is missing")
                            if len(pairs) != 1:
                                raise Phase4BError("ACTIVE_FULL requires one exact route")
                            total_energy = float(
                                base_row["total_gpu_gross_energy_j"]
                            )
                            logical_requests = int(base_row["completed_requests"])
                            snapshot = active_costs.observe(
                                workload_id,
                                pairs[0],
                                total_system_gross_energy_j=total_energy,
                                logical_requests=logical_requests,
                                timestamp_s=float(base_row["timing_end_unix_s"]),
                                sequence=block_sequence,
                            )
                            route_cost_rows.append({
                                "dynamic_repeat": dynamic_repeat,
                                "trace_id": trace_id,
                                "window_id": iteration_id,
                                "decision_context": feedback_context,
                                "observation_context": workload_id,
                                "route": "%s->%s" % tuple(pairs[0]),
                                "decision_mode": route_decision.mode,
                                "total_four_gpu_gross_energy_j": total_energy,
                                "logical_requests": logical_requests,
                                "observed_system_joules_per_request": (
                                    total_energy / logical_requests
                                ),
                                "ewma_system_joules_per_request": (
                                    snapshot.ewma_system_joules_per_request
                                ),
                                "sample_count": snapshot.sample_count,
                                "timestamp_s": snapshot.last_observation_s,
                                "sequence": block_sequence,
                                "window_amortized_not_request_attribution": True,
                            })
                        if policy in {"PASSIVE_FULL", "ACTIVE_FULL"}:
                            self._observe_window(
                                iteration_id, window_dir,
                                context.registry, context.telemetry,
                            )
                        previous_observed_context = workload_id
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            restoration_valid = True
            restore_context = self._new_context(capabilities)
            try:
                self._actuate(
                    actuator, restore_context, high,
                    "phase4b_dynamic_final_safe_high_restoration",
                    controller_action=False, policy="RESTORE", workload="all",
                    iteration_id="dynamic-final-restore",
                )
                restoration_valid = all(
                    actuator.read(endpoint_id).graphics_clock_mhz == target
                    for endpoint_id, target in high.items()
                )
            except Exception:
                restoration_valid = False

        adaptation = summarize_dynamic_adaptation(
            rows, self.control_rows, request_rows, trace_specs, policies,
            float(self.settings["slo"]["ttft_ms"]),
            float(self.settings["slo"]["tpot_ms"]),
        )
        adaptation_aggregates = []
        for trace in trace_specs:
            trace_id = str(trace["id"])
            for policy in policies:
                for transition_index in range(1, len(trace["states"])):
                    group = [
                        item for item in adaptation
                        if item["trace_id"] == trace_id
                        and item["policy"] == policy
                        and int(item["transition_index"]) == transition_index
                    ]
                    reaction_times = [
                        item["time_to_first_reaction_s"] for item in group
                        if item["time_to_first_reaction_s"] is not None
                    ]
                    adaptation_aggregates.append({
                        "policy": policy,
                        "trace_id": trace_id,
                        "transition_index": transition_index,
                        "from_workload": str(
                            trace["states"][transition_index - 1]
                        ),
                        "to_workload": str(trace["states"][transition_index]),
                        "repeat_count": len(group),
                        "reaction_observed_count": sum(
                            bool(item["reaction_observed"]) for item in group
                        ),
                        "reaction_observed_rate": (
                            sum(bool(item["reaction_observed"]) for item in group)
                            / len(group) if group else None
                        ),
                        "reaction_time_mean_s": _mean(reaction_times),
                        "reaction_time_std_s": _std(reaction_times),
                        "reaction_time_ci95_half_width_s": _ci95(reaction_times),
                        "settling_time_mean_s": _mean([
                            item["settling_time_s"] for item in group
                        ]),
                        "settling_time_std_s": _std([
                            item["settling_time_s"] for item in group
                        ]),
                        "slo_violation_count": sum(
                            int(item["slo_violation_count"]) for item in group
                        ),
                        "routing_change_count": sum(
                            int(item["routing_change_count"]) for item in group
                        ),
                        "oscillation_or_reversal_count": sum(
                            int(item["oscillation_or_reversal_count"])
                            for item in group
                        ),
                        "fallback_event_count": sum(
                            int(item["fallback_event_count"]) for item in group
                        ),
                        "stale_telemetry_event_count": sum(
                            int(item["stale_telemetry_event_count"])
                            for item in group
                        ),
                    })
        policy_summaries = []
        for policy in policies:
            for trace in trace_specs:
                trace_id = str(trace["id"])
                for dynamic_repeat in range(1, dynamic_repeats + 1):
                    group = [
                        item for item in rows
                        if item["policy"] == policy
                        and item["trace_id"] == trace_id
                        and int(item["dynamic_repeat"]) == dynamic_repeat
                    ]
                    group_requests = [
                        item for item in request_rows
                        if item["policy"] == policy
                        and item["trace_id"] == trace_id
                        and int(item["dynamic_repeat"]) == dynamic_repeat
                    ]
                    total_energy = sum(
                        float(item["total_gpu_gross_energy_j"]) for item in group
                    )
                    offline_piecewise_oracle = sum(
                        float(reference["oracles"][str(item["workload"])]["J_per_request"])
                        * int(item["completed_requests"])
                        for item in group
                    )
                    violations = sum(
                        float(item["ttft_ms"])
                        > float(self.settings["slo"]["ttft_ms"])
                        or float(item["tpot_ms"])
                        > float(self.settings["slo"]["tpot_ms"])
                        for item in group_requests
                    )
                    probe_rows = [item for item in group if item["route_probe"]]
                    exploit_rows = [
                        item for item in group
                        if item["routing_decision_mode"] == "EXPLOIT_RECENT"
                    ]
                    signatures = [
                        str(item["control_state_signature"])
                        for item in sorted(
                            group, key=lambda value: int(value["block_sequence"])
                        )
                    ]
                    signature_changes = sum(
                        signatures[index] != signatures[index - 1]
                        for index in range(1, len(signatures))
                    )
                    probe_energy = sum(
                        float(item["total_gpu_gross_energy_j"])
                        for item in probe_rows
                    )
                    policy_summaries.append({
                        "policy": policy,
                        "trace_id": trace_id,
                        "dynamic_repeat": dynamic_repeat,
                        "window_count": len(group),
                        "request_count": len(group_requests),
                        "total_gpu_gross_energy_j": total_energy,
                        "joules_per_request": (
                            total_energy / len(group_requests)
                            if group_requests else None
                        ),
                        "offline_piecewise_phase4a_oracle_energy_j": (
                            offline_piecewise_oracle
                        ),
                        "offline_piecewise_oracle_gap": (
                            (total_energy - offline_piecewise_oracle)
                            / offline_piecewise_oracle
                            if offline_piecewise_oracle else None
                        ),
                        "slo_violation_count": violations,
                        "slo_violation_rate": (
                            violations / len(group_requests)
                            if group_requests else None
                        ),
                        "ttft_mean_ms": _mean([
                            item.get("mean_ttft_ms") for item in group
                        ]),
                        "tpot_mean_ms": _mean([
                            item.get("mean_tpot_ms") for item in group
                        ]),
                        "itl_mean_ms": _mean([
                            item.get("mean_itl_ms") for item in group
                        ]),
                        "e2e_mean_ms": _mean([
                            item.get("mean_e2e_latency_ms") for item in group
                        ]),
                        "measurement_valid": bool(group) and all(
                            item["measurement_valid"] for item in group
                        ),
                        "dvfs_action_count": sum(
                            int(item["dvfs_action_count"]) for item in group
                        ),
                        "routing_change_count": sum(
                            bool(item["routing_changed"]) for item in group
                        ),
                        "unique_route_count": len({
                            str(item["selected_pairs_json"]) for item in group
                        }),
                        "probe_window_count": len(probe_rows),
                        "probe_unseen_count": sum(
                            item["routing_decision_mode"] == "PROBE_UNSEEN"
                            for item in group
                        ),
                        "probe_stale_count": sum(
                            item["routing_decision_mode"] == "PROBE_STALE"
                            for item in group
                        ),
                        "exploit_window_count": len(exploit_rows),
                        "probe_window_energy_j": probe_energy,
                        "probe_window_energy_fraction": (
                            probe_energy / total_energy if total_energy else None
                        ),
                        "fallback_event_count": sum(
                            bool(item["fallback_reason"]) for item in group
                        ),
                        "stale_telemetry_event_count": sum(
                            bool(json.loads(item["stale_endpoint_ids_json"]))
                            for item in group
                        ),
                        "control_oscillation_count": max(
                            0, signature_changes - 1
                        ),
                    })

        policy_aggregates = []
        for policy in policies:
            for trace in trace_specs:
                trace_id = str(trace["id"])
                group = [
                    item for item in policy_summaries
                    if item["policy"] == policy and item["trace_id"] == trace_id
                ]
                policy_aggregates.append({
                    "policy": policy,
                    "trace_id": trace_id,
                    "repeat_count": len(group),
                    "energy_mean_j": _mean([
                        item["total_gpu_gross_energy_j"] for item in group
                    ]),
                    "energy_std_j": _std([
                        item["total_gpu_gross_energy_j"] for item in group
                    ]),
                    "energy_ci95_half_width_j": _ci95([
                        item["total_gpu_gross_energy_j"] for item in group
                    ]),
                    "joules_per_request_mean": _mean([
                        item["joules_per_request"] for item in group
                    ]),
                    "joules_per_request_std": _std([
                        item["joules_per_request"] for item in group
                    ]),
                    "slo_violation_count": sum(
                        int(item["slo_violation_count"]) for item in group
                    ),
                    "ttft_mean_ms": _mean([
                        item["ttft_mean_ms"] for item in group
                    ]),
                    "ttft_std_ms": _std([
                        item["ttft_mean_ms"] for item in group
                    ]),
                    "tpot_mean_ms": _mean([
                        item["tpot_mean_ms"] for item in group
                    ]),
                    "tpot_std_ms": _std([
                        item["tpot_mean_ms"] for item in group
                    ]),
                    "itl_mean_ms": _mean([
                        item["itl_mean_ms"] for item in group
                    ]),
                    "itl_std_ms": _std([
                        item["itl_mean_ms"] for item in group
                    ]),
                    "e2e_mean_ms": _mean([
                        item["e2e_mean_ms"] for item in group
                    ]),
                    "probe_window_count": sum(
                        int(item["probe_window_count"]) for item in group
                    ),
                    "probe_unseen_count": sum(
                        int(item["probe_unseen_count"]) for item in group
                    ),
                    "probe_stale_count": sum(
                        int(item["probe_stale_count"]) for item in group
                    ),
                    "fallback_event_count": sum(
                        int(item["fallback_event_count"]) for item in group
                    ),
                    "stale_telemetry_event_count": sum(
                        int(item["stale_telemetry_event_count"]) for item in group
                    ),
                    "control_oscillation_count": sum(
                        int(item["control_oscillation_count"]) for item in group
                    ),
                })

        dynamic_fields = (
            "run_id", "trace_id", "dynamic_repeat", "sequence_index",
            "block_sequence", "policy", "state_index",
            "window_in_state", "workload", "workload_changed",
            "transition_timestamp_unix_s", "control_iteration_id",
            "control_timestamp_unix_s", "feedback_context_key",
            "route_cost_observation_context",
            "control_state_signature", "timing_start_unix_s", "timing_end_unix_s",
            "duration_s", "completed_requests", "output_tokens",
            "measurement_valid", "slo_pass", "total_gpu_gross_energy_j",
            "joules_per_request", "joules_per_output_token", "mean_ttft_ms",
            "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms",
            "mean_e2e_latency_ms", "throughput_requests_s",
            "route_distribution_json", "selected_pairs_json",
            "P0_requested_freq_mhz", "P0_requested_level", "P0_observed_freq_mhz",
            "P1_requested_freq_mhz", "P1_requested_level", "P1_observed_freq_mhz",
            "D0_requested_freq_mhz", "D0_requested_level", "D0_observed_freq_mhz",
            "D1_requested_freq_mhz", "D1_requested_level", "D1_observed_freq_mhz",
            "dvfs_action_count", "routing_changed", "routing_decision_mode",
            "routing_score_basis", "route_probe", "dvfs_frozen_for_probe",
            "probe_frequency_state_unchanged", "fallback_reason",
            "stale_endpoint_ids_json", "cumulative_trace_energy_j",
            "oracle_gap", "oracle_gap_class",
            "hard_gates_json",
        )
        _csv(self.run_dir / "phase4b_dynamic_results.csv", rows, dynamic_fields)
        _csv(self.run_dir / "phase4b_dynamic_adaptation.csv", adaptation,
             tuple(adaptation[0]) if adaptation else ())
        _csv(
            self.run_dir / "phase4b_dynamic_adaptation_aggregate.csv",
            adaptation_aggregates,
            tuple(adaptation_aggregates[0]) if adaptation_aggregates else (),
        )
        _csv(self.run_dir / "phase4b_policy_summary.csv", policy_summaries,
             tuple(policy_summaries[0]) if policy_summaries else ())
        _csv(self.run_dir / "phase4b_policy_aggregate.csv", policy_aggregates,
             tuple(policy_aggregates[0]) if policy_aggregates else ())
        _csv(
            self.run_dir / "phase4b_route_cost_observations.csv",
            route_cost_rows,
            tuple(route_cost_rows[0]) if route_cost_rows else (),
        )
        oracle_rows = [{
            key: item[key] for key in (
                "policy", "trace_id", "dynamic_repeat",
                "total_gpu_gross_energy_j",
                "offline_piecewise_phase4a_oracle_energy_j",
                "offline_piecewise_oracle_gap", "slo_violation_count",
                "slo_violation_rate",
            )
        } for item in policy_summaries]
        _csv(self.run_dir / "phase4b_oracle_gap.csv", oracle_rows,
             tuple(oracle_rows[0]) if oracle_rows else ())
        _csv(self.run_dir / "phase4b_control_actions.csv", self.control_rows, (
            "control_timestamp", "control_iteration_id", "dynamic_repeat",
            "trace_id", "policy", "workload",
            "feedback_context_key", "measured_iteration", "telemetry_fresh",
            "stale_endpoint_ids_json", "stale_endpoint_ages_s_json",
            "fallback_reason", "selected_pairs_json", "requested_frequencies_json",
            "routing_changed", "dvfs_action_count", "routing_decision_mode",
            "routing_score_basis", "route_probe", "dvfs_frozen_for_probe",
            "route_cost_context", "route_costs_json",
            "probe_safety_reasons_json", "probe_frequency_state_unchanged",
            "candidate_evaluations_json",
            "dvfs_recommendations_json",
        ))
        _csv(self.run_dir / "raw" / "dynamic_requests.csv", request_rows, (
            "trace_id", "dynamic_repeat", "state_index", "window_in_state",
            "window_id", "policy",
            "workload", "repeat", "request_id", "prefill_endpoint_id",
            "decode_endpoint_id", "ttft_ms", "tpot_ms", "mean_itl_ms",
            "e2e_latency_ms",
        ))
        _csv(self.run_dir / "raw" / "actuator_actions.csv", self.actuator_rows, (
            "timestamp_unix_s", "policy", "workload", "control_iteration_id",
            "controller_action", "endpoint_id", "node", "gpu_id",
            "previous_requested_freq_mhz", "requested_freq_mhz",
            "observed_freq_before_mhz", "observed_freq_after_mhz",
            "command_status", "readback_valid", "transition_readback_latency_s",
            "reason", "error",
        ))
        _csv(self.run_dir / "raw" / "endpoint_telemetry.csv", self.telemetry_rows, (
            "window_id", "endpoint_id", "timestamp_s", "health", "queue",
            "kv_cache_usage_frac", "ttft_ms", "tpot_ms", "power_w", "energy_j",
            "observed_frequency_mhz", "telemetry_age_s_at_observation",
            "measurement_valid",
        ))
        expected = (
            dynamic_repeats
            *
            len(policies)
            * sum(len(trace["states"]) for trace in trace_specs)
            * windows_per_state
        )
        active_controls = [
            item for item in self.control_rows if item.get("policy") == "ACTIVE_FULL"
        ]
        probe_controls = [
            item for item in active_controls if item.get("route_probe")
        ]
        inherited_gates = [
            json.loads(item["hard_gates_json"]) for item in rows
        ]
        active_cost_formula_valid = (
            len(route_cost_rows)
            == sum(item["policy"] == "ACTIVE_FULL" for item in rows)
            and all(
                abs(
                    float(item["observed_system_joules_per_request"])
                    - float(item["total_four_gpu_gross_energy_j"])
                    / int(item["logical_requests"])
                ) < 1e-9
                for item in route_cost_rows
            )
        )
        active_probe_safety_valid = bool(probe_controls) and all(
            not json.loads(item["probe_safety_reasons_json"])[
                "%s->%s" % tuple(json.loads(item["selected_pairs_json"])[0])
            ]
            for item in probe_controls
        )
        hard_gates = {
            "accepted_full_stationary_audit": bool(stationary_audit.get("valid")),
            "accepted_phase4b1_active_smoke_audit": bool(
                active_smoke_audit.get("valid")
            ),
            "frozen_phase4a_reference_valid": bool(reference["summary"].get("valid")),
            "exact_three_policy_comparison": tuple(policies) == (
                "STATIC", "PASSIVE_FULL", "ACTIVE_FULL"
            ),
            "complete_dynamic_plan": len(rows) == expected,
            "all_dynamic_measurements_valid": len(rows) == expected and all(
                item.get("measurement_valid") for item in rows
            ),
            "all_dynamic_warmups_valid": bool(warmup_audits) and all(
                item.get("valid") for item in warmup_audits
            ),
            "transition_context_has_no_future_leak": all(
                item["feedback_context_key"] != item["workload"]
                for item in rows if item["workload_changed"]
            ),
            "exact_request_ids_tokens_and_routes": bool(inherited_gates) and all(
                gate.get("logical_request_id")
                and gate.get("requested_output_tokens")
                and gate.get("endpoint_assignment")
                and gate.get("explicit_compatibility")
                for gate in inherited_gates
            ),
            "active_context_route_system_cost_formula": active_cost_formula_valid,
            "active_probe_modes_are_explicit": bool(active_controls) and all(
                item["routing_decision_mode"] in {
                    "PROBE_UNSEEN", "PROBE_STALE", "EXPLOIT_RECENT",
                    "SAFE_FALLBACK_NO_PROBE", "NO_ELIGIBLE_ROUTE",
                }
                for item in active_controls
            ),
            "all_active_probes_passed_strict_safety_gate": (
                active_probe_safety_valid
            ),
            "active_probe_windows_freeze_dvfs": bool(probe_controls) and all(
                item["dvfs_frozen_for_probe"]
                and item["probe_frequency_state_unchanged"]
                and int(item["dvfs_action_count"]) == 0
                for item in probe_controls
            ),
            "safe_high_restoration": restoration_valid,
            "no_models_or_online_oracle_inputs": True,
            "no_unresolved_error": error is None,
        }
        valid = all(hard_gates.values())
        aggregate_by_policy_trace = {
            (item["policy"], item["trace_id"]): item
            for item in policy_aggregates
        }
        energy_comparison = {}
        for trace in trace_specs:
            trace_id = str(trace["id"])
            static = aggregate_by_policy_trace[("STATIC", trace_id)]
            passive = aggregate_by_policy_trace[("PASSIVE_FULL", trace_id)]
            active = aggregate_by_policy_trace[("ACTIVE_FULL", trace_id)]
            energy_comparison[trace_id] = {
                "passive_savings_vs_static": relative_energy_savings(
                    static["energy_mean_j"], passive["energy_mean_j"]
                ),
                "active_savings_vs_static": relative_energy_savings(
                    static["energy_mean_j"], active["energy_mean_j"]
                ),
                "active_savings_vs_passive": relative_energy_savings(
                    passive["energy_mean_j"], active["energy_mean_j"]
                ),
                "active_minus_passive_energy_fraction": (
                    (float(active["energy_mean_j"]) - float(passive["energy_mean_j"]))
                    / float(passive["energy_mean_j"])
                ),
                "interpretation": (
                    "net measured policy difference; not pure causal probe energy"
                ),
            }
        audit = {
            "phase": "4B_dynamic_feedback_adaptation",
            "valid": valid, "smoke": self.smoke,
            "hard_gates": hard_gates, "error": error,
            "accepted_stationary_audit": stationary_path.as_posix(),
            "accepted_stationary_sha256": hashlib.sha256(stationary_raw).hexdigest(),
            "accepted_active_smoke_audit": active_smoke_path.as_posix(),
            "accepted_active_smoke_sha256": hashlib.sha256(
                active_smoke_raw
            ).hexdigest(),
            "warmup_audits": warmup_audits,
            "models_trained_or_used": [],
            "oracle_usage": "post-hoc offline piecewise reference only; never a controller input",
        }
        _write_json(self.run_dir / "phase4b_audit.json", audit)
        summary = {
            "phase": "4B_dynamic_feedback_adaptation",
            "valid": valid, "smoke": self.smoke,
            "hard_gates": hard_gates, "error": error,
            "dynamic_window_count": len(rows),
            "policy_summaries": policy_summaries,
            "policy_aggregates": policy_aggregates,
            "adaptation_summaries": adaptation,
            "adaptation_aggregates": adaptation_aggregates,
            "energy_comparison": energy_comparison,
            "probe_overhead_claim_boundary": (
                "probe-window energy and ACTIVE-minus-PASSIVE net energy are "
                "descriptive; neither is request-level physical attribution "
                "or a pure causal probe-energy estimate"
            ),
            "models_trained_or_used": [],
            "decision_gate": "READY_FOR_COMBINED_PHASE4B_ANALYSIS" if valid else "BLOCKED",
            "claim_boundary": "small measured dynamic traces; no theoretical oracle or model claim",
        }
        _write_json(self.run_dir / "phase4b_summary.json", summary)
        lines = [
            "# XpYd Phase 4B dynamic feedback evaluation", "",
            "Verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
            "The first decision after every transition used only pre-transition observed context.", "",
            "| Policy | Trace | Repeat | Energy (J) | Offline oracle gap | SLO violations | Probes | Stale events |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in policy_summaries:
            lines.append(
                "| %s | %s | %d | %.2f | %.2f%% | %d | %d | %d |" % (
                    item["policy"], item["trace_id"],
                    int(item["dynamic_repeat"]),
                    float(item["total_gpu_gross_energy_j"]),
                    100.0 * float(item["offline_piecewise_oracle_gap"]),
                    int(item["slo_violation_count"]),
                    int(item["probe_window_count"]),
                    int(item["stale_telemetry_event_count"]),
                )
            )
        lines.extend([
            "", "Mean-energy comparisons: `%s`." % energy_comparison,
            "", "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), "",
        ])
        if error:
            lines.extend(["Failure: `%s`" % error, ""])
        (self.run_dir / "phase4b_summary.md").write_text("\n".join(lines), encoding="utf-8")
        if not valid:
            raise Phase4BError(error or "Phase 4B dynamic hard-gate audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def load_config(path: Path) -> dict[str, Any]:
    config = load_phase3d_config(path)
    settings = config.get("phase4b")
    if not isinstance(settings, dict):
        raise Phase4BError("Phase 4B config requires a phase4b object")
    if int(settings.get("repeats", 0)) < 3:
        raise Phase4BError("Phase 4B requires at least three stationary repeats")
    if int(settings.get("requests_per_repeat", 0)) < 1:
        raise Phase4BError("Phase 4B requests_per_repeat must be positive")
    policies = tuple(settings.get("policies", ()))
    if policies != POLICIES:
        raise Phase4BError("Phase 4B must compare exactly the four required runtime policies")
    endpoints = {str(item["endpoint_id"]): item for item in config["endpoints"]}
    if set(endpoints) != {"P0", "P1", "D0", "D1"}:
        raise Phase4BError("Phase 4B requires exactly P0/P1/D0/D1")
    if any(endpoints[key]["node"] != "neptune" for key in ("P0", "P1")):
        raise Phase4BError("Phase 4B prefill endpoints must remain on Neptune")
    if any(endpoints[key]["node"] != "io" for key in ("D0", "D1")):
        raise Phase4BError("Phase 4B decode endpoints must remain on IO")
    if any(int(item["tp_degree"]) != 1 for item in endpoints.values()):
        raise Phase4BError("Phase 4B fixes every endpoint at TP1")
    dynamic = settings.get("dynamic")
    if not isinstance(dynamic, dict):
        raise Phase4BError("Phase 4B config requires a dynamic object")
    if tuple(dynamic.get("policies", ())) != (
        "STATIC", "PASSIVE_FULL", "ACTIVE_FULL"
    ):
        raise Phase4BError(
            "dynamic evaluation must compare exactly STATIC, PASSIVE_FULL, ACTIVE_FULL"
        )
    if int(dynamic.get("repeats", 0)) < 3:
        raise Phase4BError("formal dynamic evaluation requires at least three repeats")
    if not str(dynamic.get("accepted_active_smoke_audit", "")):
        raise Phase4BError("dynamic evaluation requires an accepted active smoke audit")
    workload_ids = {str(item["id"]) for item in config["workloads"]}
    for trace in dynamic.get("traces", []):
        if len(trace.get("states", [])) != 3:
            raise Phase4BError("each dynamic trace must contain exactly three states")
        if not set(map(str, trace["states"])).issubset(workload_ids):
            raise Phase4BError("dynamic trace references an unknown workload")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("stationary", "dynamic"), default="stationary")
    args = parser.parse_args()
    harness = Phase4BEvaluationHarness(
        load_config(Path(args.config)), args.run_id,
        smoke=args.smoke, stage=args.stage,
    )
    if args.stage == "dynamic":
        harness.run_dynamic()
    else:
        harness.run_stationary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
