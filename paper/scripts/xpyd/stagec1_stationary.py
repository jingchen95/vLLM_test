"""Stage C1c stationary, evidence-window-bounded safe-search validation.

This runner deliberately reuses the Phase 3C measurement path.  Routing
remains the proxy's fixed four-pair round robin.  ONE-STEP-FEEDBACK and
ADAPTIVE-SAFE-SEARCH share evidence, SLOs, safety gates, action space, and cold
start; only the permitted downward jump size differs.  This stage does not
claim event-driven safety between physical measurement windows.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from xpyd.feedback_scheduler import DVFSAction
from xpyd.adaptive_exploration import (
    AdaptiveExplorationConfig,
    AdaptivePoolController,
    evidence_from_window,
)
from xpyd.phase3c_substrate import Phase3CSubstrateHarness, _read_json, _write_json
from xpyd.phase3d_control import (
    EndpointClockCapability,
    NvidiaSmiClockBackend,
    PerEndpointClockActuator,
    Phase3DClosedLoopHarness,
)
from xpyd.stagec0_capacity import analyze_point, percentile
from xpyd.workload_harness import build_trace, load_config as load_workload_config
from xpyd.phase3c_substrate import load_config as load_runtime_config


class StageC1Error(RuntimeError):
    """A Stage C1 semantic, measurement, or physical-control gate failed."""


POLICIES = ("STATIC-HIGH", "ONE-STEP-FEEDBACK", "ADAPTIVE-SAFE-SEARCH")
POOLS = {"P": ("P0", "P1"), "D": ("D0", "D1")}


@dataclass
class ControllerContext:
    registry: Any
    telemetry: Any
    scheduler: Any
    targets: dict[str, int]
    adaptive_pools: dict[str, AdaptivePoolController] | None = None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _trace_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    semantic = [
        {
            "arrival_time_s": float(row["arrival_time_s"]),
            "input_len": int(row["input_len"]),
            "output_len": int(row["output_len"]),
            "workload_class": str(row["workload_class"]),
            "segment": str(row["segment"]),
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_configs(runtime_path: Path, experiment_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_runtime_config(runtime_path)
    experiment = load_workload_config(experiment_path)
    settings = experiment.get("stagec1")
    if not isinstance(settings, Mapping):
        raise StageC1Error("stagec1 settings are required")
    headroom_diagnostic = "stagec1_headroom" in experiment
    expected_policies = ["STATIC-HIGH", "FEEDBACK-DVFS"] if headroom_diagnostic else list(POLICIES)
    if settings.get("policies") != expected_policies:
        raise StageC1Error("Stage C1c compares exactly STATIC-HIGH, ONE-STEP-FEEDBACK, and ADAPTIVE-SAFE-SEARCH")
    if int(settings.get("max_concurrency", 0)) != 4:
        raise StageC1Error("Stage C1 requires max concurrency 4")
    if settings.get("routing") != "fixed four-pair round-robin":
        raise StageC1Error("Stage C1 routing must remain fixed four-pair round-robin")
    if settings.get("pool_frequency_state") != {"P": ["P0", "P1"], "D": ["D0", "D1"]}:
        raise StageC1Error("Stage C1 requires shared P-pool and D-pool frequency states")
    slo_regimes = {
        "TIGHT": (600.0, 65.0, 0.95),
        "MODERATE": (625.0, 66.0, 0.95),
        "LOOSE": (650.0, 70.0, 0.95),
    }
    slo_regime = str(settings.get("slo_regime", "TIGHT")).upper()
    if slo_regime not in slo_regimes:
        raise StageC1Error("unsupported Stage C1 SLO regime")
    if not headroom_diagnostic and slo_regime != "TIGHT":
        raise StageC1Error("full Stage C1c must use the TIGHT 600/65 ms SLO")
    slo = settings.get("slo", {})
    expected_slo = slo_regimes[slo_regime]
    observed_slo = (
        float(slo.get("ttft_ms", 0)), float(slo.get("tpot_ms", 0)),
        float(slo.get("attainment", 0)),
    )
    if observed_slo != expected_slo:
        raise StageC1Error("Stage C1 SLO does not match selected calibrated regime")
    if not headroom_diagnostic:
        if not isinstance(settings.get("adaptive_exploration", {}), Mapping):
            raise StageC1Error("Stage C1c requires adaptive_exploration settings")
        if int(settings.get("repetitions", 0)) != 3:
            raise StageC1Error("full Stage C1c requires exactly 3 repetitions")
        if int(settings.get("windows_per_block", 0)) != 4:
            raise StageC1Error("full Stage C1c requires 4 windows per block")
        expected_levels = {
            "P0": {"LOW": 1260, "MID": 1890, "HIGH": 2520},
            "P1": {"LOW": 1260, "MID": 1890, "HIGH": 2520},
            "D0": {"LOW": 750, "MID": 1125, "HIGH": 1500},
            "D1": {"LOW": 750, "MID": 1125, "HIGH": 1500},
        }
        if settings.get("frequency_levels_mhz") != expected_levels:
            raise StageC1Error("Stage C1c action space must be the validated three-level P/D clocks")
        scenarios = {str(item["id"]): item for item in experiment.get("scenarios", [])}
        for scenario_id in ("stationary_medium", "stationary_high_sustainable"):
            segment = scenarios[scenario_id]["segments"][0]
            if int(segment["request_count"]) != 20 or segment["class_weights"] != {
                "SS": 0.35, "SL": 0.15, "LS": 0.35, "LL": 0.15,
            }:
                raise StageC1Error("Stage C1c requires 20 requests/window and 35/15/35/15 mix")
    points = settings.get("load_points", [])
    observed = {str(item["id"]): float(item["offered_rate_rps"]) for item in points}
    if observed != {"MEDIUM": 0.10, "HIGH-SUSTAINABLE": 0.20}:
        raise StageC1Error("Stage C1 requires calibrated 0.10 and 0.20 req/s loads")
    endpoints = {str(item["endpoint_id"]): item for item in runtime["endpoints"]}
    expected_nodes = {"P0": "neptune", "P1": "neptune", "D0": "europa", "D1": "europa"}
    if {key: str(value["node"]) for key, value in endpoints.items()} != expected_nodes:
        raise StageC1Error("runtime topology must be Neptune L40S + Europa L4")
    if any(int(value["tp_degree"]) != 1 for value in endpoints.values()):
        raise StageC1Error("Stage C1 requires TP=1")
    if any(bool(settings.get(key, False)) for key in ("request_aware_routing", "probing", "predictor", "oracle")):
        raise StageC1Error("Stage C1 forbids routing intelligence, probing, predictors, and oracle use")

    # Reuse the accepted scheduler thresholds, but align them to the calibrated
    # contract and enable the already-tested direct latency-risk recovery path.
    runtime["phase3d"] = {
        "minimum_dwell_s": float(settings.get("minimum_dwell_s", 1.0)),
        "feedback": {
            "ttft_slo_ms": expected_slo[0],
            "tpot_slo_ms": expected_slo[1],
            "safety_fraction": 0.9,
            "telemetry_max_age_s": float(settings.get("telemetry_max_age_s", 300.0)),
            "max_queue_depth": int(settings.get("max_queue_depth", 1)),
            "max_kv_cache_usage_frac": float(settings.get("max_kv_cache_usage_frac", 0.90)),
            "step_down_fraction": float(settings.get("step_down_fraction", 0.50)),
            "step_up_fraction": float(settings.get("step_up_fraction", 0.80)),
            "severe_queue_depth": int(settings.get("severe_queue_depth", 4)),
            "severe_kv_cache_usage_frac": float(settings.get("severe_kv_cache_usage_frac", 0.97)),
            "low_kv_cache_usage_frac": float(settings.get("low_kv_cache_usage_frac", 0.50)),
            "ewma_alpha": float(settings.get("ewma_alpha", 0.50)),
            "latency_risk_to_max": True,
        },
    }
    return runtime, experiment


def policy_order_for_repeat(repeat: int) -> tuple[str, ...]:
    """Balanced cyclic order for the three full-C1c policies."""
    offset = (int(repeat) - 1) % len(POLICIES)
    return POLICIES[offset:] + POLICIES[:offset]


def coordinate_pool(
    pool: str, endpoint_ids: Sequence[str], recommendations: Sequence[Any],
    current: int, high: int, admission_risk: bool,
) -> dict[str, Any]:
    """Conservatively combine endpoint recommendations into one pool action."""
    by_id = {item.endpoint_id: item for item in recommendations}
    items = [by_id[item] for item in endpoint_ids]
    actions = [item.action for item in items]
    reasons = [str(item.reason) for item in items]
    if admission_risk:
        target, action = high, "RECOVER_HIGH" if current != high else "HOLD"
        reason = "fresh client admission/backlog risk; pool fail-safe HIGH"
    elif any(action in {DVFSAction.RECOVER_HIGH, DVFSAction.FALLBACK_MAX} for action in actions):
        target, action = high, "RECOVER_HIGH" if current != high else "HOLD"
        reason = "at least one pool endpoint requested direct safe recovery"
    elif any(action == DVFSAction.STEP_UP for action in actions):
        target = max(int(item.target_freq_mhz) for item in items)
        action, reason = "STEP_UP", "at least one pool endpoint reported pressure"
    elif all(action == DVFSAction.STEP_DOWN for action in actions):
        targets = {int(item.target_freq_mhz) for item in items}
        target = targets.pop() if len(targets) == 1 else current
        action = "STEP_DOWN" if target != current else "HOLD"
        reason = "both pool endpoints reported stable measured headroom"
    else:
        target, action = current, "HOLD"
        reason = "pool downshift requires agreement from both endpoints"
    return {
        "pool": pool, "action": action, "current_freq_mhz": current,
        "target_freq_mhz": target, "reason": reason,
        "endpoint_actions_json": json.dumps(
            {item.endpoint_id: item.action.value for item in items}, sort_keys=True
        ),
        "endpoint_reasons_json": json.dumps(
            {item.endpoint_id: str(item.reason) for item in items}, sort_keys=True
        ),
    }


def admission_risk(metrics: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("backlog_growing")
        or float(metrics.get("client_waiting_at_offered_close") or 0) > 0
        or float(metrics.get("admission_delay_p95_ms") or 0)
        > float(settings.get("admission_risk_p95_ms", 250.0))
    )


def frequency_residency(
    rows: Sequence[Mapping[str, Any]], endpoint_id: str,
    levels: Mapping[str, int],
) -> dict[str, float]:
    """Return measurement-window time fractions at requested LOW/MID/HIGH."""
    durations = {level: 0.0 for level in ("LOW", "MID", "HIGH")}
    inverse = {int(value): key for key, value in levels.items()}
    for row in rows:
        frequency = int(row["%s_freq_mhz" % endpoint_id])
        if frequency not in inverse:
            raise StageC1Error("unrecognized requested frequency in residency audit")
        durations[inverse[frequency]] += float(row["observation_duration_s"])
    total = sum(durations.values())
    return {key: value / total if total else 0.0 for key, value in durations.items()}


def oscillation_count(decisions: Sequence[Mapping[str, Any]]) -> int:
    """Count reversals between non-HOLD up/down controller actions."""
    directions = []
    for row in decisions:
        action = str(row["action"])
        if action == "STEP_DOWN":
            directions.append(-1)
        elif action in {"STEP_UP", "RECOVER_HIGH"}:
            directions.append(1)
    return sum(left != right for left, right in zip(directions, directions[1:]))


def observed_metric_stats(
    windows: Sequence[Mapping[str, Any]], endpoint_id: str, metric: str,
) -> dict[str, Any]:
    """Summarize only explicitly observed endpoint values; never fill zero."""
    values = [
        float(window[endpoint_id][metric])
        for window in windows
        if endpoint_id in window and window[endpoint_id].get(metric) not in (None, "")
    ]
    return {
        "observed_windows": len(values),
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


class StageC1Harness(Phase3DClosedLoopHarness):
    def __init__(
        self, runtime: Mapping[str, Any], experiment: Mapping[str, Any],
        run_id: str | None = None, backend: NvidiaSmiClockBackend | None = None,
        smoke: bool = False,
    ) -> None:
        config = copy.deepcopy(dict(runtime))
        # The inherited constructor only needs a path for a control file; C1
        # never publishes it and the proxy therefore keeps default RR routing.
        config["routing_control_file"] = "/tmp/xpyd_stagec1_unused_routes.json"
        super().__init__(config, run_id=run_id, backend=backend)
        self.experiment = copy.deepcopy(dict(experiment))
        self.settings = self.experiment["stagec1"]
        self.smoke = smoke
        self.run_dir = Path(os.path.expandvars(str(experiment["output_root"]))) / self.run_id
        self.action_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.block_search_rows: list[dict[str, Any]] = []

    @staticmethod
    def _levels(capability: EndpointClockCapability) -> dict[str, int]:
        return {
            "LOW": capability.selected_low_mhz,
            "MID": capability.selected_mid_mhz,
            "HIGH": capability.selected_high_mhz,
        }

    def _actuate_targets(
        self, actuator: PerEndpointClockActuator, context: ControllerContext,
        targets: Mapping[str, int], reason: str, policy: str, block_id: str,
    ) -> dict[str, dict[str, Any]]:
        action_by_endpoint: dict[str, dict[str, Any]] = {}
        for endpoint_id in sorted(targets):
            target = int(targets[endpoint_id])
            changed = actuator.requested.get(endpoint_id) != target
            if changed:
                row = actuator.actuate(endpoint_id, target, reason)
                row.update({"policy": policy, "block_id": block_id})
                if row.get("timestamp_unix_s") is not None and row.get("transition_readback_latency_s") is not None:
                    row["readback_timestamp_unix_s"] = (
                        float(row["timestamp_unix_s"])
                        + float(row["transition_readback_latency_s"])
                    )
                self.action_rows.append(row)
                action_by_endpoint[endpoint_id] = row
                if row["command_status"] != "success" or not row["readback_valid"]:
                    raise StageC1Error("physical actuation failed for %s" % endpoint_id)
            state = context.registry.get_state(endpoint_id)
            state.freq_mhz = target
            context.registry.update_state(state)
            context.targets[endpoint_id] = target
            # FeedbackScheduler validates that its registry state already
            # reflects the successful hardware readback.  Update state before
            # recording the actuation (important for fixed-point diagnostics
            # and for controller recovery transitions).
            if changed:
                readback_timestamp = float(
                    action_by_endpoint[endpoint_id].get(
                        "readback_timestamp_unix_s",
                        action_by_endpoint[endpoint_id].get("timestamp_unix_s", time.time()),
                    )
                )
                context.scheduler.record_dvfs_actuation(
                    endpoint_id, observed_freq_mhz=target,
                    timestamp_s=readback_timestamp,
                )
        return action_by_endpoint

    def _adaptive_pools(
        self, capabilities: Mapping[str, EndpointClockCapability], policy: str,
    ) -> dict[str, AdaptivePoolController] | None:
        if policy not in {"ONE-STEP-FEEDBACK", "ADAPTIVE-SAFE-SEARCH"}:
            return None
        feedback = self.config["phase3d"]["feedback"]
        adaptive = AdaptiveExplorationConfig.from_mapping(
            self.settings.get("adaptive_exploration", {}),
            ttft_slo_ms=float(feedback["ttft_slo_ms"]),
            tpot_slo_ms=float(feedback["tpot_slo_ms"]),
        )
        if policy == "ONE-STEP-FEEDBACK":
            adaptive = replace(adaptive, large_jump_levels=1)
        regime = str(self.settings.get("slo_regime", "TIGHT")).upper()
        return {
            "P": AdaptivePoolController(
                "P", "prefill",
                [capabilities[item].selected_low_mhz for item in POOLS["P"]]
                + [capabilities[item].selected_mid_mhz for item in POOLS["P"]]
                + [capabilities[item].selected_high_mhz for item in POOLS["P"]],
                adaptive, slo_regime=regime,
            ),
            "D": AdaptivePoolController(
                "D", "decode",
                [capabilities[item].selected_low_mhz for item in POOLS["D"]]
                + [capabilities[item].selected_mid_mhz for item in POOLS["D"]]
                + [capabilities[item].selected_high_mhz for item in POOLS["D"]],
                adaptive, slo_regime=regime,
            ),
        }

    def _adaptive_control(
        self, context: ControllerContext,
        capabilities: Mapping[str, EndpointClockCapability],
        actuator: PerEndpointClockActuator,
        metrics: Mapping[str, Any],
        requests: Sequence[Mapping[str, Any]],
        policy: str,
        block_id: str,
        window_index: int,
        *,
        allow_exploration: bool = True,
    ) -> None:
        """Run one evidence epoch for independent P and D pool searches."""
        if not context.adaptive_pools:
            raise StageC1Error("adaptive controller state was not initialized")
        now = time.time()
        targets: dict[str, int] = {}
        decisions_by_pool: dict[str, Any] = {}
        for pool, endpoint_ids in POOLS.items():
            controller = context.adaptive_pools[pool]
            current_values = {context.targets[item] for item in endpoint_ids}
            if len(current_values) != 1:
                raise StageC1Error("%s pool entered an asymmetric frequency state" % pool)
            current = current_values.pop()
            controller.set_current_frequency(current)
            evidence = evidence_from_window(
                pool, metrics, requests,
                observed_frequency_mhz=current,
                slo_regime=str(self.settings.get("slo_regime", "TIGHT")).upper(),
                ttft_slo_ms=float(self.settings["slo"]["ttft_ms"]),
                tpot_slo_ms=float(self.settings["slo"]["tpot_ms"]),
                context_signature="stationary:%s" % str(metrics.get("load_point", "unknown")),
                captured_at_s=now,
            )
            decision = controller.decide(
                evidence, now_s=now, allow_exploration=allow_exploration
            )
            decisions_by_pool[pool] = decision
            targets.update({endpoint_id: int(decision.requested_frequency_mhz) for endpoint_id in endpoint_ids})
        actions = self._actuate_targets(
            actuator, context, targets, "stagec1_adaptive_pool_feedback", policy, block_id
        )
        for pool, endpoint_ids in POOLS.items():
            controller = context.adaptive_pools[pool]
            target = context.targets[endpoint_ids[0]]
            decision = decisions_by_pool[pool]
            if int(target) != int(decision.current_frequency_mhz):
                pool_actions = [actions[item] for item in endpoint_ids if item in actions]
                if len(pool_actions) != len(endpoint_ids):
                    raise StageC1Error("pool actuation did not cover every shared-frequency endpoint")
                readback_timestamp = max(float(item["readback_timestamp_unix_s"]) for item in pool_actions)
                controller.record_actuation(
                    target, target, readback_valid=True,
                    timestamp_s=readback_timestamp,
                )
            else:
                controller.set_current_frequency(target)
            row = decision.as_log()
            state = controller.state
            pool_actions = [actions[item] for item in endpoint_ids if item in actions]
            if pool_actions:
                row["action_timestamp_s"] = min(float(item["timestamp_unix_s"]) for item in pool_actions)
                row["readback_timestamp_s"] = max(float(item["readback_timestamp_unix_s"]) for item in pool_actions)
                row["readback_freq_mhz"] = target
                row["readback_valid"] = all(bool(item["readback_valid"]) for item in pool_actions)
            row.update({
                "timestamp_unix_s": now,
                "policy": policy,
                "block_id": block_id,
                "before_window_index": window_index,
                "evidence_window_index": window_index if not allow_exploration else window_index - 1,
                "finalization": not allow_exploration,
                "prior_ttft_mean_ms": metrics.get("ttft_mean_ms"),
                "prior_ttft_p95_ms": metrics.get("ttft_p95_ms"),
                "prior_tpot_mean_ms": metrics.get("tpot_mean_ms"),
                "prior_tpot_p95_ms": metrics.get("tpot_p95_ms"),
                "prior_admission_p95_ms": metrics.get("admission_delay_p95_ms"),
                "prior_client_waiting_close": metrics.get("client_waiting_at_offered_close"),
                "prior_backlog_growing": metrics.get("backlog_growing"),
                "evidence_since_action": state.evidence_since_action,
                "last_action_reason": state.last_action_reason,
                "last_actual_action_timestamp_s": state.last_action_timestamp_s,
                "current_index": state.current_index,
                "last_validated_safe_index": state.last_validated_safe_index,
                "unsafe_index": state.unsafe_index,
                "safe_energy_by_frequency_j_json": json.dumps(state.safe_energy_by_frequency_j, sort_keys=True),
                "best_measured_safe_frequency_mhz": state.best_measured_safe_frequency,
                "best_measured_safe_energy_j": state.best_measured_safe_energy_j,
                "context_signature": state.context_signature,
                "confidence": state.confidence,
                "transition_reason": state.transition_reason,
                "converged": state.converged,
                "convergence_reason": state.convergence_reason,
                "convergence_window_count": state.convergence_window_count,
                "convergence_request_count": state.convergence_request_count,
            })
            self.decision_rows.append(row)

    def _record_block_search_summary(
        self, context: ControllerContext, *, policy: str, block_id: str,
        repeat: int, load_point: str,
    ) -> None:
        if not context.adaptive_pools:
            for pool, endpoint_ids in POOLS.items():
                frequency = context.targets[endpoint_ids[0]]
                self.block_search_rows.append({
                    "block_id": block_id, "policy": policy, "repeat": repeat,
                    "load_point": load_point, "pool": pool,
                    "probe_count": 0, "unsafe_probe_count": 0,
                    "recovery_action_count": 0,
                    "windows_until_convergence": 0,
                    "requests_until_convergence": 0,
                    "converged": True, "convergence_reason": "STATIC_HIGH",
                    "final_validated_safe_frequency_mhz": frequency,
                    "final_selected_frequency_mhz": frequency,
                    "best_measured_safe_frequency_mhz": frequency,
                    "best_measured_safe_energy_j": None,
                    "safe_energy_by_frequency_j_json": "{}",
                })
            return
        for pool, controller in context.adaptive_pools.items():
            state = controller.state
            decisions = [
                row for row in self.decision_rows
                if row.get("block_id") == block_id and row.get("pool") == pool
            ]
            self.block_search_rows.append({
                "block_id": block_id, "policy": policy, "repeat": repeat,
                "load_point": load_point, "pool": pool,
                "probe_count": sum(row.get("decision") == "EXPLORE_DOWN" for row in decisions),
                "unsafe_probe_count": sum(
                    row.get("evidence_classification") == "UNSAFE"
                    and row.get("candidate_freq_mhz") not in (None, "")
                    for row in decisions
                ),
                "recovery_action_count": sum(row.get("decision") == "RECOVER" for row in decisions),
                "windows_until_convergence": state.convergence_window_count,
                "requests_until_convergence": state.convergence_request_count,
                "converged": state.converged,
                "convergence_reason": state.convergence_reason,
                "final_validated_safe_frequency_mhz": state.known_safe_frequency,
                "final_selected_frequency_mhz": state.current_frequency,
                "best_measured_safe_frequency_mhz": state.best_measured_safe_frequency,
                "best_measured_safe_energy_j": state.best_measured_safe_energy_j,
                "safe_energy_by_frequency_j_json": json.dumps(
                    state.safe_energy_by_frequency_j, sort_keys=True
                ),
            })

    def _control(
        self, context: ControllerContext, capabilities: Mapping[str, EndpointClockCapability],
        actuator: PerEndpointClockActuator, metrics: Mapping[str, Any], policy: str,
        block_id: str, window_index: int,
        requests: Sequence[Mapping[str, Any]] | None = None,
        *, allow_exploration: bool = True,
    ) -> None:
        if context.adaptive_pools is not None:
            self._adaptive_control(
                context, capabilities, actuator, metrics, requests or [],
                policy, block_id, window_index,
                allow_exploration=allow_exploration,
            )
            return
        risk = admission_risk(metrics, self.settings)
        recommendations = [
            context.scheduler.choose_frequency_adjustment(endpoint_id, now_s=time.time())
            for endpoint_id in ("P0", "P1", "D0", "D1")
        ]
        targets: dict[str, int] = {}
        for pool, endpoint_ids in POOLS.items():
            current_values = {context.targets[item] for item in endpoint_ids}
            if len(current_values) != 1:
                raise StageC1Error("%s pool entered an asymmetric frequency state" % pool)
            decision = coordinate_pool(
                pool, endpoint_ids, recommendations, current_values.pop(),
                capabilities[endpoint_ids[0]].selected_high_mhz, risk,
            )
            decision.update({
                "timestamp_unix_s": time.time(), "policy": policy,
                "block_id": block_id, "before_window_index": window_index,
                "admission_risk": risk,
                "prior_ttft_mean_ms": metrics.get("ttft_mean_ms"),
                "prior_ttft_p95_ms": metrics.get("ttft_p95_ms"),
                "prior_tpot_mean_ms": metrics.get("tpot_mean_ms"),
                "prior_tpot_p95_ms": metrics.get("tpot_p95_ms"),
                "prior_admission_p95_ms": metrics.get("admission_delay_p95_ms"),
                "prior_client_waiting_close": metrics.get("client_waiting_at_offered_close"),
                "prior_backlog_growing": metrics.get("backlog_growing"),
            })
            self.decision_rows.append(decision)
            targets.update({endpoint_id: int(decision["target_freq_mhz"]) for endpoint_id in endpoint_ids})
        self._actuate_targets(
            actuator, context, targets, "stagec1_pool_feedback", policy, block_id
        )

    def _run_window(
        self, window_id: str, trace_rows: Sequence[Mapping[str, Any]],
        requested: Mapping[str, int], windows_root: Path,
    ) -> Path:
        config = copy.deepcopy(self.config)
        config["output_root"] = windows_root.as_posix()
        config["client"]["max_concurrency"] = 4
        config["server_logs"] = {}
        config["coverage_policy"] = {
            "required_endpoint_ids": ["P0", "P1", "D0", "D1"],
            "required_pairs": [["P0", "D0"], ["P0", "D1"], ["P1", "D0"], ["P1", "D1"]],
        }
        for endpoint_id, frequency in requested.items():
            config["fixed_clocks"][endpoint_id]["graphics_mhz"] = int(frequency)
        Phase3CSubstrateHarness(config, run_id=window_id, trace_rows=trace_rows).run()
        return windows_root / window_id

    def _measurement(
        self, point: Mapping[str, Any], window_dir: Path, *, policy: str,
        repeat: int, window_index: int, block_id: str, context: ControllerContext,
        signature: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metrics = analyze_point(point, window_dir, self.settings["slo"], self.settings["trend_thresholds"])
        requests = _read_csv(window_dir / "requests.csv")
        ttft_limit = float(self.settings["slo"]["ttft_ms"])
        tpot_limit = float(self.settings["slo"]["tpot_ms"])
        ttft_attainment = sum(float(row["ttft_ms"]) <= ttft_limit for row in requests) / len(requests)
        tpot_attainment = sum(float(row["tpot_ms"]) <= tpot_limit for row in requests) / len(requests)
        row = {key: value for key, value in metrics.items() if key != "endpoint_load"}
        row.update({
            "run_id": self.run_id, "block_id": block_id, "policy": policy,
            "repeat": repeat, "window_index": window_index,
            "trace_signature": signature,
            "ttft_attainment": ttft_attainment, "tpot_attainment": tpot_attainment,
            "slo_contract_pass": ttft_attainment >= float(self.settings["slo"]["attainment"]) and tpot_attainment >= float(self.settings["slo"]["attainment"]),
            "P0_freq_mhz": context.targets["P0"], "P1_freq_mhz": context.targets["P1"],
            "D0_freq_mhz": context.targets["D0"], "D1_freq_mhz": context.targets["D1"],
            "endpoint_load_json": json.dumps(metrics["endpoint_load"], sort_keys=True),
        })
        for request in requests:
            request.update({
                "block_id": block_id, "policy": policy, "repeat": repeat,
                "window_index": window_index, "load_point": point["id"],
            })
        return row, requests

    def run(self) -> Path:
        if self.run_dir.exists():
            raise StageC1Error("run directory already exists: %s" % self.run_dir)
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        expected = self.settings["frequency_levels_mhz"]
        discovered = {key: self._levels(value) for key, value in capabilities.items()}
        if discovered != expected:
            raise StageC1Error("discovered clocks differ from validated Stage 3D-A levels")
        _write_json(self.run_dir / "capabilities.json", {
            "endpoints": {key: asdict(value) for key, value in capabilities.items()},
            "frequency_levels_mhz": discovered,
        })
        actuator = PerEndpointClockActuator(
            self.backend, capabilities, float(self.config["phase3d"]["minimum_dwell_s"])
        )
        high = {key: value.selected_high_mhz for key, value in capabilities.items()}
        rows: list[dict[str, Any]] = []
        request_rows: list[dict[str, Any]] = []
        repetitions = 1 if self.smoke else int(self.settings["repetitions"])
        windows = int(self.settings["smoke_windows"] if self.smoke else self.settings["windows_per_block"])
        points = [dict(item) for item in self.settings["load_points"]]
        valid = False
        error: str | None = None
        restoration_valid = False
        try:
            block_number = 0
            for repeat in range(1, repetitions + 1):
                policy_order = policy_order_for_repeat(repeat)
                for point in points:
                    for policy in policy_order:
                        block_number += 1
                        block_id = "r%02d-%s-%s" % (repeat, point["id"].lower(), policy.lower())
                        registry, telemetry, scheduler = self._build_scheduler(capabilities)
                        context = ControllerContext(
                            registry, telemetry, scheduler, dict(high),
                            self._adaptive_pools(capabilities, policy),
                        )
                        self._actuate_targets(
                            actuator, context, high, "stagec1_cold_start_high", policy, block_id
                        )
                        previous: dict[str, Any] | None = None
                        previous_requests: list[dict[str, Any]] = []
                        for window_index in range(1, windows + 1):
                            if policy != "STATIC-HIGH" and previous is not None:
                                self._control(
                                    context, capabilities, actuator, previous, policy,
                                    block_id, window_index, previous_requests,
                                )
                            scenario = str(point["scenario"])
                            trace = build_trace(
                                self.experiment, scenario,
                                "%s-w%02d" % (block_id, window_index),
                            )
                            signature = _trace_signature(trace)
                            window_id = "%s-w%02d" % (block_id, window_index)
                            window_dir = self._run_window(
                                window_id, trace, context.targets, windows_root
                            )
                            row, requests = self._measurement(
                                point, window_dir, policy=policy, repeat=repeat,
                                window_index=window_index, block_id=block_id,
                                context=context, signature=signature,
                            )
                            rows.append(row)
                            request_rows.extend(requests)
                            self._observe_window(
                                window_id, window_dir, registry, telemetry
                            )
                            previous = row
                            previous_requests = requests
                        if policy != "STATIC-HIGH" and previous is not None:
                            self._control(
                                context, capabilities, actuator, previous, policy,
                                block_id, windows + 1, previous_requests,
                                allow_exploration=False,
                            )
                        self._record_block_search_summary(
                            context, policy=policy, block_id=block_id,
                            repeat=repeat, load_point=str(point["id"]),
                        )
            valid = True
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            raise
        finally:
            try:
                if "actuator" in locals() and "capabilities" in locals():
                    if "context" not in locals():
                        registry, telemetry, scheduler = self._build_scheduler(capabilities)
                        context = ControllerContext(
                            registry, telemetry, scheduler, dict(high),
                            None,
                        )
                    self._actuate_targets(
                        actuator, context, high, "stagec1_final_restore_high",
                        "RESTORE", "final",
                    )
                    restoration_valid = all(
                        actuator.read(key).graphics_clock_mhz == value
                        for key, value in high.items()
                    )
            finally:
                if self.run_dir.exists():
                    self._write_reports(rows, request_rows, valid, restoration_valid, error)
        print(self.run_dir.as_posix())
        return self.run_dir

    def _write_reports(
        self, rows: list[dict[str, Any]], requests: list[dict[str, Any]],
        run_valid: bool, restoration_valid: bool, error: str | None,
    ) -> None:
        ttft_limit = float(self.settings["slo"]["ttft_ms"])
        tpot_limit = float(self.settings["slo"]["tpot_ms"])
        attainment_gate = float(self.settings["slo"]["attainment"])
        for block_id in sorted({str(row["block_id"]) for row in rows}):
            cumulative_energy = 0.0
            cumulative_requests = 0
            for row in sorted(
                (item for item in rows if item["block_id"] == block_id),
                key=lambda item: int(item["window_index"]),
            ):
                cumulative_energy += float(row["total_gross_energy_j"])
                cumulative_requests += int(row["request_count"])
                row["cumulative_energy_j"] = cumulative_energy
                row["cumulative_requests"] = cumulative_requests
                row["cumulative_joules_per_request"] = cumulative_energy / cumulative_requests
        _write_csv(self.run_dir / "stagec1_measurements.csv", rows)
        _write_csv(self.run_dir / "stagec1_requests.csv", requests)
        _write_csv(self.run_dir / "controller_decisions.csv", self.decision_rows)
        _write_csv(self.run_dir / "clock_actions.csv", self.action_rows)
        _write_csv(self.run_dir / "telemetry_observations.csv", self.telemetry_rows)
        _write_csv(self.run_dir / "stagec1_search_convergence.csv", self.block_search_rows)
        repeat_summaries = []
        for point in self.settings["load_points"]:
            for policy in POLICIES:
                repeats = sorted({
                    int(row["repeat"]) for row in rows
                    if row["load_point"] == point["id"] and row["policy"] == policy
                })
                for repeat in repeats:
                    selected = [
                        row for row in rows
                        if row["load_point"] == point["id"]
                        and row["policy"] == policy and int(row["repeat"]) == repeat
                    ]
                    selected_requests = [
                        row for row in requests
                        if row["load_point"] == point["id"]
                        and row["policy"] == policy and int(row["repeat"]) == repeat
                    ]
                    energy = sum(float(row["total_gross_energy_j"]) for row in selected)
                    ttft = [float(row["ttft_ms"]) for row in selected_requests]
                    tpot = [float(row["tpot_ms"]) for row in selected_requests]
                    repeat_summaries.append({
                        "load_point": point["id"], "policy": policy, "repeat": repeat,
                        "request_count": len(selected_requests), "total_gross_energy_j": energy,
                        "joules_per_request": energy / len(selected_requests),
                        "ttft_attainment": sum(value <= ttft_limit for value in ttft) / len(ttft),
                        "tpot_attainment": sum(value <= tpot_limit for value in tpot) / len(tpot),
                    })
        for row in repeat_summaries:
            if row["policy"] == "STATIC-HIGH":
                continue
            baseline = next(
                item for item in repeat_summaries
                if item["load_point"] == row["load_point"]
                and item["policy"] == "STATIC-HIGH" and item["repeat"] == row["repeat"]
            )
            row["paired_energy_saving_fraction"] = (
                (baseline["joules_per_request"] - row["joules_per_request"])
                / baseline["joules_per_request"]
                if row["ttft_attainment"] >= attainment_gate and row["tpot_attainment"] >= attainment_gate
                else None
            )
        _write_csv(self.run_dir / "stagec1_repeat_summary.csv", repeat_summaries)
        aggregates = []
        for point in self.settings["load_points"]:
            for policy in POLICIES:
                selected = [row for row in rows if row["load_point"] == point["id"] and row["policy"] == policy]
                selected_requests = [row for row in requests if row["load_point"] == point["id"] and row["policy"] == policy]
                if not selected:
                    continue
                ttft = [float(row["ttft_ms"]) for row in selected_requests]
                tpot = [float(row["tpot_ms"]) for row in selected_requests]
                total_energy = sum(float(row["total_gross_energy_j"]) for row in selected)
                count = len(selected_requests)
                transitions = [row for row in self.decision_rows if row["policy"] == policy and point["id"].lower() in row["block_id"] and row["action"] != "HOLD"]
                decisions = [row for row in self.decision_rows if row["policy"] == policy and point["id"].lower() in row["block_id"]]
                search_rows = [
                    row for row in self.block_search_rows
                    if row["policy"] == policy and row["load_point"] == point["id"]
                ]
                endpoint_load = [json.loads(row["endpoint_load_json"]) for row in selected]
                aggregate = {
                    "load_point": point["id"], "policy": policy,
                    "request_count": count, "offered_rate_rps": point["offered_rate_rps"],
                    "completed_rate_rps_mean": statistics.fmean(float(row["completed_rate_rps"]) for row in selected),
                    "total_gross_energy_j": total_energy,
                    "joules_per_request": total_energy / count,
                    "P0_energy_j": sum(float(row["P0_energy_j"]) for row in selected),
                    "P1_energy_j": sum(float(row["P1_energy_j"]) for row in selected),
                    "D0_energy_j": sum(float(row["D0_energy_j"]) for row in selected),
                    "D1_energy_j": sum(float(row["D1_energy_j"]) for row in selected),
                    "ttft_mean_ms": statistics.fmean(ttft), "ttft_p95_ms": percentile(ttft, 0.95),
                    "ttft_p99_ms": percentile(ttft, 0.99),
                    "ttft_attainment": sum(value <= ttft_limit for value in ttft) / count,
                    "tpot_mean_ms": statistics.fmean(tpot), "tpot_p95_ms": percentile(tpot, 0.95),
                    "tpot_p99_ms": percentile(tpot, 0.99),
                    "tpot_attainment": sum(value <= tpot_limit for value in tpot) / count,
                    "admission_delay_mean_ms": statistics.fmean(float(row["client_admission_delay_ms"]) for row in selected_requests),
                    "admission_delay_p95_ms": percentile([float(row["client_admission_delay_ms"]) for row in selected_requests], 0.95),
                    "admission_delay_p99_ms": percentile([float(row["client_admission_delay_ms"]) for row in selected_requests], 0.99),
                    "client_waiting_mean": statistics.fmean(float(row["client_waiting_mean"]) for row in selected),
                    "client_waiting_max": max(float(row["client_waiting_max"]) for row in selected),
                    "proxy_inflight_mean": statistics.fmean(float(row["proxy_inflight_mean"]) for row in selected),
                    "proxy_inflight_max": max(float(row["proxy_inflight_max"]) for row in selected),
                    "frequency_transitions": len(transitions),
                    "control_oscillations": oscillation_count(decisions),
                    "probe_count": sum(int(row["probe_count"]) for row in search_rows),
                    "unsafe_probe_count": sum(int(row["unsafe_probe_count"]) for row in search_rows),
                    "emergency_recovery_action_count": sum(int(row["recovery_action_count"]) for row in search_rows),
                    "converged_pool_blocks": sum(bool(row["converged"]) for row in search_rows),
                    "total_pool_blocks": len(search_rows),
                    "windows_until_convergence_json": json.dumps([
                        row["windows_until_convergence"] for row in search_rows
                    ]),
                    "requests_until_convergence_json": json.dumps([
                        row["requests_until_convergence"] for row in search_rows
                    ]),
                    "final_validated_safe_frequency_json": json.dumps({
                        row["block_id"] + ":" + row["pool"]: row["final_validated_safe_frequency_mhz"]
                        for row in search_rows
                    }, sort_keys=True),
                    "best_measured_safe_frequency_json": json.dumps({
                        row["block_id"] + ":" + row["pool"]: row["best_measured_safe_frequency_mhz"]
                        for row in search_rows
                    }, sort_keys=True),
                    "P_frequency_residency_json": json.dumps(
                        frequency_residency(selected, "P0", self.settings["frequency_levels_mhz"]["P0"]), sort_keys=True
                    ),
                    "D_frequency_residency_json": json.dumps(
                        frequency_residency(selected, "D0", self.settings["frequency_levels_mhz"]["D0"]), sort_keys=True
                    ),
                    "endpoint_running_waiting_json": json.dumps({
                        endpoint_id: {
                            "running_mean": observed_metric_stats(
                                endpoint_load, endpoint_id, "running_mean"
                            ),
                            "running_max": observed_metric_stats(
                                endpoint_load, endpoint_id, "running_max"
                            ),
                            "waiting_mean": observed_metric_stats(
                                endpoint_load, endpoint_id, "waiting_mean"
                            ),
                            "waiting_max": observed_metric_stats(
                                endpoint_load, endpoint_id, "waiting_max"
                            ),
                        } for endpoint_id in ("P0", "P1", "D0", "D1")
                    }, sort_keys=True),
                    "measurement_valid": all(bool(row["measurement_valid"]) for row in selected),
                }
                aggregate["slo_contract_pass"] = aggregate["ttft_attainment"] >= attainment_gate and aggregate["tpot_attainment"] >= attainment_gate
                repeat_joules = [
                    float(item["joules_per_request"]) for item in repeat_summaries
                    if item["load_point"] == point["id"] and item["policy"] == policy
                ]
                aggregate["joules_per_request_repeat_std"] = (
                    statistics.stdev(repeat_joules) if len(repeat_joules) >= 2 else None
                )
                aggregates.append(aggregate)
        for point in self.settings["load_points"]:
            static = next((row for row in aggregates if row["load_point"] == point["id"] and row["policy"] == "STATIC-HIGH"), None)
            for feedback in (
                row for row in aggregates
                if row["load_point"] == point["id"] and row["policy"] != "STATIC-HIGH"
            ):
                if static:
                    feedback["energy_saving_vs_static_fraction"] = (
                        (static["joules_per_request"] - feedback["joules_per_request"])
                        / static["joules_per_request"] if feedback["slo_contract_pass"] else None
                    )
                    paired = [
                        float(row["paired_energy_saving_fraction"])
                        for row in repeat_summaries
                        if row["load_point"] == point["id"]
                        and row["policy"] == feedback["policy"]
                        and row.get("paired_energy_saving_fraction") is not None
                    ]
                    feedback["paired_energy_saving_mean_fraction"] = statistics.fmean(paired) if paired else None
                    feedback["paired_energy_saving_std_fraction"] = statistics.stdev(paired) if len(paired) >= 2 else None
        _write_csv(self.run_dir / "stagec1_policy_summary.csv", aggregates)
        signatures: dict[tuple[int, str, int], set[str]] = {}
        for row in rows:
            signatures.setdefault((int(row["repeat"]), str(row["load_point"]), int(row["window_index"])), set()).add(str(row["trace_signature"]))
        matched = bool(signatures) and all(len(value) == 1 for value in signatures.values())
        pool_symmetric = all(
            int(row["P0_freq_mhz"]) == int(row["P1_freq_mhz"])
            and int(row["D0_freq_mhz"]) == int(row["D1_freq_mhz"])
            for row in rows
        )
        hard_valid = run_valid and restoration_valid and matched and pool_symmetric and all(bool(row["measurement_valid"]) for row in rows)
        summary = {
            "schema_version": 1, "phase": "Stage_C1_stationary_feedback_DVFS",
            "smoke": self.smoke, "valid": hard_valid, "error": error,
            "policy_matrix": list(POLICIES),
            "slo_regime": self.settings.get("slo_regime", "TIGHT"),
            "matched_trace_audit": matched, "pool_frequency_symmetry_audit": pool_symmetric,
            "final_high_restoration_audit": restoration_valid,
            "routing": "fixed four-pair round-robin", "max_concurrency": 4,
            "slo": self.settings["slo"], "policy_results": aggregates,
            "search_convergence": self.block_search_rows,
            "controller_contract": {
                "base": "shared evidence/safety state machine; one-step restricts maximum downward jump to one level",
                "safety_timing": "window-bounded in C1c; no event-driven between-window claim",
                "fast_up": "at an evidence boundary, phase-specific SLO/pressure/telemetry/readback risk cancels exploration and recovers a validated-safe point",
                "slow_down": "sufficient fresh phase evidence selects a configurable coarse or one-level lower candidate; candidate is not safe until observed",
                "p_running_zero_is_idle_evidence": False,
            },
            "claim_boundary": "stationary, evidence-window-bounded C1c safe-search validation; no event-driven safety, routing change, predictor, or physical request-energy attribution claim",
        }
        _write_json(self.run_dir / "stagec1_summary.json", summary)
        lines = [
            "# XPYD Stage C1 stationary feedback-only DVFS", "",
            "Matched fixed four-pair round-robin traces; P and D pools use shared frequency states.", "",
            "| Load | Policy | J/request | TTFT p95 / attain | TPOT p95 / attain | Transitions | SLO contract |", "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in aggregates:
            lines.append(
                "| %s | %s | %.2f | %.1f / %.1f%% | %.1f / %.1f%% | %d | %s |" % (
                    row["load_point"], row["policy"], row["joules_per_request"],
                    row["ttft_p95_ms"], 100 * row["ttft_attainment"],
                    row["tpot_p95_ms"], 100 * row["tpot_attainment"],
                    row["frequency_transitions"], "PASS" if row["slo_contract_pass"] else "FAIL",
                )
            )
        lines += ["", "Energy saving is reported only where a feedback policy satisfies both 95% attainment gates.", "", "C1c safety actions are evidence-window-bounded; event-driven between-window safety is not implemented here.", "", "Hard-gate audit: **%s**." % ("PASS" if hard_valid else "FAIL")]
        (self.run_dir / "stagec1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    runtime, experiment = load_configs(args.runtime_config, args.experiment_config)
    StageC1Harness(runtime, experiment, args.run_id, smoke=args.smoke).run()


if __name__ == "__main__":
    main()
