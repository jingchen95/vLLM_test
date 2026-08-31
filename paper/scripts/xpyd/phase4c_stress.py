"""Phase 4C robustness evaluation of the unchanged passive controller.

This module deliberately compares only STATIC and PASSIVE_FULL.  It reuses the
Phase 4B controller and physical measurement path, raises offered rate without
exceeding the validated connector concurrency, and applies a tighter SLO.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.phase3c_substrate import _read_json, _write_json, build_registry_and_compatibility
from xpyd.phase3d_control import PerEndpointClockActuator, _csv, load_config as load_phase3d_config
from xpyd.phase4b_evaluation import (
    Phase4BError,
    Phase4BEvaluationHarness,
    relative_energy_savings,
    summarize_dynamic_adaptation,
)


class Phase4CError(RuntimeError):
    """A fail-closed Phase 4C configuration, measurement, or audit error."""


POLICIES = ("STATIC", "PASSIVE_FULL")


def _mean(values: Sequence[Any]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _std(values: Sequence[Any]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def _ci95(values: Sequence[Any]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return 1.96 * statistics.stdev(clean) / math.sqrt(len(clean)) if len(clean) >= 2 else None


def _format_optional_float(value: Any, digits: int = 2) -> str:
    """Format an incomplete failure summary without masking its root error."""
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return ("%%.%df" % int(digits)) % number


def oracle_reference_status(
    stress: Mapping[str, Any], oracle: Mapping[str, Any], *, reference_rate_rps: float,
    ttft_slo_ms: float, tpot_slo_ms: float,
) -> tuple[bool, str]:
    """Allow the frozen oracle only for an exactly covered rate and feasible SLO."""
    if abs(float(stress["rate_rps"]) - float(reference_rate_rps)) > 1e-9:
        return False, "oracle reference unavailable: offered rate is outside Phase 4A coverage"
    if float(oracle["TTFT"]) > ttft_slo_ms or float(oracle["TPOT"]) > tpot_slo_ms:
        return False, "oracle reference unavailable: Phase 4A winner fails the tighter SLO"
    return True, "exact Phase 4A shape/rate match and measured winner satisfies tighter SLO"


def paired_energy_comparisons(
    summaries: Sequence[Mapping[str, Any]], trace_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Compare policies within trace/repeat blocks and retain run variance."""
    result = []
    for trace_id in trace_ids:
        paired = []
        repeats = sorted({
            int(item["repeat"]) for item in summaries if item["trace_id"] == trace_id
        })
        for repeat in repeats:
            by_policy = {
                item["policy"]: item for item in summaries
                if item["trace_id"] == trace_id and int(item["repeat"]) == repeat
            }
            if set(by_policy) == set(POLICIES):
                paired.append(relative_energy_savings(
                    by_policy["STATIC"]["total_gpu_gross_energy_j"],
                    by_policy["PASSIVE_FULL"]["total_gpu_gross_energy_j"],
                ))
        result.append({
            "trace_id": trace_id,
            "paired_repeat_count": len(paired),
            "passive_energy_saving_vs_static_mean": _mean(paired),
            "passive_energy_saving_vs_static_std": _std(paired),
            "passive_energy_saving_vs_static_ci95_half_width": _ci95(paired),
        })
    return result


def classify_robustness(
    comparisons: Sequence[Mapping[str, Any]], passive_summaries: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    """Apply the predeclared Phase 4C decision gate to measured evidence."""
    reasons: list[str] = []
    if not comparisons or not passive_summaries or any(
        int(item["paired_repeat_count"]) < 3 for item in comparisons
    ):
        return "D. INCONCLUSIVE", ["required paired repeats are incomplete"]
    if not all(bool(item["measurement_valid"]) for item in passive_summaries):
        return "D. INCONCLUSIVE", ["one or more physical measurements are invalid"]
    savings = [float(item["passive_energy_saving_vs_static_mean"]) for item in comparisons]
    requests = sum(int(item["request_count"]) for item in passive_summaries)
    violations = sum(int(item["slo_violation_count"]) for item in passive_summaries)
    violation_rate = violations / requests if requests else 1.0
    stale_fallback = sum(
        int(item["stale_telemetry_event_count"]) + int(item["fallback_event_count"])
        for item in passive_summaries
    )
    windows = sum(int(item["window_count"]) for item in passive_summaries)
    comparable = [item for item in passive_summaries if item["oracle_reference_available"]]
    oracle_gaps = [
        float(item["oracle_gap"]) for item in comparable if item.get("oracle_gap") is not None
    ]
    reasons.extend([
        "minimum trace-level energy saving %.2f%%" % (100.0 * min(savings)),
        "PASSIVE request-level SLO violation rate %.2f%%" % (100.0 * violation_rate),
        "stale plus fallback events per window %.2f%%" % (
            100.0 * stale_fallback / windows if windows else 100.0
        ),
        (
            "maximum comparable oracle gap %.2f%%" % (100.0 * max(oracle_gaps))
            if oracle_gaps else "oracle reference unavailable for all measured stress traces"
        ),
    ])
    oracle_ok = not oracle_gaps or max(oracle_gaps) <= 0.10
    stable = stale_fallback / windows <= 0.10 if windows else False
    if min(savings) >= 0.05 and violation_rate <= 0.01 and oracle_ok and stable:
        return "A. PASSIVE ROBUSTLY SUFFICIENT", reasons
    if violation_rate > 0.10 or max(savings) <= 0.0:
        return "C. PASSIVE INSUFFICIENT", reasons
    if violation_rate > 0.01 or min(savings) < 0.05 or not oracle_ok or not stable:
        return "B. PASSIVE SUFFICIENT ONLY IN EASY REGIMES", reasons
    return "D. INCONCLUSIVE", reasons


def load_config(path: Path) -> dict[str, Any]:
    # Phase 4C reuses the Phase 4B runtime implementation, not its ACTIVE_FULL
    # experiment gate.  Loading through the Phase 3D substrate validator avoids
    # incorrectly requiring a Phase 4B.1 active-probing audit for this exact
    # STATIC/PASSIVE-only experiment.
    config = load_phase3d_config(path)
    phase4b = config.get("phase4b")
    if not isinstance(phase4b, dict) or not str(phase4b.get("phase4a_summary", "")):
        raise Phase4CError("Phase 4C requires the frozen Phase 4A reference path")
    settings = config.get("phase4c")
    if not isinstance(settings, dict):
        raise Phase4CError("Phase 4C config requires a phase4c object")
    if tuple(settings.get("policies", ())) != POLICIES:
        raise Phase4CError("Phase 4C compares exactly STATIC and PASSIVE_FULL")
    if int(settings.get("repeats", 0)) < 3 or int(settings.get("windows_per_state", 0)) < 3:
        raise Phase4CError("Phase 4C requires at least three repeats and three windows per state")
    if int(config["client"].get("max_concurrency", 0)) != 1:
        raise Phase4CError("Phase 4C must remain at validated connector concurrency one")
    if int(config["phase3d"].get("validated_connector_max_concurrency", 0)) != 1:
        raise Phase4CError("connector concurrency evidence must remain exactly one")
    endpoints = {str(item["endpoint_id"]): item for item in config["endpoints"]}
    if set(endpoints) != {"P0", "P1", "D0", "D1"}:
        raise Phase4CError("Phase 4C requires exactly P0/P1/D0/D1")
    if any(str(endpoints[key]["node"]) != "neptune" for key in ("P0", "P1")):
        raise Phase4CError("Phase 4C prefill endpoints must remain on Neptune")
    if any(str(endpoints[key]["node"]) != "io" for key in ("D0", "D1")):
        raise Phase4CError("Phase 4C decode endpoints must remain on io")
    if any(int(item["tp_degree"]) != 1 for item in endpoints.values()):
        raise Phase4CError("Phase 4C fixes every endpoint at TP1")
    if len(settings.get("traces", ())) != 3:
        raise Phase4CError("Phase 4C requires exactly three compact stress traces")
    base_ids = {str(item["id"]) for item in config["workloads"]}
    stress = {str(item["id"]): item for item in settings.get("stress_workloads", ())}
    if not stress or any(str(item["base_workload"]) not in base_ids for item in stress.values()):
        raise Phase4CError("Phase 4C stress workload references an unknown validated shape")
    for trace in settings["traces"]:
        if len(trace.get("states", ())) != 3 or not set(trace["states"]).issubset(stress):
            raise Phase4CError("each Phase 4C trace must contain three known stress states")
    rates = {float(item["rate_rps"]) for item in stress.values()}
    levels = {str(item["stress_level"]) for item in stress.values()}
    if max(rates) <= 0.5 or not {"moderate", "high"}.issubset(levels):
        raise Phase4CError("Phase 4C must include higher-rate moderate and high regimes")
    slo = settings.get("slo", {})
    phase4b_slo = config["phase4b"]["slo"]
    if not (0 < float(slo.get("ttft_ms", 0)) < float(phase4b_slo["ttft_ms"])):
        raise Phase4CError("Phase 4C TTFT SLO must be positive and tighter than Phase 4B")
    if not (0 < float(slo.get("tpot_ms", 0)) < float(phase4b_slo["tpot_ms"])):
        raise Phase4CError("Phase 4C TPOT SLO must be positive and tighter than Phase 4B")
    return config


class Phase4CStressHarness(Phase4BEvaluationHarness):
    """Physical STATIC/PASSIVE stress comparison with unchanged decisions."""

    def __init__(self, config: Mapping[str, Any], run_id: Optional[str] = None, backend: Any = None) -> None:
        super().__init__(config, run_id=run_id, backend=backend, smoke=False, stage="dynamic")
        self.phase4c = dict(self.config["phase4c"])
        self.settings["slo"] = dict(self.phase4c["slo"])
        self.run_dir = Path(os.path.expandvars(str(self.phase4c["output_root"]))) / self.run_id

    def run(self) -> Path:
        if self.run_dir.exists():
            raise Phase4CError("run directory already exists: %s" % self.run_dir)
        accepted_path = Path(os.path.expandvars(str(self.phase4c["accepted_phase4b_dynamic_audit"])))
        if not accepted_path.is_file():
            raise Phase4CError("missing accepted formal Phase 4B dynamic audit")
        accepted_raw = accepted_path.read_bytes()
        accepted_audit = json.loads(accepted_raw)
        if not accepted_audit.get("valid") or accepted_audit.get("smoke"):
            raise Phase4CError("Phase 4C requires a valid non-smoke Phase 4B dynamic audit")

        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        reference = self._reference()
        accepted_freq = self._accepted_frequencies(reference)
        capabilities = self._discover_exact(accepted_freq)
        _write_json(self.run_dir / "raw" / "capabilities.json", {
            key: asdict(value) for key, value in capabilities.items()
        })
        _write_json(self.run_dir / "raw" / "accepted_phase4b_dynamic_reference.json", {
            "path": accepted_path.as_posix(), "sha256": hashlib.sha256(accepted_raw).hexdigest(),
            "valid": True,
        })
        _write_json(self.run_dir / "raw" / "frozen_phase4a_reference.json", {
            "path": reference["path"], "sha256": reference["sha256"],
            "oracles": reference["oracles"],
        })
        high = {endpoint: values["HIGH"] for endpoint, values in accepted_freq.items()}
        actuator = PerEndpointClockActuator(
            self.backend, capabilities, float(self.config["phase3d"].get("minimum_dwell_s", 1.0))
        )
        actuator.requested.update(high)
        _, _, all_pairs = build_registry_and_compatibility(self.config)
        base_by_id = {str(item["id"]): dict(item) for item in self.config["workloads"]}
        stress_by_id = {str(item["id"]): dict(item) for item in self.phase4c["stress_workloads"]}
        traces = [dict(item) for item in self.phase4c["traces"]]
        repeats = int(self.phase4c["repeats"])
        windows_per_state = int(self.phase4c["windows_per_state"])
        request_count = int(self.phase4c["requests_per_window"])
        blocks = [
            (repeat, policy, trace) for repeat in range(1, repeats + 1)
            for policy in POLICIES for trace in traces
        ]
        random.Random(int(self.phase4c.get("order_seed", 53))).shuffle(blocks)
        _write_json(self.run_dir / "raw" / "phase4c_plan.json", {
            "policies": list(POLICIES), "traces": traces,
            "stress_workloads": list(stress_by_id.values()), "repeats": repeats,
            "windows_per_state": windows_per_state, "requests_per_window": request_count,
            "max_concurrency": 1, "slo": self.phase4c["slo"],
            "block_order": [
                {"repeat": repeat, "policy": policy, "trace_id": trace["id"]}
                for repeat, policy, trace in blocks
            ],
            "transition_information_policy": "first transition decision uses only the prior observed base-workload context",
            "models": [],
        })

        rows: list[dict[str, Any]] = []
        request_rows: list[dict[str, Any]] = []
        warmups: list[dict[str, Any]] = []
        restoration_valid = False
        error: Optional[str] = None
        sequence = 0
        try:
            global_context = self._new_context(capabilities)
            self._actuate(
                actuator, global_context, high, "phase4c_global_warmup_high",
                controller_action=False, policy="WARMUP", workload="all",
                iteration_id="phase4c-global-warmup",
            )
            self._write_routes(all_pairs, "phase4c_global_all_route_warmup")
            warm_dir = self._run_window(
                "phase4c-global-warmup",
                {"id": "phase4c_global_warmup", "input_len": 128, "output_len": 16, "rate_rps": 0.5},
                4, high, all_pairs, windows_root, require_all_pairs=True,
            )
            warm_audit = _read_json(warm_dir / "audit.json")
            warmups.append({"window_id": "phase4c-global-warmup", **warm_audit})
            if not warm_audit.get("valid"):
                raise Phase4CError("Phase 4C global all-route warmup failed")

            for block_index, (repeat, policy, trace) in enumerate(blocks, 1):
                states = [stress_by_id[str(item)] for item in trace["states"]]
                context = self._new_context(capabilities)
                if policy == "PASSIVE_FULL":
                    initial = states[0]
                    base = copy.deepcopy(base_by_id[str(initial["base_workload"])])
                    base["rate_rps"] = float(initial["rate_rps"])
                    warmup_id = "p4c-r%02d-b%02d-warm-%s" % (repeat, block_index, trace["id"])
                    self._actuate(
                        actuator, context, high, "phase4c_passive_warmup_high",
                        controller_action=False, policy=policy,
                        workload=str(initial["base_workload"]), iteration_id=warmup_id,
                    )
                    context.selected_pairs = copy.deepcopy(all_pairs)
                    self._write_routes(all_pairs, "phase4c_passive_all_route_observation_warmup")
                    warm_dir = self._run_window(
                        warmup_id, base, 4, high, all_pairs, windows_root, require_all_pairs=True,
                    )
                    warm_audit = _read_json(warm_dir / "audit.json")
                    warmups.append({"window_id": warmup_id, **warm_audit})
                    if not warm_audit.get("valid"):
                        raise Phase4CError("Phase 4C passive warmup failed: %s" % warmup_id)
                    self._observe_window(warmup_id, warm_dir, context.registry, context.telemetry)

                prior_context = str(states[0]["base_workload"])
                cumulative_energy = 0.0
                block_sequence = 0
                for state_index, stress in enumerate(states):
                    workload_id = str(stress["base_workload"])
                    workload = copy.deepcopy(base_by_id[workload_id])
                    workload["rate_rps"] = float(stress["rate_rps"])
                    transition_s = time.time()
                    for window_in_state in range(1, windows_per_state + 1):
                        sequence += 1
                        block_sequence += 1
                        iteration_id = "p4c-r%02d-b%02d-s%02d-w%02d-%s-%s" % (
                            repeat, block_index, state_index, window_in_state,
                            policy.lower(), trace["id"],
                        )
                        feedback_context = (
                            prior_context if state_index > 0 and window_in_state == 1 else workload_id
                        )
                        pairs = self._decide(
                            policy, workload_id, iteration_id, context, capabilities,
                            actuator, all_pairs, measured_iteration=True,
                            feedback_context_key=feedback_context,
                        )
                        control = self.control_rows[-1]
                        control.update({"repeat": repeat, "trace_id": trace["id"]})
                        window_dir = self._run_window(
                            iteration_id, workload, request_count,
                            context.target_frequencies, pairs, windows_root,
                        )
                        row, observed_requests = self._measurement_row(
                            iteration_id, policy, workload, repeat, context,
                            accepted_freq, reference, window_dir,
                        )
                        oracle_ok, oracle_reason = oracle_reference_status(
                            stress, reference["oracles"][workload_id],
                            reference_rate_rps=float(self.phase4c["oracle_reference_rate_rps"]),
                            ttft_slo_ms=float(self.phase4c["slo"]["ttft_ms"]),
                            tpot_slo_ms=float(self.phase4c["slo"]["tpot_ms"]),
                        )
                        if not oracle_ok:
                            row["phase4a_oracle_joules_per_request"] = None
                            row["oracle_gap"] = None
                            row["oracle_gap_class"] = None
                            row["phase4a_near_optimal_match"] = None
                        cumulative_energy += float(row["total_gpu_gross_energy_j"])
                        signature = json.dumps({
                            "pairs": context.selected_pairs,
                            "frequencies": context.target_frequencies,
                        }, sort_keys=True)
                        row.update({
                            "trace_id": str(trace["id"]), "stress_workload": str(stress["id"]),
                            "stress_level": str(stress["stress_level"]), "repeat": repeat,
                            "dynamic_repeat": repeat,
                            "sequence_index": sequence, "block_sequence": block_sequence,
                            "state_index": state_index, "window_in_state": window_in_state,
                            "workload_changed": state_index > 0 and window_in_state == 1,
                            "transition_timestamp_unix_s": transition_s,
                            "control_iteration_id": iteration_id,
                            "control_timestamp_unix_s": control["control_timestamp"],
                            "feedback_context_key": feedback_context,
                            "control_state_signature": signature,
                            "dvfs_action_count": control["dvfs_action_count"],
                            "routing_changed": control["routing_changed"],
                            "fallback_reason": control["fallback_reason"],
                            "stale_endpoint_ids_json": control["stale_endpoint_ids_json"],
                            "cumulative_trace_energy_j": cumulative_energy,
                            "oracle_reference_available": oracle_ok,
                            "oracle_reference_reason": oracle_reason,
                        })
                        rows.append(row)
                        request_rows.extend({
                            **item, "trace_id": str(trace["id"]), "stress_workload": str(stress["id"]),
                            "stress_level": str(stress["stress_level"]), "dynamic_repeat": repeat,
                            "state_index": state_index, "window_in_state": window_in_state,
                        } for item in observed_requests)
                        if not row["measurement_valid"]:
                            raise Phase4CError("Phase 4C window audit failed: %s" % iteration_id)
                        if policy == "PASSIVE_FULL":
                            self._observe_window(iteration_id, window_dir, context.registry, context.telemetry)
                        prior_context = workload_id
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            try:
                restore = self._new_context(capabilities)
                self._actuate(
                    actuator, restore, high, "phase4c_final_safe_high_restoration",
                    controller_action=False, policy="RESTORE", workload="all",
                    iteration_id="phase4c-final-restore",
                )
                restoration_valid = all(
                    actuator.read(endpoint).graphics_clock_mhz == target
                    for endpoint, target in high.items()
                )
            except Exception:
                restoration_valid = False

        return self._report(
            rows, request_rows, traces, warmups, accepted_audit, accepted_path,
            accepted_raw, reference, restoration_valid, error,
        )

    def _report(
        self, rows: list[dict[str, Any]], requests: list[dict[str, Any]],
        traces: list[dict[str, Any]], warmups: list[dict[str, Any]],
        accepted_audit: Mapping[str, Any], accepted_path: Path, accepted_raw: bytes,
        reference: Mapping[str, Any], restoration_valid: bool, error: Optional[str],
    ) -> Path:
        controls_by_id = {str(item["control_iteration_id"]): item for item in self.control_rows}
        summaries: list[dict[str, Any]] = []
        for policy in POLICIES:
            for trace in traces:
                for repeat in range(1, int(self.phase4c["repeats"]) + 1):
                    group = [item for item in rows if item["policy"] == policy and item["trace_id"] == trace["id"] and int(item["repeat"]) == repeat]
                    reqs = [item for item in requests if item["policy"] == policy and item["trace_id"] == trace["id"] and int(item["dynamic_repeat"]) == repeat]
                    controls = [controls_by_id[item["control_iteration_id"]] for item in group]
                    energy = sum(float(item["total_gpu_gross_energy_j"]) for item in group)
                    output_tokens = sum(int(item["output_tokens"]) for item in group)
                    violations = sum(
                        float(item["ttft_ms"]) > float(self.phase4c["slo"]["ttft_ms"])
                        or float(item["tpot_ms"]) > float(self.phase4c["slo"]["tpot_ms"])
                        for item in reqs
                    )
                    route_counts: dict[str, int] = {}
                    for item in reqs:
                        route = "%s->%s" % (item["prefill_endpoint_id"], item["decode_endpoint_id"])
                        route_counts[route] = route_counts.get(route, 0) + 1
                    residency: dict[str, dict[str, float]] = {}
                    total_duration = sum(float(item["duration_s"]) for item in group)
                    for endpoint in ("P0", "P1", "D0", "D1"):
                        duration_by_level: dict[str, float] = {}
                        for item in group:
                            level = str(item[endpoint + "_requested_level"])
                            duration_by_level[level] = duration_by_level.get(level, 0.0) + float(item["duration_s"])
                        residency[endpoint] = {
                            level: duration / total_duration for level, duration in duration_by_level.items()
                        } if total_duration else {}
                    comparable = [item for item in group if item["oracle_reference_available"]]
                    oracle_energy = sum(
                        float(reference["oracles"][item["workload"]]["J_per_request"])
                        * int(item["completed_requests"]) for item in comparable
                    )
                    comparable_energy = sum(float(item["total_gpu_gross_energy_j"]) for item in comparable)
                    summaries.append({
                        "policy": policy, "trace_id": str(trace["id"]), "repeat": repeat,
                        "window_count": len(group), "request_count": len(reqs),
                        "total_gpu_gross_energy_j": energy,
                        "joules_per_request": energy / len(reqs) if reqs else None,
                        "joules_per_output_token": energy / output_tokens if output_tokens else None,
                        "throughput_requests_s": len(reqs) / sum(float(item["duration_s"]) for item in group) if group else None,
                        "ttft_mean_ms": _mean([item["ttft_ms"] for item in reqs]),
                        "tpot_mean_ms": _mean([item["tpot_ms"] for item in reqs]),
                        "itl_mean_ms": _mean([item["mean_itl_ms"] for item in reqs]),
                        "e2e_mean_ms": _mean([item["e2e_latency_ms"] for item in reqs]),
                        "slo_violation_count": violations,
                        "slo_violation_rate": violations / len(reqs) if reqs else None,
                        "route_distribution_json": json.dumps(route_counts, sort_keys=True),
                        "frequency_residency_json": json.dumps(residency, sort_keys=True),
                        "dvfs_action_count": sum(int(item["dvfs_action_count"]) for item in controls),
                        "routing_change_count": sum(bool(item["routing_changed"]) for item in controls),
                        "fallback_event_count": sum(bool(item["fallback_reason"]) for item in controls),
                        "stale_telemetry_event_count": sum(not bool(item["telemetry_fresh"]) for item in controls),
                        "control_oscillation_count": max(0, sum(
                            group[index]["control_state_signature"] != group[index - 1]["control_state_signature"]
                            for index in range(1, len(group))
                        ) - 1),
                        "oracle_reference_available": bool(comparable),
                        "oracle_comparable_window_count": len(comparable),
                        "oracle_comparable_measured_energy_j": comparable_energy if comparable else None,
                        "oracle_reference_energy_j": oracle_energy if comparable else None,
                        "oracle_gap": (comparable_energy - oracle_energy) / oracle_energy if oracle_energy else None,
                        "measurement_valid": bool(group) and all(item["measurement_valid"] for item in group),
                    })

        comparisons = paired_energy_comparisons(summaries, [str(item["id"]) for item in traces])
        passive = [item for item in summaries if item["policy"] == "PASSIVE_FULL"]
        classification, classification_reasons = classify_robustness(comparisons, passive)
        adaptation_traces = [{
            "id": trace["id"],
            "states": [str(next(item for item in self.phase4c["stress_workloads"] if item["id"] == state)["base_workload"]) for state in trace["states"]],
        } for trace in traces]
        adaptation = summarize_dynamic_adaptation(
            rows, self.control_rows, requests, adaptation_traces, POLICIES,
            float(self.phase4c["slo"]["ttft_ms"]), float(self.phase4c["slo"]["tpot_ms"]),
        )
        expected = int(self.phase4c["repeats"]) * len(POLICIES) * sum(len(item["states"]) for item in traces) * int(self.phase4c["windows_per_state"])
        inherited = [json.loads(item["hard_gates_json"]) for item in rows]
        hard_gates = {
            "accepted_formal_phase4b_dynamic_audit": bool(accepted_audit.get("valid")),
            "exact_static_vs_unchanged_passive_comparison": tuple(POLICIES) == ("STATIC", "PASSIVE_FULL"),
            "complete_stress_plan": len(rows) == expected,
            "all_measurements_valid": len(rows) == expected and all(item["measurement_valid"] for item in rows),
            "all_warmups_valid": bool(warmups) and all(item.get("valid") for item in warmups),
            "transition_context_has_no_future_leak": all(item["feedback_context_key"] != item["workload"] for item in rows if item["workload_changed"]),
            "exact_request_ids_tokens_routes_and_energy": bool(inherited) and all(
                gate.get("logical_request_id") and gate.get("requested_output_tokens")
                and gate.get("endpoint_assignment") and gate.get("explicit_compatibility")
                and gate.get("nvml_energy_windows") for gate in inherited
            ),
            "validated_connector_concurrency_preserved": int(self.config["client"]["max_concurrency"]) == 1,
            "higher_rate_moderate_and_high_regimes_present": max(float(item["rate_rps"]) for item in self.phase4c["stress_workloads"]) >= 2.0,
            "phase4a_oracle_used_post_hoc_only_when_covered": all(
                item["oracle_reference_available"] == (abs(float(item["rate_rps"]) - float(self.phase4c["oracle_reference_rate_rps"])) <= 1e-9)
                for item in rows
            ),
            "safe_high_restoration": restoration_valid,
            "no_models_or_new_controller_mechanisms": True,
            "no_unresolved_error": error is None,
        }
        valid = all(hard_gates.values())
        if not valid:
            classification = "D. INCONCLUSIVE"
            classification_reasons.append("one or more Phase 4C measurement hard gates failed")

        result_fields = tuple(rows[0]) if rows else ()
        _csv(self.run_dir / "phase4c_results.csv", rows, result_fields)
        _csv(self.run_dir / "phase4c_summary.csv", summaries, tuple(summaries[0]) if summaries else ())
        _csv(self.run_dir / "raw" / "phase4c_requests.csv", requests, tuple(requests[0]) if requests else ())
        _csv(self.run_dir / "raw" / "phase4c_controls.csv", self.control_rows, tuple(self.control_rows[0]) if self.control_rows else ())
        _csv(self.run_dir / "raw" / "phase4c_adaptation.csv", adaptation, tuple(adaptation[0]) if adaptation else ())
        _csv(self.run_dir / "raw" / "phase4c_actuator_actions.csv", self.actuator_rows, tuple(self.actuator_rows[0]) if self.actuator_rows else ())
        _csv(self.run_dir / "raw" / "phase4c_endpoint_telemetry.csv", self.telemetry_rows, tuple(self.telemetry_rows[0]) if self.telemetry_rows else ())

        audit = {
            "phase": "4C_passive_robustness_stress", "valid": valid,
            "hard_gates": hard_gates, "error": error,
            "accepted_phase4b_dynamic_audit": accepted_path.as_posix(),
            "accepted_phase4b_dynamic_sha256": hashlib.sha256(accepted_raw).hexdigest(),
            "warmup_audits": warmups, "models_trained_or_used": [],
            "controller_change": "none",
            "oracle_usage": "frozen Phase 4A post-hoc reference only for exact shape/rate/SLO-covered windows",
        }
        _write_json(self.run_dir / "phase4c_audit.json", audit)
        summary = {
            "phase": "4C_passive_robustness_stress", "valid": valid,
            "classification": classification, "classification_reasons": classification_reasons,
            "hard_gates": hard_gates, "error": error, "slo": self.phase4c["slo"],
            "policy_repeat_summaries": summaries, "energy_comparisons": comparisons,
            "adaptation_summaries": adaptation, "models_trained_or_used": [],
            "claim_boundary": (
                "max_concurrency remains one; offered-rate pressure reduces idle gaps, "
                "and client semaphore waiting before send is not part of reported TTFT. "
                "Oracle gaps are partial and only reported for exact Phase 4A-covered windows."
            ),
        }
        _write_json(self.run_dir / "phase4c_summary.json", summary)
        lines = [
            "# XpYd Phase 4C passive robustness / stress evaluation", "",
            "Measurement verdict: **%s**" % ("PASS" if valid else "FAIL"),
            "", "Decision classification: **%s**" % classification, "",
            "Tighter SLO: TTFT %.0f ms / TPOT %.0f ms. Connector concurrency remains 1." % (
                float(self.phase4c["slo"]["ttft_ms"]), float(self.phase4c["slo"]["tpot_ms"]),
            ), "",
            "| Policy | Trace | Repeat | Energy (J) | J/request | SLO violations | Stale | Fallback |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in summaries:
            lines.append("| %s | %s | %d | %s | %s | %d | %d | %d |" % (
                item["policy"], item["trace_id"], int(item["repeat"]),
                _format_optional_float(item["total_gpu_gross_energy_j"]),
                _format_optional_float(item["joules_per_request"]),
                int(item["slo_violation_count"]), int(item["stale_telemetry_event_count"]),
                int(item["fallback_event_count"]),
            ))
        lines.extend(["", "Energy comparison: `%s`" % json.dumps(comparisons, sort_keys=True), "",
                      "Classification evidence: `%s`" % json.dumps(classification_reasons), "",
                      "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), ""])
        if error:
            lines.extend(["Failure: `%s`" % error, ""])
        (self.run_dir / "phase4c_summary.md").write_text("\n".join(lines), encoding="utf-8")
        if not valid:
            raise Phase4CError(error or "Phase 4C measurement hard-gate audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    Phase4CStressHarness(load_config(Path(args.config)), args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
