"""Minimal fixed-HIGH matched-routing experiment for Phase 4C.2.

This experiment isolates block-level route concentration from DVFS.  It runs
only two fixed routing modes, two previously validated heavy workloads, and
three counterbalanced repeats.  No controller decision is made.
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
from xpyd.phase4b_evaluation import Phase4BEvaluationHarness


POLICIES = ("BALANCED_HIGH", "P0D0_HIGH")


class Phase4CMatchedError(RuntimeError):
    """Fail-closed matched-routing configuration or measurement error."""


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(float(value) for value in values)


def counterbalanced_blocks(
    workloads: Sequence[Mapping[str, Any]], repeats: int, seed: int,
) -> list[tuple[int, int, str, dict[str, Any]]]:
    """Return adjacent A/B pairs with globally balanced first-policy counts."""
    pairs = [
        (repeat, workload_index, dict(workload))
        for repeat in range(1, repeats + 1)
        for workload_index, workload in enumerate(workloads)
    ]
    random.Random(seed).shuffle(pairs)
    blocks: list[tuple[int, int, str, dict[str, Any]]] = []
    for pair_index, (repeat, workload_index, workload) in enumerate(pairs, 1):
        order = POLICIES if (repeat + workload_index) % 2 == 0 else tuple(reversed(POLICIES))
        blocks.extend((pair_index, repeat, policy, workload) for policy in order)
    return blocks


def load_config(path: Path) -> dict[str, Any]:
    config = load_phase3d_config(path)
    settings = config.get("phase4c_matched")
    if not isinstance(settings, dict):
        raise Phase4CMatchedError("config requires phase4c_matched")
    if tuple(settings.get("policies", ())) != POLICIES:
        raise Phase4CMatchedError("matched routing compares exactly BALANCED_HIGH and P0D0_HIGH")
    if int(settings.get("repeats", 0)) != 3:
        raise Phase4CMatchedError("matched routing requires exactly three repeats")
    if int(settings.get("requests_per_block", 0)) != 3:
        raise Phase4CMatchedError("matched routing requires exactly three measured requests per block")
    if int(settings.get("priming_requests_per_block", 0)) != 3:
        raise Phase4CMatchedError("matched routing requires exactly three priming requests per block")
    if int(config["client"].get("max_concurrency", 0)) != 1:
        raise Phase4CMatchedError("matched routing preserves connector concurrency one")
    endpoints = {str(item["endpoint_id"]): item for item in config["endpoints"]}
    if set(endpoints) != {"P0", "P1", "D0", "D1"}:
        raise Phase4CMatchedError("matched routing requires exactly P0/P1/D0/D1")
    if any(str(endpoints[key]["node"]) != "neptune" for key in ("P0", "P1")):
        raise Phase4CMatchedError("matched routing requires Neptune P endpoints")
    if any(str(endpoints[key]["node"]) != "io" for key in ("D0", "D1")):
        raise Phase4CMatchedError("matched routing requires io D endpoints")
    if any(int(item["tp_degree"]) != 1 for item in endpoints.values()):
        raise Phase4CMatchedError("matched routing fixes all endpoints at TP1")
    workloads = settings.get("workloads", ())
    if len(workloads) != 2 or {
        str(item.get("base_workload")) for item in workloads
    } != {"prefill_heavy", "both_heavy"}:
        raise Phase4CMatchedError("matched routing measures only prefill-heavy and both-heavy")
    if any(float(item.get("rate_rps", 0)) != 2.0 for item in workloads):
        raise Phase4CMatchedError("matched routing reuses the validated 2 request/s heavy rate")
    slo = settings.get("slo", {})
    if float(slo.get("ttft_ms", 0)) != 550.0 or float(slo.get("tpot_ms", 0)) != 65.0:
        raise Phase4CMatchedError("matched routing preserves the official 550/65 ms SLO")
    if list(settings.get("concentrated_pair", ())) != ["P0", "D0"]:
        raise Phase4CMatchedError("matched routing concentrated pair must be P0->D0")
    return config


class Phase4CMatchedRoutingHarness(Phase4BEvaluationHarness):
    """Physical HIGH-only balanced-versus-concentrated routing harness."""

    def __init__(
        self, config: Mapping[str, Any], run_id: Optional[str] = None,
        backend: Any = None,
    ) -> None:
        super().__init__(config, run_id=run_id, backend=backend, smoke=False, stage="stationary")
        self.matched = dict(self.config["phase4c_matched"])
        self.settings["slo"] = dict(self.matched["slo"])
        self.run_dir = Path(os.path.expandvars(str(self.matched["output_root"]))) / self.run_id

    def _policy_pairs(
        self, policy: str, all_pairs: Sequence[Sequence[str]],
    ) -> list[list[str]]:
        if policy == "BALANCED_HIGH":
            return [list(pair) for pair in all_pairs]
        if policy == "P0D0_HIGH":
            return [list(self.matched["concentrated_pair"])]
        raise Phase4CMatchedError("unknown matched-routing policy: %s" % policy)

    def run(self) -> Path:
        if self.run_dir.exists():
            raise Phase4CMatchedError("run directory already exists: %s" % self.run_dir)
        accepted_path = Path(os.path.expandvars(str(self.matched["accepted_phase4c1_audit"])))
        if not accepted_path.is_file():
            raise Phase4CMatchedError("missing accepted Phase 4C.1 audit")
        accepted_raw = accepted_path.read_bytes()
        accepted_audit = json.loads(accepted_raw)
        if not accepted_audit.get("valid") or not str(accepted_audit.get("phase", "")).startswith("4C1"):
            raise Phase4CMatchedError("matched routing requires a valid Phase 4C.1 audit")

        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        reference = self._reference()
        accepted_freq = self._accepted_frequencies(reference)
        capabilities = self._discover_exact(accepted_freq)
        _write_json(self.run_dir / "raw" / "capabilities.json", {
            endpoint: asdict(capability) for endpoint, capability in capabilities.items()
        })
        _write_json(self.run_dir / "raw" / "accepted_phase4c1_reference.json", {
            "path": accepted_path.as_posix(),
            "sha256": hashlib.sha256(accepted_raw).hexdigest(),
            "valid": True,
        })
        high = {endpoint: values["HIGH"] for endpoint, values in accepted_freq.items()}
        actuator = PerEndpointClockActuator(
            self.backend, capabilities,
            float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        actuator.requested.update(high)
        _, _, all_pairs = build_registry_and_compatibility(self.config)
        concentrated = list(self.matched["concentrated_pair"])
        if concentrated not in [list(pair) for pair in all_pairs]:
            raise Phase4CMatchedError("P0->D0 is not in the validated compatibility graph")
        base_by_id = {str(item["id"]): dict(item) for item in self.config["workloads"]}
        workload_specs = [dict(item) for item in self.matched["workloads"]]
        repeats = int(self.matched["repeats"])
        request_count = int(self.matched["requests_per_block"])
        blocks = counterbalanced_blocks(
            workload_specs, repeats, int(self.matched.get("order_seed", 71))
        )
        _write_json(self.run_dir / "raw" / "phase4c_matched_plan.json", {
            "policies": list(POLICIES), "workloads": workload_specs,
            "repeats": repeats, "requests_per_block": request_count,
            "priming_requests_per_block": int(self.matched["priming_requests_per_block"]),
            "priming_output_len": int(self.matched["priming_output_len"]),
            "max_concurrency": 1, "slo": self.matched["slo"],
            "frequencies": "all endpoints fixed at validated HIGH",
            "block_order": [
                {
                    "block_sequence": index, "pair_sequence": pair_index,
                    "repeat": repeat, "policy": policy,
                    "workload": str(workload["base_workload"]),
                }
                for index, (pair_index, repeat, policy, workload) in enumerate(blocks, 1)
            ],
            "models": [], "controller_decisions": [],
        })

        rows: list[dict[str, Any]] = []
        request_rows: list[dict[str, Any]] = []
        endpoint_rows: list[dict[str, Any]] = []
        warmups: list[dict[str, Any]] = []
        restoration_valid = False
        error: Optional[str] = None
        try:
            context = self._new_context(capabilities)
            self._actuate(
                actuator, context, high, "phase4c_matched_global_high",
                controller_action=False, policy="WARMUP", workload="all",
                iteration_id="phase4c-matched-global-warmup",
            )
            context.selected_pairs = [list(pair) for pair in all_pairs]
            self._write_routes(all_pairs, "phase4c_matched_global_all_route_warmup")
            global_dir = self._run_window(
                "phase4c-matched-global-warmup",
                {"id": "phase4c_matched_global_warmup", "input_len": 128,
                 "output_len": 16, "rate_rps": 0.5},
                4, high, all_pairs, windows_root, require_all_pairs=True,
            )
            global_audit = _read_json(global_dir / "audit.json")
            warmups.append({"window_id": "phase4c-matched-global-warmup", **global_audit})
            if not global_audit.get("valid"):
                raise Phase4CMatchedError("global all-route warmup failed")

            for block_sequence, (pair_sequence, repeat, policy, spec) in enumerate(blocks, 1):
                workload_id = str(spec["base_workload"])
                workload = copy.deepcopy(base_by_id[workload_id])
                workload["rate_rps"] = float(spec["rate_rps"])
                pairs = self._policy_pairs(policy, all_pairs)
                context = self._new_context(capabilities)
                self._actuate(
                    actuator, context, high, "phase4c_matched_fixed_high",
                    controller_action=False, policy=policy, workload=workload_id,
                    iteration_id="phase4c-matched-b%02d-high" % block_sequence,
                )
                context.selected_pairs = copy.deepcopy(pairs)
                self._write_routes(pairs, "phase4c_matched_%s" % policy.lower())

                priming_id = "p4cm-b%02d-r%02d-%s-%s-prime" % (
                    block_sequence, repeat, policy.lower(), workload_id,
                )
                priming_workload = copy.deepcopy(workload)
                priming_workload["output_len"] = int(self.matched["priming_output_len"])
                priming_dir = self._run_window(
                    priming_id, priming_workload,
                    int(self.matched["priming_requests_per_block"]), high, pairs,
                    windows_root,
                )
                priming_audit = _read_json(priming_dir / "audit.json")
                warmups.append({"window_id": priming_id, **priming_audit})
                if not priming_audit.get("valid"):
                    raise Phase4CMatchedError("route-matched priming failed: %s" % priming_id)

                window_id = "p4cm-b%02d-r%02d-%s-%s" % (
                    block_sequence, repeat, policy.lower(), workload_id,
                )
                self._write_routes(pairs, "phase4c_matched_measured_%s" % policy.lower())
                window_dir = self._run_window(
                    window_id, workload, request_count, high, pairs, windows_root,
                )
                row, observed_requests = self._measurement_row(
                    window_id, policy, workload, repeat, context,
                    accepted_freq, reference, window_dir,
                )
                row.update({
                    "matched_workload_id": str(spec["id"]),
                    "pair_sequence": pair_sequence,
                    "block_sequence": block_sequence,
                    "order_within_pair": 1 if block_sequence % 2 else 2,
                    "routing_mode": (
                        "all_validated_pairs_round_robin"
                        if policy == "BALANCED_HIGH" else "fixed_P0_to_D0"
                    ),
                    "controller_decisions": 0,
                    "phase4a_oracle_joules_per_request": None,
                    "oracle_gap": None, "oracle_gap_class": None,
                    "phase4a_near_optimal_match": None,
                })
                rows.append(row)
                request_rows.extend({
                    **item, "matched_workload_id": str(spec["id"]),
                    "pair_sequence": pair_sequence, "block_sequence": block_sequence,
                    "request_ordinal": ordinal,
                } for ordinal, item in enumerate(observed_requests, 1))
                with (window_dir / "endpoint_summary.csv").open(
                    newline="", encoding="utf-8"
                ) as stream:
                    endpoint_rows.extend({
                        "window_id": window_id, "policy": policy,
                        "workload": workload_id, "repeat": repeat, **item,
                    } for item in csv.DictReader(stream))
                if not row["measurement_valid"]:
                    raise Phase4CMatchedError("measured window audit failed: %s" % window_id)
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            try:
                restore = self._new_context(capabilities)
                self._actuate(
                    actuator, restore, high, "phase4c_matched_final_safe_high",
                    controller_action=False, policy="RESTORE", workload="all",
                    iteration_id="phase4c-matched-final-restore",
                )
                restoration_valid = all(
                    actuator.read(endpoint).graphics_clock_mhz == frequency
                    for endpoint, frequency in high.items()
                )
            except Exception:
                restoration_valid = False

        return self._report(
            rows, request_rows, endpoint_rows, warmups, accepted_path,
            accepted_raw, accepted_audit, all_pairs, restoration_valid, error,
        )

    def _report(
        self, rows: list[dict[str, Any]], requests: list[dict[str, Any]],
        endpoint_rows: list[dict[str, Any]], warmups: list[dict[str, Any]],
        accepted_path: Path, accepted_raw: bytes, accepted_audit: Mapping[str, Any],
        all_pairs: Sequence[Sequence[str]], restoration_valid: bool,
        error: Optional[str],
    ) -> Path:
        summaries: list[dict[str, Any]] = []
        for workload in ("prefill_heavy", "both_heavy"):
            for policy in POLICIES:
                group = [row for row in requests if row["policy"] == policy and row["workload"] == workload]
                windows = [row for row in rows if row["policy"] == policy and row["workload"] == workload]
                if not group or not windows:
                    continue
                values = [float(row["ttft_ms"]) for row in group]
                repeat_means = [
                    _mean([row["ttft_ms"] for row in group if int(row["repeat"]) == repeat])
                    for repeat in range(1, int(self.matched["repeats"]) + 1)
                ]
                route_counts: dict[str, int] = {}
                p_counts: dict[str, int] = {}
                for item in group:
                    route = "%s->%s" % (item["prefill_endpoint_id"], item["decode_endpoint_id"])
                    route_counts[route] = route_counts.get(route, 0) + 1
                    endpoint = str(item["prefill_endpoint_id"])
                    p_counts[endpoint] = p_counts.get(endpoint, 0) + 1
                summaries.append({
                    "policy": policy, "workload": workload,
                    "repeat_count": len(repeat_means), "request_count": len(group),
                    "mean_ttft_ms": _mean(values), "p50_ttft_ms": _percentile(values, 0.50),
                    "p95_ttft_ms": _percentile(values, 0.95),
                    "p99_ttft_ms": _percentile(values, 0.99),
                    "min_ttft_ms": min(values), "max_ttft_ms": max(values),
                    "ttft_violation_count": sum(value > 550.0 for value in values),
                    "ttft_violation_rate": sum(value > 550.0 for value in values) / len(values),
                    "tpot_violation_count": sum(float(item["tpot_ms"]) > 65.0 for item in group),
                    "repeat_mean_ttft_std_ms": statistics.stdev(repeat_means),
                    "repeat_mean_ttft_min_ms": min(repeat_means),
                    "repeat_mean_ttft_max_ms": max(repeat_means),
                    "total_gpu_gross_energy_j": sum(float(item["total_gpu_gross_energy_j"]) for item in windows),
                    "joules_per_request": sum(float(item["total_gpu_gross_energy_j"]) for item in windows) / len(group),
                    "route_distribution_json": json.dumps(route_counts, sort_keys=True),
                    "prefill_distribution_json": json.dumps(p_counts, sort_keys=True),
                })

        comparisons: list[dict[str, Any]] = []
        paired_categories: dict[str, int] = {}
        for workload in ("prefill_heavy", "both_heavy"):
            by_policy = {
                item["policy"]: item for item in summaries if item["workload"] == workload
            }
            if set(by_policy) != set(POLICIES):
                continue
            balanced = by_policy["BALANCED_HIGH"]
            concentrated = by_policy["P0D0_HIGH"]
            categories = {"both_pass": 0, "balanced_pass_concentrated_fail": 0,
                          "balanced_fail_concentrated_pass": 0, "both_fail": 0}
            for repeat in range(1, int(self.matched["repeats"]) + 1):
                groups = {}
                for policy in POLICIES:
                    values = [
                        row for row in requests
                        if row["policy"] == policy and row["workload"] == workload
                        and int(row["repeat"]) == repeat
                    ]
                    values.sort(key=lambda row: int(row["request_ordinal"]))
                    groups[policy] = values
                for balanced_request, concentrated_request in zip(
                    groups["BALANCED_HIGH"], groups["P0D0_HIGH"]
                ):
                    balanced_ok = (
                        float(balanced_request["ttft_ms"]) <= 550.0
                        and float(balanced_request["tpot_ms"]) <= 65.0
                    )
                    concentrated_ok = (
                        float(concentrated_request["ttft_ms"]) <= 550.0
                        and float(concentrated_request["tpot_ms"]) <= 65.0
                    )
                    category = (
                        "both_pass" if balanced_ok and concentrated_ok else
                        "balanced_pass_concentrated_fail" if balanced_ok else
                        "balanced_fail_concentrated_pass" if concentrated_ok else "both_fail"
                    )
                    categories[category] += 1
                    paired_categories[category] = paired_categories.get(category, 0) + 1
            comparisons.append({
                "workload": workload,
                "balanced_mean_ttft_ms": balanced["mean_ttft_ms"],
                "concentrated_mean_ttft_ms": concentrated["mean_ttft_ms"],
                "concentrated_minus_balanced_ttft_ms": (
                    concentrated["mean_ttft_ms"] - balanced["mean_ttft_ms"]
                ),
                "balanced_violation_rate": balanced["ttft_violation_rate"],
                "concentrated_violation_rate": concentrated["ttft_violation_rate"],
                "concentrated_minus_balanced_violation_rate": (
                    concentrated["ttft_violation_rate"] - balanced["ttft_violation_rate"]
                ),
                "balanced_joules_per_request": balanced["joules_per_request"],
                "concentrated_joules_per_request": concentrated["joules_per_request"],
                "paired_categories_json": json.dumps(categories, sort_keys=True),
            })

        expected_windows = len(POLICIES) * 2 * int(self.matched["repeats"])
        expected_requests = expected_windows * int(self.matched["requests_per_block"])
        inherited = [json.loads(row["hard_gates_json"]) for row in rows]
        concentrated_requests = [row for row in requests if row["policy"] == "P0D0_HIGH"]
        balanced_requests = [row for row in requests if row["policy"] == "BALANCED_HIGH"]
        high_and_stable = bool(rows) and all(
            row[f"{endpoint}_requested_level"] == "HIGH"
            and float(row[f"{endpoint}_frequency_match_fraction"]) >= 0.95
            for row in rows for endpoint in ("P0", "P1", "D0", "D1")
        )
        hard_gates = {
            "accepted_phase4c1_audit": bool(accepted_audit.get("valid")),
            "complete_12_window_36_request_plan": (
                len(rows) == expected_windows and len(requests) == expected_requests
            ),
            "all_measurements_valid": bool(rows) and all(row["measurement_valid"] for row in rows),
            "all_priming_and_global_warmups_valid": (
                len(warmups) == expected_windows + 1 and all(item.get("valid") for item in warmups)
            ),
            "all_endpoints_fixed_high_with_readback": high_and_stable,
            "concentrated_route_exactly_p0d0": bool(concentrated_requests) and all(
                row["prefill_endpoint_id"] == "P0" and row["decode_endpoint_id"] == "D0"
                for row in concentrated_requests
            ),
            "balanced_route_uses_both_prefill_endpoints": bool(balanced_requests) and {
                row["prefill_endpoint_id"] for row in balanced_requests
            } == {"P0", "P1"},
            "exact_request_ids_tokens_routes_and_energy": bool(inherited) and all(
                gate.get("logical_request_id") and gate.get("requested_output_tokens")
                and gate.get("endpoint_assignment") and gate.get("explicit_compatibility")
                and gate.get("nvml_energy_windows") for gate in inherited
            ),
            "no_invalidating_thermal_or_hw_slowdown": bool(inherited) and all(
                gate.get("no_invalidating_thermal_or_hw_slowdown") for gate in inherited
            ),
            "validated_connector_concurrency_one": int(self.config["client"]["max_concurrency"]) == 1,
            "no_controller_dvfs_models_or_probing": all(
                int(row["controller_decisions"]) == 0 for row in rows
            ),
            "safe_high_restoration": restoration_valid,
            "no_unresolved_error": error is None,
        }
        valid = all(hard_gates.values())
        if not valid:
            classification = "INCONCLUSIVE: MEASUREMENT HARD GATE FAILED"
        elif all(
            item["concentrated_minus_balanced_ttft_ms"] > 0
            and item["concentrated_minus_balanced_violation_rate"] > 0
            for item in comparisons
        ):
            classification = "ROUTING CONCENTRATION PENALTY OBSERVED"
        elif any(item["concentrated_minus_balanced_ttft_ms"] > 0 for item in comparisons):
            classification = "MIXED ROUTING EFFECT"
        else:
            classification = "NO ROUTING CONCENTRATION PENALTY OBSERVED"

        if rows:
            _csv(self.run_dir / "phase4c_matched_measurements.csv", rows, tuple(rows[0]))
        if summaries:
            _csv(self.run_dir / "phase4c_matched_summary.csv", summaries, tuple(summaries[0]))
        if requests:
            _csv(self.run_dir / "raw" / "phase4c_matched_requests.csv", requests, tuple(requests[0]))
        if endpoint_rows:
            _csv(
                self.run_dir / "raw" / "phase4c_matched_endpoint_summary.csv",
                endpoint_rows, tuple(endpoint_rows[0]),
            )
        if self.actuator_rows:
            _csv(
                self.run_dir / "raw" / "phase4c_matched_actuator_actions.csv",
                self.actuator_rows, tuple(self.actuator_rows[0]),
            )
        audit = {
            "phase": "4C2_fixed_high_matched_routing", "valid": valid,
            "hard_gates": hard_gates, "error": error,
            "accepted_phase4c1_audit": accepted_path.as_posix(),
            "accepted_phase4c1_sha256": hashlib.sha256(accepted_raw).hexdigest(),
            "warmup_audits": warmups, "models_trained_or_used": [],
            "controller_decisions": 0,
            "claim_boundary": (
                "small empirical fixed-HIGH comparison within the measured route action space; "
                "not a theoretical optimum and not a controller evaluation"
            ),
        }
        _write_json(self.run_dir / "phase4c_matched_audit.json", audit)
        summary = {
            "phase": "4C2_fixed_high_matched_routing", "valid": valid,
            "classification": classification, "hard_gates": hard_gates,
            "error": error, "slo": self.matched["slo"],
            "policy_workload_summaries": summaries,
            "matched_comparisons": comparisons,
            "paired_request_categories": paired_categories,
            "validated_compatible_pairs": [list(pair) for pair in all_pairs],
            "models_trained_or_used": [], "controller_decisions": 0,
        }
        _write_json(self.run_dir / "phase4c_matched_summary.json", summary)
        lines = [
            "# Phase 4C.2 fixed-HIGH matched-routing result", "",
            "Measurement verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
            "Classification: **%s**" % classification, "",
            "| Policy | Workload | N | Mean TTFT | P95 TTFT | Violations | J/request | P distribution |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for item in summaries:
            lines.append(
                "| %s | %s | %d | %.1f | %.1f | %d | %.1f | `%s` |" % (
                    item["policy"], item["workload"], item["request_count"],
                    item["mean_ttft_ms"], item["p95_ttft_ms"],
                    item["ttft_violation_count"], item["joules_per_request"],
                    item["prefill_distribution_json"],
                )
            )
        lines.extend(["", "Matched differences:", "", "```json",
                      json.dumps(comparisons, indent=2, sort_keys=True), "```", "",
                      "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), ""])
        (self.run_dir / "phase4c_matched_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        if error is not None:
            raise Phase4CMatchedError(error)
        if not valid:
            raise Phase4CMatchedError("one or more matched-routing hard gates failed")
        return self.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(args.config)
    run_id = args.run_id or "phase4c_matched_%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    result = Phase4CMatchedRoutingHarness(config, run_id=run_id).run()
    print(result)


if __name__ == "__main__":
    main()
