"""Analysis-only Phase 4C/4C.1 routing and SLO diagnosis.

The script consumes compact accepted artifacts copied from jobs 255354 and
255370.  It does not contact the cluster, mutate the controller, or launch a
workload.  Optional ``window_endpoints`` directories contain the existing
per-window ``endpoint_summary.csv`` files used for queue/KV checks.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable


TTFT_SLO_MS = 550.0
TPOT_SLO_MS = 65.0
SENSITIVITY_THRESHOLDS_MS = (500.0, 550.0, 600.0, 650.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def descriptive(values: list[float]) -> dict[str, Any]:
    return {
        "request_count": len(values),
        "mean_ttft_ms": mean(values),
        "p50_ttft_ms": percentile(values, 0.50),
        "p95_ttft_ms": percentile(values, 0.95),
        "p99_ttft_ms": percentile(values, 0.99),
        "min_ttft_ms": min(values),
        "max_ttft_ms": max(values),
        **{
            f"violation_rate_at_{int(threshold)}ms": (
                sum(value > threshold for value in values) / len(values)
            )
            for threshold in SENSITIVITY_THRESHOLDS_MS
        },
    }


def request_pass(row: dict[str, str]) -> bool:
    return (
        float(row["ttft_ms"]) <= TTFT_SLO_MS
        and float(row["tpot_ms"]) <= TPOT_SLO_MS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(label: str, root: Path) -> dict[str, Any]:
    required = (
        "phase4c_results.csv",
        "phase4c_requests.csv",
        "phase4c_controls.csv",
        "phase4c_endpoint_telemetry.csv",
        "phase4c_plan.json",
        "phase4c_summary.json",
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    results = read_csv(root / "phase4c_results.csv")
    requests = read_csv(root / "phase4c_requests.csv")
    controls = read_csv(root / "phase4c_controls.csv")
    telemetry = read_csv(root / "phase4c_endpoint_telemetry.csv")
    plan = json.loads((root / "phase4c_plan.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "phase4c_summary.json").read_text(encoding="utf-8"))
    if len(results) != 162 or len(requests) != 486:
        raise ValueError(f"{label}: unexpected result/request counts")
    if not summary.get("valid"):
        raise ValueError(f"{label}: accepted summary is not valid")
    windows = {row["window_id"]: row for row in results}
    request_windows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in requests:
        request_windows[row["window_id"]].append(row)
    for rows in request_windows.values():
        rows.sort(key=lambda row: row["request_id"])

    endpoint_summaries: dict[str, list[dict[str, str]]] = {}
    endpoint_root = root / "window_endpoints"
    if endpoint_root.is_dir():
        for path in endpoint_root.glob("*/endpoint_summary.csv"):
            endpoint_summaries[path.parent.name] = read_csv(path)

    return {
        "label": label,
        "root": root,
        "results": results,
        "requests": requests,
        "controls": controls,
        "telemetry": telemetry,
        "plan": plan,
        "summary": summary,
        "windows": windows,
        "request_windows": request_windows,
        "endpoint_summaries": endpoint_summaries,
        "hashes": {name: sha256(root / name) for name in required},
    }


def static_feasibility(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        grouped: dict[tuple[str, int, str, str, int], list[float]] = defaultdict(list)
        for request in run["requests"]:
            if request["policy"] != "STATIC":
                continue
            key = (
                request["trace_id"], int(request["state_index"]),
                request["stress_workload"], request["workload"], int(request["repeat"]),
            )
            value = float(request["ttft_ms"])
            grouped[key].append(value)
            pooled[request["workload"]].append(value)
        for key, values in sorted(grouped.items()):
            trace, state, segment, workload, repeat = key
            rows.append({
                "job": run["label"], "trace": trace, "state_index": state,
                "segment": segment, "workload": workload, "repeat": repeat,
                **descriptive(values),
            })
    pooled_summary: dict[str, Any] = {}
    for workload, values in sorted(pooled.items()):
        stats = descriptive(values)
        pooled_summary[workload] = stats
        rows.append({
            "job": "POOLED", "trace": "ALL", "state_index": "ALL",
            "segment": "ALL", "workload": workload, "repeat": "ALL", **stats,
        })
    all_values = [value for values in pooled.values() for value in values]
    pooled_summary["ALL"] = descriptive(all_values)
    rows.append({
        "job": "POOLED", "trace": "ALL", "state_index": "ALL",
        "segment": "ALL", "workload": "ALL", "repeat": "ALL",
        **pooled_summary["ALL"],
    })
    return rows, pooled_summary


def endpoint_window_load(run: dict[str, Any], window_ids: set[str]) -> dict[str, Any]:
    endpoint_rows = [
        row for window_id in window_ids
        for row in run["endpoint_summaries"].get(window_id, ())
    ]
    if not endpoint_rows:
        return {
            "endpoint_summary_available": False,
            "max_mean_queue_depth": None,
            "nonzero_queue_endpoint_windows": None,
            "max_mean_kv_cache_usage_frac": None,
        }
    return {
        "endpoint_summary_available": True,
        "max_mean_queue_depth": max(float(row["mean_queue_depth"]) for row in endpoint_rows),
        "nonzero_queue_endpoint_windows": sum(
            float(row["mean_queue_depth"]) > 0 for row in endpoint_rows
        ),
        "max_mean_kv_cache_usage_frac": max(
            float(row["mean_kv_cache_usage_frac"]) for row in endpoint_rows
        ),
    }


def routing_load(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    overall: dict[str, Any] = {}
    for run in runs:
        grouped: dict[tuple[str, int, str, str, int, str], list[dict[str, str]]] = defaultdict(list)
        for request in run["requests"]:
            key = (
                request["trace_id"], int(request["state_index"]),
                request["stress_workload"], request["workload"],
                int(request["repeat"]), request["policy"],
            )
            grouped[key].append(request)
        for key, requests in sorted(grouped.items()):
            trace, state, segment, workload, repeat, policy = key
            window_ids = {request["window_id"] for request in requests}
            p_counts = Counter(request["prefill_endpoint_id"] for request in requests)
            d_counts = Counter(request["decode_endpoint_id"] for request in requests)
            route_counts = Counter(
                f'{request["prefill_endpoint_id"]}->{request["decode_endpoint_id"]}'
                for request in requests
            )
            frequency_counts: Counter[str] = Counter()
            endpoint_ttft: dict[str, list[float]] = defaultdict(list)
            clock_matches: list[float] = []
            for request in requests:
                window = run["windows"][request["window_id"]]
                endpoint = request["prefill_endpoint_id"]
                frequency_counts[window[f"{endpoint}_requested_level"]] += 1
                endpoint_ttft[endpoint].append(float(request["ttft_ms"]))
            for window_id in window_ids:
                window = run["windows"][window_id]
                clock_matches.extend(
                    float(window[f"{endpoint}_frequency_match_fraction"])
                    for endpoint in ("P0", "P1", "D0", "D1")
                )
            output.append({
                "job": run["label"], "trace": trace, "state_index": state,
                "segment": segment, "workload": workload, "repeat": repeat,
                "policy": policy, "request_count": len(requests),
                "P0_requests": p_counts["P0"], "P1_requests": p_counts["P1"],
                "D0_requests": d_counts["D0"], "D1_requests": d_counts["D1"],
                "P0_D0_requests": route_counts["P0->D0"],
                "P0_D1_requests": route_counts["P0->D1"],
                "P1_D0_requests": route_counts["P1->D0"],
                "P1_D1_requests": route_counts["P1->D1"],
                "P0_mean_ttft_ms": mean(endpoint_ttft["P0"]) if endpoint_ttft["P0"] else None,
                "P1_mean_ttft_ms": mean(endpoint_ttft["P1"]) if endpoint_ttft["P1"] else None,
                "selected_P_LOW_requests": frequency_counts["LOW"],
                "selected_P_MID_requests": frequency_counts["MID"],
                "selected_P_HIGH_requests": frequency_counts["HIGH"],
                "all_prefill_on_one_endpoint": max(p_counts.values()) == len(requests),
                "minimum_clock_match_fraction": min(clock_matches),
                **endpoint_window_load(run, window_ids),
            })
        for policy in ("STATIC", "PASSIVE_FULL"):
            requests = [row for row in run["requests"] if row["policy"] == policy]
            p_counts = Counter(row["prefill_endpoint_id"] for row in requests)
            routes = Counter(
                f'{row["prefill_endpoint_id"]}->{row["decode_endpoint_id"]}'
                for row in requests
            )
            overall[f'{run["label"]}_{policy}'] = {
                "request_count": len(requests),
                "prefill_counts": dict(sorted(p_counts.items())),
                "route_counts": dict(sorted(routes.items())),
            }
    return output, overall


def build_pairs(run: dict[str, Any]) -> list[dict[str, Any]]:
    result_index = {
        (
            row["trace_id"], int(row["repeat"]), int(row["state_index"]),
            int(row["window_in_state"]), row["policy"],
        ): row
        for row in run["results"]
    }
    pairs: list[dict[str, Any]] = []
    base_keys = sorted({key[:4] for key in result_index})
    for trace, repeat, state, window_in_state in base_keys:
        static_window = result_index[(trace, repeat, state, window_in_state, "STATIC")]
        passive_window = result_index[(trace, repeat, state, window_in_state, "PASSIVE_FULL")]
        static_requests = run["request_windows"][static_window["window_id"]]
        passive_requests = run["request_windows"][passive_window["window_id"]]
        for ordinal, (static, passive) in enumerate(zip(static_requests, passive_requests), 1):
            selected_p = passive["prefill_endpoint_id"]
            route = f'{passive["prefill_endpoint_id"]}->{passive["decode_endpoint_id"]}'
            static_route = f'{static["prefill_endpoint_id"]}->{static["decode_endpoint_id"]}'
            level = passive_window[f"{selected_p}_requested_level"]
            static_ok, passive_ok = request_pass(static), request_pass(passive)
            category = (
                "both_pass" if static_ok and passive_ok else
                "static_pass_passive_fail" if static_ok else
                "static_fail_passive_pass" if passive_ok else "both_fail"
            )
            pairs.append({
                "trace": trace, "repeat": repeat, "state_index": state,
                "window_in_state": window_in_state, "request_ordinal": ordinal,
                "segment": passive["stress_workload"], "workload": passive["workload"],
                "category": category, "static_route": static_route,
                "passive_route": route, "same_route": static_route == route,
                "same_prefill_endpoint": (
                    static["prefill_endpoint_id"] == passive["prefill_endpoint_id"]
                ),
                "passive_P_level": level,
                "static_ttft_ms": float(static["ttft_ms"]),
                "passive_ttft_ms": float(passive["ttft_ms"]),
                "ttft_difference_ms": float(passive["ttft_ms"]) - float(static["ttft_ms"]),
            })
    return pairs


def matched_conditions(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    definitions = {
        "exact_route_and_P_HIGH": lambda row: row["same_route"] and row["passive_P_level"] == "HIGH",
        "same_P_and_P_HIGH": lambda row: row["same_prefill_endpoint"] and row["passive_P_level"] == "HIGH",
    }
    for definition, predicate in definitions.items():
        selected = [row for row in pairs if predicate(row)]
        workloads = ["ALL"] + sorted({row["workload"] for row in selected})
        for workload in workloads:
            rows = selected if workload == "ALL" else [
                row for row in selected if row["workload"] == workload
            ]
            if not rows:
                continue
            static = [row["static_ttft_ms"] for row in rows]
            passive = [row["passive_ttft_ms"] for row in rows]
            differences = [row["ttft_difference_ms"] for row in rows]
            output.append({
                "match_definition": definition, "workload": workload,
                "pair_count": len(rows), "static_mean_ttft_ms": mean(static),
                "passive_mean_ttft_ms": mean(passive),
                "mean_passive_minus_static_ttft_ms": mean(differences),
                "median_passive_minus_static_ttft_ms": median(differences),
                "static_violation_count": sum(value > TTFT_SLO_MS for value in static),
                "passive_violation_count": sum(value > TTFT_SLO_MS for value in passive),
            })
    return output


def paired_breakdown(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in pairs:
        key = (
            row["trace"], row["repeat"], row["state_index"], row["workload"],
            row["passive_P_level"], row["passive_route"],
        )
        grouped[key][row["category"]] += 1
    output = []
    for key, counts in sorted(grouped.items()):
        trace, repeat, state, workload, level, route = key
        output.append({
            "trace": trace, "repeat": repeat, "state_index": state,
            "workload": workload, "passive_P_level": level,
            "passive_route": route, "paired_requests": sum(counts.values()),
            "both_pass": counts["both_pass"],
            "static_pass_passive_fail": counts["static_pass_passive_fail"],
            "static_fail_passive_pass": counts["static_fail_passive_pass"],
            "both_fail": counts["both_fail"],
        })
    totals = Counter(row["category"] for row in pairs)
    candidates = [row for row in pairs if row["category"] == "static_pass_passive_fail"]
    summary = {
        "categories": dict(totals),
        "static_pass_passive_fail_by_P_level": dict(
            Counter(row["passive_P_level"] for row in candidates)
        ),
        "static_pass_passive_fail_by_workload": dict(
            Counter(row["workload"] for row in candidates)
        ),
        "static_pass_passive_fail_by_route": dict(
            Counter(row["passive_route"] for row in candidates)
        ),
    }
    return output, summary


def action_counts(run: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for control in run["controls"]:
        if control["policy"] != "PASSIVE_FULL":
            continue
        for recommendation in json.loads(control["dvfs_recommendations_json"] or "[]"):
            counts[recommendation["action"]] += 1
    return dict(counts)


def recovery_effect(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for run in runs:
        grouped: dict[tuple[str, int, str, str], list[dict[str, str]]] = defaultdict(list)
        for request in run["requests"]:
            if request["policy"] == "PASSIVE_FULL":
                grouped[(
                    request["trace_id"], int(request["state_index"]),
                    request["stress_workload"], request["workload"],
                )].append(request)
        for key, requests in sorted(grouped.items()):
            trace, state, segment, workload = key
            window_ids = sorted({row["window_id"] for row in requests})
            windows = [run["windows"][window_id] for window_id in window_ids]
            levels: Counter[str] = Counter()
            for request in requests:
                window = run["windows"][request["window_id"]]
                levels[window[f'{request["prefill_endpoint_id"]}_requested_level']] += 1
            values = [float(row["ttft_ms"]) for row in requests]
            output.append({
                "job": run["label"], "trace": trace, "state_index": state,
                "segment": segment, "workload": workload,
                "request_count": len(requests), "mean_ttft_ms": mean(values),
                "p50_ttft_ms": percentile(values, 0.50),
                "p95_ttft_ms": percentile(values, 0.95),
                "ttft_violation_count": sum(value > TTFT_SLO_MS for value in values),
                "ttft_violation_rate": sum(value > TTFT_SLO_MS for value in values) / len(values),
                "total_gpu_gross_energy_j": sum(
                    float(window["total_gpu_gross_energy_j"]) for window in windows
                ),
                "selected_P_LOW_requests": levels["LOW"],
                "selected_P_MID_requests": levels["MID"],
                "selected_P_HIGH_requests": levels["HIGH"],
            })
        passive_summary = [
            row for row in run["summary"]["policy_repeat_summaries"]
            if row["policy"] == "PASSIVE_FULL"
        ]
        total_energy = sum(row["total_gpu_gross_energy_j"] for row in passive_summary)
        total_requests = sum(row["request_count"] for row in passive_summary)
        total_violations = sum(row["slo_violation_count"] for row in passive_summary)
        summary[run["label"]] = {
            "total_gpu_gross_energy_j": total_energy,
            "joules_per_request": total_energy / total_requests,
            "slo_violation_count": total_violations,
            "slo_violation_rate": total_violations / total_requests,
            "actions": action_counts(run),
        }
    target = {
        run["label"]: next(
            row for row in output
            if row["job"] == run["label"]
            and row["trace"] == "high_prefill_decode_prefill"
            and row["state_index"] == 2
        )
        for run in runs
    }
    summary["decode_to_prefill_target"] = target
    return output, summary


def noise_order(runs: list[dict[str, Any]]) -> dict[str, Any]:
    orders_equal = runs[0]["plan"]["block_order"] == runs[1]["plan"]["block_order"]
    block_positions: dict[str, dict[str, float]] = {}
    for run in runs:
        positions: dict[str, list[int]] = defaultdict(list)
        for index, block in enumerate(run["plan"]["block_order"], 1):
            positions[block["policy"]].append(index)
        block_positions[run["label"]] = {
            policy: mean(values) for policy, values in positions.items()
        }

    static_cell_means: dict[tuple[str, str, int, int], float] = {}
    repeat_variation: dict[str, dict[str, Any]] = {}
    for run in runs:
        grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
        for request in run["requests"]:
            if request["policy"] == "STATIC":
                grouped[(
                    request["trace_id"], int(request["state_index"]), int(request["repeat"])
                )].append(float(request["ttft_ms"]))
        for key, values in grouped.items():
            static_cell_means[(run["label"],) + key] = mean(values)
        repeat_sds: list[float] = []
        repeat_ranges: list[float] = []
        for trace, state in sorted({key[:2] for key in grouped}):
            means = [mean(grouped[(trace, state, repeat)]) for repeat in (1, 2, 3)]
            repeat_sds.append(stdev(means))
            repeat_ranges.append(max(means) - min(means))
        repeat_variation[run["label"]] = {
            "median_repeat_mean_sd_ms": median(repeat_sds),
            "maximum_repeat_mean_sd_ms": max(repeat_sds),
            "median_repeat_mean_range_ms": median(repeat_ranges),
            "maximum_repeat_mean_range_ms": max(repeat_ranges),
        }
    base_keys = sorted({key[1:] for key in static_cell_means})
    cross_differences = [
        static_cell_means[(runs[1]["label"],) + key]
        - static_cell_means[(runs[0]["label"],) + key]
        for key in base_keys
    ]
    timing: dict[str, Any] = {}
    clock: dict[str, Any] = {}
    for run in runs:
        starts = [float(row["timing_start_unix_s"]) for row in run["results"]]
        ends = [float(row["timing_end_unix_s"]) for row in run["results"]]
        timing[run["label"]] = {
            "start_utc": datetime.fromtimestamp(min(starts), timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(max(ends), timezone.utc).isoformat(),
            "span_hours": (max(ends) - min(starts)) / 3600.0,
        }
        fractions = [
            float(row[f"{endpoint}_frequency_match_fraction"])
            for row in run["results"] for endpoint in ("P0", "P1", "D0", "D1")
        ]
        clock[run["label"]] = {
            "minimum_frequency_match_fraction": min(fractions),
            "windows_below_95_percent_match": sum(value < 0.95 for value in fractions),
        }
    return {
        "block_order_identical_between_jobs": orders_equal,
        "mean_block_position": block_positions,
        "static_cross_job_cell_mean_difference_ms": {
            "cell_count": len(cross_differences),
            "mean": mean(cross_differences),
            "median": median(cross_differences),
            "mean_absolute": mean(abs(value) for value in cross_differences),
            "standard_deviation": stdev(cross_differences),
            "minimum": min(cross_differences),
            "maximum": max(cross_differences),
        },
        "static_repeat_variation": repeat_variation,
        "timing": timing,
        "clock_stability": clock,
        "temperature_samples_available": False,
        "background_load_samples_available": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4c_dir", type=Path)
    parser.add_argument("phase4c1_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    runs = [load_run("255354", args.phase4c_dir), load_run("255370", args.phase4c1_dir)]
    static_rows, feasibility = static_feasibility(runs)
    routing_rows, routing_summary = routing_load(runs)
    fixed_pairs = build_pairs(runs[1])
    matched_rows = matched_conditions(fixed_pairs)
    paired_rows, paired_summary = paired_breakdown(fixed_pairs)
    recovery_rows, recovery_summary = recovery_effect(runs)
    order_noise = noise_order(runs)

    write_csv(args.output_dir / "static_high_ttft.csv", static_rows)
    write_csv(args.output_dir / "routing_load.csv", routing_rows)
    write_csv(args.output_dir / "matched_conditions.csv", matched_rows)
    write_csv(args.output_dir / "paired_fixed_breakdown.csv", paired_rows)
    write_csv(args.output_dir / "recovery_effect.csv", recovery_rows)

    diagnosis = {
        "phase": "Phase 4C/4C.1 analysis-only routing/SLO diagnosis",
        "source_jobs": {
            run["label"]: {
                "valid": run["summary"]["valid"],
                "classification": run["summary"]["classification"],
                "input_sha256": run["hashes"],
            }
            for run in runs
        },
        "official_slo": {"ttft_ms": TTFT_SLO_MS, "tpot_ms": TPOT_SLO_MS},
        "static_high_feasibility": feasibility,
        "routing": routing_summary,
        "matched_conditions": matched_rows,
        "paired_fixed": paired_summary,
        "recovery_effect": recovery_summary,
        "order_and_noise": order_noise,
        "decision": "E. MIXED",
        "decision_basis": {
            "slo_feasibility": "strong: STATIC@HIGH violates 550 ms on heavy workloads",
            "routing_load_distribution": (
                "moderate: PASSIVE sends every prefill request to P0 while STATIC is balanced; "
                "existing blocks cannot isolate concentration from time/order history"
            ),
            "remaining_dvfs_control": (
                "weak as dominant cause: most fixed-PASSIVE excess failures occur at P HIGH"
            ),
            "noise_order": (
                "meaningful: cross-job and repeat variation is comparable to near-boundary gaps, "
                "but identical interleaved block order provides no directional order explanation"
            ),
        },
        "smallest_next_action": "one minimal matched-routing experiment",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase4c_routing_slo_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
