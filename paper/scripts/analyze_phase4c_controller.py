"""Build compact Phase 4C controller-diagnosis tables from an accepted run.

The decision-visible EWMA/queue state is reconstructed only from the passive
block's warmup and completed prior windows.  The current window's measurements
are appended as outcomes and are never used as inputs to its decision record.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


ENDPOINTS = ("P0", "P1", "D0", "D1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def as_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def update_snapshot(
    state: dict[str, dict[str, Any]], rows: Iterable[dict[str, str]], alpha: float,
) -> None:
    for row in rows:
        endpoint = row["endpoint_id"]
        current = state[endpoint]
        for key, source in (("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms")):
            measurement = as_optional_float(row[source])
            if measurement is not None:
                previous = current[key]
                current[key] = (
                    measurement if previous is None
                    else alpha * measurement + (1.0 - alpha) * previous
                )
        queue = as_optional_float(row["queue"])
        kv = as_optional_float(row["kv_cache_usage_frac"])
        if queue is not None:
            current["queue"] = int(round(queue))
        if kv is not None:
            current["kv_cache_usage_frac"] = kv
        current["timestamp_s"] = float(row["timestamp_s"])


def reconstruct_decision_state(
    controls: list[dict[str, str]], telemetry: list[dict[str, str]], alpha: float,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, int]]]:
    telemetry_by_window: dict[str, list[dict[str, str]]] = {}
    for row in telemetry:
        telemetry_by_window.setdefault(row["window_id"], []).append(row)

    blocks: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in controls:
        if row["policy"] != "PASSIVE_FULL":
            continue
        match = re.match(r"p4c-r(\d+)-b(\d+)-", row["control_iteration_id"])
        if match is None:
            raise ValueError("unrecognized Phase 4C control ID: %s" % row["control_iteration_id"])
        blocks.setdefault((int(match.group(1)), int(match.group(2))), []).append(row)

    snapshots: dict[str, dict[str, dict[str, Any]]] = {}
    pre_frequencies: dict[str, dict[str, int]] = {}
    safe_high = {endpoint: 0 for endpoint in ENDPOINTS}
    for row in controls:
        for endpoint, value in json.loads(row["requested_frequencies_json"]).items():
            safe_high[endpoint] = max(safe_high[endpoint], int(value))
    for (repeat, block), rows in blocks.items():
        rows.sort(key=lambda item: float(item["control_timestamp"]))
        trace = rows[0]["trace_id"]
        warmup = "p4c-r%02d-b%02d-warm-%s" % (repeat, block, trace)
        state = {
            endpoint: {
                "ttft_ms": None, "tpot_ms": None, "queue": None,
                "kv_cache_usage_frac": None, "timestamp_s": None,
            }
            for endpoint in ENDPOINTS
        }
        update_snapshot(state, telemetry_by_window.get(warmup, ()), alpha)
        # Every passive block receives the documented all-route HIGH warmup.
        frequencies = dict(safe_high)
        for row in rows:
            iteration = row["control_iteration_id"]
            snapshots[iteration] = json.loads(json.dumps(state))
            pre_frequencies[iteration] = dict(frequencies)
            frequencies = {
                key: int(value)
                for key, value in json.loads(row["requested_frequencies_json"]).items()
            }
            update_snapshot(state, telemetry_by_window.get(iteration, ()), alpha)
    return snapshots, pre_frequencies


def request_pass(row: dict[str, str], ttft_slo: float, tpot_slo: float) -> bool:
    return float(row["ttft_ms"]) <= ttft_slo and float(row["tpot_ms"]) <= tpot_slo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ttft-slo-ms", type=float, default=550.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=65.0)
    parser.add_argument("--ewma-alpha", type=float, default=0.5)
    parser.add_argument("--source-label")
    args = parser.parse_args()

    root = args.run_dir
    results = read_csv(root / "phase4c_results.csv")
    controls = read_csv(root / "phase4c_controls.csv")
    requests = read_csv(root / "phase4c_requests.csv")
    telemetry = read_csv(root / "phase4c_endpoint_telemetry.csv")
    snapshots, pre_frequencies = reconstruct_decision_state(
        controls, telemetry, args.ewma_alpha
    )
    controls_by_id = {row["control_iteration_id"]: row for row in controls}

    requests_by_window: dict[str, list[dict[str, str]]] = {}
    for row in requests:
        requests_by_window.setdefault(row["window_id"], []).append(row)
    for rows in requests_by_window.values():
        rows.sort(key=lambda item: item["request_id"])

    window_rows: list[dict[str, Any]] = []
    result_index: dict[tuple[str, int, int, int, str], dict[str, str]] = {}
    for row in results:
        key = (
            row["trace_id"], int(row["repeat"]), int(row["state_index"]),
            int(row["window_in_state"]), row["policy"],
        )
        result_index[key] = row
        control = controls_by_id[row["control_iteration_id"]]
        pairs = json.loads(control["selected_pairs_json"])
        request_rows = requests_by_window[row["window_id"]]
        decision = snapshots.get(row["control_iteration_id"], {})
        window_rows.append({
            "trace": row["trace_id"], "repeat": row["repeat"],
            "state_index": row["state_index"], "window_in_state": row["window_in_state"],
            "segment": row["stress_workload"], "workload": row["workload"],
            "policy": row["policy"], "window_id": row["window_id"],
            "window_start_unix_s": row["timing_start_unix_s"],
            "window_end_unix_s": row["timing_end_unix_s"], "duration_s": row["duration_s"],
            "mean_ttft_ms": row["mean_ttft_ms"], "ttft_slo_ms": args.ttft_slo_ms,
            "mean_tpot_ms": row["mean_tpot_ms"], "tpot_slo_ms": args.tpot_slo_ms,
            "mean_window_slo_pass": row["slo_pass"],
            "strict_request_window_slo_pass": all(
                request_pass(item, args.ttft_slo_ms, args.tpot_slo_ms)
                for item in request_rows
            ),
            "request_slo_violation_count": sum(
                not request_pass(item, args.ttft_slo_ms, args.tpot_slo_ms)
                for item in request_rows
            ),
            "route": ";".join("%s->%s" % tuple(pair) for pair in pairs),
            "P0_frequency_level": row["P0_requested_level"],
            "P0_frequency_mhz": row["P0_requested_freq_mhz"],
            "P0_observed_frequency_mhz": row["P0_observed_freq_mhz"],
            "P1_frequency_level": row["P1_requested_level"],
            "P1_frequency_mhz": row["P1_requested_freq_mhz"],
            "P1_observed_frequency_mhz": row["P1_observed_freq_mhz"],
            "P0_decision_queue": decision.get("P0", {}).get("queue"),
            "P1_decision_queue": decision.get("P1", {}).get("queue"),
            "feedback_context": control["feedback_context_key"],
        })

    window_fields = list(window_rows[0])
    write_csv(args.output_dir / "phase4c_window_summary.csv", window_rows, window_fields)

    transition_rows: list[dict[str, Any]] = []
    target_controls = [
        row for row in controls
        if row["policy"] == "PASSIVE_FULL"
        and row["trace_id"] == "high_prefill_decode_prefill"
    ]
    target_controls.sort(key=lambda row: (int(row["repeat"]), float(row["control_timestamp"])))
    results_by_control = {row["control_iteration_id"]: row for row in results}
    for control in target_controls:
        result = results_by_control[control["control_iteration_id"]]
        if int(result["state_index"]) not in (1, 2):
            continue
        iteration = control["control_iteration_id"]
        decision = snapshots[iteration]
        requested = {key: int(value) for key, value in json.loads(control["requested_frequencies_json"]).items()}
        recommendations = {
            item["endpoint_id"]: item
            for item in json.loads(control["dvfs_recommendations_json"])
        }
        pairs = json.loads(control["selected_pairs_json"])
        selected_p = pairs[0][0]
        reqs = requests_by_window[iteration]
        item: dict[str, Any] = {
            "repeat": result["repeat"], "state_index": result["state_index"],
            "window_in_state": result["window_in_state"], "segment": result["stress_workload"],
            "decision_timestamp_unix_s": control["control_timestamp"],
            "feedback_context": control["feedback_context_key"],
            "route": "%s->%s" % tuple(pairs[0]), "selected_P": selected_p,
            "selected_P_decision_ewma_ttft_ms": decision[selected_p]["ttft_ms"],
            "selected_P_decision_queue": decision[selected_p]["queue"],
            "selected_P_decision_kv_frac": decision[selected_p]["kv_cache_usage_frac"],
            "observed_after_decision_mean_ttft_ms": result["mean_ttft_ms"],
            "observed_after_decision_ttft_violation_count": sum(
                float(row["ttft_ms"]) > args.ttft_slo_ms for row in reqs
            ),
        }
        for endpoint in ("P0", "P1"):
            item.update({
                endpoint + "_decision_ewma_ttft_ms": decision[endpoint]["ttft_ms"],
                endpoint + "_decision_queue": decision[endpoint]["queue"],
                endpoint + "_pre_action_freq_mhz": pre_frequencies[iteration][endpoint],
                endpoint + "_requested_freq_mhz": requested[endpoint],
                endpoint + "_observed_freq_mhz": result[endpoint + "_observed_freq_mhz"],
                endpoint + "_action": recommendations[endpoint]["action"],
                endpoint + "_reason": recommendations[endpoint]["reason"],
            })
        transition_rows.append(item)
    write_csv(
        args.output_dir / "phase4c_decode_to_prefill_controls.csv",
        transition_rows, list(transition_rows[0]),
    )

    paired_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    grouped_keys = sorted({key[:4] for key in result_index})
    aggregates: dict[tuple[str, int, str], dict[str, Any]] = {}
    for trace, repeat, state_index, window_in_state in grouped_keys:
        static = result_index[(trace, repeat, state_index, window_in_state, "STATIC")]
        passive = result_index[(trace, repeat, state_index, window_in_state, "PASSIVE_FULL")]
        static_requests = requests_by_window[static["window_id"]]
        passive_requests = requests_by_window[passive["window_id"]]
        aggregate_key = (trace, repeat, passive["stress_workload"])
        aggregate = aggregates.setdefault(aggregate_key, {
            "trace": trace, "repeat": repeat, "segment": passive["stress_workload"],
            "paired_requests": 0, "both_pass": 0, "static_pass_passive_fail": 0,
            "static_fail_passive_pass": 0, "both_fail": 0,
            "paired_windows": 0,
            "mean_window_both_pass": 0,
            "mean_window_static_pass_passive_fail": 0,
            "mean_window_static_fail_passive_pass": 0,
            "mean_window_both_fail": 0,
            "strict_window_both_pass": 0,
            "strict_window_static_pass_passive_fail": 0,
            "strict_window_static_fail_passive_pass": 0,
            "strict_window_both_fail": 0,
            "candidate_frequency_counts": {}, "candidate_queue_counts": {},
            "candidate_route_counts": {},
        })
        control = controls_by_id[passive["control_iteration_id"]]
        pair = json.loads(control["selected_pairs_json"])[0]
        selected_p = pair[0]
        decision_queue = snapshots[passive["control_iteration_id"]][selected_p]["queue"]
        aggregate["paired_windows"] += 1
        for prefix, static_ok, passive_ok in (
            ("mean_window", as_bool(static["slo_pass"]), as_bool(passive["slo_pass"])),
            (
                "strict_window",
                all(request_pass(row, args.ttft_slo_ms, args.tpot_slo_ms) for row in static_requests),
                all(request_pass(row, args.ttft_slo_ms, args.tpot_slo_ms) for row in passive_requests),
            ),
        ):
            category = (
                "both_pass" if static_ok and passive_ok else
                "static_pass_passive_fail" if static_ok else
                "static_fail_passive_pass" if passive_ok else "both_fail"
            )
            aggregate[prefix + "_" + category] += 1
        for request_ordinal, (static_request, passive_request) in enumerate(
            zip(static_requests, passive_requests), 1
        ):
            static_ok = request_pass(static_request, args.ttft_slo_ms, args.tpot_slo_ms)
            passive_ok = request_pass(passive_request, args.ttft_slo_ms, args.tpot_slo_ms)
            aggregate["paired_requests"] += 1
            category = (
                "both_pass" if static_ok and passive_ok else
                "static_pass_passive_fail" if static_ok else
                "static_fail_passive_pass" if passive_ok else "both_fail"
            )
            aggregate[category] += 1
            if category == "static_pass_passive_fail":
                freq_key = "%s:%s:%s" % (
                    selected_p, passive[selected_p + "_requested_level"],
                    passive[selected_p + "_requested_freq_mhz"],
                )
                queue_key = "%s:%s" % (selected_p, decision_queue)
                route_key = "%s->%s" % tuple(pair)
                for field, value in (
                    ("candidate_frequency_counts", freq_key),
                    ("candidate_queue_counts", queue_key),
                    ("candidate_route_counts", route_key),
                ):
                    aggregate[field][value] = aggregate[field].get(value, 0) + 1
                candidate_rows.append({
                    "trace": trace, "repeat": repeat,
                    "state_index": state_index, "window_in_state": window_in_state,
                    "segment": passive["stress_workload"],
                    "passive_window_id": passive["window_id"],
                    "passive_window_start_unix_s": passive["timing_start_unix_s"],
                    "request_ordinal": request_ordinal,
                    "static_ttft_ms": static_request["ttft_ms"],
                    "passive_ttft_ms": passive_request["ttft_ms"],
                    "static_tpot_ms": static_request["tpot_ms"],
                    "passive_tpot_ms": passive_request["tpot_ms"],
                    "passive_route": route_key, "passive_selected_P": selected_p,
                    "passive_P_frequency_level": passive[selected_p + "_requested_level"],
                    "passive_P_frequency_mhz": passive[selected_p + "_requested_freq_mhz"],
                    "passive_P_decision_queue": decision_queue,
                    "passive_P_decision_ewma_ttft_ms": snapshots[
                        passive["control_iteration_id"]
                    ][selected_p]["ttft_ms"],
                })
    for aggregate in aggregates.values():
        for field in (
            "candidate_frequency_counts", "candidate_queue_counts", "candidate_route_counts",
        ):
            aggregate[field + "_json"] = json.dumps(aggregate.pop(field), sort_keys=True)
        paired_rows.append(aggregate)
    paired_rows.sort(key=lambda row: (row["trace"], int(row["repeat"]), row["segment"]))
    write_csv(
        args.output_dir / "phase4c_paired_violation_breakdown.csv",
        paired_rows, list(paired_rows[0]),
    )
    write_csv(
        args.output_dir / "phase4c_paired_violation_candidates.csv",
        candidate_rows, list(candidate_rows[0]),
    )

    totals = {
        field: sum(int(row[field]) for row in paired_rows)
        for field in (
            "paired_requests", "both_pass", "static_pass_passive_fail",
            "static_fail_passive_pass", "both_fail",
        )
    }
    window_totals = {
        field: sum(int(row[field]) for row in paired_rows)
        for field in (
            "paired_windows", "mean_window_both_pass",
            "mean_window_static_pass_passive_fail",
            "mean_window_static_fail_passive_pass", "mean_window_both_fail",
            "strict_window_both_pass", "strict_window_static_pass_passive_fail",
            "strict_window_static_fail_passive_pass", "strict_window_both_fail",
        )
    }
    summary = {
        "source_run": args.source_label or root.as_posix(),
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "ewma_alpha": args.ewma_alpha,
        "window_count": len(window_rows),
        "transition_control_count": len(transition_rows),
        "paired_request_counts": totals,
        "paired_window_counts": window_totals,
        "decision_state_reconstruction": (
            "warmup plus completed prior windows only; current-window metrics are outcomes"
        ),
    }
    (args.output_dir / "phase4c_controller_diagnosis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
