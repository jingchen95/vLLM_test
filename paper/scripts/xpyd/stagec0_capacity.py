"""Stage C0 fixed-HIGH capacity calibration over the Stage-B substrate.

This runner reuses Phase 3C measurements and the Stage-B mixed trace path.  It
contains no controller, adaptive routing, or GPU-state decision.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    load_config as load_runtime_config,
)
from xpyd.workload_harness import build_trace, load_config as load_workload_config


class StageC0Error(RuntimeError):
    """A Stage C0 contract or required physical measurement failed."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def time_weighted_state(
    events: Sequence[tuple[float, float]], start: float, end: float,
) -> tuple[float, float]:
    """Return time-weighted mean and maximum of a step function."""
    if end <= start:
        return 0.0, 0.0
    current = 0.0
    cursor = start
    area = 0.0
    maximum = 0.0
    for timestamp, value in sorted(events):
        if timestamp < start:
            current = value
            continue
        if timestamp > end:
            break
        area += current * (timestamp - cursor)
        cursor = timestamp
        current = value
        maximum = max(maximum, current)
    area += current * (end - cursor)
    return area / (end - start), maximum


def load_configs(
    runtime_path: Path, workload_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_runtime_config(runtime_path)
    workload = load_workload_config(workload_path)
    calibration = workload.get("capacity_calibration")
    if not isinstance(calibration, Mapping):
        raise StageC0Error("capacity_calibration is required")
    points = calibration.get("load_points")
    if not isinstance(points, list) or len(points) < 4:
        raise StageC0Error("Stage C0 requires at least four ordered load points")
    rates = [float(item["offered_rate_rps"]) for item in points]
    if rates != sorted(rates) or len(rates) != len(set(rates)):
        raise StageC0Error("Stage C0 load rates must be unique and increasing")
    scenarios = {str(item["id"]): item for item in workload["scenarios"]}
    for point in points:
        scenario = scenarios.get(str(point["scenario"]))
        if scenario is None or len(scenario["segments"]) != 1:
            raise StageC0Error("each load point requires one stationary scenario")
        segment = scenario["segments"][0]
        if float(segment["rate_rps"]) != float(point["offered_rate_rps"]):
            raise StageC0Error("load point and scenario rates differ")
    endpoints = {str(item["endpoint_id"]): item for item in runtime["endpoints"]}
    topology = workload.get("physical_topology", {})
    expected = {
        "P0": str(topology.get("prefill_node", "neptune")),
        "P1": str(topology.get("prefill_node", "neptune")),
        "D0": str(topology.get("decode_node", "europa")),
        "D1": str(topology.get("decode_node", "europa")),
    }
    if {key: str(value["node"]) for key, value in endpoints.items()} != expected:
        raise StageC0Error("runtime endpoints do not match physical_topology")
    if int(calibration.get("max_concurrency", 0)) != 4:
        raise StageC0Error("Stage C0 starts from validated max_concurrency=4")
    if calibration.get("control") != "disabled":
        raise StageC0Error("Stage C0 forbids controller decisions")
    if calibration.get("routing") != "round-robin over four validated pairs":
        raise StageC0Error("Stage C0 requires four-pair round-robin")
    if calibration.get("frequencies") != "all validated HIGH":
        raise StageC0Error("Stage C0 requires fixed validated HIGH frequencies")
    return runtime, workload


def analyze_point(
    point: Mapping[str, Any], point_dir: Path, slo: Mapping[str, Any],
    trend: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _read_json(point_dir / "audit.json")
    requests = _read_csv(point_dir / "requests.csv")
    load_events = _read_jsonl(point_dir / "client" / "load_events.jsonl")
    routes = _read_csv(point_dir / "routes.csv")
    endpoint_load = _read_csv(point_dir / "derived" / "endpoint_load.csv")
    energy = _read_csv(point_dir / "energy_summary.csv")
    if not audit.get("valid") or not requests:
        raise StageC0Error("%s has invalid Phase 3C evidence" % point["id"])

    rate = float(point["offered_rate_rps"])
    scheduled = [float(row["scheduled_arrival_unix_s"]) for row in requests]
    admitted = [float(row["client_admitted_unix_s"]) for row in requests]
    completed = [
        admitted_time + float(row["e2e_latency_ms"]) / 1000.0
        for admitted_time, row in zip(admitted, requests)
    ]
    start = min(scheduled)
    offered_end = start + len(requests) / rate
    observation_end = max(completed)
    admission = [float(row["client_admission_delay_ms"]) for row in requests]
    relative_arrivals = [value - start for value in scheduled]
    admission_slope = linear_slope(relative_arrivals, admission)
    quartile = max(1, len(admission) // 4)
    first_q = statistics.fmean(admission[:quartile])
    last_q = statistics.fmean(admission[-quartile:])

    client_steps = [
        (float(event["timestamp_unix_s"]), float(event["client_waiting_requests"]))
        for event in load_events
    ]
    client_mean, client_max = time_weighted_state(
        client_steps, start, offered_end
    )
    close_waiting = 0.0
    for timestamp, value in sorted(client_steps):
        if timestamp <= offered_end:
            close_waiting = value

    proxy_events: list[tuple[float, float]] = []
    inflight = 0.0
    changes = []
    for row in routes:
        changes.append((float(row["route_timestamp_wall_s"]), 1.0))
        changes.append((float(row["decode_completion_wall_s"]), -1.0))
    for timestamp, delta in sorted(changes, key=lambda item: (item[0], -item[1])):
        inflight += delta
        proxy_events.append((timestamp, inflight))
    proxy_mean, proxy_max = time_weighted_state(proxy_events, start, offered_end)

    endpoint_stats: dict[str, dict[str, float | None]] = {}
    for endpoint_id in ("P0", "P1", "D0", "D1"):
        rows = [
            row for row in endpoint_load
            if row["endpoint_id"] == endpoint_id and row["valid"] == "True"
            and start <= float(row["actual_start_wall_s"]) <= observation_end
        ]
        running = [_float(row["num_requests_running"]) for row in rows]
        waiting = [_float(row["num_requests_waiting"]) for row in rows]
        running_values = [value for value in running if value is not None]
        waiting_values = [value for value in waiting if value is not None]
        endpoint_stats[endpoint_id] = {
            "running_mean": statistics.fmean(running_values) if running_values else None,
            "running_max": max(running_values) if running_values else None,
            "waiting_mean": statistics.fmean(waiting_values) if waiting_values else None,
            "waiting_max": max(waiting_values) if waiting_values else None,
        }

    ttft = [float(row["ttft_ms"]) for row in requests]
    tpot = [float(row["tpot_ms"]) for row in requests]
    violations = sum(
        left > float(slo["ttft_ms"]) or right > float(slo["tpot_ms"])
        for left, right in zip(ttft, tpot)
    )
    energy_by_scope = {
        row["scope"]: _float(row["gross_energy_j"]) for row in energy
    }
    completed_rate = len(requests) / (observation_end - start)
    throughput_ratio = completed_rate / rate
    growing = (
        admission_slope > float(trend["admission_delay_slope_ms_per_s"])
        and last_q - first_q > float(trend["quartile_delay_increase_ms"])
        and close_waiting >= float(trend["minimum_close_waiting"])
    )
    little_queue = (
        client_mean <= float(trend["low_mean_client_waiting"])
        and (percentile(admission, 0.95) or 0.0)
        <= float(trend["low_p95_admission_delay_ms"])
    )
    sustainable = not growing and throughput_ratio >= float(
        trend["minimum_sustainable_throughput_ratio"]
    )
    backlog = "grows" if growing else ("drains" if little_queue else "stabilizes")

    return {
        "load_point": str(point["id"]),
        "offered_rate_rps": rate,
        "completed_rate_rps": completed_rate,
        "throughput_ratio": throughput_ratio,
        "request_count": len(requests),
        "observation_duration_s": observation_end - start,
        "admission_delay_mean_ms": statistics.fmean(admission),
        "admission_delay_p95_ms": percentile(admission, 0.95),
        "admission_delay_p99_ms": percentile(admission, 0.99),
        "admission_delay_slope_ms_per_s": admission_slope,
        "first_quartile_admission_delay_mean_ms": first_q,
        "last_quartile_admission_delay_mean_ms": last_q,
        "client_waiting_mean": client_mean,
        "client_waiting_max": client_max,
        "client_waiting_at_offered_close": close_waiting,
        "proxy_inflight_mean": proxy_mean,
        "proxy_inflight_max": proxy_max,
        "ttft_mean_ms": statistics.fmean(ttft),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_mean_ms": statistics.fmean(tpot),
        "tpot_p95_ms": percentile(tpot, 0.95),
        "tpot_p99_ms": percentile(tpot, 0.99),
        "slo_violation_count": violations,
        "slo_violation_rate": violations / len(requests),
        "total_gross_energy_j": energy_by_scope.get("all_GPU_boards"),
        "P0_energy_j": energy_by_scope.get("P0"),
        "P1_energy_j": energy_by_scope.get("P1"),
        "D0_energy_j": energy_by_scope.get("D0"),
        "D1_energy_j": energy_by_scope.get("D1"),
        "joules_per_request": (
            energy_by_scope["all_GPU_boards"] / len(requests)
            if energy_by_scope.get("all_GPU_boards") is not None else None
        ),
        "endpoint_load": endpoint_stats,
        "backlog_behavior": backlog,
        "backlog_growing": growing,
        "little_or_no_sustained_queue": little_queue,
        "sustainable": sustainable,
        "measurement_valid": True,
        "classification": "UNCLASSIFIED",
    }


def classify_points(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Choose capacity regimes independently of the later SLO admission gate."""
    for row in rows:
        if not row["sustainable"]:
            row["classification"] = "OVERLOAD"
    sustainable = [row for row in rows if row["sustainable"]]
    low_candidates = [
        row for row in sustainable if row["little_or_no_sustained_queue"]
    ]
    low = min(low_candidates, key=lambda row: row["offered_rate_rps"]) if low_candidates else None
    remaining = [row for row in sustainable if low is None or row is not low]
    high = max(remaining, key=lambda row: row["offered_rate_rps"]) if remaining else None
    between = [
        row for row in remaining if high is not None and row is not high
        and (low is None or row["offered_rate_rps"] > low["offered_rate_rps"])
    ]
    target = (
        math.sqrt(low["offered_rate_rps"] * high["offered_rate_rps"])
        if low is not None and high is not None else None
    )
    medium = min(
        between, key=lambda row: abs(row["offered_rate_rps"] - target)
    ) if between and target is not None else None
    for row in sustainable:
        if row is low:
            row["classification"] = "LOW"
        elif row is high:
            row["classification"] = "HIGH-SUSTAINABLE"
        else:
            row["classification"] = "MEDIUM"
    for row in rows:
        if row["classification"] == "UNCLASSIFIED":
            row["classification"] = "OVERLOAD"
    return {"LOW": low, "MEDIUM": medium, "HIGH-SUSTAINABLE": high}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened = []
    for row in rows:
        item = dict(row)
        item["endpoint_load_json"] = json.dumps(item.pop("endpoint_load"), sort_keys=True)
        flattened.append(item)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


class StageC0Harness:
    def __init__(
        self, runtime: Mapping[str, Any], workload: Mapping[str, Any],
        run_id: str | None = None,
    ) -> None:
        self.runtime = copy.deepcopy(dict(runtime))
        self.workload = copy.deepcopy(dict(workload))
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(os.path.expandvars(str(workload["output_root"]))) / self.run_id

    def run(self) -> Path:
        if self.run_dir.exists():
            raise StageC0Error("Stage C0 run directory already exists: %s" % self.run_dir)
        self.run_dir.mkdir(parents=True)
        calibration = self.workload["capacity_calibration"]
        points = [dict(item) for item in calibration["load_points"]]
        _write_json(self.run_dir / "stagec0_plan.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "load_points": points,
            "max_concurrency": 4,
            "frequencies": "all validated HIGH",
            "routing": "round-robin over four validated pairs",
            "controller": "disabled",
            "slo": calibration["slo"],
        })
        rows = []
        for point in points:
            trace_rows = build_trace(
                self.workload, str(point["scenario"]),
                "%s-%s" % (self.run_id, point["id"]),
            )
            config = copy.deepcopy(self.runtime)
            config["output_root"] = self.run_dir.as_posix()
            config["client"]["max_concurrency"] = 4
            config["coverage_policy"] = {
                "required_endpoint_ids": ["P0", "P1", "D0", "D1"],
                "required_pairs": [
                    ["P0", "D0"], ["P0", "D1"],
                    ["P1", "D0"], ["P1", "D1"],
                ],
            }
            Phase3CSubstrateHarness(
                config, run_id=str(point["id"]), trace_rows=trace_rows
            ).run()
            rows.append(analyze_point(
                point, self.run_dir / str(point["id"]),
                calibration["slo"], calibration["trend_thresholds"],
            ))
        write_reports(self.run_dir, rows, calibration)
        print(self.run_dir.as_posix())
        return self.run_dir


def write_reports(
    run_dir: Path, rows: list[dict[str, Any]], calibration: Mapping[str, Any],
) -> None:
        chosen = classify_points(rows)
        maximum_slo_rate = float(
            calibration["maximum_selected_slo_violation_rate"]
        )
        regimes_complete = all(chosen.values())
        selected_slo_safe = regimes_complete and all(
            value is not None
            and value["slo_violation_rate"] <= maximum_slo_rate
            for value in chosen.values()
        )
        ready = (
            regimes_complete and selected_slo_safe
            and all(row["measurement_valid"] for row in rows)
        )
        blockers = []
        if not regimes_complete:
            blockers.append("LOW/MEDIUM/HIGH-SUSTAINABLE capacity regimes are incomplete")
        if regimes_complete and not selected_slo_safe:
            blockers.append(
                "fixed-HIGH baseline exceeds the %.1f%% selected-regime SLO-violation gate"
                % (100.0 * maximum_slo_rate)
            )
        _write_csv(run_dir / "stagec0_measurements.csv", rows)
        summary = {
            "schema_version": 1,
            "phase": "Stage_C0_capacity_load_calibration",
            "valid": all(row["measurement_valid"] for row in rows),
            "capacity_regimes_complete": bool(regimes_complete),
            "selected_regimes_slo_safe": bool(selected_slo_safe),
            "ready_for_stage_c1": bool(ready),
            "stage_c1_blockers": blockers,
            "chosen_load_points_rps": {
                key: value["offered_rate_rps"] if value else None
                for key, value in chosen.items()
            },
            "load_points": rows,
            "claim_boundary": (
                "fixed-HIGH capacity/load calibration only; no controller, "
                "DVFS-policy, or intelligent-routing result"
            ),
            "next_action": (
                "start Stage C1 with the selected LOW/MEDIUM/HIGH-SUSTAINABLE rates"
                if ready else "resolve the fixed-HIGH SLO baseline before Stage C1"
            ),
        }
        _write_json(run_dir / "stagec0_summary.json", summary)
        lines = [
            "# XPYD Stage C0 capacity/load calibration", "",
            "All endpoints were fixed at validated HIGH; routing was four-pair round-robin; no controller was active.", "",
            "| Point | Offered | Completed | Admission p95/p99 (ms) | Client waiting mean/max | TTFT p95 | TPOT p95 | SLO violations | Backlog | Class |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in rows:
            lines.append(
                "| {load_point} | {offered_rate_rps:.3f} | {completed_rate_rps:.3f} | "
                "{admission_delay_p95_ms:.1f}/{admission_delay_p99_ms:.1f} | "
                "{client_waiting_mean:.2f}/{client_waiting_max:.0f} | "
                "{ttft_p95_ms:.1f} | {tpot_p95_ms:.1f} | {slo_violation_rate:.1%} | "
                "{backlog_behavior} | {classification} |".format(**row)
            )
        lines.extend([
            "", "Chosen rates: `%s`" % json.dumps(
                summary["chosen_load_points_rps"], sort_keys=True
            ),
            "", "Stage C1 recommendation: **%s**" % (
                "READY" if ready else "NOT READY"
            ), "", summary["claim_boundary"], "",
        ])
        (run_dir / "stagec0_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def reanalyze_existing(run_dir: Path, workload: Mapping[str, Any]) -> None:
    calibration = workload["capacity_calibration"]
    rows = [
        analyze_point(
            point, run_dir / str(point["id"]), calibration["slo"],
            calibration["trend_thresholds"],
        )
        for point in calibration["load_points"]
    ]
    write_reports(run_dir, rows, calibration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config")
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--reanalyze-run-dir")
    args = parser.parse_args()
    if args.reanalyze_run_dir:
        workload = load_workload_config(Path(args.workload_config))
        reanalyze_existing(Path(args.reanalyze_run_dir), workload)
        return 0
    if not args.runtime_config:
        parser.error("--runtime-config is required unless --reanalyze-run-dir is used")
    runtime, workload = load_configs(
        Path(args.runtime_config), Path(args.workload_config)
    )
    StageC0Harness(runtime, workload, args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
