"""Deterministic (or seeded-Poisson) mixed-workload traces for the post-Phase-4 architecture audit.

This module only defines offered load. It does not select routes, change GPU
state, or make controller decisions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence


REQUEST_CLASSES = ("SS", "SL", "LS", "LL")
ARRIVAL_PROCESSES = ("uniform", "poisson")


class WorkloadHarnessError(ValueError):
    """The mixed-workload harness configuration is invalid."""


@dataclass(frozen=True)
class RequestBuckets:
    short_input_max_tokens: int
    short_output_max_tokens: int

    def __post_init__(self) -> None:
        if self.short_input_max_tokens <= 0 or self.short_output_max_tokens <= 0:
            raise WorkloadHarnessError("request bucket thresholds must be positive")

    def classify(self, input_len: int, output_len: int) -> str:
        if input_len <= 0 or output_len <= 0:
            raise WorkloadHarnessError("request lengths must be positive")
        left = "S" if input_len <= self.short_input_max_tokens else "L"
        right = "S" if output_len <= self.short_output_max_tokens else "L"
        return left + right


def _safe_id(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text) is None:
        raise WorkloadHarnessError("%s must be non-empty and filesystem-safe" % name)
    return text


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise WorkloadHarnessError("schema_version must be 1")
    buckets = RequestBuckets(**dict(config["bucket_definition"]))
    shapes = config.get("shapes")
    if not isinstance(shapes, Mapping) or set(shapes) != set(REQUEST_CLASSES):
        raise WorkloadHarnessError("shapes must define exactly SS, SL, LS, and LL")
    for request_class in REQUEST_CLASSES:
        shape = shapes[request_class]
        observed = buckets.classify(
            int(shape["input_len"]), int(shape["output_len"])
        )
        if observed != request_class:
            raise WorkloadHarnessError(
                "shape %s classifies as %s" % (request_class, observed)
            )
    window_s = float(config.get("control_window_s", 0.0))
    if not math.isfinite(window_s) or window_s <= 0:
        raise WorkloadHarnessError("control_window_s must be positive")

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise WorkloadHarnessError("at least one scenario is required")
    scenario_ids = set()
    has_mixed_segment = False
    for scenario in scenarios:
        scenario_id = _safe_id(scenario.get("id"), "scenario id")
        if scenario_id in scenario_ids:
            raise WorkloadHarnessError("scenario IDs must be unique")
        scenario_ids.add(scenario_id)
        segments = scenario.get("segments")
        if not isinstance(segments, list) or not segments:
            raise WorkloadHarnessError("each scenario requires segments")
        segment_ids = set()
        for segment in segments:
            segment_id = _safe_id(segment.get("id"), "segment id")
            if segment_id in segment_ids:
                raise WorkloadHarnessError("segment IDs must be unique per scenario")
            segment_ids.add(segment_id)
            request_count = int(segment.get("request_count", 0))
            rate_rps = float(segment.get("rate_rps", 0.0))
            if request_count <= 0 or not math.isfinite(rate_rps) or rate_rps <= 0:
                raise WorkloadHarnessError(
                    "segment request_count and rate_rps must be positive"
                )
            weights = segment.get("class_weights")
            if not isinstance(weights, Mapping) or set(weights) != set(REQUEST_CLASSES):
                raise WorkloadHarnessError(
                    "class_weights must define exactly SS, SL, LS, and LL"
                )
            numeric = [float(weights[key]) for key in REQUEST_CLASSES]
            if any(not math.isfinite(value) or value < 0 for value in numeric):
                raise WorkloadHarnessError("class weights must be finite and nonnegative")
            if sum(numeric) <= 0:
                raise WorkloadHarnessError("class weights must have positive mass")
            has_mixed_segment = has_mixed_segment or sum(value > 0 for value in numeric) > 1
            process = segment.get("arrival_process", "uniform")
            if process not in ARRIVAL_PROCESSES:
                raise WorkloadHarnessError("arrival_process must be uniform or poisson")
            if process == "poisson" and not isinstance(segment.get("seed"), int):
                raise WorkloadHarnessError("poisson segments require an integer seed")
    if not has_mixed_segment:
        raise WorkloadHarnessError("at least one segment must mix request classes")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _apportion(count: int, weights: Mapping[str, Any]) -> dict[str, int]:
    total = sum(float(weights[key]) for key in REQUEST_CLASSES)
    raw = {key: count * float(weights[key]) / total for key in REQUEST_CLASSES}
    allocated = {key: math.floor(raw[key]) for key in REQUEST_CLASSES}
    remainder = count - sum(allocated.values())
    order = sorted(
        REQUEST_CLASSES,
        key=lambda key: (-(raw[key] - allocated[key]), REQUEST_CLASSES.index(key)),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated


def _interleave(counts: Mapping[str, int]) -> list[str]:
    total = sum(counts.values())
    used = {key: 0 for key in REQUEST_CLASSES}
    sequence = []
    for index in range(total):
        eligible = [key for key in REQUEST_CLASSES if used[key] < counts[key]]
        selected = max(
            eligible,
            key=lambda key: (
                ((index + 1) * counts[key] / total) - used[key],
                -REQUEST_CLASSES.index(key),
            ),
        )
        used[selected] += 1
        sequence.append(selected)
    return sequence


def build_trace(
    config: Mapping[str, Any], scenario_id: str, namespace: str | None = None
) -> list[dict[str, Any]]:
    validate_config(config)
    scenario_id = _safe_id(scenario_id, "scenario id")
    namespace = _safe_id(namespace or scenario_id, "request ID namespace")
    try:
        scenario = next(
            item for item in config["scenarios"] if str(item["id"]) == scenario_id
        )
    except StopIteration as exc:
        raise WorkloadHarnessError("unknown scenario: %s" % scenario_id) from exc

    rows: list[dict[str, Any]] = []
    segment_start_s = 0.0
    for segment in scenario["segments"]:
        count = int(segment["request_count"])
        rate = float(segment["rate_rps"])
        counts = _apportion(count, segment["class_weights"])
        # Uniform arrivals keep the original deterministic grid.  Poisson
        # arrivals draw seeded exponential gaps; the class mix stays exact.
        poisson = segment.get("arrival_process", "uniform") == "poisson"
        rng = random.Random(int(segment["seed"])) if poisson else None
        offset_s = 0.0
        for offset, request_class in enumerate(_interleave(counts)):
            shape = config["shapes"][request_class]
            if rng is not None and offset > 0:
                offset_s += rng.expovariate(rate)
            rows.append({
                "request_id": "%s-%04d" % (namespace, len(rows)),
                "arrival_time_s": segment_start_s + (offset_s if poisson else offset / rate),
                "input_len": int(shape["input_len"]),
                "output_len": int(shape["output_len"]),
                "workload_class": request_class,
                "segment": str(segment["id"]),
            })
        # Start the next segment on the next point in the offered-arrival grid.
        segment_start_s += (offset_s + 1.0 / rate) if poisson else count / rate
    return rows


def offered_window_summaries(
    rows: Sequence[Mapping[str, Any]], window_s: float
) -> list[dict[str, Any]]:
    if not math.isfinite(window_s) or window_s <= 0:
        raise WorkloadHarnessError("window_s must be positive")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        index = int(math.floor(float(row["arrival_time_s"]) / window_s))
        grouped.setdefault(index, []).append(row)
    summaries = []
    for index, window_rows in sorted(grouped.items()):
        counts = {
            key: sum(row["workload_class"] == key for row in window_rows)
            for key in REQUEST_CLASSES
        }
        total = len(window_rows)
        summaries.append({
            "window_index": index,
            "start_s": index * window_s,
            "end_s": (index + 1) * window_s,
            "logical_request_count": total,
            "input_tokens": sum(int(row["input_len"]) for row in window_rows),
            "requested_output_tokens": sum(
                int(row["output_len"]) for row in window_rows
            ),
            "class_counts": counts,
            "class_fractions": {
                key: counts[key] / total for key in REQUEST_CLASSES
            },
        })
    return summaries


def write_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "request_id", "arrival_time_s", "input_len", "output_len",
            "workload_class", "segment",
        ))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--request-id-namespace")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    rows = build_trace(config, args.scenario, args.request_id_namespace)
    write_trace(Path(args.output_csv), rows)
    summary = {
        "schema_version": 1,
        "scenario": args.scenario,
        "request_count": len(rows),
        "control_window_s": float(config["control_window_s"]),
        "offered_windows": offered_window_summaries(
            rows, float(config["control_window_s"])
        ),
        "claim_boundary": "offered-load definition only; no measured GPU result",
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
