"""Fixed-frequency XpYd characterization at sustainable hardware operating points.

The GPU clock actuation is deliberately outside this module, in the guarded
Slurm lifecycle.  This process only orchestrates the accepted Phase 3A client
and Prometheus audits plus the read-only Phase 3B NVML monitors.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.nvml_readonly import structured_error
from xpyd.phase3a_observability import Phase3AHarness, load_config as load_phase3a_config
from xpyd.phase3b_energy import (
    Phase3BEnergyError,
    WindowSpec,
    _git_commit,
    _launch_monitor,
    _read_jsonl,
    _stop_monitors,
    _wait_capabilities,
    _write_json,
    add_idle_adjustment,
    aggregate_endpoint_workload,
    summarize_window,
)


class CharacterizationError(RuntimeError):
    """Configuration, runtime, or audit failure in characterization."""


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CharacterizationError("characterization config must be a JSON object")
    required = {
        "output_root", "phase3a_config", "sampling", "windows", "endpoints",
        "fixed_clocks", "workloads", "experiment",
    }
    missing = required.difference(value)
    if missing:
        raise CharacterizationError("characterization config missing fields: %s" % sorted(missing))
    endpoints = value["endpoints"]
    if {item.get("endpoint_id") for item in endpoints} != {"P0", "D0"}:
        raise CharacterizationError("characterization requires exactly P0 and D0")
    workload_ids = [str(item.get("id", "")) for item in value["workloads"]]
    if not workload_ids or any(not item for item in workload_ids):
        raise CharacterizationError("each workload requires a non-empty id")
    if len(set(workload_ids)) != len(workload_ids):
        raise CharacterizationError("workload ids must be unique")
    for item in value["workloads"]:
        if int(item["input_len"]) <= 0 or int(item["output_len"]) <= 0:
            raise CharacterizationError("workload token lengths must be positive")
    for endpoint_id in ("P0", "D0"):
        clock = value["fixed_clocks"].get(endpoint_id, {})
        if int(clock.get("graphics_mhz", 0)) <= 0 or int(clock.get("memory_mhz", 0)) <= 0:
            raise CharacterizationError("fixed clocks missing for %s" % endpoint_id)
    return value


def balanced_schedule(workloads: Sequence[Mapping[str, Any]], repeats: int) -> list[dict[str, Any]]:
    """Return deterministic cyclic blocks so workload order changes per repeat."""
    if repeats <= 0:
        raise CharacterizationError("repeat count must be positive")
    items = [dict(item) for item in workloads]
    schedule = []
    for repeat_index in range(repeats):
        offset = repeat_index % len(items)
        rotated = items[offset:] + items[:offset]
        for order_index, workload in enumerate(rotated):
            schedule.append({
                "repeat_index": repeat_index + 1,
                "order_index": order_index + 1,
                "workload": workload,
            })
    return schedule


def _clock_audit(
    records: Sequence[Mapping[str, Any]],
    start_wall_s: float,
    end_wall_s: float,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [
        item for item in records
        if item.get("status") == "success"
        and isinstance(item.get("actual_start_wall_s"), (int, float))
        and start_wall_s <= float(item["actual_start_wall_s"]) <= end_wall_s
    ]
    graphics = [float(item["graphics_clock_mhz"]) for item in selected if item.get("graphics_clock_mhz") is not None]
    memory = [float(item["memory_clock_mhz"]) for item in selected if item.get("memory_clock_mhz") is not None]
    target_graphics = float(target["graphics_mhz"])
    target_memory = float(target["memory_mhz"])

    def describe(values: Sequence[float], expected: float) -> dict[str, Any]:
        matches = sum(math.isclose(value, expected, abs_tol=0.5) for value in values)
        return {
            "target_mhz": expected,
            "sample_count": len(values),
            "minimum_mhz": min(values) if values else None,
            "maximum_mhz": max(values) if values else None,
            "mean_mhz": statistics.fmean(values) if values else None,
            "target_match_count": matches,
            "target_match_fraction": matches / len(values) if values else 0.0,
        }

    graphics_result = describe(graphics, target_graphics)
    memory_result = describe(memory, target_memory)
    required_fraction = float(target.get("minimum_workload_match_fraction", 0.95))
    valid = (
        bool(selected)
        and graphics_result["target_match_fraction"] >= required_fraction
        and memory_result["target_match_fraction"] >= required_fraction
    )
    return {
        "valid": valid,
        "required_match_fraction": required_fraction,
        "energy_sample_count": len(selected),
        "graphics": graphics_result,
        "memory": memory_result,
    }


def _token_audit(load_record: Mapping[str, Any], client: Mapping[str, Any]) -> dict[str, Any]:
    requests = int(client["successful_requests"])
    input_tokens = int(client["input_tokens_total"])
    output_tokens = int(client["output_tokens_total"])
    expected = {
        "P0": {
            "delta_completed_requests": requests,
            "delta_prompt_tokens": input_tokens + requests,
            "delta_generation_tokens": requests,
        },
        "D0": {
            "delta_completed_requests": requests,
            "delta_prompt_tokens": input_tokens + requests,
            "delta_generation_tokens": output_tokens,
        },
    }
    checks = []
    violations = []
    for endpoint_id, fields in expected.items():
        window = load_record["endpoints"][endpoint_id]["window"]
        for field, expected_value in fields.items():
            observed = window.get(field)
            valid = bool(window.get("valid")) and observed == expected_value
            checks.append({
                "endpoint": endpoint_id,
                "field": field,
                "observed": observed,
                "expected": expected_value,
                "valid": valid,
            })
            if not valid:
                violations.append("%s %s=%r expected %r" % (endpoint_id, field, observed, expected_value))
    return {"valid": not violations, "checks": checks, "violations": violations}


def _queue_mean_ms(endpoint_record: Mapping[str, Any]) -> Optional[float]:
    histogram = endpoint_record.get("histogram_deltas", {}).get(
        "vllm:request_queue_time_seconds"
    )
    if not histogram or not histogram.get("count"):
        return None
    return 1000.0 * float(histogram["sum_seconds"]) / float(histogram["count"])


def _bootstrap_median_ci(values: Sequence[float], seed: str) -> Optional[list[float]]:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(5000)
    )
    return [medians[int(0.025 * (len(medians) - 1))], medians[int(0.975 * (len(medians) - 1))]]


def _metric_summary(values: Sequence[float], seed: str) -> dict[str, Any]:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "std": statistics.stdev(clean) if len(clean) >= 2 else None,
        "median": statistics.median(clean) if clean else None,
        "minimum": min(clean) if clean else None,
        "maximum": max(clean) if clean else None,
        "bootstrap_median_ci95": _bootstrap_median_ci(clean, seed),
    }


def _sample_interval_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the observed endpoint-local NVML sampling cadence."""
    times = sorted(
        float(item["actual_start_local_monotonic_s"])
        for item in records
        if item.get("status") == "success"
        and isinstance(item.get("actual_start_local_monotonic_s"), (int, float))
    )
    intervals = [right - left for left, right in zip(times, times[1:]) if right > left]
    if not intervals:
        return {
            "n_intervals": 0,
            "median_s": None,
            "mean_s": None,
            "minimum_s": None,
            "maximum_s": None,
            "p95_s": None,
        }
    ordered = sorted(intervals)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "n_intervals": len(intervals),
        "median_s": statistics.median(intervals),
        "mean_s": statistics.fmean(intervals),
        "minimum_s": min(intervals),
        "maximum_s": max(intervals),
        "p95_s": ordered[p95_index],
    }


def _interpolated_counter_value(
    records: Sequence[Mapping[str, Any]], wall_time_s: float
) -> Optional[float]:
    """Linearly interpolate the cumulative energy counter at a wall time."""
    points = sorted(
        (
            float(item["actual_start_wall_s"]),
            float(item["total_energy_mj"]),
        )
        for item in records
        if item.get("status") == "success"
        and isinstance(item.get("actual_start_wall_s"), (int, float))
        and isinstance(item.get("total_energy_mj"), (int, float))
    )
    if not points:
        return None
    times = [item[0] for item in points]
    index = bisect.bisect_left(times, wall_time_s)
    if index < len(points) and math.isclose(points[index][0], wall_time_s, abs_tol=1e-9):
        return points[index][1]
    if index == 0 or index >= len(points):
        return None
    left_time, left_value = points[index - 1]
    right_time, right_value = points[index]
    if right_time <= left_time:
        return None
    fraction = (wall_time_s - left_time) / (right_time - left_time)
    return left_value + fraction * (right_value - left_value)


def _prefill_request_energy(
    p0_records: Sequence[Mapping[str, Any]],
    start_wall_s: float,
    end_wall_s: float,
) -> dict[str, Any]:
    """Estimate one phase-aligned P0 energy interval without clamping."""
    successful = [
        item for item in p0_records
        if item.get("status") == "success"
        and isinstance(item.get("actual_start_wall_s"), (int, float))
        and isinstance(item.get("actual_finish_wall_s"), (int, float))
    ]
    overlap = [
        item for item in successful
        if float(item["actual_start_wall_s"]) <= end_wall_s
        and float(item["actual_finish_wall_s"]) >= start_wall_s
    ]
    start_counter = _interpolated_counter_value(p0_records, start_wall_s)
    end_counter = _interpolated_counter_value(p0_records, end_wall_s)
    gross = (
        (end_counter - start_counter) / 1000.0
        if start_counter is not None and end_counter is not None
        else None
    )
    return {
        "prefill_start_wall_s": start_wall_s,
        "prefill_complete_wall_s": end_wall_s,
        "prefill_duration_ms": max(0.0, end_wall_s - start_wall_s) * 1000.0,
        "p0_samples_during_prefill": len(overlap),
        "p0_counter_start_mj": start_counter,
        "p0_counter_complete_mj": end_counter,
        "prefill_gross_energy_j": gross,
        "energy_method": "linear_interpolation_of_cumulative_nvml_counter",
        "sampling_support_sufficient": len(overlap) >= 3,
        "valid": gross is not None,
    }


def _prefill_identifiability(
    phase3a_dir: Path,
    probe_id: str,
    expected_requests: int,
    p0_records: Sequence[Mapping[str, Any]],
    idle: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract phase-aligned prefill intervals and P0 energy candidates."""
    diagnostics_path = phase3a_dir / "derived" / "proxy_diagnostics.jsonl"
    reasons: list[str] = []
    diagnostics = _read_jsonl(diagnostics_path) if diagnostics_path.exists() else []
    prefix = "%s-" % probe_id
    requests = sorted(
        [
            item for item in diagnostics
            if str(item.get("logical_request_id", "")).startswith(prefix)
            and item.get("outcome") == "completed"
        ],
        key=lambda item: float(
            (item.get("timestamps_wall_s") or {}).get("prefill_started", float("inf"))
        ),
    )
    if len(requests) != expected_requests:
        reasons.append(
            "prefill diagnostic request count %d != expected %d"
            % (len(requests), expected_requests)
        )

    request_windows = []
    for diagnostic in requests:
        stamps = diagnostic.get("timestamps_wall_s") or {}
        start = stamps.get("prefill_started")
        end = stamps.get("prefill_completed")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            reasons.append("prefill wall timestamps unavailable")
            continue
        window = _prefill_request_energy(p0_records, float(start), float(end))
        window["logical_request_id"] = diagnostic.get("logical_request_id")
        window["reported_prefill_duration_ms"] = (
            diagnostic.get("durations_ms") or {}
        ).get("prefill")
        request_windows.append(window)

    gross_values = [
        item["prefill_gross_energy_j"]
        for item in request_windows
        if isinstance(item.get("prefill_gross_energy_j"), (int, float))
    ]
    duration_s = sum(item["prefill_duration_ms"] for item in request_windows) / 1000.0
    idle_power = idle.get("mean_power_w") if idle.get("valid") else None
    if not idle.get("valid"):
        reasons.append("immediate idle window invalid")
    if len(gross_values) != len(request_windows):
        reasons.append("one or more prefill counter intervals cannot be interpolated")
    gross = sum(gross_values) if len(gross_values) == len(request_windows) else None
    incremental = (
        gross - float(idle_power) * duration_s
        if gross is not None and isinstance(idle_power, (int, float))
        else None
    )
    sample_counts = [item["p0_samples_during_prefill"] for item in request_windows]
    return {
        "valid": not reasons and len(request_windows) == expected_requests,
        "reasons": sorted(set(reasons)),
        "logical_request_count": len(request_windows),
        "requests": request_windows,
        "p0_idle_power_w": idle_power,
        "prefill_duration_s_total": duration_s,
        "prefill_duration_ms_mean": (
            statistics.fmean(item["prefill_duration_ms"] for item in request_windows)
            if request_windows else None
        ),
        "p0_samples_during_prefill_total": sum(sample_counts),
        "p0_samples_during_prefill_mean": (
            statistics.fmean(sample_counts) if sample_counts else None
        ),
        "p0_samples_during_prefill_min": min(sample_counts) if sample_counts else None,
        "p0_samples_during_prefill_max": max(sample_counts) if sample_counts else None,
        "p_prefill_gross_energy_j": gross,
        "p_prefill_incremental_energy_j": incremental,
        "positive_incremental": incremental > 0 if incremental is not None else None,
        "energy_method": "linear_interpolation_of_cumulative_nvml_counter",
    }


def _write_report(run_dir: Path, audit: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        return ("%%.%df" % digits) % float(value) if isinstance(value, (int, float)) else "unavailable"

    lines = [
        "# XpYd Phase 3B fixed-frequency characterization",
        "",
        "Verdict: **%s**" % ("PASS" if audit["valid"] else "FAIL"),
        "",
        "Fixed-frequency characterization using sustainable hardware operating points. GPU-board telemetry only; measurement-only/model-free with no adaptive routing, training, or DVFS policy.",
        "",
        "| Workload | Valid repeats | P energy median (J) | D energy median (J) | P share median | TTFT median (ms) | TPOT median (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for workload_id, item in summary["workloads"].items():
        metrics = item["metrics"]
        lines.append("| %s | %d/%d | %s | %s | %s | %s | %s |" % (
            workload_id,
            item["valid_repeats"], item["expected_repeats"],
            fmt(metrics["p_energy_j"]["median"]),
            fmt(metrics["d_energy_j"]["median"]),
            fmt(metrics["p_energy_share"]["median"], 4),
            fmt(metrics["mean_ttft_ms"]["median"]),
            fmt(metrics["mean_tpot_ms"]["median"]),
        ))
    lines.extend([
        "",
        "Repeat-level statistics are the inference unit; 200 ms telemetry samples are not treated as independent repeats.",
        "Gross GPU-board energy is primary. Idle-adjusted values remain separately labelled estimates.",
        "CPU/node, NIC/network, KV-transfer-specific, cooling, and facility energy are unobserved.",
    ])
    (run_dir / "characterization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_prefill_identifiability_outputs(
    run_dir: Path,
    config: Mapping[str, Any],
    repeats_output: Sequence[Mapping[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Write the compact Phase 3B.1 analysis products."""
    p0_idle = [
        float(item["prefill_identifiability"]["p0_idle_power_w"])
        for item in repeats_output
        if item.get("prefill_identifiability", {}).get("p0_idle_power_w") is not None
    ]
    d0_idle = [
        float(item["endpoints"]["D0"]["idle"]["mean_power_w"])
        for item in repeats_output
        if item["endpoints"]["D0"]["idle"].get("valid")
        and item["endpoints"]["D0"]["idle"].get("mean_power_w") is not None
    ]
    p0_idle_mean = statistics.fmean(p0_idle) if p0_idle else None
    p0_idle_std = statistics.stdev(p0_idle) if len(p0_idle) >= 2 else None
    sampling = {
        endpoint: _sample_interval_summary(records[endpoint])
        for endpoint in ("P0", "D0")
    }
    workloads = {}
    csv_rows = []
    for workload in config["workloads"]:
        workload_id = str(workload["id"])
        selected = [
            item["prefill_identifiability"]
            for item in repeats_output
            if item["workload_id"] == workload_id
            and item.get("prefill_identifiability", {}).get("valid")
        ]
        durations = [item["prefill_duration_ms_mean"] for item in selected]
        samples = [item["p0_samples_during_prefill_mean"] for item in selected]
        gross = [item["p_prefill_gross_energy_j"] for item in selected]
        incremental = [item["p_prefill_incremental_energy_j"] for item in selected]
        positive_fraction = (
            sum(item["positive_incremental"] is True for item in selected) / len(selected)
            if selected else None
        )
        mean_duration_s = (
            statistics.fmean(item["prefill_duration_s_total"] for item in selected)
            if selected else None
        )
        baseline_uncertainty = (
            p0_idle_std * mean_duration_s
            if p0_idle_std is not None and mean_duration_s is not None
            else None
        )
        mean_incremental = statistics.fmean(incremental) if incremental else None
        mean_samples = statistics.fmean(samples) if samples else None
        if not selected:
            classification = "UNDER-RESOLVED"
            classification_reason = "no valid phase-aligned prefill intervals"
        elif (
            mean_samples is not None and mean_samples < 3
        ) or (
            baseline_uncertainty is not None
            and mean_incremental is not None
            and abs(mean_incremental) <= baseline_uncertainty
        ) or (positive_fraction is not None and positive_fraction < 0.8):
            classification = "UNDER-RESOLVED"
            classification_reason = (
                "fewer than 3 overlapping P0 samples per request, or signal is within "
                "repeated-idle baseline uncertainty"
            )
        elif (
            mean_samples is not None and mean_samples >= 3
            and positive_fraction is not None and positive_fraction >= 0.8
            and baseline_uncertainty is not None
            and mean_incremental is not None
            and mean_incremental > 2.0 * baseline_uncertainty
        ):
            classification = "MEASURABLE"
            classification_reason = "positive signal exceeds twice baseline uncertainty"
        else:
            classification = "MARGINAL"
            classification_reason = "signal is visible but repeat support or variance is insufficient"
        workloads[workload_id] = {
            "workload": workload,
            "valid_repeats": len(selected),
            "prefill_duration_ms": _metric_summary(durations, workload_id + ":duration"),
            "p0_samples_during_prefill": _metric_summary(samples, workload_id + ":samples"),
            "p_prefill_gross_energy_j": _metric_summary(gross, workload_id + ":gross"),
            "p_prefill_incremental_energy_j": _metric_summary(incremental, workload_id + ":incremental"),
            "positive_incremental_fraction": positive_fraction,
            "baseline_uncertainty_j": baseline_uncertainty,
            "classification": classification,
            "classification_reason": classification_reason,
        }
        for item in selected:
            csv_rows.append({
                "workload_id": workload_id,
                "input_len": workload["input_len"],
                "output_len": workload["output_len"],
                "prefill_duration_ms_mean": item["prefill_duration_ms_mean"],
                "prefill_duration_s_total": item["prefill_duration_s_total"],
                "p0_idle_power_w": item["p0_idle_power_w"],
                "p0_samples_during_prefill_total": item["p0_samples_during_prefill_total"],
                "p0_samples_during_prefill_mean": item["p0_samples_during_prefill_mean"],
                "p_prefill_gross_energy_j": item["p_prefill_gross_energy_j"],
                "p_prefill_incremental_energy_j": item["p_prefill_incremental_energy_j"],
                "positive_incremental": item["positive_incremental"],
            })

    analysis = {
        "phase": "3B.1_prefill_energy_identifiability",
        "run_id": run_dir.name,
        "fixed_frequency": {
            "P0_graphics_mhz": config["fixed_clocks"]["P0"]["graphics_mhz"],
            "D0_graphics_mhz": config["fixed_clocks"]["D0"]["graphics_mhz"],
        },
        "measurement_method": {
            "prefill_time_source": "proxy diagnostics timestamps_wall_s",
            "energy_source": "P0 cumulative NVML counter",
            "energy_interpolation": "linear between neighboring P0 counter samples",
            "baseline_source": "same-repeat immediate idle window mean_power_w",
            "negative_incremental_values_clamped": False,
        },
        "nvml_sampling_interval": sampling,
        "baseline": {
            "P0_idle_power_w": {
                "n_windows": len(p0_idle),
                "mean": p0_idle_mean,
                "std": p0_idle_std,
                "minimum": min(p0_idle) if p0_idle else None,
                "maximum": max(p0_idle) if p0_idle else None,
            },
            "D0_idle_power_w": {
                "n_windows": len(d0_idle),
                "mean": statistics.fmean(d0_idle) if d0_idle else None,
                "std": statistics.stdev(d0_idle) if len(d0_idle) >= 2 else None,
                "minimum": min(d0_idle) if d0_idle else None,
                "maximum": max(d0_idle) if d0_idle else None,
            },
        },
        "workloads": workloads,
        "skipped_input_lengths": config.get("analysis", {}).get("skipped_input_lengths", []),
        "identifiability_heuristic": {
            "minimum_overlapping_samples_for_measurable": 3,
            "positive_repeat_fraction_threshold": 0.8,
            "signal_to_baseline_uncertainty_threshold": 2.0,
            "approximate_min_duration_s_from_P0_median_interval": (
                3.0 * sampling["P0"]["median_s"]
                if sampling["P0"]["median_s"] is not None else None
            ),
        },
    }
    _write_json(run_dir / "summary.json", analysis)
    with (run_dir / "prefill_identifiability.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(csv_rows[0]) if csv_rows else [
            "workload_id", "input_len", "output_len", "prefill_duration_ms_mean",
            "prefill_duration_s_total", "p0_idle_power_w",
            "p0_samples_during_prefill_total", "p0_samples_during_prefill_mean",
            "p_prefill_gross_energy_j", "p_prefill_incremental_energy_j",
            "positive_incremental",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# Phase 3B.1 prefill energy identifiability",
        "",
        "Analysis-only phase-aligned P0 prefill energy using existing fixed-frequency instrumentation.",
        "",
        "P0 idle baseline: %s W mean, %s W repeat standard deviation across %d valid idle windows." % (
            "%.3f" % p0_idle_mean if p0_idle_mean is not None else "unavailable",
            "%.3f" % p0_idle_std if p0_idle_std is not None else "unavailable",
            len(p0_idle),
        ),
        "NVML interval (P0): median %s ms, mean %s ms, p95 %s ms." % tuple(
            "%.3f" % (1000.0 * sampling["P0"][key])
            if sampling["P0"][key] is not None else "unavailable"
            for key in ("median_s", "mean_s", "p95_s")
        ),
        "",
        "| IL | Prefill ms mean±std | P0 samples/request mean±std | Gross P J mean±std | Incremental P J mean±std | Positive fraction | Classification |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for workload_id, item in workloads.items():
        def fmt(metric: Mapping[str, Any], digits: int = 2) -> str:
            if metric.get("mean") is None:
                return "unavailable"
            return "%.{0}f ± %.{0}f".format(digits) % (metric["mean"], metric["std"] or 0.0)
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |" % (
                item["workload"]["input_len"],
                fmt(item["prefill_duration_ms"]),
                fmt(item["p0_samples_during_prefill"]),
                fmt(item["p_prefill_gross_energy_j"]),
                fmt(item["p_prefill_incremental_energy_j"]),
                "%.2f" % (100.0 * item["positive_incremental_fraction"])
                if item["positive_incremental_fraction"] is not None else "unavailable",
                item["classification"],
            )
        )
    lines.extend([
        "",
        "Negative incremental values are retained as estimator residuals; they are not physical negative energy.",
        "Classification heuristic: at least three overlapping P0 samples per request, at least 80% positive repeats, and mean signal above twice the repeated-idle baseline uncertainty.",
        "",
    ])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return analysis


def analyze_characterization(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    capabilities = {
        endpoint_id: json.loads((run_dir / endpoint_id / "capability.json").read_text(encoding="utf-8"))
        for endpoint_id in ("P0", "D0")
    }
    records = {
        endpoint_id: _read_jsonl(run_dir / endpoint_id / "samples.jsonl")
        for endpoint_id in ("P0", "D0")
    }
    sampling = config["sampling"]
    repeats_output = []
    violations = []
    expected_repeats = int(progress["repeats"])
    expected_cells = expected_repeats * len(config["workloads"])
    if len(progress["cells"]) != expected_cells:
        violations.append("completed cells %d != expected %d" % (len(progress["cells"]), expected_cells))

    for cell in progress["cells"]:
        phase3a_dir = Path(cell["phase3a_run_dir"])
        load_record = json.loads((phase3a_dir / "derived" / "load_runs.json").read_text(encoding="utf-8"))[0]
        phase3a_summary = json.loads((phase3a_dir / "derived" / "summary.json").read_text(encoding="utf-8"))
        client = load_record["client"]
        idle_spec = WindowSpec(
            "idle", float(cell["idle_start_wall_s"]), float(cell["idle_end_wall_s"]), False
        )
        workload_spec = WindowSpec(
            "workload", float(client["timing_start_unix_s"]), float(client["timing_end_unix_s"]), True,
            int(client["successful_requests"]), int(client["input_tokens_total"]), int(client["output_tokens_total"]),
        )
        endpoints = {}
        endpoint_validity = []
        for endpoint_id in ("P0", "D0"):
            idle = summarize_window(
                endpoint_id, records[endpoint_id], idle_spec, capabilities[endpoint_id],
                max_gap_s=float(sampling["maximum_gap_s"]),
                min_coverage_ratio=float(sampling["minimum_coverage_ratio"]),
                max_boundary_gap_s=float(sampling["maximum_boundary_gap_s"]),
            )
            workload = summarize_window(
                endpoint_id, records[endpoint_id], workload_spec, capabilities[endpoint_id],
                max_gap_s=float(sampling["maximum_gap_s"]),
                min_coverage_ratio=float(sampling["minimum_coverage_ratio"]),
                max_boundary_gap_s=float(sampling["maximum_boundary_gap_s"]),
            )
            add_idle_adjustment(workload, idle)
            clock = _clock_audit(
                records[endpoint_id], workload_spec.start_wall_s, workload_spec.end_wall_s,
                config["fixed_clocks"][endpoint_id],
            )
            endpoint_valid = (
                idle["valid"] and workload["valid"] and clock["valid"]
                and idle["sample_counts"]["missed_slots"] == 0
                and workload["sample_counts"]["missed_slots"] == 0
                and idle["sample_counts"]["error_slots"] == 0
                and workload["sample_counts"]["error_slots"] == 0
            )
            endpoint_validity.append(endpoint_valid)
            endpoints[endpoint_id] = {"idle": idle, "workload": workload, "clock_audit": clock}

        prefill_identifiability = _prefill_identifiability(
            phase3a_dir,
            str(cell["probe_id"]),
            int(client["successful_requests"]),
            records["P0"],
            endpoints["P0"]["idle"],
        )
        token_audit = _token_audit(load_record, client)
        proxy_audit = phase3a_summary.get("proxy_diagnostics_audit") or {}
        phase3a_valid = (
            bool(phase3a_summary.get("completed"))
            and bool(proxy_audit.get("valid"))
        )
        auxiliary_telemetry = {
            "claim_boundary": "auxiliary telemetry; not a Phase 3B hard gate",
            "scrape_error_count": int(phase3a_summary.get("scrape_error_count", 0)),
            "missed_scrape_count": int(phase3a_summary.get("missed_scrape_count", 0)),
            "load_monitoring": phase3a_summary.get("load_monitoring", []),
        }
        aggregate = aggregate_endpoint_workload(
            [endpoints["P0"]["workload"], endpoints["D0"]["workload"]],
            logical_requests=int(client["successful_requests"]),
            output_tokens=int(client["output_tokens_total"]),
        )
        total = float(aggregate["gross_gpu_energy_j"]) if aggregate["valid"] else None
        p_energy = endpoints["P0"]["workload"]["gross_gpu_energy_j"]
        d_energy = endpoints["D0"]["workload"]["gross_gpu_energy_j"]
        valid = all(endpoint_validity) and token_audit["valid"] and phase3a_valid and aggregate["valid"]
        if not valid:
            violations.append("%s repeat %s invalid" % (cell["workload_id"], cell["repeat_index"]))
        behavior = phase3a_summary["endpoint_window_behavior"]
        repeat = {
            **cell,
            "valid": valid,
            "client": client,
            "phase3a_valid": phase3a_valid,
            "phase3a_auxiliary_telemetry": auxiliary_telemetry,
            "proxy_audit": proxy_audit,
            "token_audit": token_audit,
            "prefill_identifiability": prefill_identifiability,
            "endpoints": endpoints,
            "aggregate": aggregate,
            "metrics": {
                "p_energy_j": p_energy,
                "d_energy_j": d_energy,
                "total_energy_j": total,
                "p_energy_share": p_energy / total if total else None,
                "d_energy_share": d_energy / total if total else None,
                "j_per_request": aggregate.get("energy_j_per_request"),
                "j_per_output_token": aggregate.get("energy_j_per_output_token"),
                "mean_ttft_ms": client.get("mean_ttft_ms"),
                "p99_ttft_ms": client.get("p99_ttft_ms"),
                "mean_tpot_ms": client.get("mean_tpot_ms"),
                "p99_tpot_ms": client.get("p99_tpot_ms"),
                "mean_itl_ms": client.get("mean_itl_ms"),
                "p0_queue_mean_ms": _queue_mean_ms(load_record["endpoints"]["P0"]),
                "d0_queue_mean_ms": _queue_mean_ms(load_record["endpoints"]["D0"]),
                "p0_max_kv_cache_usage_frac": behavior["P0"].get("max_kv_cache_usage_frac"),
                "d0_max_kv_cache_usage_frac": behavior["D0"].get("max_kv_cache_usage_frac"),
                "p_prefill_gross_energy_j": prefill_identifiability.get("p_prefill_gross_energy_j"),
                "p_prefill_incremental_energy_j": prefill_identifiability.get("p_prefill_incremental_energy_j"),
                "prefill_duration_ms_mean": prefill_identifiability.get("prefill_duration_ms_mean"),
                "p0_samples_during_prefill_mean": prefill_identifiability.get("p0_samples_during_prefill_mean"),
            },
        }
        repeats_output.append(repeat)

    metric_names = (
        "p_energy_j", "d_energy_j", "total_energy_j", "p_energy_share", "d_energy_share",
        "j_per_request", "j_per_output_token", "mean_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "p0_queue_mean_ms",
        "d0_queue_mean_ms", "p0_max_kv_cache_usage_frac", "d0_max_kv_cache_usage_frac",
    )
    workloads_summary = {}
    for workload in config["workloads"]:
        workload_id = str(workload["id"])
        selected = [item for item in repeats_output if item["workload_id"] == workload_id and item["valid"]]
        workloads_summary[workload_id] = {
            "workload": workload,
            "expected_repeats": expected_repeats,
            "valid_repeats": len(selected),
            "metrics": {
                name: _metric_summary(
                    [item["metrics"][name] for item in selected if item["metrics"].get(name) is not None],
                    "%s:%s" % (workload_id, name),
                )
                for name in metric_names
            },
        }
        if len(selected) != expected_repeats:
            violations.append("%s valid repeats %d != expected %d" % (workload_id, len(selected), expected_repeats))

    audit = {
        "valid": not violations,
        "violations": violations,
        "expected_cells": expected_cells,
        "completed_cells": len(progress["cells"]),
        "valid_cells": sum(item["valid"] for item in repeats_output),
        "identity_mapping_valid": all(
            config["fixed_clocks"][endpoint_id]["expected_gpu_name"].lower()
            in capabilities[endpoint_id]["identity"]["gpu_name"].lower()
            for endpoint_id in ("P0", "D0")
        ),
        "fixed_clock_policy": "sustainable hardware operating points; set once before serving; no runtime changes; reset on job exit",
        "feedback_or_model_updates": False,
    }
    if not audit["identity_mapping_valid"]:
        audit["valid"] = False
        audit["violations"].append("physical identity mapping mismatch")
    summary = {
        "valid": audit["valid"],
        "run_id": run_dir.name,
        "workloads": workloads_summary,
        "claim_boundary": "controlled characterization, not routing/DVFS optimization",
    }
    _write_json(run_dir / "repeats.json", repeats_output)
    _write_json(run_dir / "characterization_audit.json", audit)
    _write_json(run_dir / "characterization_summary.json", summary)
    _write_report(run_dir, audit, summary)
    if config.get("analysis", {}).get("kind") == "phase3b1_prefill_identifiability":
        _write_prefill_identifiability_outputs(run_dir, config, repeats_output, records)
    if not audit["valid"]:
        raise CharacterizationError("characterization audit failed: %s" % "; ".join(audit["violations"]))
    return summary


def run_characterization(
    config_path: Path,
    run_id: Optional[str] = None,
    repeats_override: Optional[int] = None,
    requests_override: Optional[int] = None,
) -> Path:
    config = load_config(config_path)
    repeats = int(repeats_override or config["experiment"]["repeats"])
    requests = int(requests_override or config["experiment"]["requests_per_repeat"])
    if requests <= 0:
        raise CharacterizationError("requests per repeat must be positive")
    run_id = run_id or _utc_run_id()
    run_dir = Path(os.path.expandvars(str(config["output_root"]))) / run_id
    if run_dir.exists():
        raise CharacterizationError("non-overwriting run directory exists: %s" % run_dir)
    for relative in ("P0", "D0"):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "metadata.json", {
        "schema_version": 1,
        "phase": "3B_controlled_fixed_clock_characterization",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": config,
        "repeats": repeats,
        "requests_per_repeat": requests,
        "actuated_fixed_clocks": True,
        "adaptive_dvfs": False,
        "routing_changed": False,
        "model_trained_or_updated": False,
        "measurement_scope": "P0_and_D0_GPU_boards_only",
        "fixed_clock_evidence": os.path.expandvars(os.environ.get("XPYD_FIXED_CLOCK_EVIDENCE", "")),
    })

    processes = {}
    logs = []
    stop_files = {}
    progress = {"run_id": run_id, "repeats": repeats, "requests_per_repeat": requests, "cells": []}
    _write_json(run_dir / "progress.json", progress)
    failure: Optional[BaseException] = None
    stop_errors = []
    try:
        for endpoint in config["endpoints"]:
            endpoint_id = str(endpoint["endpoint_id"])
            process, log = _launch_monitor(endpoint, run_dir / endpoint_id, config["sampling"])
            processes[endpoint_id] = process
            logs.append(log)
            stop_files[endpoint_id] = run_dir / endpoint_id / "stop.requested"
        _wait_capabilities(
            processes, run_dir,
            timeout_s=float(config["sampling"].get("readiness_timeout_s", 120.0)),
        )
        time.sleep(float(config["windows"].get("capability_s", 1.0)))
        base_phase3a = load_phase3a_config(Path(config["phase3a_config"]))
        proxy_log = Path(os.path.expandvars(str(base_phase3a["proxy_diagnostics_log"])))
        schedule = balanced_schedule(config["workloads"], repeats)
        for item in schedule:
            workload = dict(item["workload"])
            repeat_index = int(item["repeat_index"])
            workload_id = str(workload["id"])
            probe_id = "%s_r%02d" % (workload_id, repeat_index)
            time.sleep(float(config["windows"].get("cooldown_s", 3.0)))
            idle_start = time.time()
            time.sleep(float(config["windows"]["idle_s"]))
            idle_end = time.time()
            proxy_log.parent.mkdir(parents=True, exist_ok=True)
            proxy_log.write_text("", encoding="utf-8")
            phase3a_config = deepcopy(base_phase3a)
            phase3a_config["semantic_probes"] = []
            phase3a_config["server_logs"] = {}
            phase3a_config["phase_warmup"] = {
                "enabled": True,
                "input_len": int(workload["input_len"]),
                "output_len": int(workload["output_len"]),
                "count": 1,
                "max_concurrency": 1,
            }
            phase3a_config["load_probes"] = [{
                "id": probe_id,
                "input_len": int(workload["input_len"]),
                "output_len": int(workload["output_len"]),
                "rate_rps": float(config["experiment"].get("offered_rate_rps", 1.0)),
                "count": requests,
                "max_concurrency": int(config["experiment"].get("max_concurrency", 1)),
            }]
            phase3a_config["physical_context"] = {
                **dict(phase3a_config.get("physical_context", {})),
                "characterization_run_id": run_id,
                "workload_id": workload_id,
                "repeat_index": repeat_index,
                "fixed_clock_state_external": True,
                "adaptive_dvfs": False,
            }
            phase3a_config["phase3b_acceptance"] = {
                "prometheus_scrapes_auxiliary": True,
                "hard_gates": [
                    "real_sse",
                    "logical_request_id_and_token_accounting",
                    "client_ttft_tpot_e2e",
                    "endpoint_energy_windows_and_sampling_coverage",
                    "fixed_clock_audit",
                    "no_invalidating_thermal_or_hw_slowdown",
                ],
            }
            phase3a_run_id = "%s-%s-r%02d-phase3a" % (run_id, workload_id, repeat_index)
            phase3a_run_dir = Phase3AHarness(phase3a_config, run_id=phase3a_run_id).run(("load",))
            progress["cells"].append({
                "workload_id": workload_id,
                "input_len": int(workload["input_len"]),
                "output_len": int(workload["output_len"]),
                "repeat_index": repeat_index,
                "order_index": int(item["order_index"]),
                "probe_id": probe_id,
                "idle_start_wall_s": idle_start,
                "idle_end_wall_s": idle_end,
                "phase3a_run_dir": phase3a_run_dir.as_posix(),
            })
            _write_json(run_dir / "progress.json", progress)
    except BaseException as exc:
        failure = exc
    finally:
        stop_errors = _stop_monitors(processes, logs, stop_files)

    if failure is not None or stop_errors:
        error = failure or CharacterizationError("; ".join(stop_errors))
        _write_json(run_dir / "failure.json", {
            "error": structured_error(error),
            "monitor_stop_errors": stop_errors,
            "completed_cells": len(progress["cells"]),
            "fixed_clock_reset_owned_by_outer_job_trap": True,
        })
        raise error
    analyze_characterization(run_dir, config)
    print(run_dir.as_posix())
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--run-id")
    run.add_argument("--repeats", type=int)
    run.add_argument("--requests-per-repeat", type=int)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--run-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_characterization(
            Path(args.config), args.run_id, args.repeats, args.requests_per_repeat
        )
    else:
        analyze_characterization(Path(args.run_dir), load_config(Path(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
