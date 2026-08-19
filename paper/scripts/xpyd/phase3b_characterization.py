"""Controlled fixed-clock XpYd workload characterization.

The GPU clock actuation is deliberately outside this module, in the guarded
Slurm lifecycle.  This process only orchestrates the accepted Phase 3A client
and Prometheus audits plus the read-only Phase 3B NVML monitors.
"""

from __future__ import annotations

import argparse
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
        "median": statistics.median(clean) if clean else None,
        "minimum": min(clean) if clean else None,
        "maximum": max(clean) if clean else None,
        "bootstrap_median_ci95": _bootstrap_median_ci(clean, seed),
    }


def _write_report(run_dir: Path, audit: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        return ("%%.%df" % digits) % float(value) if isinstance(value, (int, float)) else "unavailable"

    lines = [
        "# XpYd Phase 3B controlled fixed-clock characterization",
        "",
        "Verdict: **%s**" % ("PASS" if audit["valid"] else "FAIL"),
        "",
        "GPU-board telemetry only. Measurement-only/model-free; no adaptive routing, training, or DVFS policy.",
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

        token_audit = _token_audit(load_record, client)
        proxy_audit = phase3a_summary.get("proxy_diagnostics_audit") or {}
        phase3a_valid = (
            bool(phase3a_summary.get("completed"))
            and bool(proxy_audit.get("valid"))
            and int(phase3a_summary.get("scrape_error_count", -1)) == 0
            and int(phase3a_summary.get("missed_scrape_count", -1)) == 0
        )
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
            "proxy_audit": proxy_audit,
            "token_audit": token_audit,
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
        "fixed_clock_policy": "set once before serving; no runtime changes; reset on job exit",
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
