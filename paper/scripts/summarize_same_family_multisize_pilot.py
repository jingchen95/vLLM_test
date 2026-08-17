#!/usr/bin/env python3
"""
Summarize same-family multi-size phase-only pilot data collected via EXP=CD.

This parser understands the extended pilot filenames emitted by
run_disagg_benchmark.sh when CD_EXTENDED_NAMES=1:

  prefill_<label>_f<freq>_tp<tp>_il<il>_r<rate>.txt
  decode_<label>_f<freq>_tp<tp>_il<il>_ol<ol>_r<rate>.txt

Legacy CD filenames are also accepted for convenience, with TP defaulting to 1
and decode input_len defaulting to 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

from paths import analysis_prefix


PREFILL_PATTERN = re.compile(
    r"^prefill_(?P<label>.+?)_f(?P<freq>\d+)(?:_tp(?P<tp>\d+))?_il(?P<il>\d+)_r(?P<rate>\d+)\.txt$"
)
DECODE_PATTERN = re.compile(
    r"^decode_(?P<label>.+?)_f(?P<freq>\d+)(?:_tp(?P<tp>\d+))?(?:_il(?P<il>\d+))?_ol(?P<ol>\d+)_r(?P<rate>\d+)\.txt$"
)


def parse_metric(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def parse_window_value(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def infer_gpu_type(label: str) -> str:
    lower = label.lower()
    if "l40s" in lower:
        return "l40s"
    if re.search(r"(^|_)l4($|_)", lower):
        return "l4"
    raise ValueError(f"Could not infer gpu_type from label '{label}'")


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def iter_monitor_rows(results_dir: Path, monitor_stem: str) -> Iterable[dict]:
    for path in sorted(results_dir.glob(f"{monitor_stem}_gpu*.csv")):
        with path.open(encoding="utf-8") as f:
            yield from csv.DictReader(f)


def compute_monitor_metrics(results_dir: Path,
                            monitor_stem: str,
                            start_ts: float | None,
                            end_ts: float | None) -> dict:
    total_avg_power = 0.0
    total_energy_j = 0.0
    total_max_power = 0.0
    avg_utils: list[float] = []
    total_samples = 0

    for path in sorted(results_dir.glob(f"{monitor_stem}_gpu*.csv")):
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        if start_ts is not None and end_ts is not None:
            rows = [r for r in rows if start_ts <= float(r["timestamp"]) <= end_ts]
        if not rows:
            continue
        powers = [v for v in (safe_float(r.get("power_w")) for r in rows) if v is not None]
        utils = [v for v in (safe_float(r.get("gpu_util_pct")) for r in rows) if v is not None]
        energy_vals = [v for v in (safe_float(r.get("total_energy_mj")) for r in rows) if v is not None]
        if not powers or not energy_vals:
            continue
        energy_start = energy_vals[0]
        energy_end = energy_vals[-1]
        total_avg_power += sum(powers) / len(powers)
        total_max_power += max(powers)
        total_energy_j += max(0.0, (energy_end - energy_start) / 1000.0)
        if utils:
            avg_utils.append(sum(utils) / len(utils))
        total_samples += len(rows)

    avg_gpu_util = sum(avg_utils) / len(avg_utils) if avg_utils else None
    return {
        "avg_power_w": total_avg_power if total_samples else None,
        "max_power_w": total_max_power if total_samples else None,
        "cluster_energy_j": total_energy_j if total_samples else None,
        "avg_gpu_util_pct": avg_gpu_util,
        "power_samples": total_samples,
    }


def monitor_stem_candidates(phase: str,
                            label: str,
                            *,
                            freq: int,
                            tp: int,
                            il: int,
                            ol: int | None,
                            rate: int,
                            stem: str) -> list[str]:
    candidates = [f"monitor_{stem}"]
    if phase == "prefill":
        candidates.append(f"monitor_prefill_f{freq}_il{il}_r{rate}")
        candidates.append(f"monitor_prefill_{label}_f{freq}_il{il}_r{rate}")
        candidates.append(f"monitor_prefill_{label}_f{freq}_tp{tp}_il{il}_r{rate}")
    else:
        candidates.append(f"monitor_decode_f{freq}_ol{ol}_r{rate}")
        candidates.append(f"monitor_decode_f{freq}_il{il}_ol{ol}_r{rate}")
        candidates.append(f"monitor_decode_{label}_f{freq}_ol{ol}_r{rate}")
        candidates.append(f"monitor_decode_{label}_f{freq}_tp{tp}_il{il}_ol{ol}_r{rate}")
    deduped = []
    for cand in candidates:
        if cand not in deduped:
            deduped.append(cand)
    return deduped


def summarize_file(path: Path,
                   *,
                   model_id: str,
                   hf_name: str,
                   family: str,
                   param_count_b: float) -> dict | None:
    name = path.name
    phase = None
    match = PREFILL_PATTERN.match(name)
    if match:
        phase = "prefill"
        label = match.group("label")
        freq = int(match.group("freq"))
        tp = int(match.group("tp") or 1)
        il = int(match.group("il"))
        ol = 1
        rate = int(match.group("rate"))
    else:
        match = DECODE_PATTERN.match(name)
        if not match:
            return None
        phase = "decode"
        label = match.group("label")
        freq = int(match.group("freq"))
        tp = int(match.group("tp") or 1)
        il = int(match.group("il") or 2)
        ol = int(match.group("ol"))
        rate = int(match.group("rate"))

    text = path.read_text(encoding="utf-8", errors="replace")
    start_ts = parse_window_value(text, "Timing window start (unix_s)")
    end_ts = parse_window_value(text, "Timing window end (unix_s)")
    power_line = re.search(
        r"Measured cluster avg_power_w,energy_j,samples:\s*([0-9.]+),([0-9.]+),([0-9]+)",
        text,
    )

    monitor_metrics = {
        "avg_power_w": None,
        "max_power_w": None,
        "cluster_energy_j": None,
        "avg_gpu_util_pct": None,
        "power_samples": 0,
    }
    monitor_stem = f"monitor_{path.stem}"
    for candidate in monitor_stem_candidates(
        phase,
        label,
        freq=freq,
        tp=tp,
        il=il,
        ol=ol if phase == "decode" else None,
        rate=rate,
        stem=path.stem,
    ):
        metrics = compute_monitor_metrics(path.parent, candidate, start_ts, end_ts)
        if metrics["power_samples"]:
            monitor_stem = candidate
            monitor_metrics = metrics
            break

    avg_power_w = monitor_metrics["avg_power_w"]
    cluster_energy_j = monitor_metrics["cluster_energy_j"]
    power_samples = monitor_metrics["power_samples"]
    if power_line:
        avg_power_w = float(power_line.group(1))
        cluster_energy_j = float(power_line.group(2))
        power_samples = int(power_line.group(3))

    succ = parse_metric(text, "Successful requests")
    benchmark_duration_s = parse_metric(text, "Benchmark duration (s)")
    row = {
        "step": "pilot_phase_only",
        "run": 1,
        "phase": phase,
        "gpu_type": infer_gpu_type(label),
        "gpu_label": label,
        "gpu_freq_mhz": freq,
        "tp_degree": tp,
        "input_len": il,
        "output_len": ol,
        "request_rate": rate,
        "mean_ttft_ms": parse_metric(text, "Mean TTFT (ms)"),
        "median_ttft_ms": parse_metric(text, "Median TTFT (ms)"),
        "p99_ttft_ms": parse_metric(text, "P99 TTFT (ms)"),
        "mean_tpot_ms": parse_metric(text, "Mean TPOT (ms)"),
        "median_tpot_ms": parse_metric(text, "Median TPOT (ms)"),
        "p99_tpot_ms": parse_metric(text, "P99 TPOT (ms)"),
        "avg_power_w": avg_power_w,
        "max_power_w": monitor_metrics["max_power_w"],
        "request_throughput_rps": parse_metric(text, "Request throughput (req/s)"),
        "output_token_throughput_tps": parse_metric(text, "Output token throughput (tok/s)"),
        "total_token_throughput_tps": parse_metric(text, "Total token throughput (tok/s)"),
        "benchmark_duration_s": benchmark_duration_s,
        "avg_gpu_util_pct": monitor_metrics["avg_gpu_util_pct"],
        "successful_requests": succ,
        "cluster_energy_j": cluster_energy_j,
        "j_per_req": (cluster_energy_j / succ) if cluster_energy_j is not None and succ else None,
        "timing_start_unix_s": start_ts,
        "timing_end_unix_s": end_ts,
        "power_samples": power_samples,
        "result_file": path.as_posix(),
        "monitor_stem": monitor_stem,
        "model_id": model_id,
        "hf_name": hf_name,
        "model_family": family,
        "param_count_b": param_count_b,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hf-name", required=True)
    parser.add_argument("--family", default="mistral")
    parser.add_argument("--param-count-b", type=float, default=12.0)
    parser.add_argument("--output-prefix", default=str(analysis_prefix("same_family_multisize_pilot")))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for path in sorted(results_dir.glob("*.txt")):
        row = summarize_file(
            path,
            model_id=args.model_id,
            hf_name=args.hf_name,
            family=args.family,
            param_count_b=args.param_count_b,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No pilot result files found under {results_dir}")

    rows.sort(key=lambda r: (r["phase"], r["gpu_type"], r["tp_degree"], r["gpu_freq_mhz"], r["input_len"], r["output_len"], r["request_rate"]))
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = prefix.with_name(prefix.name + "_phase_measurements.csv")
    pd_fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pd_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "prefill_rows": sum(1 for r in rows if r["phase"] == "prefill"),
        "decode_rows": sum(1 for r in rows if r["phase"] == "decode"),
        "gpu_types": sorted({r["gpu_type"] for r in rows}),
        "phase_counts": {
            "prefill": sum(1 for r in rows if r["phase"] == "prefill"),
            "decode": sum(1 for r in rows if r["phase"] == "decode"),
        },
    }
    prefix.with_name(prefix.name + "_phase_measurements_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(out_csv)
    print(prefix.with_name(prefix.name + "_phase_measurements_summary.json"))


if __name__ == "__main__":
    main()
