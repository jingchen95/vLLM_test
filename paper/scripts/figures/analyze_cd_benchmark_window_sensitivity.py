#!/usr/bin/env python3
"""
Benchmark-aware sensitivity analysis for phase-only C/C'/D/D' energy accounting.

Compares:
  1. Full monitor-window hardware energy counter
  2. Benchmark-aware threshold-window energy counter

Benchmark-aware rule:
  - Parse the first benchmark-log INFO timestamp from the result .txt file
  - Ignore monitor samples before that time
  - Find the first and last sustained-utilization blocks after that time
  - Compute energy from the first qualifying sample to the last qualifying sample

Default thresholds:
  - Prefill: gpu_util_pct > 50 for 4 consecutive samples
  - Decode:  gpu_util_pct > 90 for 5 consecutive samples

Outputs:
  - cd_benchmark_window_sensitivity_rows.csv
  - cd_benchmark_window_sensitivity_winners.csv
  - cd_benchmark_window_sensitivity_summary.txt
"""

from __future__ import annotations

import csv
import glob
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
REPO_DIR = REPO_ROOT
ROOT_CD = REPO_DIR / "results" / "disagg_20260321_v2" / "CD_prefill_decode_only"
ROOT_CROSS = REPO_DIR / "results" / "disagg_20260405_CD_cross_gpu_phase_only" / "CD_prefill_decode_only"

PREFILL_INPUT_LENS = [128, 256, 512, 1024, 2048]
DECODE_OUTPUT_LENS = [64, 128, 256, 512, 1024]
REQUEST_RATES = [5, 10, 20, 30, 50]
PREFILL_SLOS = [100, 5000, 10000]
DECODE_SLOS = [40, 80, 120]

INFO_RE = re.compile(r"INFO\s+(\d{2})-(\d{2})\s+(\d{2}:\d{2}:\d{2})")


def parse_metric(text: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def parse_info_timestamp(text: str, sample_year: int):
    match = INFO_RE.search(text)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    hms = match.group(3)
    return datetime.strptime(f"{sample_year:04d}-{month:02d}-{day:02d} {hms}", "%Y-%m-%d %H:%M:%S")


def monitor_paths(root: Path, prefix: str):
    return sorted(glob.glob(str(root / f"{prefix}*.csv")))


def load_monitor_rows(path: str):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for idx, row in enumerate(rows):
        try:
            dt = datetime.fromisoformat(row["datetime"])
            energy = float(row["total_energy_mj"])
            gpu_util = float(row["gpu_util_pct"])
            mem_util = float(row["mem_util_pct"])
        except (TypeError, ValueError, KeyError):
            continue
        parsed.append(
            {
                "idx": idx,
                "datetime": dt,
                "total_energy_mj": energy,
                "gpu_util_pct": gpu_util,
                "mem_util_pct": mem_util,
            }
        )
    return parsed


def full_window_energy_j(root: Path, prefix: str):
    total = 0.0
    found = 0
    for path in monitor_paths(root, prefix):
        rows = load_monitor_rows(path)
        if not rows:
            continue
        total += (rows[-1]["total_energy_mj"] - rows[0]["total_energy_mj"]) / 1000.0
        found += 1
    return total if found else None


def qualifying_segments(rows: list[dict], lower_bound: datetime, util_threshold: float, min_consecutive: int, mem_threshold: float | None):
    flags = []
    indices = []
    for idx, row in enumerate(rows):
        if row["datetime"] < lower_bound:
            continue
        active = row["gpu_util_pct"] > util_threshold
        if mem_threshold is not None:
            active = active and row["mem_util_pct"] > mem_threshold
        flags.append(active)
        indices.append(idx)

    segments = []
    run_start = None
    run_len = 0
    for pos, active in enumerate(flags):
        if active:
            if run_start is None:
                run_start = pos
            run_len += 1
        else:
            if run_start is not None and run_len >= min_consecutive:
                segments.append((indices[run_start], indices[pos - 1]))
            run_start = None
            run_len = 0
    if run_start is not None and run_len >= min_consecutive:
        segments.append((indices[run_start], indices[len(flags) - 1]))
    return segments


def benchmark_window_energy_j(root: Path, prefix: str, txt_path: Path, util_threshold: float, min_consecutive: int = 5, mem_threshold: float | None = None):
    txt = txt_path.read_text(encoding="utf-8", errors="ignore")
    total = 0.0
    found = 0
    for path in monitor_paths(root, prefix):
        rows = load_monitor_rows(path)
        if not rows:
            continue
        lower_bound = parse_info_timestamp(txt, rows[0]["datetime"].year)
        if lower_bound is None:
            continue
        segments = qualifying_segments(rows, lower_bound, util_threshold, min_consecutive, mem_threshold)
        if not segments:
            continue
        start_idx = segments[0][0]
        end_idx = segments[-1][1]
        total += (rows[end_idx]["total_energy_mj"] - rows[start_idx]["total_energy_mj"]) / 1000.0
        found += 1
    return total if found else None


def build_rows(
    prefill_util_threshold: int = 50,
    decode_util_threshold: int = 90,
    prefill_min_consecutive: int = 4,
    decode_min_consecutive: int = 5,
):
    rows = []

    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        for il in PREFILL_INPUT_LENS:
            for rate in REQUEST_RATES:
                txt_path = ROOT_CD / f"prefill_l40s_f{freq}_il{il}_r{rate}.txt"
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_prefill_f{freq}_il{il}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CD, prefix)
                bench_e = benchmark_window_energy_j(
                    ROOT_CD, prefix, txt_path, prefill_util_threshold, prefill_min_consecutive
                )
                rows.append(
                    {
                        "phase": "prefill",
                        "hardware": "L40S",
                        "freq_mhz": freq,
                        "token_len": il,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TTFT (ms)"),
                        "p99_latency_ms": parse_metric(text, "P99 TTFT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_bench_window": (bench_e / succ) if bench_e is not None and succ else None,
                    }
                )

    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        for il in PREFILL_INPUT_LENS:
            for rate in REQUEST_RATES:
                txt_path = ROOT_CROSS / f"prefill_l4_f{freq}_il{il}_r{rate}.txt"
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_prefill_l4_f{freq}_il{il}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CROSS, prefix)
                bench_e = benchmark_window_energy_j(
                    ROOT_CROSS, prefix, txt_path, prefill_util_threshold, prefill_min_consecutive
                )
                rows.append(
                    {
                        "phase": "prefill",
                        "hardware": "L4",
                        "freq_mhz": freq,
                        "token_len": il,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TTFT (ms)"),
                        "p99_latency_ms": parse_metric(text, "P99 TTFT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_bench_window": (bench_e / succ) if bench_e is not None and succ else None,
                    }
                )

    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        for ol in DECODE_OUTPUT_LENS:
            for rate in REQUEST_RATES:
                txt_path = ROOT_CD / f"decode_l4_f{freq}_ol{ol}_r{rate}.txt"
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_decode_f{freq}_ol{ol}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CD, prefix)
                bench_e = benchmark_window_energy_j(
                    ROOT_CD, prefix, txt_path, decode_util_threshold, decode_min_consecutive
                )
                rows.append(
                    {
                        "phase": "decode",
                        "hardware": "L4",
                        "freq_mhz": freq,
                        "token_len": ol,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TPOT (ms)"),
                        "p99_latency_ms": parse_metric(text, "P99 TPOT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_bench_window": (bench_e / succ) if bench_e is not None and succ else None,
                    }
                )

    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        for ol in DECODE_OUTPUT_LENS:
            for rate in REQUEST_RATES:
                txt_path = ROOT_CROSS / f"decode_l40s_f{freq}_ol{ol}_r{rate}.txt"
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_decode_l40s_f{freq}_ol{ol}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CROSS, prefix)
                bench_e = benchmark_window_energy_j(
                    ROOT_CROSS, prefix, txt_path, decode_util_threshold, decode_min_consecutive
                )
                rows.append(
                    {
                        "phase": "decode",
                        "hardware": "L40S",
                        "freq_mhz": freq,
                        "token_len": ol,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TPOT (ms)"),
                        "p99_latency_ms": parse_metric(text, "P99 TPOT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_bench_window": (bench_e / succ) if bench_e is not None and succ else None,
                    }
                )

    return rows


def winner(rows: list[dict], phase: str, token_len: int, request_rate: int, metric_key: str, slo_ms: int | None):
    vals = [
        row for row in rows
        if row["phase"] == phase and row["token_len"] == token_len and row["request_rate"] == request_rate
    ]
    if slo_ms is not None:
        vals = [row for row in vals if row["latency_ms"] is not None and row["latency_ms"] <= slo_ms]
    vals = [row for row in vals if row[metric_key] is not None]
    if not vals:
        return {"hardware": "none", "freq_mhz": "", metric_key: None}
    return min(vals, key=lambda row: row[metric_key])


def compare_winners(rows: list[dict]):
    out = []

    for il in PREFILL_INPUT_LENS:
        for rate in REQUEST_RATES:
            best_full = winner(rows, "prefill", il, rate, "j_per_req_full", None)
            best_bench = winner(rows, "prefill", il, rate, "j_per_req_bench_window", None)
            out.append(
                {
                    "scope": "prefill_unconstrained",
                    "token_len": il,
                    "request_rate": rate,
                    "slo_ms": "",
                    "winner_full_hardware": best_full["hardware"],
                    "winner_full_freq_mhz": best_full["freq_mhz"],
                    "winner_bench_hardware": best_bench["hardware"],
                    "winner_bench_freq_mhz": best_bench["freq_mhz"],
                    "changed": int((best_full["hardware"], best_full["freq_mhz"]) != (best_bench["hardware"], best_bench["freq_mhz"])),
                }
            )

    for ol in DECODE_OUTPUT_LENS:
        for rate in REQUEST_RATES:
            best_full = winner(rows, "decode", ol, rate, "j_per_req_full", None)
            best_bench = winner(rows, "decode", ol, rate, "j_per_req_bench_window", None)
            out.append(
                {
                    "scope": "decode_unconstrained",
                    "token_len": ol,
                    "request_rate": rate,
                    "slo_ms": "",
                    "winner_full_hardware": best_full["hardware"],
                    "winner_full_freq_mhz": best_full["freq_mhz"],
                    "winner_bench_hardware": best_bench["hardware"],
                    "winner_bench_freq_mhz": best_bench["freq_mhz"],
                    "changed": int((best_full["hardware"], best_full["freq_mhz"]) != (best_bench["hardware"], best_bench["freq_mhz"])),
                }
            )

    for slo in PREFILL_SLOS:
        for il in PREFILL_INPUT_LENS:
            for rate in REQUEST_RATES:
                best_full = winner(rows, "prefill", il, rate, "j_per_req_full", slo)
                best_bench = winner(rows, "prefill", il, rate, "j_per_req_bench_window", slo)
                out.append(
                    {
                        "scope": "prefill_slo",
                        "token_len": il,
                        "request_rate": rate,
                        "slo_ms": slo,
                        "winner_full_hardware": best_full["hardware"],
                        "winner_full_freq_mhz": best_full["freq_mhz"],
                        "winner_bench_hardware": best_bench["hardware"],
                        "winner_bench_freq_mhz": best_bench["freq_mhz"],
                        "changed": int((best_full["hardware"], best_full["freq_mhz"]) != (best_bench["hardware"], best_bench["freq_mhz"])),
                    }
                )

    for slo in DECODE_SLOS:
        for ol in DECODE_OUTPUT_LENS:
            for rate in REQUEST_RATES:
                best_full = winner(rows, "decode", ol, rate, "j_per_req_full", slo)
                best_bench = winner(rows, "decode", ol, rate, "j_per_req_bench_window", slo)
                out.append(
                    {
                        "scope": "decode_slo",
                        "token_len": ol,
                        "request_rate": rate,
                        "slo_ms": slo,
                        "winner_full_hardware": best_full["hardware"],
                        "winner_full_freq_mhz": best_full["freq_mhz"],
                        "winner_bench_hardware": best_bench["hardware"],
                        "winner_bench_freq_mhz": best_bench["freq_mhz"],
                        "changed": int((best_full["hardware"], best_full["freq_mhz"]) != (best_bench["hardware"], best_bench["freq_mhz"])),
                    }
                )

    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    rows: list[dict],
    compares: list[dict],
    prefill_min_consecutive: int,
    decode_min_consecutive: int,
):
    counts = Counter(row["scope"] for row in compares if row["changed"])
    total = Counter(row["scope"] for row in compares)
    available = Counter()
    for row in rows:
        if row["j_per_req_bench_window"] is not None:
            available[row["phase"]] += 1

    bench_ratios = []
    for row in rows:
        full_v = row["j_per_req_full"]
        bench_v = row["j_per_req_bench_window"]
        if full_v and bench_v is not None:
            bench_ratios.append(bench_v / full_v)
    bench_ratios.sort()

    with path.open("w", encoding="utf-8") as f:
        f.write("Benchmark-aware window sensitivity summary\n")
        f.write(f"prefill threshold: gpu_util_pct > 50 for {prefill_min_consecutive} consecutive samples\n")
        f.write(f"decode threshold: gpu_util_pct > 90 for {decode_min_consecutive} consecutive samples\n")
        f.write("lower bound: first INFO timestamp in result .txt\n\n")
        for scope in ["prefill_unconstrained", "decode_unconstrained", "prefill_slo", "decode_slo"]:
            f.write(f"{scope}: {counts[scope]} / {total[scope]} winner changes\n")
        f.write("\nRows with non-empty benchmark-aware energy\n")
        for phase in ["prefill", "decode"]:
            f.write(f"  {phase}: {available[phase]}\n")
        if bench_ratios:
            n = len(bench_ratios)
            f.write("\nbench/full J-per-request ratio stats\n")
            f.write(f"  count: {n}\n")
            f.write(f"  min: {bench_ratios[0]:.4f}\n")
            f.write(f"  median: {bench_ratios[n // 2]:.4f}\n")
            f.write(f"  max: {bench_ratios[-1]:.4f}\n")


def main():
    prefill_min_consecutive = 4
    decode_min_consecutive = 5
    rows = build_rows(
        prefill_min_consecutive=prefill_min_consecutive,
        decode_min_consecutive=decode_min_consecutive,
    )
    compares = compare_winners(rows)

    rows_csv = THIS_DIR / "cd_benchmark_window_sensitivity_rows.csv"
    winners_csv = THIS_DIR / "cd_benchmark_window_sensitivity_winners.csv"
    summary_txt = THIS_DIR / "cd_benchmark_window_sensitivity_summary.txt"

    write_csv(
        rows_csv,
        rows,
        [
            "phase",
            "hardware",
            "freq_mhz",
            "token_len",
            "request_rate",
            "latency_ms",
            "p99_latency_ms",
            "successful_requests",
            "j_per_req_full",
            "j_per_req_bench_window",
        ],
    )
    write_csv(
        winners_csv,
        compares,
        [
            "scope",
            "token_len",
            "request_rate",
            "slo_ms",
            "winner_full_hardware",
            "winner_full_freq_mhz",
            "winner_bench_hardware",
            "winner_bench_freq_mhz",
            "changed",
        ],
    )
    write_summary(summary_txt, rows, compares, prefill_min_consecutive, decode_min_consecutive)

    print(rows_csv)
    print(winners_csv)
    print(summary_txt)


if __name__ == "__main__":
    main()
