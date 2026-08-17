#!/usr/bin/env python3
"""
Summarize Experiment C+D from results/disagg_20260321_v2/CD_prefill_decode_only.

This script uses the full monitor-window energy counter rather than the
utilization-derived active window. For the short prefill-only runs in
experiment C, GPU utilization samples are often too sparse to define a stable
active interval, whereas the monitor process starts shortly before the benchmark
and stops shortly after it.

Outputs:
  - experiment_C_prefill_summary.csv
  - experiment_D_decode_summary.csv
  - experiment_CD_winner_summary.txt
"""

from __future__ import annotations

import csv
import glob
import os
import re
from collections import Counter, defaultdict

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT
THIS_DIR = os.fspath(PAPER_RESULTS_FIGURES_DIR)
REPO_DIR = os.fspath(REPO_ROOT)
ROOT = os.path.join(REPO_DIR, "results", "disagg_20260321_v2", "CD_prefill_decode_only")


def parse_metric(text: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def full_window_energy_j(prefix: str):
    total = 0.0
    found = 0
    for path in glob.glob(os.path.join(ROOT, f"{prefix}*.csv")):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        start = float(rows[0]["total_energy_mj"])
        end = float(rows[-1]["total_energy_mj"])
        total += (end - start) / 1000.0
        found += 1
    return total if found else None


def build_prefill_rows():
    rows = []
    pattern = re.compile(r"prefill_l40s_f(\d+)_il(\d+)_r(\d+)\.txt$")
    for path in glob.glob(os.path.join(ROOT, "prefill_l40s_f*_il*_r*.txt")):
        match = pattern.match(os.path.basename(path))
        if not match:
            continue
        freq, il, rate = map(int, match.groups())
        text = open(path, encoding="utf-8").read()
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(f"monitor_prefill_f{freq}_il{il}_r{rate}_gpu")
        rows.append(
            {
                "freq_mhz": freq,
                "input_len": il,
                "request_rate": rate,
                "successful_requests": succ,
                "benchmark_duration_s": parse_metric(text, "Benchmark duration (s)"),
                "req_tput": parse_metric(text, "Request throughput (req/s)"),
                "mean_ttft_ms": parse_metric(text, "Mean TTFT (ms)"),
                "mean_tpot_ms": parse_metric(text, "Mean TPOT (ms)"),
                "cluster_energy_j": energy_j,
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    return sorted(rows, key=lambda r: (r["input_len"], r["request_rate"], r["freq_mhz"]))


def build_decode_rows():
    rows = []
    pattern = re.compile(r"decode_l4_f(\d+)_ol(\d+)_r(\d+)\.txt$")
    for path in glob.glob(os.path.join(ROOT, "decode_l4_f*_ol*_r*.txt")):
        match = pattern.match(os.path.basename(path))
        if not match:
            continue
        freq, ol, rate = map(int, match.groups())
        text = open(path, encoding="utf-8").read()
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(f"monitor_decode_f{freq}_ol{ol}_r{rate}_gpu")
        rows.append(
            {
                "freq_mhz": freq,
                "output_len": ol,
                "request_rate": rate,
                "successful_requests": succ,
                "benchmark_duration_s": parse_metric(text, "Benchmark duration (s)"),
                "req_tput": parse_metric(text, "Request throughput (req/s)"),
                "mean_ttft_ms": parse_metric(text, "Mean TTFT (ms)"),
                "mean_tpot_ms": parse_metric(text, "Mean TPOT (ms)"),
                "cluster_energy_j": energy_j,
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    return sorted(rows, key=lambda r: (r["output_len"], r["request_rate"], r["freq_mhz"]))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def winner_summary(rows: list[dict], dim_key: str):
    by_workload = defaultdict(list)
    for row in rows:
        by_workload[(row[dim_key], row["request_rate"])].append(row)
    winners = []
    counts = Counter()
    for workload, vals in sorted(by_workload.items()):
        best = min(vals, key=lambda r: r["j_per_req"])
        counts[best["freq_mhz"]] += 1
        winners.append((workload, best))
    return winners, counts


def main():
    prefill_rows = build_prefill_rows()
    decode_rows = build_decode_rows()

    prefill_csv = os.path.join(THIS_DIR, "experiment_C_prefill_summary.csv")
    decode_csv = os.path.join(THIS_DIR, "experiment_D_decode_summary.csv")
    write_csv(
        prefill_csv,
        prefill_rows,
        [
            "freq_mhz",
            "input_len",
            "request_rate",
            "successful_requests",
            "benchmark_duration_s",
            "req_tput",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "cluster_energy_j",
            "j_per_req",
        ],
    )
    write_csv(
        decode_csv,
        decode_rows,
        [
            "freq_mhz",
            "output_len",
            "request_rate",
            "successful_requests",
            "benchmark_duration_s",
            "req_tput",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "cluster_energy_j",
            "j_per_req",
        ],
    )

    prefill_winners, prefill_counts = winner_summary(prefill_rows, "input_len")
    decode_winners, decode_counts = winner_summary(decode_rows, "output_len")

    summary_path = os.path.join(THIS_DIR, "experiment_CD_winner_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Experiment C prefill-only energy winners by workload\n")
        for (il, rate), best in prefill_winners:
            f.write(
                f"(il={il}, r={rate}) -> f={best['freq_mhz']} MHz, "
                f"J/req={best['j_per_req']:.3f}, req/s={best['req_tput']:.3f}, "
                f"TTFT={best['mean_ttft_ms']:.1f} ms\n"
            )
        f.write(f"\nPrefill winner counts: {dict(prefill_counts)}\n")

        f.write("\nExperiment D decode-only energy winners by workload\n")
        for (ol, rate), best in decode_winners:
            f.write(
                f"(ol={ol}, r={rate}) -> f={best['freq_mhz']} MHz, "
                f"J/req={best['j_per_req']:.3f}, req/s={best['req_tput']:.3f}, "
                f"TPOT={best['mean_tpot_ms']:.1f} ms\n"
            )
        f.write(f"\nDecode winner counts: {dict(decode_counts)}\n")

    print(prefill_csv)
    print(decode_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
