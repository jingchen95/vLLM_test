#!/usr/bin/env python3
"""
Summarize Experiment A monolithic L40S runs and compare datasets.

Outputs for a single dataset:
  - <prefix>_full_results.csv
  - <prefix>_best_config_by_workload.csv

Optional comparison outputs:
  - <compare_prefix>_comparison.csv
  - <compare_prefix>_winner_counts.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import Counter, defaultdict

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT
THIS_DIR = os.fspath(PAPER_RESULTS_FIGURES_DIR)
REPO_DIR = os.fspath(REPO_ROOT)


BENCH_RE = re.compile(r"bench_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r(\d+)\.txt$")
MONITOR_RE = re.compile(r"monitor_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r(\d+)_gpu(\d+)\.csv$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, help="Directory containing bench_*.txt and monitor_*.csv")
    parser.add_argument("--prefix", required=True, help="Output prefix for dataset-specific CSVs")
    parser.add_argument("--compare-csv", help="Existing full_results.csv to compare against")
    parser.add_argument("--compare-prefix", help="Prefix for comparison CSV outputs")
    return parser.parse_args()


def _extract_metric(text: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def parse_txt_results(result_dir: str):
    metrics = {}
    for path in glob.glob(os.path.join(result_dir, "bench_*.txt")):
        match = BENCH_RE.match(os.path.basename(path))
        if not match:
            continue
        freq, tp, il, ol, rate = map(int, match.groups())
        text = open(path, encoding="utf-8").read()
        metrics[(freq, tp, il, ol, rate)] = {
            "successful_requests": _extract_metric(text, "Successful requests"),
            "failed_requests": _extract_metric(text, "Failed requests"),
            "benchmark_duration_s": _extract_metric(text, "Benchmark duration (s)"),
            "req_tput": _extract_metric(text, "Request throughput (req/s)"),
            "tok_tput": _extract_metric(text, "Output token throughput (tok/s)"),
            "mean_ttft_ms": _extract_metric(text, "Mean TTFT (ms)"),
            "p99_ttft_ms": _extract_metric(text, "P99 TTFT (ms)"),
            "mean_tpot_ms": _extract_metric(text, "Mean TPOT (ms)"),
        }
    return metrics


def _sample_is_active(sample):
    return sample["gpu_util_pct"] > 0 or sample["mem_util_pct"] > 0


def _sample_is_hi(sample):
    return sample["gpu_util_pct"] >= 90 or sample["mem_util_pct"] >= 95


def _active_energy_from_samples(samples):
    if not samples:
        return None

    hi = [i for i, s in enumerate(samples) if _sample_is_hi(s)]
    nz = [i for i, s in enumerate(samples) if _sample_is_active(s)]
    if not nz:
        return None

    start = hi[0] if hi else nz[0]
    end = hi[-1] if hi else nz[-1]

    while start > 0 and _sample_is_active(samples[start - 1]):
        start -= 1
    while end + 1 < len(samples) and _sample_is_active(samples[end + 1]):
        end += 1

    start_idx = max(start - 1, 0)
    end_idx = min(end + 1, len(samples) - 1)
    start_energy = samples[start_idx]["total_energy_mj"]
    end_energy = samples[end_idx]["total_energy_mj"]
    if start_energy is None or end_energy is None:
        return None
    return (end_energy - start_energy) / 1000.0


def parse_monitor_stats(result_dir: str):
    cluster_power = defaultdict(list)
    cluster_energy = defaultdict(list)

    for path in glob.glob(os.path.join(result_dir, "monitor_*.csv")):
        match = MONITOR_RE.match(os.path.basename(path))
        if not match:
            continue
        freq, tp, il, ol, rate, _gpu = map(int, match.groups())
        samples = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                power = row.get("power_w")
                if power in ("", None, "None"):
                    continue
                samples.append(
                    {
                        "power_w": float(power),
                        "total_energy_mj": (
                            float(row["total_energy_mj"])
                            if row.get("total_energy_mj") not in ("", None, "None")
                            else None
                        ),
                        "gpu_util_pct": (
                            float(row["gpu_util_pct"])
                            if row.get("gpu_util_pct") not in ("", None, "None")
                            else 0.0
                        ),
                        "mem_util_pct": (
                            float(row["mem_util_pct"])
                            if row.get("mem_util_pct") not in ("", None, "None")
                            else 0.0
                        ),
                    }
                )
        if not samples:
            continue
        key = (freq, tp, il, ol, rate)
        cluster_power[key].append(sum(s["power_w"] for s in samples) / len(samples))
        active_energy_j = _active_energy_from_samples(samples)
        if active_energy_j is not None:
            cluster_energy[key].append(active_energy_j)

    return (
        {k: sum(v) for k, v in cluster_power.items()},
        {k: sum(v) for k, v in cluster_energy.items()},
    )


def build_table(result_dir: str):
    metrics = parse_txt_results(result_dir)
    cluster_power, cluster_energy = parse_monitor_stats(result_dir)
    rows = []
    for key, metric in sorted(metrics.items(), key=lambda x: (x[0][2], x[0][3], x[0][4], x[0][1], x[0][0])):
        freq, tp, il, ol, rate = key
        if key not in cluster_power:
            continue
        successful_requests = metric["successful_requests"]
        req_tput = metric["req_tput"]
        tok_tput = metric["tok_tput"]
        duration = metric["benchmark_duration_s"]
        active_energy_j = cluster_energy.get(key)
        row = {
            "input_len": il,
            "output_len": ol,
            "request_rate": rate,
            "tp": tp,
            "freq_mhz": freq,
            "successful_requests": successful_requests,
            "failed_requests": metric["failed_requests"],
            "benchmark_duration_s": duration,
            "req_tput": req_tput,
            "req_tput_from_counts": (successful_requests / duration) if successful_requests and duration else None,
            "tok_tput": tok_tput,
            "mean_ttft_ms": metric["mean_ttft_ms"],
            "p99_ttft_ms": metric["p99_ttft_ms"],
            "mean_tpot_ms": metric["mean_tpot_ms"],
            "cluster_power_w": cluster_power[key],
            "cluster_active_energy_j": active_energy_j,
            "j_per_req_proxy": (cluster_power[key] / req_tput) if req_tput else None,
            "j_per_req_active": (active_energy_j / successful_requests) if active_energy_j and successful_requests else None,
            "j_per_tok_proxy": (cluster_power[key] / tok_tput) if tok_tput else None,
        }
        rows.append(row)
    return rows


def write_full_results(rows, prefix: str):
    path = os.path.join(THIS_DIR, f"{prefix}_full_results.csv")
    fieldnames = [
        "input_len",
        "output_len",
        "request_rate",
        "tp",
        "freq_mhz",
        "successful_requests",
        "failed_requests",
        "benchmark_duration_s",
        "req_tput",
        "req_tput_from_counts",
        "tok_tput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "cluster_power_w",
        "cluster_active_energy_j",
        "j_per_req_proxy",
        "j_per_req_active",
        "j_per_tok_proxy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_best_configs(rows, prefix: str):
    by_workload = defaultdict(list)
    for row in rows:
        by_workload[(row["input_len"], row["output_len"], row["request_rate"])].append(row)

    path = os.path.join(THIS_DIR, f"{prefix}_best_config_by_workload.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "input_len",
            "output_len",
            "request_rate",
            "best_energy_config",
            "best_energy_j_per_req_active",
            "best_energy_req_tput",
            "best_throughput_config",
            "best_throughput_j_per_req_active",
            "best_throughput_req_tput",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for workload in sorted(by_workload):
            vals = by_workload[workload]
            energy_vals = [r for r in vals if r["j_per_req_active"] is not None]
            tput_vals = [r for r in vals if r["req_tput"] is not None]
            if not energy_vals or not tput_vals:
                continue
            best_energy = min(energy_vals, key=lambda r: r["j_per_req_active"])
            best_tput = max(tput_vals, key=lambda r: r["req_tput"])
            writer.writerow(
                {
                    "input_len": workload[0],
                    "output_len": workload[1],
                    "request_rate": workload[2],
                    "best_energy_config": f"TP{best_energy['tp']}@{best_energy['freq_mhz']}",
                    "best_energy_j_per_req_active": best_energy["j_per_req_active"],
                    "best_energy_req_tput": best_energy["req_tput"],
                    "best_throughput_config": f"TP{best_tput['tp']}@{best_tput['freq_mhz']}",
                    "best_throughput_j_per_req_active": best_tput["j_per_req_active"],
                    "best_throughput_req_tput": best_tput["req_tput"],
                }
            )
    return path


def load_full_results(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "input_len": int(row["input_len"]),
                    "output_len": int(row["output_len"]),
                    "request_rate": int(row["request_rate"]),
                    "tp": int(row["tp"]),
                    "freq_mhz": int(row["freq_mhz"]),
                    "req_tput": float(row["req_tput"]),
                    "j_per_req_active": float(row["j_per_req_active"]),
                    "cluster_active_energy_j": float(row["cluster_active_energy_j"]),
                    "successful_requests": float(row["successful_requests"]),
                }
            )
    return rows


def write_comparison(rows_a, rows_b, prefix: str):
    map_a = {
        (r["input_len"], r["output_len"], r["request_rate"], r["tp"], r["freq_mhz"]): r
        for r in rows_a
    }
    map_b = {
        (r["input_len"], r["output_len"], r["request_rate"], r["tp"], r["freq_mhz"]): r
        for r in rows_b
    }
    keys = sorted(set(map_a) & set(map_b))

    comparison_path = os.path.join(THIS_DIR, f"{prefix}_comparison.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "input_len",
            "output_len",
            "request_rate",
            "tp",
            "freq_mhz",
            "j_per_req_active_ref",
            "j_per_req_active_new",
            "j_per_req_active_ratio_new_over_ref",
            "req_tput_ref",
            "req_tput_new",
            "req_tput_ratio_new_over_ref",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in keys:
            a = map_a[key]
            b = map_b[key]
            writer.writerow(
                {
                    "input_len": key[0],
                    "output_len": key[1],
                    "request_rate": key[2],
                    "tp": key[3],
                    "freq_mhz": key[4],
                    "j_per_req_active_ref": a["j_per_req_active"],
                    "j_per_req_active_new": b["j_per_req_active"],
                    "j_per_req_active_ratio_new_over_ref": b["j_per_req_active"] / a["j_per_req_active"],
                    "req_tput_ref": a["req_tput"],
                    "req_tput_new": b["req_tput"],
                    "req_tput_ratio_new_over_ref": b["req_tput"] / a["req_tput"],
                }
            )

    def winner_map(rows):
        by_workload = defaultdict(list)
        for row in rows:
            by_workload[(row["input_len"], row["output_len"], row["request_rate"])].append(row)
        out = {}
        for workload, vals in by_workload.items():
            energy_vals = [r for r in vals if r["j_per_req_active"] is not None]
            tput_vals = [r for r in vals if r["req_tput"] is not None]
            if not energy_vals or not tput_vals:
                continue
            be = min(energy_vals, key=lambda r: r["j_per_req_active"])
            bt = max(tput_vals, key=lambda r: r["req_tput"])
            out[workload] = {
                "best_energy_config": f"TP{be['tp']}@{be['freq_mhz']}",
                "best_throughput_config": f"TP{bt['tp']}@{bt['freq_mhz']}",
            }
        return out

    winners_a = winner_map(rows_a)
    winners_b = winner_map(rows_b)
    common_workloads = sorted(set(winners_a) & set(winners_b))

    winner_counts = Counter()
    for workload in common_workloads:
        same_e = winners_a[workload]["best_energy_config"] == winners_b[workload]["best_energy_config"]
        same_t = winners_a[workload]["best_throughput_config"] == winners_b[workload]["best_throughput_config"]
        winner_counts[("energy_same" if same_e else "energy_changed")] += 1
        winner_counts[("throughput_same" if same_t else "throughput_changed")] += 1

    winner_path = os.path.join(THIS_DIR, f"{prefix}_winner_counts.csv")
    with open(winner_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "count"])
        writer.writeheader()
        for metric, count in sorted(winner_counts.items()):
            writer.writerow({"metric": metric, "count": count})

    return comparison_path, winner_path


def main():
    args = parse_args()
    rows = build_table(args.result_dir)
    full_results = write_full_results(rows, args.prefix)
    best_configs = write_best_configs(rows, args.prefix)
    print(full_results)
    print(best_configs)

    if args.compare_csv and args.compare_prefix:
        ref_rows = load_full_results(args.compare_csv)
        comparison_path, winner_path = write_comparison(ref_rows, rows, args.compare_prefix)
        print(comparison_path)
        print(winner_path)


if __name__ == "__main__":
    main()
