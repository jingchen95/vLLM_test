#!/usr/bin/env python3
"""
Sensitivity analysis for phase-only C/C'/D/D' energy accounting.

Compares:
  1. Full monitor-window hardware energy counter
  2. Threshold-based active-window hardware energy counter

Default active-window thresholds:
  - Prefill: gpu_util_pct >= 50
  - Decode:  gpu_util_pct >= 90

Outputs:
  - cd_active_window_sensitivity_rows.csv
  - cd_active_window_sensitivity_winners.csv
  - cd_active_window_sensitivity_summary.txt
"""

from __future__ import annotations

import csv
import glob
import os
import re
from collections import Counter
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


def parse_metric(text: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def monitor_paths(root: Path, prefix: str):
    return sorted(glob.glob(str(root / f"{prefix}*.csv")))


def full_window_energy_j(root: Path, prefix: str):
    total = 0.0
    found = 0
    for path in monitor_paths(root, prefix):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        start = float(rows[0]["total_energy_mj"])
        end = float(rows[-1]["total_energy_mj"])
        total += (end - start) / 1000.0
        found += 1
    return total if found else None


def active_window_energy_j(root: Path, prefix: str, util_threshold: int):
    total = 0.0
    found = 0
    for path in monitor_paths(root, prefix):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        active_idx = []
        for idx, row in enumerate(rows):
            try:
                util = float(row["gpu_util_pct"])
            except (TypeError, ValueError):
                continue
            if util >= util_threshold:
                active_idx.append(idx)
        if not active_idx:
            continue
        start = float(rows[active_idx[0]]["total_energy_mj"])
        end = float(rows[active_idx[-1]]["total_energy_mj"])
        total += (end - start) / 1000.0
        found += 1
    return total if found else None


def build_rows(prefill_util_threshold: int = 50, decode_util_threshold: int = 90):
    rows = []

    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        for il in PREFILL_INPUT_LENS:
            for rate in REQUEST_RATES:
                path = ROOT_CD / f"prefill_l40s_f{freq}_il{il}_r{rate}.txt"
                text = path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_prefill_f{freq}_il{il}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CD, prefix)
                active_e = active_window_energy_j(ROOT_CD, prefix, prefill_util_threshold)
                rows.append(
                    {
                        "phase": "prefill",
                        "hardware": "L40S",
                        "freq_mhz": freq,
                        "token_len": il,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TTFT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_active": (active_e / succ) if active_e is not None and succ else None,
                    }
                )

    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        for il in PREFILL_INPUT_LENS:
            for rate in REQUEST_RATES:
                path = ROOT_CROSS / f"prefill_l4_f{freq}_il{il}_r{rate}.txt"
                text = path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_prefill_l4_f{freq}_il{il}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CROSS, prefix)
                active_e = active_window_energy_j(ROOT_CROSS, prefix, prefill_util_threshold)
                rows.append(
                    {
                        "phase": "prefill",
                        "hardware": "L4",
                        "freq_mhz": freq,
                        "token_len": il,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TTFT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_active": (active_e / succ) if active_e is not None and succ else None,
                    }
                )

    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        for ol in DECODE_OUTPUT_LENS:
            for rate in REQUEST_RATES:
                path = ROOT_CD / f"decode_l4_f{freq}_ol{ol}_r{rate}.txt"
                text = path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_decode_f{freq}_ol{ol}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CD, prefix)
                active_e = active_window_energy_j(ROOT_CD, prefix, decode_util_threshold)
                rows.append(
                    {
                        "phase": "decode",
                        "hardware": "L4",
                        "freq_mhz": freq,
                        "token_len": ol,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TPOT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_active": (active_e / succ) if active_e is not None and succ else None,
                    }
                )

    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        for ol in DECODE_OUTPUT_LENS:
            for rate in REQUEST_RATES:
                path = ROOT_CROSS / f"decode_l40s_f{freq}_ol{ol}_r{rate}.txt"
                text = path.read_text(encoding="utf-8", errors="ignore")
                succ = parse_metric(text, "Successful requests")
                prefix = f"monitor_decode_l40s_f{freq}_ol{ol}_r{rate}_gpu"
                full_e = full_window_energy_j(ROOT_CROSS, prefix)
                active_e = active_window_energy_j(ROOT_CROSS, prefix, decode_util_threshold)
                rows.append(
                    {
                        "phase": "decode",
                        "hardware": "L40S",
                        "freq_mhz": freq,
                        "token_len": ol,
                        "request_rate": rate,
                        "latency_ms": parse_metric(text, "Mean TPOT (ms)"),
                        "successful_requests": succ,
                        "j_per_req_full": (full_e / succ) if full_e is not None and succ else None,
                        "j_per_req_active": (active_e / succ) if active_e is not None and succ else None,
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
    best = min(vals, key=lambda row: row[metric_key])
    return best


def compare_winners(rows: list[dict]):
    out = []

    for il in PREFILL_INPUT_LENS:
        for rate in REQUEST_RATES:
            best_full = winner(rows, "prefill", il, rate, "j_per_req_full", None)
            best_active = winner(rows, "prefill", il, rate, "j_per_req_active", None)
            out.append(
                {
                    "scope": "prefill_unconstrained",
                    "token_len": il,
                    "request_rate": rate,
                    "slo_ms": "",
                    "winner_full_hardware": best_full["hardware"],
                    "winner_full_freq_mhz": best_full["freq_mhz"],
                    "winner_active_hardware": best_active["hardware"],
                    "winner_active_freq_mhz": best_active["freq_mhz"],
                    "changed": int(
                        (best_full["hardware"], best_full["freq_mhz"]) !=
                        (best_active["hardware"], best_active["freq_mhz"])
                    ),
                }
            )

    for ol in DECODE_OUTPUT_LENS:
        for rate in REQUEST_RATES:
            best_full = winner(rows, "decode", ol, rate, "j_per_req_full", None)
            best_active = winner(rows, "decode", ol, rate, "j_per_req_active", None)
            out.append(
                {
                    "scope": "decode_unconstrained",
                    "token_len": ol,
                    "request_rate": rate,
                    "slo_ms": "",
                    "winner_full_hardware": best_full["hardware"],
                    "winner_full_freq_mhz": best_full["freq_mhz"],
                    "winner_active_hardware": best_active["hardware"],
                    "winner_active_freq_mhz": best_active["freq_mhz"],
                    "changed": int(
                        (best_full["hardware"], best_full["freq_mhz"]) !=
                        (best_active["hardware"], best_active["freq_mhz"])
                    ),
                }
            )

    for slo in PREFILL_SLOS:
        for il in PREFILL_INPUT_LENS:
            for rate in REQUEST_RATES:
                best_full = winner(rows, "prefill", il, rate, "j_per_req_full", slo)
                best_active = winner(rows, "prefill", il, rate, "j_per_req_active", slo)
                out.append(
                    {
                        "scope": "prefill_slo",
                        "token_len": il,
                        "request_rate": rate,
                        "slo_ms": slo,
                        "winner_full_hardware": best_full["hardware"],
                        "winner_full_freq_mhz": best_full["freq_mhz"],
                        "winner_active_hardware": best_active["hardware"],
                        "winner_active_freq_mhz": best_active["freq_mhz"],
                        "changed": int(
                            (best_full["hardware"], best_full["freq_mhz"]) !=
                            (best_active["hardware"], best_active["freq_mhz"])
                        ),
                    }
                )

    for slo in DECODE_SLOS:
        for ol in DECODE_OUTPUT_LENS:
            for rate in REQUEST_RATES:
                best_full = winner(rows, "decode", ol, rate, "j_per_req_full", slo)
                best_active = winner(rows, "decode", ol, rate, "j_per_req_active", slo)
                out.append(
                    {
                        "scope": "decode_slo",
                        "token_len": ol,
                        "request_rate": rate,
                        "slo_ms": slo,
                        "winner_full_hardware": best_full["hardware"],
                        "winner_full_freq_mhz": best_full["freq_mhz"],
                        "winner_active_hardware": best_active["hardware"],
                        "winner_active_freq_mhz": best_active["freq_mhz"],
                        "changed": int(
                            (best_full["hardware"], best_full["freq_mhz"]) !=
                            (best_active["hardware"], best_active["freq_mhz"])
                        ),
                    }
                )

    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], winners: list[dict], prefill_util_threshold: int, decode_util_threshold: int):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Active-window sensitivity analysis\n")
        f.write(f"Prefill active threshold: gpu_util_pct >= {prefill_util_threshold}\n")
        f.write(f"Decode active threshold: gpu_util_pct >= {decode_util_threshold}\n\n")

        for scope in ["prefill_unconstrained", "decode_unconstrained", "prefill_slo", "decode_slo"]:
            vals = [row for row in winners if row["scope"] == scope]
            changed = sum(int(row["changed"]) for row in vals)
            f.write(f"{scope}: {changed}/{len(vals)} winner changes\n")

        f.write("\nChanged winner entries:\n")
        for row in winners:
            if not row["changed"]:
                continue
            f.write(
                f"{row['scope']} token_len={row['token_len']} rate={row['request_rate']} "
                f"slo={row['slo_ms']} full={row['winner_full_hardware']}@{row['winner_full_freq_mhz']} "
                f"active={row['winner_active_hardware']}@{row['winner_active_freq_mhz']}\n"
            )

        ratios = []
        for row in rows:
            full = row["j_per_req_full"]
            active = row["j_per_req_active"]
            if full and active:
                ratios.append(active / full)
        if ratios:
            f.write("\nActive/full J-per-request ratio statistics:\n")
            f.write(f"count={len(ratios)} min={min(ratios):.4f} median={sorted(ratios)[len(ratios)//2]:.4f} max={max(ratios):.4f}\n")


def main():
    prefill_util_threshold = 50
    decode_util_threshold = 90
    rows = build_rows(prefill_util_threshold=prefill_util_threshold, decode_util_threshold=decode_util_threshold)
    winners = compare_winners(rows)

    rows_csv = THIS_DIR / "cd_active_window_sensitivity_rows.csv"
    winners_csv = THIS_DIR / "cd_active_window_sensitivity_winners.csv"
    summary_txt = THIS_DIR / "cd_active_window_sensitivity_summary.txt"

    write_csv(rows_csv, rows)
    write_csv(winners_csv, winners)
    write_summary(summary_txt, rows, winners, prefill_util_threshold, decode_util_threshold)

    print(rows_csv)
    print(winners_csv)
    print(summary_txt)


if __name__ == "__main__":
    main()
