#!/usr/bin/env python3
"""
Generate p99-latency versions of the benchmark-aware Observation 1 SLO-region figures.

Uses:
  - cd_benchmark_window_sensitivity_rows.csv

Outputs:
  - fig_obs1_prefill_slo_regions_bench_window_p99.pdf/png
  - fig_obs1_decode_slo_regions_bench_window_p99.pdf/png
  - fig_obs1_slo_regions_bench_window_p99_summary.csv
  - fig_obs1_slo_regions_bench_window_mean_vs_p99.txt
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from paths import PAPER_RESULTS_FIGURES_DIR

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
ROWS_CSV = THIS_DIR / "cd_benchmark_window_sensitivity_rows.csv"

PREFILL_INPUT_LENS = [128, 256, 512, 1024, 2048]
DECODE_OUTPUT_LENS = [64, 128, 256, 512, 1024]
REQUEST_RATES = [5, 10, 20, 30, 50]
PREFILL_SLOS = [100, 5000, 10000]
DECODE_SLOS = [40, 80, 120]

L4_COLOR = "#4C78A8"
L40S_COLOR = "#E17C05"
NONE_COLOR = "#D9D9D9"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def load_rows():
    rows = []
    with ROWS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["freq_mhz"] = int(row["freq_mhz"])
            row["token_len"] = int(row["token_len"])
            row["request_rate"] = int(row["request_rate"])
            row["latency_ms"] = float(row["latency_ms"]) if row["latency_ms"] not in ("", "None") else None
            row["p99_latency_ms"] = float(row["p99_latency_ms"]) if row["p99_latency_ms"] not in ("", "None") else None
            row["j_per_req_bench_window"] = (
                float(row["j_per_req_bench_window"]) if row["j_per_req_bench_window"] not in ("", "None") else None
            )
            rows.append(row)
    return rows


def winner(rows, phase: str, token_len: int, rate: int, slo_ms: int, latency_key: str):
    vals = [
        row for row in rows
        if row["phase"] == phase
        and row["token_len"] == token_len
        and row["request_rate"] == rate
        and row[latency_key] is not None
        and row[latency_key] <= slo_ms
        and row["j_per_req_bench_window"] is not None
    ]
    if not vals:
        return {"hardware": "none", "freq_mhz": None, "j_per_req_bench_window": None}
    return min(vals, key=lambda row: row["j_per_req_bench_window"])


def build_summary(rows, latency_key: str):
    out = []
    for slo in PREFILL_SLOS:
        for rate in REQUEST_RATES:
            for token_len in PREFILL_INPUT_LENS:
                best = winner(rows, "prefill", token_len, rate, slo, latency_key)
                out.append(
                    {
                        "phase": "prefill",
                        "slo_ms": slo,
                        "token_len": token_len,
                        "request_rate": rate,
                        "winner_hardware": best["hardware"],
                        "winner_freq_mhz": best["freq_mhz"] if best["freq_mhz"] is not None else "",
                        "winner_j_per_req_bench_window": (
                            round(best["j_per_req_bench_window"], 6) if best["j_per_req_bench_window"] is not None else ""
                        ),
                    }
                )
    for slo in DECODE_SLOS:
        for rate in REQUEST_RATES:
            for token_len in DECODE_OUTPUT_LENS:
                best = winner(rows, "decode", token_len, rate, slo, latency_key)
                out.append(
                    {
                        "phase": "decode",
                        "slo_ms": slo,
                        "token_len": token_len,
                        "request_rate": rate,
                        "winner_hardware": best["hardware"],
                        "winner_freq_mhz": best["freq_mhz"] if best["freq_mhz"] is not None else "",
                        "winner_j_per_req_bench_window": (
                            round(best["j_per_req_bench_window"], 6) if best["j_per_req_bench_window"] is not None else ""
                        ),
                    }
                )
    return out


def fill_grid(summary_rows, phase: str, thresholds, xvals):
    grids = []
    labels = []
    phase_rows = [r for r in summary_rows if r["phase"] == phase]
    for thr in thresholds:
        grid = np.zeros((len(REQUEST_RATES), len(xvals)))
        lab = [["" for _ in xvals] for _ in REQUEST_RATES]
        for i, rate in enumerate(REQUEST_RATES):
            for j, x in enumerate(xvals):
                row = next(
                    r for r in phase_rows
                    if int(r["slo_ms"]) == thr and int(r["request_rate"]) == rate and int(r["token_len"]) == x
                )
                hw = row["winner_hardware"]
                if hw == "L4":
                    grid[i, j] = 0
                    lab[i][j] = str(row["winner_freq_mhz"])
                elif hw == "L40S":
                    grid[i, j] = 1
                    lab[i][j] = str(row["winner_freq_mhz"])
                else:
                    grid[i, j] = 2
                    lab[i][j] = "N/A"
        grids.append(grid)
        labels.append(lab)
    return grids, labels


def plot_multi(summary_rows, phase: str, thresholds, xvals, xlabel: str, outfile: str, titles):
    cmap = ListedColormap([L4_COLOR, L40S_COLOR, NONE_COLOR])
    grids, labels = fill_grid(summary_rows, phase, thresholds, xvals)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(8.1, 3.6))
    if len(thresholds) == 1:
        axes = [axes]
    for idx, (ax, thr, grid, lab) in enumerate(zip(axes, thresholds, grids, labels)):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="upper")
        ax.set_xticks(range(len(xvals)))
        ax.set_xticklabels([str(x) for x in xvals])
        ax.set_yticks(range(len(REQUEST_RATES)))
        if idx == 0:
            ax.set_yticklabels([str(r) for r in REQUEST_RATES])
            ax.set_ylabel("Request rate (req/s)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(xlabel)
        ax.set_title(titles[idx].format(thr=thr))
        ax.set_xticks(np.arange(-0.5, len(xvals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(REQUEST_RATES), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_box_aspect(1.0)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                color = "white" if val == 0 else "black"
                ax.text(j, i, lab[i][j], ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(THIS_DIR / f"{outfile}.pdf")
    fig.savefig(THIS_DIR / f"{outfile}.png")
    plt.close(fig)


def write_summary(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "slo_ms",
                "token_len",
                "request_rate",
                "winner_hardware",
                "winner_freq_mhz",
                "winner_j_per_req_bench_window",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def compare_summaries(mean_rows: list[dict], p99_rows: list[dict], out_path: Path):
    counts = Counter()
    for m, p in zip(mean_rows, p99_rows):
        scope = f"{m['phase']}_slo"
        changed = (m["winner_hardware"], m["winner_freq_mhz"]) != (p["winner_hardware"], p["winner_freq_mhz"])
        counts[(scope, "total")] += 1
        if changed:
            counts[(scope, "changed")] += 1
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Mean-vs-p99 winner comparison on benchmark-aware energy\n")
        for scope in ["prefill_slo", "decode_slo"]:
            f.write(f"{scope}: {counts[(scope,'changed')]} / {counts[(scope,'total')]} winner changes\n")


def main():
    rows = load_rows()
    mean_summary = build_summary(rows, "latency_ms")
    p99_summary = build_summary(rows, "p99_latency_ms")
    write_summary(THIS_DIR / "fig_obs1_slo_regions_bench_window_p99_summary.csv", p99_summary)
    compare_summaries(mean_summary, p99_summary, THIS_DIR / "fig_obs1_slo_regions_bench_window_mean_vs_p99.txt")
    plot_multi(
        p99_summary,
        phase="prefill",
        thresholds=PREFILL_SLOS,
        xvals=PREFILL_INPUT_LENS,
        xlabel="Input length",
        outfile="fig_obs1_prefill_slo_regions_bench_window_p99",
        titles=[
            r"(a) P99 TTFT $\leq {thr}$ ms",
            r"(b) P99 TTFT $\leq {thr}$ ms",
            r"(c) P99 TTFT $\leq {thr}$ ms",
        ],
    )
    plot_multi(
        p99_summary,
        phase="decode",
        thresholds=DECODE_SLOS,
        xvals=DECODE_OUTPUT_LENS,
        xlabel="Output length",
        outfile="fig_obs1_decode_slo_regions_bench_window_p99",
        titles=[
            r"(a) P99 TPOT $\leq {thr}$ ms",
            r"(b) P99 TPOT $\leq {thr}$ ms",
            r"(c) P99 TPOT $\leq {thr}$ ms",
        ],
    )
    print(THIS_DIR / "fig_obs1_prefill_slo_regions_bench_window_p99.pdf")
    print(THIS_DIR / "fig_obs1_decode_slo_regions_bench_window_p99.pdf")
    print(THIS_DIR / "fig_obs1_slo_regions_bench_window_p99_summary.csv")
    print(THIS_DIR / "fig_obs1_slo_regions_bench_window_mean_vs_p99.txt")


if __name__ == "__main__":
    main()
