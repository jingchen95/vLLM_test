#!/usr/bin/env python3
"""
Generate active-window versions of the Observation 1 SLO-region figures.

Uses the sensitivity-analysis rows from analyze_cd_active_window_sensitivity.py,
with:
  - prefill active window: gpu_util_pct >= 50
  - decode active window:  gpu_util_pct >= 90

Outputs:
  - fig_obs1_prefill_slo_regions_active.pdf/png
  - fig_obs1_decode_slo_regions_active.pdf/png
  - fig_obs1_slo_regions_active_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from paths import PAPER_RESULTS_FIGURES_DIR

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
ROWS_CSV = THIS_DIR / "cd_active_window_sensitivity_rows.csv"

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
            row["latency_ms"] = float(row["latency_ms"])
            row["j_per_req_active"] = (
                float(row["j_per_req_active"]) if row["j_per_req_active"] not in ("", "None") else None
            )
            rows.append(row)
    return rows


def winner(rows, phase: str, token_len: int, rate: int, slo_ms: int):
    vals = [
        row for row in rows
        if row["phase"] == phase
        and row["token_len"] == token_len
        and row["request_rate"] == rate
        and row["latency_ms"] <= slo_ms
        and row["j_per_req_active"] is not None
    ]
    if not vals:
        return {"hardware": "none", "freq_mhz": None, "j_per_req_active": None}
    return min(vals, key=lambda row: row["j_per_req_active"])


def build_summary(rows):
    out = []
    for slo in PREFILL_SLOS:
        for rate in REQUEST_RATES:
            for token_len in PREFILL_INPUT_LENS:
                best = winner(rows, "prefill", token_len, rate, slo)
                out.append(
                    {
                        "phase": "prefill",
                        "slo_ms": slo,
                        "token_len": token_len,
                        "request_rate": rate,
                        "winner_hardware": best["hardware"],
                        "winner_freq_mhz": best["freq_mhz"] if best["freq_mhz"] is not None else "",
                        "winner_j_per_req_active": (
                            round(best["j_per_req_active"], 6) if best["j_per_req_active"] is not None else ""
                        ),
                    }
                )
    for slo in DECODE_SLOS:
        for rate in REQUEST_RATES:
            for token_len in DECODE_OUTPUT_LENS:
                best = winner(rows, "decode", token_len, rate, slo)
                out.append(
                    {
                        "phase": "decode",
                        "slo_ms": slo,
                        "token_len": token_len,
                        "request_rate": rate,
                        "winner_hardware": best["hardware"],
                        "winner_freq_mhz": best["freq_mhz"] if best["freq_mhz"] is not None else "",
                        "winner_j_per_req_active": (
                            round(best["j_per_req_active"], 6) if best["j_per_req_active"] is not None else ""
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


def write_summary(summary_rows):
    out = THIS_DIR / "fig_obs1_slo_regions_active_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "slo_ms",
                "token_len",
                "request_rate",
                "winner_hardware",
                "winner_freq_mhz",
                "winner_j_per_req_active",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    rows = load_rows()
    summary_rows = build_summary(rows)
    write_summary(summary_rows)
    plot_multi(
        summary_rows,
        phase="prefill",
        thresholds=PREFILL_SLOS,
        xvals=PREFILL_INPUT_LENS,
        xlabel="Input length",
        outfile="fig_obs1_prefill_slo_regions_active",
        titles=[
            r"(a) TTFT $\leq {thr}$ ms",
            r"(b) TTFT $\leq {thr}$ ms",
            r"(c) TTFT $\leq {thr}$ ms",
        ],
    )
    plot_multi(
        summary_rows,
        phase="decode",
        thresholds=DECODE_SLOS,
        xvals=DECODE_OUTPUT_LENS,
        xlabel="Output length",
        outfile="fig_obs1_decode_slo_regions_active",
        titles=[
            r"(a) TPOT $\leq {thr}$ ms",
            r"(b) TPOT $\leq {thr}$ ms",
            r"(c) TPOT $\leq {thr}$ ms",
        ],
    )
    print(THIS_DIR / "fig_obs1_prefill_slo_regions_active.pdf")
    print(THIS_DIR / "fig_obs1_prefill_slo_regions_active.png")
    print(THIS_DIR / "fig_obs1_decode_slo_regions_active.pdf")
    print(THIS_DIR / "fig_obs1_decode_slo_regions_active.png")
    print(THIS_DIR / "fig_obs1_slo_regions_active_summary.csv")


if __name__ == "__main__":
    main()
