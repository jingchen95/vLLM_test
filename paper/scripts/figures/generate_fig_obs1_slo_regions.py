#!/usr/bin/env python3
"""
Generate SLO-constrained Observation 1 figure assets.

Outputs:
  - fig_obs1_prefill_slo_regions.pdf/png
  - fig_obs1_decode_slo_regions.pdf/png
  - fig_obs1_slo_regions_summary.csv
"""

from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
REPO_DIR = REPO_ROOT
ROOT_OLD = REPO_DIR / "results" / "disagg_20260321_v2" / "CD_prefill_decode_only"
ROOT_NEW = REPO_DIR / "results" / "disagg_20260405_CD_cross_gpu_phase_only" / "CD_prefill_decode_only"

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


def parse_metric(text: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def full_window_energy_j(root: Path, prefix: str):
    total = 0.0
    found = 0
    for path in glob.glob(str(root / f"{prefix}*.csv")):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        start = float(rows[0]["total_energy_mj"])
        end = float(rows[-1]["total_energy_mj"])
        total += (end - start) / 1000.0
        found += 1
    return total if found else None


def prefill_candidates(input_len: int, rate: int):
    rows = []
    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        path = ROOT_OLD / f"prefill_l40s_f{freq}_il{input_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_OLD, f"monitor_prefill_f{freq}_il{input_len}_r{rate}_gpu")
        rows.append(
            {
                "hardware": "L40S",
                "freq_mhz": freq,
                "latency_ms": parse_metric(text, "Mean TTFT (ms)"),
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        path = ROOT_NEW / f"prefill_l4_f{freq}_il{input_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_NEW, f"monitor_prefill_l4_f{freq}_il{input_len}_r{rate}_gpu")
        rows.append(
            {
                "hardware": "L4",
                "freq_mhz": freq,
                "latency_ms": parse_metric(text, "Mean TTFT (ms)"),
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    return rows


def decode_candidates(output_len: int, rate: int):
    rows = []
    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        path = ROOT_OLD / f"decode_l4_f{freq}_ol{output_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_OLD, f"monitor_decode_f{freq}_ol{output_len}_r{rate}_gpu")
        rows.append(
            {
                "hardware": "L4",
                "freq_mhz": freq,
                "latency_ms": parse_metric(text, "Mean TPOT (ms)"),
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        path = ROOT_NEW / f"decode_l40s_f{freq}_ol{output_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_NEW, f"monitor_decode_l40s_f{freq}_ol{output_len}_r{rate}_gpu")
        rows.append(
            {
                "hardware": "L40S",
                "freq_mhz": freq,
                "latency_ms": parse_metric(text, "Mean TPOT (ms)"),
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    return rows


def winner(rows, threshold_ms: int):
    feasible = [row for row in rows if row["latency_ms"] is not None and row["latency_ms"] <= threshold_ms]
    if not feasible:
        return {"hardware": "none", "freq_mhz": None, "j_per_req": None}
    return min(feasible, key=lambda row: row["j_per_req"])


def build_summary():
    out = []
    for thr in PREFILL_SLOS:
        for rate in REQUEST_RATES:
            for il in PREFILL_INPUT_LENS:
                best = winner(prefill_candidates(il, rate), thr)
                out.append(
                    {
                        "phase": "prefill",
                        "slo_ms": thr,
                        "token_len": il,
                        "request_rate": rate,
                        "winner_hardware": best["hardware"],
                        "winner_freq_mhz": best["freq_mhz"] if best["freq_mhz"] is not None else "",
                        "winner_j_per_req": round(best["j_per_req"], 6) if best["j_per_req"] is not None else "",
                    }
                )
    for thr in DECODE_SLOS:
        for rate in REQUEST_RATES:
            for ol in DECODE_OUTPUT_LENS:
                best = winner(decode_candidates(ol, rate), thr)
                out.append(
                    {
                        "phase": "decode",
                        "slo_ms": thr,
                        "token_len": ol,
                        "request_rate": rate,
                        "winner_hardware": best["hardware"],
                        "winner_freq_mhz": best["freq_mhz"] if best["freq_mhz"] is not None else "",
                        "winner_j_per_req": round(best["j_per_req"], 6) if best["j_per_req"] is not None else "",
                    }
                )
    return out


def fill_grid(rows, xvals, thresholds):
    grid_data = []
    label_data = []
    for thr in thresholds:
        grid = np.zeros((len(REQUEST_RATES), len(xvals)))
        labels = [["" for _ in xvals] for _ in REQUEST_RATES]
        for i, rate in enumerate(REQUEST_RATES):
            for j, x in enumerate(xvals):
                match = next(row for row in rows if int(row["slo_ms"]) == thr and int(row["request_rate"]) == rate and int(row["token_len"]) == x)
                hw = match["winner_hardware"]
                if hw == "L4":
                    grid[i, j] = 0
                    labels[i][j] = str(match["winner_freq_mhz"])
                elif hw == "L40S":
                    grid[i, j] = 1
                    labels[i][j] = str(match["winner_freq_mhz"])
                else:
                    grid[i, j] = 2
                    labels[i][j] = "N/A"
        grid_data.append(grid)
        label_data.append(labels)
    return grid_data, label_data


def plot_multi(rows, phase, thresholds, xvals, xlabel, outfile, panel_titles):
    cmap = ListedColormap([L4_COLOR, L40S_COLOR, NONE_COLOR])
    grid_data, label_data = fill_grid([r for r in rows if r["phase"] == phase], xvals, thresholds)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(8.1, 3.6))
    if len(thresholds) == 1:
        axes = [axes]
    for idx, (ax, thr, grid, labels) in enumerate(zip(axes, thresholds, grid_data, label_data)):
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
        ax.set_title(panel_titles[idx].format(thr=thr))
        ax.set_xticks(np.arange(-0.5, len(xvals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(REQUEST_RATES), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_box_aspect(1.0)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                color = "white" if val == 0 else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(THIS_DIR / f"{outfile}.pdf")
    fig.savefig(THIS_DIR / f"{outfile}.png")
    plt.close(fig)


def write_summary(rows):
    out_path = THIS_DIR / "fig_obs1_slo_regions_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "slo_ms",
                "token_len",
                "request_rate",
                "winner_hardware",
                "winner_freq_mhz",
                "winner_j_per_req",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = build_summary()
    write_summary(rows)
    plot_multi(
        rows,
        phase="prefill",
        thresholds=PREFILL_SLOS,
        xvals=PREFILL_INPUT_LENS,
        xlabel="Input length",
        outfile="fig_obs1_prefill_slo_regions",
        panel_titles=[
            r"(a) TTFT $\leq {thr}$ ms",
            r"(b) TTFT $\leq {thr}$ ms",
            r"(c) TTFT $\leq {thr}$ ms",
        ],
    )
    plot_multi(
        rows,
        phase="decode",
        thresholds=DECODE_SLOS,
        xvals=DECODE_OUTPUT_LENS,
        xlabel="Output length",
        outfile="fig_obs1_decode_slo_regions",
        panel_titles=[
            r"(a) TPOT $\leq {thr}$ ms",
            r"(b) TPOT $\leq {thr}$ ms",
            r"(c) TPOT $\leq {thr}$ ms",
        ],
    )
    print(THIS_DIR / "fig_obs1_prefill_slo_regions.pdf")
    print(THIS_DIR / "fig_obs1_prefill_slo_regions.png")
    print(THIS_DIR / "fig_obs1_decode_slo_regions.pdf")
    print(THIS_DIR / "fig_obs1_decode_slo_regions.png")
    print(THIS_DIR / "fig_obs1_slo_regions_summary.csv")


if __name__ == "__main__":
    main()
