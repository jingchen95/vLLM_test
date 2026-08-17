#!/usr/bin/env python3
"""
Generate Observation 1 figure assets from experiments C/C' and D/D'.

Outputs:
  - fig_obs1_phase_regions.pdf
  - fig_obs1_phase_regions.png
  - fig_obs1_phase_regions_summary.csv
"""

from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
REPO_DIR = REPO_ROOT
ROOT_OLD = REPO_DIR / "results" / "disagg_20260321_v2" / "CD_prefill_decode_only"
ROOT_NEW = REPO_DIR / "results" / "disagg_20260405_CD_cross_gpu_phase_only" / "CD_prefill_decode_only"
OUT_BASENAME = "fig_obs1_phase_regions"

PREFILL_INPUT_LENS = [128, 256, 512, 1024, 2048]
DECODE_OUTPUT_LENS = [64, 128, 256, 512, 1024]
REQUEST_RATES = [5, 10, 20, 30, 50]

L4_COLOR = "#4C78A8"
L40S_COLOR = "#E17C05"


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


def winner_prefill(input_len: int, rate: int):
    candidates = []
    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        path = ROOT_OLD / f"prefill_l40s_f{freq}_il{input_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_OLD, f"monitor_prefill_f{freq}_il{input_len}_r{rate}_gpu")
        candidates.append(
            {
                "hardware": "L40S",
                "freq_mhz": freq,
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        path = ROOT_NEW / f"prefill_l4_f{freq}_il{input_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_NEW, f"monitor_prefill_l4_f{freq}_il{input_len}_r{rate}_gpu")
        candidates.append(
            {
                "hardware": "L4",
                "freq_mhz": freq,
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    return min(candidates, key=lambda row: row["j_per_req"])


def winner_decode(output_len: int, rate: int):
    candidates = []
    for freq in [990, 1200, 1410, 1620, 1830, 2040]:
        path = ROOT_OLD / f"decode_l4_f{freq}_ol{output_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_OLD, f"monitor_decode_f{freq}_ol{output_len}_r{rate}_gpu")
        candidates.append(
            {
                "hardware": "L4",
                "freq_mhz": freq,
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    for freq in [1245, 1500, 1755, 2010, 2265, 2520]:
        path = ROOT_NEW / f"decode_l40s_f{freq}_ol{output_len}_r{rate}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        succ = parse_metric(text, "Successful requests")
        energy_j = full_window_energy_j(ROOT_NEW, f"monitor_decode_l40s_f{freq}_ol{output_len}_r{rate}_gpu")
        candidates.append(
            {
                "hardware": "L40S",
                "freq_mhz": freq,
                "j_per_req": (energy_j / succ) if energy_j is not None and succ else None,
            }
        )
    return min(candidates, key=lambda row: row["j_per_req"])


def build_summary():
    rows = []
    for rate in REQUEST_RATES:
        for input_len in PREFILL_INPUT_LENS:
            best = winner_prefill(input_len, rate)
            rows.append(
                {
                    "phase": "prefill",
                    "token_len": input_len,
                    "request_rate": rate,
                    "winner_hardware": best["hardware"],
                    "winner_freq_mhz": best["freq_mhz"],
                    "winner_j_per_req": round(best["j_per_req"], 6),
                }
            )
        for output_len in DECODE_OUTPUT_LENS:
            best = winner_decode(output_len, rate)
            rows.append(
                {
                    "phase": "decode",
                    "token_len": output_len,
                    "request_rate": rate,
                    "winner_hardware": best["hardware"],
                    "winner_freq_mhz": best["freq_mhz"],
                    "winner_j_per_req": round(best["j_per_req"], 6),
                }
            )
    return rows


def plot_panel(ax, grid, freq_labels, xlabels, title, xlabel):
    cmap = ListedColormap([L4_COLOR, L40S_COLOR])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="upper")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels([str(x) for x in xlabels])
    ax.set_yticks(range(len(REQUEST_RATES)))
    ax.set_yticklabels([str(r) for r in REQUEST_RATES])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Request rate (req/s)")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(REQUEST_RATES), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            text_color = "white" if grid[i, j] == 0 else "black"
            ax.text(
                j,
                i,
                f"{freq_labels[i][j]}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )


def plot(rows):
    prefill_grid = np.zeros((len(REQUEST_RATES), len(PREFILL_INPUT_LENS)))
    prefill_freqs = [["" for _ in PREFILL_INPUT_LENS] for _ in REQUEST_RATES]
    decode_grid = np.zeros((len(REQUEST_RATES), len(DECODE_OUTPUT_LENS)))
    decode_freqs = [["" for _ in DECODE_OUTPUT_LENS] for _ in REQUEST_RATES]

    for row in rows:
        r_idx = REQUEST_RATES.index(int(row["request_rate"]))
        if row["phase"] == "prefill":
            c_idx = PREFILL_INPUT_LENS.index(int(row["token_len"]))
            prefill_grid[r_idx, c_idx] = 0 if row["winner_hardware"] == "L4" else 1
            prefill_freqs[r_idx][c_idx] = str(int(row["winner_freq_mhz"]))
        else:
            c_idx = DECODE_OUTPUT_LENS.index(int(row["token_len"]))
            decode_grid[r_idx, c_idx] = 0 if row["winner_hardware"] == "L4" else 1
            decode_freqs[r_idx][c_idx] = str(int(row["winner_freq_mhz"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    plot_panel(
        axes[0],
        prefill_grid,
        prefill_freqs,
        PREFILL_INPUT_LENS,
        "(a) Prefill winner hardware and frequency",
        "Input length",
    )
    plot_panel(
        axes[1],
        decode_grid,
        decode_freqs,
        DECODE_OUTPUT_LENS,
        "(b) Decode winner hardware and frequency",
        "Output length",
    )

    legend_handles = [
        mpatches.Patch(color=L4_COLOR, label="L4 wins"),
        mpatches.Patch(color=L40S_COLOR, label="L40S wins"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    pdf_path = THIS_DIR / f"{OUT_BASENAME}.pdf"
    png_path = THIS_DIR / f"{OUT_BASENAME}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)


def write_summary(rows):
    out_path = THIS_DIR / f"{OUT_BASENAME}_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
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
    plot(rows)
    print(THIS_DIR / f"{OUT_BASENAME}.pdf")
    print(THIS_DIR / f"{OUT_BASENAME}.png")
    print(THIS_DIR / f"{OUT_BASENAME}_summary.csv")


if __name__ == "__main__":
    main()
