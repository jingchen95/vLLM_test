#!/usr/bin/env python3
"""
Generate Observation 3 figure assets from experiments A, C, and D.

Outputs:
  - fig_obs3_knob_interaction.pdf
  - fig_obs3_knob_interaction.png
  - fig_obs3_knob_interaction_summary.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from paths import PAPER_RESULTS_FIGURES_DIR

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
OUT_BASENAME = "fig_obs3_knob_interaction"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


WORKLOAD_ROWS = [
    ((1024, 128), "Prefill-heavy\n1024/128"),
    ((512, 128), "Mid-prefill\n512/128"),
    ((128, 128), "Balanced\n128/128"),
    ((128, 1024), "Decode-heavy\n128/1024"),
]
RATE_COLS = [2, 5, 10, 20, 30, 50]
TP_TO_CODE = {1: 1, 2: 2, 4: 3}
CODE_TO_TP = {1: 1, 2: 2, 3: 4}
TP_COLORS = {1: "#4c78a8", 2: "#f58518", 4: "#54a24b"}


def load_a_rows():
    rows = []
    with (THIS_DIR / "experiment_A_full_results.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "input_len": int(float(row["input_len"])),
                    "output_len": int(float(row["output_len"])),
                    "request_rate": int(float(row["request_rate"])),
                    "tp": int(float(row["tp"])),
                    "freq_mhz": int(float(row["freq_mhz"])),
                    "req_tput": float(row["req_tput"]) if row["req_tput"] not in ("", "None") else None,
                    "j_per_req_active": (
                        float(row["j_per_req_active"])
                        if row["j_per_req_active"] not in ("", "None")
                        else None
                    ),
                }
            )
    return rows


def load_cd_curve(path: Path, dim_name: str, dim_value: int, rate: int):
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row[dim_name])) != dim_value or int(float(row["request_rate"])) != rate:
                continue
            rows.append(
                {
                    "freq_mhz": int(float(row["freq_mhz"])),
                    "j_per_req": float(row["j_per_req"]),
                }
            )
    rows.sort(key=lambda r: r["freq_mhz"])
    min_j = min(r["j_per_req"] for r in rows)
    for row in rows:
        row["j_norm"] = row["j_per_req"] / min_j
    return rows


def build_winner_map(rows):
    by_workload = defaultdict(list)
    for row in rows:
        key = (row["input_len"], row["output_len"], row["request_rate"])
        by_workload[key].append(row)

    grid = [[0 for _ in RATE_COLS] for _ in WORKLOAD_ROWS]
    labels = [["" for _ in RATE_COLS] for _ in WORKLOAD_ROWS]
    summary_rows = []

    for i, ((il, ol), family_label) in enumerate(WORKLOAD_ROWS):
        for j, rate in enumerate(RATE_COLS):
            vals = [
                row
                for row in by_workload.get((il, ol, rate), [])
                if row["j_per_req_active"] is not None
            ]
            if not vals:
                continue
            best = min(vals, key=lambda r: r["j_per_req_active"])
            grid[i][j] = TP_TO_CODE[best["tp"]]
            labels[i][j] = f"TP{best['tp']}\n{best['freq_mhz']}"
            summary_rows.append(
                {
                    "family": family_label.replace("\n", " "),
                    "input_len": il,
                    "output_len": ol,
                    "request_rate": rate,
                    "best_tp": best["tp"],
                    "best_freq_mhz": best["freq_mhz"],
                    "best_j_per_req_active": round(best["j_per_req_active"], 6),
                    "best_req_tput": round(best["req_tput"], 4),
                }
            )

    return grid, labels, summary_rows


def write_summary(rows):
    path = THIS_DIR / f"{OUT_BASENAME}_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "input_len",
                "output_len",
                "request_rate",
                "best_tp",
                "best_freq_mhz",
                "best_j_per_req_active",
                "best_req_tput",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot():
    a_rows = load_a_rows()
    winner_grid, winner_labels, summary_rows = build_winner_map(a_rows)
    write_summary(summary_rows)

    prefill_curve = load_cd_curve(
        THIS_DIR / "experiment_C_prefill_summary.csv",
        "input_len",
        128,
        30,
    )
    decode_curve = load_cd_curve(
        THIS_DIR / "experiment_D_decode_summary.csv",
        "output_len",
        128,
        20,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    ax = axes[0]
    cmap = ListedColormap(["#f1f1f1", TP_COLORS[1], TP_COLORS[2], TP_COLORS[4]])
    ax.imshow(winner_grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(RATE_COLS)))
    ax.set_xticklabels([str(r) for r in RATE_COLS])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_yticks(range(len(WORKLOAD_ROWS)))
    ax.set_yticklabels([label for _, label in WORKLOAD_ROWS])
    ax.set_title("(a) Energy-optimal $(\\mathrm{TP}, f)$ shifts with workload and load")

    for i in range(len(WORKLOAD_ROWS) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.0)
    for j in range(len(RATE_COLS) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.0)

    for i in range(len(WORKLOAD_ROWS)):
        for j in range(len(RATE_COLS)):
            if winner_grid[i][j] == 0:
                ax.text(j, i, "—", ha="center", va="center", color="#777777", fontsize=8)
                continue
            ax.text(j, i, winner_labels[i][j], ha="center", va="center", color="white", fontsize=7)

    tp_legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TP_COLORS[tp], markersize=8, label=f"TP{tp}")
        for tp in (1, 2, 4)
    ]
    ax.legend(handles=tp_legend, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20))

    ax = axes[1]
    ax.plot(
        [row["freq_mhz"] for row in prefill_curve],
        [row["j_norm"] for row in prefill_curve],
        color="#d55e00",
        marker="o",
        linewidth=1.8,
        label="Prefill-only L40S\n$(il{=}128, r{=}30)$",
    )
    ax.plot(
        [row["freq_mhz"] for row in decode_curve],
        [row["j_norm"] for row in decode_curve],
        color="#0072b2",
        marker="s",
        linewidth=1.8,
        label="Decode-only L4\n$(ol{=}128, r{=}20)$",
    )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Energy/request normalized to minimum")
    ax.set_title("(b) Phase placement changes the efficient DVFS region")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout(w_pad=1.3)

    pdf_path = THIS_DIR / f"{OUT_BASENAME}.pdf"
    png_path = THIS_DIR / f"{OUT_BASENAME}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Wrote {pdf_path}")
    print(f"[OK] Wrote {png_path}")


if __name__ == "__main__":
    plot()
