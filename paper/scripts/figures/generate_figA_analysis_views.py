#!/usr/bin/env python3
"""
Generate internal analysis views for Experiment A using active-energy metrics.

Outputs:
  - figA_pareto_all_workloads.pdf/png
  - figA_energy_heatmap_all_workloads.pdf/png
  - figA_best_config_by_workload.csv
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from paths import PAPER_RESULTS_FIGURES_DIR

THIS_DIR = os.fspath(PAPER_RESULTS_FIGURES_DIR)
CSV_PATH = os.path.join(THIS_DIR, "experiment_A_full_results.csv")

OUT_PARETO = "figA_pareto_all_workloads"
OUT_HEATMAP = "figA_energy_heatmap_all_workloads"
OUT_SUMMARY = "figA_best_config_by_workload.csv"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


TP_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
TP_MARKERS = {1: "o", 2: "s", 4: "D"}


def load_rows():
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
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
                    "req_tput_from_counts": float(row["req_tput_from_counts"]),
                    "j_per_req_active": float(row["j_per_req_active"]),
                    "cluster_active_energy_j": float(row["cluster_active_energy_j"]),
                    "mean_ttft_ms": float(row["mean_ttft_ms"]),
                    "mean_tpot_ms": float(row["mean_tpot_ms"]),
                }
            )
    return rows


def workload_label(w):
    il, ol, r = w
    return f"({il},{ol},r={r})"


def config_label(tp, freq):
    return f"TP{tp}@{freq}"


def pareto_frontier(points):
    # maximize throughput, minimize energy
    frontier = []
    for p in sorted(points, key=lambda x: (x["req_tput"], -x["j_per_req_active"])):
        dominated = False
        for q in points:
            if q is p:
                continue
            if (
                q["req_tput"] >= p["req_tput"]
                and q["j_per_req_active"] <= p["j_per_req_active"]
                and (
                    q["req_tput"] > p["req_tput"]
                    or q["j_per_req_active"] < p["j_per_req_active"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda x: x["req_tput"])
    return frontier


def summarize(rows):
    by_workload = defaultdict(list)
    for row in rows:
        by_workload[(row["input_len"], row["output_len"], row["request_rate"])].append(row)

    summary = []
    for workload, vals in sorted(by_workload.items()):
        best_energy = min(vals, key=lambda r: r["j_per_req_active"])
        best_tput = max(vals, key=lambda r: r["req_tput"])
        frontier = pareto_frontier(vals)
        summary.append(
            {
                "workload": workload,
                "label": workload_label(workload),
                "best_energy": best_energy,
                "best_tput": best_tput,
                "frontier": frontier,
                "all_rows": vals,
            }
        )
    return summary


def write_summary(summary):
    path = os.path.join(THIS_DIR, OUT_SUMMARY)
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
            "pareto_frontier_configs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            be = item["best_energy"]
            bt = item["best_tput"]
            frontier = ", ".join(
                config_label(p["tp"], p["freq_mhz"]) for p in item["frontier"]
            )
            writer.writerow(
                {
                    "input_len": item["workload"][0],
                    "output_len": item["workload"][1],
                    "request_rate": item["workload"][2],
                    "best_energy_config": config_label(be["tp"], be["freq_mhz"]),
                    "best_energy_j_per_req_active": round(be["j_per_req_active"], 6),
                    "best_energy_req_tput": round(be["req_tput"], 6),
                    "best_throughput_config": config_label(bt["tp"], bt["freq_mhz"]),
                    "best_throughput_j_per_req_active": round(bt["j_per_req_active"], 6),
                    "best_throughput_req_tput": round(bt["req_tput"], 6),
                    "pareto_frontier_configs": frontier,
                }
            )
    return path


def plot_pareto(summary):
    n = len(summary)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 2.4 * nrows), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, item in zip(axes, summary):
        vals = item["all_rows"]
        frontier = item["frontier"]
        for tp in sorted({r["tp"] for r in vals}):
            pts = [r for r in vals if r["tp"] == tp]
            ax.scatter(
                [r["req_tput"] for r in pts],
                [r["j_per_req_active"] for r in pts],
                s=28,
                alpha=0.8,
                color=TP_COLORS[tp],
                marker=TP_MARKERS[tp],
                label=f"TP{tp}" if ax is axes[0] else None,
            )
        ax.plot(
            [r["req_tput"] for r in frontier],
            [r["j_per_req_active"] for r in frontier],
            color="black",
            linewidth=1.2,
            alpha=0.8,
        )
        be = item["best_energy"]
        bt = item["best_tput"]
        ax.scatter([be["req_tput"]], [be["j_per_req_active"]], color="black", marker="*", s=70, zorder=5)
        ax.scatter([bt["req_tput"]], [bt["j_per_req_active"]], color="black", marker="x", s=44, zorder=5)
        ax.set_title(workload_label(item["workload"]))
        ax.set_xlabel("Throughput (req/s)")
        ax.set_ylabel("Energy / request (J)")
        ax.grid(alpha=0.25, linestyle="--")
        ax.text(
            0.98,
            0.98,
            f"best-E {config_label(be['tp'], be['freq_mhz'])}\n"
            f"best-T {config_label(bt['tp'], bt['freq_mhz'])}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
        )

    for ax in axes[n:]:
        ax.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    pdf_path = os.path.join(THIS_DIR, f"{OUT_PARETO}.pdf")
    png_path = os.path.join(THIS_DIR, f"{OUT_PARETO}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def plot_heatmap(summary):
    config_order = []
    for tp in [1, 2, 4]:
        freqs = [1245, 1500, 1755, 2010, 2265, 2520] if tp == 1 else [1500, 2010, 2520]
        for freq in freqs:
            config_order.append((tp, freq))

    matrix = []
    labels = []
    best_positions = []
    tput_positions = []
    for item in summary:
        vals = {(r["tp"], r["freq_mhz"]): r for r in item["all_rows"]}
        be = item["best_energy"]
        bt = item["best_tput"]
        best_positions.append(config_order.index((be["tp"], be["freq_mhz"])))
        tput_positions.append(config_order.index((bt["tp"], bt["freq_mhz"])))
        best_energy = be["j_per_req_active"]
        row = []
        for cfg in config_order:
            if cfg in vals:
                row.append(vals[cfg]["j_per_req_active"] / best_energy)
            else:
                row.append(float("nan"))
        matrix.append(row)
        labels.append(workload_label(item["workload"]))

    fig, ax = plt.subplots(figsize=(9.5, max(6.5, 0.35 * len(summary))))
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=1.0, vmax=3.0)
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels([config_label(tp, f) for tp, f in config_order], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Workload")
    ax.set_title(r"Active energy per request normalized to each workload's best-energy config")

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if math.isnan(val):
                continue
            text = f"{val:.2f}" if val < 10 else f"{val:.1f}"
            color = "white" if val > 1.9 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
        ax.add_patch(plt.Rectangle((best_positions[i] - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.6))
        ax.add_patch(plt.Rectangle((tput_positions[i] - 0.38, i - 0.38), 0.76, 0.76, fill=False, edgecolor="#d62728", linewidth=1.2, linestyle="--"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Relative energy / request")
    fig.tight_layout()

    pdf_path = os.path.join(THIS_DIR, f"{OUT_HEATMAP}.pdf")
    png_path = os.path.join(THIS_DIR, f"{OUT_HEATMAP}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main():
    rows = load_rows()
    summary = summarize(rows)
    summary_path = write_summary(summary)
    pareto_pdf, pareto_png = plot_pareto(summary)
    heatmap_pdf, heatmap_png = plot_heatmap(summary)
    print(f"[OK] Wrote {summary_path}")
    print(f"[OK] Wrote {pareto_pdf}")
    print(f"[OK] Wrote {pareto_png}")
    print(f"[OK] Wrote {heatmap_pdf}")
    print(f"[OK] Wrote {heatmap_png}")


if __name__ == "__main__":
    main()
