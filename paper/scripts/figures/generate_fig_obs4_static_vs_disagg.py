#!/usr/bin/env python3
"""
Generate Observation 4 figure assets from experiments A and E.

Outputs:
  - fig_obs4_static_vs_disagg.pdf
  - fig_obs4_static_vs_disagg.png
  - fig_obs4_static_vs_disagg_summary.csv
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
REPO_DIR = REPO_ROOT
RESULT_DIR = REPO_DIR / "results" / "disagg_20260321_v2"
OUT_BASENAME = "fig_obs4_static_vs_disagg"


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


REP_WORKLOADS = [
    (1024, 128, 10, "Prefill-heavy"),
    (128, 128, 20, "Balanced"),
    (128, 1024, 10, "Decode-heavy"),
    (64, 2048, 10, "Long decode"),
]


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
                    "mean_ttft_ms": (
                        float(row["mean_ttft_ms"])
                        if row["mean_ttft_ms"] not in ("", "None")
                        else None
                    ),
                    "j_per_req_active": (
                        float(row["j_per_req_active"])
                        if row["j_per_req_active"] not in ("", "None")
                        else None
                    ),
                }
            )
    return rows


def parse_metric(text: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def load_e_rows():
    base = RESULT_DIR / "E_disaggregated"
    pattern = re.compile(r"disagg_l40sf(\d+)_l4f(\d+)_il(\d+)_ol(\d+)_r(\d+)\.txt$")
    rows = []
    for path in sorted(base.glob("disagg_*.txt")):
        match = pattern.match(path.name)
        if not match:
            continue
        l40s_freq, l4_freq, il, ol, rate = map(int, match.groups())
        text = path.read_text(encoding="utf-8", errors="ignore")
        successful_requests = parse_metric(text, "Successful requests")
        if successful_requests != 200:
            continue
        rows.append(
            {
                "input_len": il,
                "output_len": ol,
                "request_rate": rate,
                "l40s_freq_mhz": l40s_freq,
                "l4_freq_mhz": l4_freq,
                "req_tput": parse_metric(text, "Request throughput (req/s)"),
                "mean_ttft_ms": parse_metric(text, "Mean TTFT (ms)"),
            }
        )
    return rows


def build_static_panel(a_rows):
    by_workload = defaultdict(list)
    for row in a_rows:
        key = (row["input_len"], row["output_len"], row["request_rate"])
        by_workload[key].append(row)

    static_low = []
    static_high = []
    adaptive = []
    rates = [5, 10, 20]
    for rate in rates:
        vals = by_workload[(1024, 128, rate)]
        best = min(vals, key=lambda r: r["j_per_req_active"])
        low = next(r for r in vals if r["tp"] == 1 and r["freq_mhz"] == 1245)
        high = next(r for r in vals if r["tp"] == 4 and r["freq_mhz"] == 2520)
        adaptive.append(best)
        static_low.append(low)
        static_high.append(high)
    return rates, adaptive, static_low, static_high


def build_disagg_panel(a_rows, e_rows):
    a_by_workload = defaultdict(list)
    for row in a_rows:
        a_by_workload[(row["input_len"], row["output_len"], row["request_rate"])].append(row)
    e_by_workload = defaultdict(list)
    for row in e_rows:
        e_by_workload[(row["input_len"], row["output_len"], row["request_rate"])].append(row)

    records = []
    for il, ol, rate, label in REP_WORKLOADS:
        best_a = max(a_by_workload[(il, ol, rate)], key=lambda r: r["req_tput"])
        best_e = max(e_by_workload[(il, ol, rate)], key=lambda r: r["req_tput"])
        records.append(
            {
                "label": label,
                "input_len": il,
                "output_len": ol,
                "request_rate": rate,
                "best_monolithic_tput": best_a["req_tput"],
                "best_monolithic_ttft_ms": best_a["mean_ttft_ms"],
                "best_e_tput": best_e["req_tput"],
                "best_e_ttft_ms": best_e["mean_ttft_ms"],
                "best_e_l40s_freq_mhz": best_e["l40s_freq_mhz"],
                "best_e_l4_freq_mhz": best_e["l4_freq_mhz"],
                "e_over_mono": best_e["req_tput"] / best_a["req_tput"],
            }
        )
    return records


def write_summary(rates, adaptive, static_low, static_high, disagg_records):
    path = THIS_DIR / f"{OUT_BASENAME}_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "section",
            "label",
            "request_rate",
            "config",
            "req_tput",
            "mean_ttft_ms",
            "j_per_req_active",
            "l40s_freq_mhz",
            "l4_freq_mhz",
            "ratio_to_best_monolithic",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rate, best, low, high in zip(rates, adaptive, static_low, static_high):
            writer.writerow(
                {
                    "section": "static_panel",
                    "label": "(1024,128)",
                    "request_rate": rate,
                    "config": f"adaptive-best TP{best['tp']}@{best['freq_mhz']}",
                    "req_tput": round(best["req_tput"], 4),
                    "mean_ttft_ms": round(best["mean_ttft_ms"], 2),
                    "j_per_req_active": round(best["j_per_req_active"], 6),
                }
            )
            writer.writerow(
                {
                    "section": "static_panel",
                    "label": "(1024,128)",
                    "request_rate": rate,
                    "config": "static-low TP1@1245",
                    "req_tput": round(low["req_tput"], 4),
                    "mean_ttft_ms": round(low["mean_ttft_ms"], 2),
                    "j_per_req_active": round(low["j_per_req_active"], 6),
                }
            )
            writer.writerow(
                {
                    "section": "static_panel",
                    "label": "(1024,128)",
                    "request_rate": rate,
                    "config": "static-high TP4@2520",
                    "req_tput": round(high["req_tput"], 4),
                    "mean_ttft_ms": round(high["mean_ttft_ms"], 2),
                    "j_per_req_active": round(high["j_per_req_active"], 6),
                }
            )

        for row in disagg_records:
            writer.writerow(
                {
                    "section": "disagg_panel",
                    "label": row["label"],
                    "request_rate": row["request_rate"],
                    "config": "best-monolithic",
                    "req_tput": round(row["best_monolithic_tput"], 4),
                    "mean_ttft_ms": round(row["best_monolithic_ttft_ms"], 2),
                }
            )
            writer.writerow(
                {
                    "section": "disagg_panel",
                    "label": row["label"],
                    "request_rate": row["request_rate"],
                    "config": "best-disaggregated",
                    "req_tput": round(row["best_e_tput"], 4),
                    "mean_ttft_ms": round(row["best_e_ttft_ms"], 2),
                    "l40s_freq_mhz": row["best_e_l40s_freq_mhz"],
                    "l4_freq_mhz": row["best_e_l4_freq_mhz"],
                    "ratio_to_best_monolithic": round(row["e_over_mono"], 6),
                }
            )
    return path


def plot():
    a_rows = load_a_rows()
    e_rows = load_e_rows()
    rates, adaptive, static_low, static_high = build_static_panel(a_rows)
    disagg_records = build_disagg_panel(a_rows, e_rows)
    write_summary(rates, adaptive, static_low, static_high, disagg_records)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    x = list(range(len(rates)))
    width = 0.34
    adaptive_color = "#4c78a8"
    static_high_color = "#e45756"
    static_low_color = "#72b7b2"

    ax.bar(
        [i - width / 2 for i in x],
        [row["j_per_req_active"] for row in adaptive],
        width=width,
        color=adaptive_color,
        alpha=0.9,
        label="Adaptive per-rate best energy",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [row["j_per_req_active"] for row in static_high],
        width=width,
        color=static_high_color,
        alpha=0.8,
        label="Static throughput-oriented TP4@2520",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(rate) for rate in rates])
    ax.set_xlabel("Request rate for $(1024,128)$")
    ax.set_ylabel("Energy per request (J)")
    ax.set_title("(a) One fixed point either wastes energy or loses latency")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax2 = ax.twinx()
    ax2.plot(
        x,
        [row["mean_ttft_ms"] for row in adaptive],
        color=adaptive_color,
        marker="o",
        linewidth=1.8,
        linestyle="--",
        label="Adaptive TTFT",
    )
    ax2.plot(
        x,
        [row["mean_ttft_ms"] for row in static_low],
        color=static_low_color,
        marker="^",
        linewidth=1.8,
        linestyle="-",
        label="Static low-power TTFT (TP1@1245)",
    )
    ax2.set_ylabel("Mean TTFT (ms)")

    for i, row in enumerate(adaptive):
        ax.text(
            i - width / 2,
            row["j_per_req_active"] + 1.8,
            f"TP{row['tp']}@{row['freq_mhz']}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color=adaptive_color,
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")

    ax = axes[1]
    x = list(range(len(disagg_records)))
    width = 0.34
    mono_color = "#f58518"
    disagg_color = "#54a24b"
    ax.bar(
        [i - width / 2 for i in x],
        [row["best_monolithic_tput"] for row in disagg_records],
        width=width,
        color=mono_color,
        alpha=0.88,
        label="Best monolithic throughput",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [row["best_e_tput"] for row in disagg_records],
        width=width,
        color=disagg_color,
        alpha=0.88,
        label="Best disaggregated throughput",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in disagg_records], rotation=12, ha="right")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("(b) Blind disaggregation can be much worse than monolithic serving")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for i, row in enumerate(disagg_records):
        ax.text(
            i + width / 2,
            row["best_e_tput"] + 0.14,
            f"{row['e_over_mono']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=7,
            color=disagg_color,
        )
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout(w_pad=1.4)

    pdf_path = THIS_DIR / f"{OUT_BASENAME}.pdf"
    png_path = THIS_DIR / f"{OUT_BASENAME}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Wrote {pdf_path}")
    print(f"[OK] Wrote {png_path}")


if __name__ == "__main__":
    plot()
