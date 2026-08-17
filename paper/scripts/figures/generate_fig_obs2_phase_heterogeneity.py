#!/usr/bin/env python3
"""
Generate Observation 2 figure assets from experiments A/B/C/D.

Outputs:
  - fig_obs2_phase_heterogeneity.pdf
  - fig_obs2_phase_heterogeneity.png
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT

THIS_DIR = PAPER_RESULTS_FIGURES_DIR
REPO_DIR = REPO_ROOT
RESULT_DIR = REPO_DIR / "results" / "disagg_20260321_v2"
OUT_BASENAME = "fig_obs2_phase_heterogeneity"


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


def load_a_results():
    rows = []
    path = THIS_DIR / "experiment_A_full_results.csv"
    with path.open(encoding="utf-8") as f:
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


def load_b_results():
    base = RESULT_DIR / "B_monolithic_l4"
    pattern = re.compile(r"bench_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r(\d+)\.txt$")
    rows = []
    for path in sorted(base.glob("bench_*.txt")):
        m = pattern.match(path.name)
        if not m:
            continue
        freq, tp, il, ol, rate = m.groups()
        text = path.read_text(encoding="utf-8", errors="ignore")

        def get(label: str):
            mm = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
            return float(mm.group(1)) if mm else None

        succ = get("Successful requests")
        if succ != 200:
            continue
        rows.append(
            {
                "input_len": int(il),
                "output_len": int(ol),
                "request_rate": int(rate),
                "tp": int(tp),
                "freq_mhz": int(freq),
                "req_tput": get("Request throughput (req/s)"),
                "mean_ttft_ms": get("Mean TTFT (ms)"),
            }
        )
    return rows


def load_cd_winner_counts():
    c_path = THIS_DIR / "experiment_C_prefill_summary.csv"
    d_path = THIS_DIR / "experiment_D_decode_summary.csv"

    c_rows = list(csv.DictReader(c_path.open(encoding="utf-8")))
    d_rows = list(csv.DictReader(d_path.open(encoding="utf-8")))

    c_groups = defaultdict(list)
    for row in c_rows:
        c_groups[(row["input_len"], row["request_rate"])].append(row)
    c_counts = Counter()
    for group in c_groups.values():
        best = min(group, key=lambda r: float(r["j_per_req"]))
        c_counts[int(best["freq_mhz"])] += 1

    d_groups = defaultdict(list)
    for row in d_rows:
        d_groups[(row["output_len"], row["request_rate"])].append(row)
    d_counts = Counter()
    for group in d_groups.values():
        best = min(group, key=lambda r: float(r["j_per_req"]))
        d_counts[int(best["freq_mhz"])] += 1

    return c_counts, d_counts


def build_representative_alignment():
    a_rows = load_a_results()
    b_rows = load_b_results()
    prefill_workloads = [
        (1024, 128, 10, "(1024,128,10)"),
        (512, 128, 20, "(512,128,20)"),
    ]
    decode_workloads = [
        (128, 1024, 10, "(128,1024,10)"),
        (64, 2048, 10, "(64,2048,10)"),
    ]

    prefill_records = []
    for il, ol, rate, label in prefill_workloads:
        vals = [
            row
            for row in a_rows
            if (row["input_len"], row["output_len"], row["request_rate"]) == (il, ol, rate)
            and row["j_per_req_active"] is not None
        ]
        best = min(vals, key=lambda r: r["j_per_req_active"])
        prefill_records.append(
            {
                "label": label,
                "freq_mhz": best["freq_mhz"],
                "tp": best["tp"],
                "j_per_req_active": best["j_per_req_active"],
            }
        )

    decode_records = []
    for il, ol, rate, label in decode_workloads:
        vals = [
            row
            for row in b_rows
            if (row["input_len"], row["output_len"], row["request_rate"]) == (il, ol, rate)
        ]
        energy_by_key = defaultdict(float)
        grouped = {}
        mon_pattern = re.compile(
            rf"monitor_f(\d+)_tp(\d+)_il{il}_ol{ol}_r{rate}_gpu(\d+)\.csv$"
        )
        for path in sorted((RESULT_DIR / "B_monolithic_l4").glob(f"monitor_f*_tp*_il{il}_ol{ol}_r{rate}_gpu*.csv")):
            match = mon_pattern.match(path.name)
            if not match:
                continue
            freq, tp, _gpu = map(int, match.groups())
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            energy_by_key[(tp, freq)] += (float(rows[-1]["total_energy_mj"]) - float(rows[0]["total_energy_mj"])) / 1000.0
        for row in vals:
            grouped[(row["tp"], row["freq_mhz"])] = row
        candidates = []
        for (tp, freq), row in grouped.items():
            succ = 200.0
            j_per_req = energy_by_key[(tp, freq)] / succ if (tp, freq) in energy_by_key else None
            if j_per_req is None:
                continue
            candidates.append(
                {
                    "label": label,
                    "freq_mhz": freq,
                    "tp": tp,
                    "j_per_req_active": j_per_req,
                }
            )
        best = min(candidates, key=lambda r: r["j_per_req_active"])
        decode_records.append(best)

    return prefill_records, decode_records


def plot():
    prefill_records, decode_records = build_representative_alignment()
    c_counts, d_counts = load_cd_winner_counts()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    l40s_color = "#d55e00"
    l4_color = "#0072b2"
    prefill_freqs = [1245, 1500, 1755]
    ax.bar(
        range(len(prefill_freqs)),
        [c_counts.get(freq, 0) for freq in prefill_freqs],
        color=l40s_color,
        alpha=0.88,
        width=0.62,
        label="Prefill-only winner count on L40S",
    )
    for i, record in enumerate(prefill_records):
        x = prefill_freqs.index(record["freq_mhz"])
        ax.scatter(
            x,
            c_counts.get(record["freq_mhz"], 0) + 1.0 + 0.5 * i,
            marker="*",
            s=110,
            color="#7f0000",
            zorder=3,
        )
        ax.text(
            x,
            c_counts.get(record["freq_mhz"], 0) + 1.35 + 0.5 * i,
            f"{record['label']}\nTP{record['tp']}@{record['freq_mhz']}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#7f0000",
        )
    ax.set_xticks(range(len(prefill_freqs)))
    ax.set_xticklabels([str(freq) for freq in prefill_freqs])
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Number of workloads")
    ax.set_title("(a) Prefill-dominant cases align with the L40S prefill region")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    decode_freqs = [990, 1200]
    ax.bar(
        range(len(decode_freqs)),
        [d_counts.get(freq, 0) for freq in decode_freqs],
        color=l4_color,
        alpha=0.88,
        width=0.62,
        label="Decode-only winner count on L4",
    )
    for i, record in enumerate(decode_records):
        x = decode_freqs.index(record["freq_mhz"])
        ax.scatter(
            x,
            d_counts.get(record["freq_mhz"], 0) + 1.0 + 0.5 * i,
            marker="*",
            s=110,
            color="#08306b",
            zorder=3,
        )
        ax.text(
            x,
            d_counts.get(record["freq_mhz"], 0) + 1.35 + 0.5 * i,
            f"{record['label']}\nTP{record['tp']}@{record['freq_mhz']}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#08306b",
        )
    ax.set_xticks(range(len(decode_freqs)))
    ax.set_xticklabels([str(freq) for freq in decode_freqs])
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Number of workloads")
    ax.set_title("(b) Decode-dominant cases align with the L4 decode region")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(frameon=False, loc="upper left")

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
