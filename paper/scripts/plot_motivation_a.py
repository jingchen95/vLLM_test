#!/usr/bin/env python3
"""Generate a motivation figure from Experiment A results.

Figure: energy per output token versus SM frequency for representative workloads,
with one line per TP degree. This is used to show that no single monolithic
configuration is universally optimal even on one GPU type.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT


RESULT_DIR = REPO_ROOT / "results" / "disagg_20260321_v2" / "A_monolithic_l40s"
FIG_DIR = PAPER_RESULTS_FIGURES_DIR
OUT_BASENAME = "figA_no_universal_optimum"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def parse_txt_results():
    metrics = {}
    pattern = re.compile(r"bench_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r(\d+)\.txt")
    for path in glob.glob(os.path.join(RESULT_DIR, "*.txt")):
        m = pattern.match(os.path.basename(path))
        if not m:
            continue
        freq, tp, il, ol, rate = map(int, m.groups())
        text = open(path).read()

        def get(label: str):
            mm = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
            return float(mm.group(1)) if mm else None

        metrics[(freq, tp, il, ol, rate)] = {
            "req_tput": get("Request throughput (req/s)"),
            "tok_tput": get("Output token throughput (tok/s)"),
            "mean_ttft": get("Mean TTFT (ms)"),
            "p99_ttft": get("P99 TTFT (ms)"),
            "mean_tpot": get("Mean TPOT (ms)"),
        }
    return metrics


def parse_monitor_power():
    powers = defaultdict(list)
    pattern = re.compile(
        r"monitor_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r(\d+)_gpu(\d+)\.csv"
    )
    for path in glob.glob(os.path.join(RESULT_DIR, "monitor_*.csv")):
        m = pattern.match(os.path.basename(path))
        if not m:
            continue
        freq, tp, il, ol, rate, _gpu = map(int, m.groups())
        vals = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                power = row.get("power_w")
                if power not in ("", None, "None"):
                    vals.append(float(power))
        if vals:
            powers[(freq, tp, il, ol, rate)].append(sum(vals) / len(vals))
    return {k: sum(v) for k, v in powers.items()}


def build_energy_table():
    metrics = parse_txt_results()
    cluster_power = parse_monitor_power()
    table = {}
    for key, metric in metrics.items():
        if key not in cluster_power or not metric["tok_tput"]:
            continue
        table[key] = {
            **metric,
            "cluster_power_w": cluster_power[key],
            "j_per_tok": cluster_power[key] / metric["tok_tput"],
        }
    return table


def plot():
    table = build_energy_table()
    workloads = [
        (1024, 128, 10, "Prefill-heavy\n(1024,128,r=10)"),
        (128, 128, 20, "Balanced short\n(128,128,r=20)"),
        (128, 1024, 10, "Decode-heavy\n(128,1024,r=10)"),
        (64, 2048, 10, "Long decode\n(64,2048,r=10)"),
    ]
    tp_colors = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
    tp_markers = {1: "o", 2: "s", 4: "D"}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (il, ol, rate, title) in zip(axes, workloads):
        subset = {
            key: val
            for key, val in table.items()
            if key[2] == il and key[3] == ol and key[4] == rate
        }
        for tp in sorted({key[1] for key in subset}):
            pts = sorted(
                [(key[0], val["j_per_tok"]) for key, val in subset.items() if key[1] == tp]
            )
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            ax.plot(
                xs,
                ys,
                marker=tp_markers[tp],
                linewidth=1.5,
                markersize=4,
                color=tp_colors[tp],
                label=f"TP{tp}",
            )

        best_key, best_val = min(subset.items(), key=lambda kv: kv[1]["j_per_tok"])
        ax.scatter(
            [best_key[0]],
            [best_val["j_per_tok"]],
            color="black",
            s=20,
            zorder=5,
        )
        ax.annotate(
            f"best: TP{best_key[1]} @ {best_key[0]}",
            xy=(best_key[0], best_val["j_per_tok"]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=7,
        )
        ax.set_title(title)
        ax.set_xlabel("SM frequency (MHz)")
        ax.set_ylabel("Energy / output token (J)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{OUT_BASENAME}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{OUT_BASENAME}.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot()
    print(f"[OK] Wrote {OUT_BASENAME}.pdf/.png")
