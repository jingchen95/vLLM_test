#!/usr/bin/env python3
"""
Generate Observation 1 motivation assets from Experiment A.

Outputs:
  - figA_no_universal_optimum.pdf
  - figA_no_universal_optimum.png
  - figA_no_universal_optimum_summary.csv
  - figA_no_universal_optimum_summary.json
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from paths import PAPER_RESULTS_FIGURES_DIR, REPO_ROOT

THIS_DIR = os.fspath(PAPER_RESULTS_FIGURES_DIR)
REPO_DIR = os.fspath(REPO_ROOT)
RESULT_DIR = os.path.join(REPO_DIR, "results", "disagg_20260321_v2", "A_monolithic_l40s")
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


WORKLOADS = [
    (1024, 128, 10, "Prefill-heavy\n(1024,128,r=10)"),
    (128, 128, 20, "Balanced short\n(128,128,r=20)"),
    (128, 1024, 10, "Decode-heavy\n(128,1024,r=10)"),
    (64, 2048, 10, "Very long decode\n(64,2048,r=10)"),
]

TP_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
TP_MARKERS = {1: "o", 2: "s", 4: "D"}


def parse_txt_results():
    metrics = {}
    pattern = re.compile(r"bench_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r(\d+)\.txt$")
    for path in glob.glob(os.path.join(RESULT_DIR, "*.txt")):
        m = pattern.match(os.path.basename(path))
        if not m:
            continue
        freq, tp, il, ol, rate = map(int, m.groups())
        text = open(path, encoding="utf-8").read()

        def get(label: str):
            mm = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", text)
            return float(mm.group(1)) if mm else None

        metrics[(freq, tp, il, ol, rate)] = {
            "successful_requests": get("Successful requests"),
            "failed_requests": get("Failed requests"),
            "benchmark_duration_s": get("Benchmark duration (s)"),
            "req_tput": get("Request throughput (req/s)"),
            "tok_tput": get("Output token throughput (tok/s)"),
            "mean_ttft": get("Mean TTFT (ms)"),
            "p99_ttft": get("P99 TTFT (ms)"),
            "mean_tpot": get("Mean TPOT (ms)"),
        }
    return metrics


def _sample_is_active(sample):
    return sample["gpu_util_pct"] > 0 or sample["mem_util_pct"] > 0


def _sample_is_hi(sample):
    return sample["gpu_util_pct"] >= 90 or sample["mem_util_pct"] >= 95


def _active_energy_from_samples(samples):
    if not samples:
        return None, None, None

    hi = [i for i, s in enumerate(samples) if _sample_is_hi(s)]
    nz = [i for i, s in enumerate(samples) if _sample_is_active(s)]
    if not nz:
        return None, None, None

    start = hi[0] if hi else nz[0]
    end = hi[-1] if hi else nz[-1]

    while start > 0 and _sample_is_active(samples[start - 1]):
        start -= 1
    while end + 1 < len(samples) and _sample_is_active(samples[end + 1]):
        end += 1

    start_energy_idx = max(start - 1, 0)
    end_energy_idx = min(end + 1, len(samples) - 1)
    start_energy = samples[start_energy_idx]["total_energy_mj"]
    end_energy = samples[end_energy_idx]["total_energy_mj"]
    if start_energy is None or end_energy is None:
        return None, start, end
    return (end_energy - start_energy) / 1000.0, start, end


def parse_monitor_stats():
    powers = defaultdict(list)
    energies = defaultdict(list)
    pattern = re.compile(
        r"monitor_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r(\d+)_gpu(\d+)\.csv$"
    )
    for path in glob.glob(os.path.join(RESULT_DIR, "monitor_*.csv")):
        m = pattern.match(os.path.basename(path))
        if not m:
            continue
        freq, tp, il, ol, rate, _gpu = map(int, m.groups())
        samples = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                power = row.get("power_w")
                energy = row.get("total_energy_mj")
                gpu_util = row.get("gpu_util_pct")
                mem_util = row.get("mem_util_pct")
                if power in ("", None, "None"):
                    continue
                samples.append({
                    "power_w": float(power),
                    "total_energy_mj": float(energy) if energy not in ("", None, "None") else None,
                    "gpu_util_pct": float(gpu_util) if gpu_util not in ("", None, "None") else 0.0,
                    "mem_util_pct": float(mem_util) if mem_util not in ("", None, "None") else 0.0,
                })
        if samples:
            powers[(freq, tp, il, ol, rate)].append(sum(s["power_w"] for s in samples) / len(samples))
            active_energy_j, _start, _end = _active_energy_from_samples(samples)
            if active_energy_j is not None:
                energies[(freq, tp, il, ol, rate)].append(active_energy_j)
    return (
        {k: sum(v) for k, v in powers.items()},
        {k: sum(v) for k, v in energies.items()},
    )


def build_energy_table():
    metrics = parse_txt_results()
    cluster_power, cluster_active_energy = parse_monitor_stats()
    table = {}
    for key, metric in metrics.items():
        tok_tput = metric["tok_tput"]
        if key not in cluster_power or not tok_tput:
            continue
        successful_requests = metric["successful_requests"]
        active_energy_j = cluster_active_energy.get(key)
        table[key] = {
            **metric,
            "cluster_power_w": cluster_power[key],
            "cluster_active_energy_j": active_energy_j,
            "req_tput_from_counts": (
                successful_requests / metric["benchmark_duration_s"]
                if successful_requests and metric["benchmark_duration_s"] else None
            ),
            "j_per_tok": cluster_power[key] / tok_tput,
            "j_per_req": cluster_power[key] / metric["req_tput"],
            "j_per_req_active": (
                active_energy_j / successful_requests
                if active_energy_j is not None and successful_requests else None
            ),
        }
    return table


def export_summary(table):
    rows = []
    summary = {}
    for il, ol, rate, title in WORKLOADS:
        subset = {
            key: val for key, val in table.items()
            if key[2] == il and key[3] == ol and key[4] == rate
        }
        if not subset:
            continue
        best_energy_key, best_energy_val = min(subset.items(), key=lambda kv: kv[1]["j_per_req_active"])
        best_tput_key, best_tput_val = max(subset.items(), key=lambda kv: kv[1]["req_tput"])
        summary[title] = {
            "best_energy": {
                "freq_mhz": best_energy_key[0],
                "tp": best_energy_key[1],
                **best_energy_val,
            },
            "best_throughput": {
                "freq_mhz": best_tput_key[0],
                "tp": best_tput_key[1],
                **best_tput_val,
            },
        }
        rows.append({
            "workload": title.replace("\n", " "),
            "input_len": il,
            "output_len": ol,
            "request_rate": rate,
            "best_energy_freq_mhz": best_energy_key[0],
            "best_energy_tp": best_energy_key[1],
            "best_energy_active_energy_j": round(best_energy_val["cluster_active_energy_j"], 6),
            "best_energy_j_per_req": round(best_energy_val["j_per_req"], 6),
            "best_energy_j_per_req_active": round(best_energy_val["j_per_req_active"], 6),
            "best_energy_j_per_tok": round(best_energy_val["j_per_tok"], 6),
            "best_energy_req_tput": round(best_energy_val["req_tput"], 4),
            "best_energy_req_tput_from_counts": round(best_energy_val["req_tput_from_counts"], 4),
            "best_energy_mean_ttft_ms": round(best_energy_val["mean_ttft"], 2),
            "best_throughput_freq_mhz": best_tput_key[0],
            "best_throughput_tp": best_tput_key[1],
            "best_throughput_active_energy_j": round(best_tput_val["cluster_active_energy_j"], 6),
            "best_throughput_j_per_req": round(best_tput_val["j_per_req"], 6),
            "best_throughput_j_per_req_active": round(best_tput_val["j_per_req_active"], 6),
            "best_throughput_j_per_tok": round(best_tput_val["j_per_tok"], 6),
            "best_throughput_req_tput": round(best_tput_val["req_tput"], 4),
            "best_throughput_req_tput_from_counts": round(best_tput_val["req_tput_from_counts"], 4),
            "best_throughput_mean_ttft_ms": round(best_tput_val["mean_ttft"], 2),
        })

    csv_path = os.path.join(THIS_DIR, f"{OUT_BASENAME}_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(THIS_DIR, f"{OUT_BASENAME}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return rows, summary


def plot(table):
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 2.9), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (il, ol, rate, title) in zip(axes, WORKLOADS):
        subset = {
            key: val for key, val in table.items()
            if key[2] == il and key[3] == ol and key[4] == rate
        }
        if not subset:
            ax.set_axis_off()
            continue

        tps = sorted({key[1] for key in subset})
        for tp in tps:
            pts = sorted(
                (val["req_tput"], val["j_per_req_active"], key[0])
                for key, val in subset.items() if key[1] == tp
            )
            xs = [x for x, _, _ in pts]
            ys = [y for _, y, _ in pts]
            ax.plot(
                xs,
                ys,
                marker=TP_MARKERS.get(tp, "o"),
                linewidth=1.2,
                markersize=4.5,
                color=TP_COLORS.get(tp, "#444444"),
                alpha=0.85,
                label=f"TP{tp}" if ax is axes[0] else None,
            )

        best_energy_key, best_energy_val = min(subset.items(), key=lambda kv: kv[1]["j_per_req_active"])
        best_tput_key, best_tput_val = max(subset.items(), key=lambda kv: kv[1]["req_tput"])
        ax.scatter(
            [best_energy_val["req_tput"]],
            [best_energy_val["j_per_req_active"]],
            facecolor="none",
            edgecolor="black",
            s=42,
            linewidth=1.1,
            zorder=6,
        )
        ax.scatter(
            [best_tput_val["req_tput"]],
            [best_tput_val["j_per_req_active"]],
            facecolor="none",
            edgecolor="#555555",
            s=42,
            linewidth=1.1,
            zorder=6,
        )
        ax.annotate(
            f"E*: TP{best_energy_key[1]}@{best_energy_key[0]}",
            (best_energy_val["req_tput"], best_energy_val["j_per_req_active"]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=6.5,
            color="black",
        )
        ax.annotate(
            f"T*: TP{best_tput_key[1]}@{best_tput_key[0]}",
            (best_tput_val["req_tput"], best_tput_val["j_per_req_active"]),
            textcoords="offset points",
            xytext=(-2, 8),
            fontsize=6.5,
            color="#444444",
        )

        ax.set_title(title)
        ax.set_xlabel("Throughput (req/s)")
        if ax is axes[0]:
            ax.set_ylabel("Energy / request (J)")
        ax.grid(alpha=0.3, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)

    pdf_path = os.path.join(THIS_DIR, f"{OUT_BASENAME}.pdf")
    png_path = os.path.join(THIS_DIR, f"{OUT_BASENAME}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main():
    table = build_energy_table()
    rows, summary = export_summary(table)
    pdf_path, png_path = plot(table)
    print(f"[OK] Wrote {pdf_path}")
    print(f"[OK] Wrote {png_path}")
    print(f"[OK] Wrote {os.path.join(THIS_DIR, f'{OUT_BASENAME}_summary.csv')}")
    for row in rows:
        print(
            f"[SUMMARY] {row['workload']}: "
            f"best-energy TP{row['best_energy_tp']}@{row['best_energy_freq_mhz']} "
            f"({row['best_energy_j_per_req_active']:.2f} J/request), "
            f"best-throughput TP{row['best_throughput_tp']}@{row['best_throughput_freq_mhz']} "
            f"({row['best_throughput_req_tput']:.2f} req/s)"
        )


if __name__ == "__main__":
    main()
