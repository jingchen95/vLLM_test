#!/usr/bin/env python3
"""
plot_section45.py — Figure for SWEEP-LLM evaluation Section 4.5 (ablation).

Reads:
  results/paper/section45_ablation/summary.csv  (knob-removal ablations)
  results/paper/section42_sweep/summary.csv     (SWEEP-LLM full as reference)

Writes:
  results/paper/section45_ablation/figures/ablation_energy.{pdf,png}
  paper/figures/section45/ablation_energy.{pdf,png}    (mirror)

The figure is a single grouped-bar panel: x = ablation, four bars per group
(one per trace) showing percent energy increase over the full SWEEP-LLM at
moderate SLO, tau_kv = 16 us/tok.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

# Match the trace colors used in plot_section42.py for consistency.
TRACE_COLORS = {"T1": "#7fb2d3", "T2": "#8dd3c9", "T3": "#ffb55f", "T4": "#fc7f71"}
# Each label lists the *active* control dimensions:
#   R = class-aware routing, D = DVFS, T = tensor parallelism, C = active capacity.
# The full SWEEP-LLM (R+D+T+C) is the normalization baseline for all ratios.
ABLATION_ORDER = [
    "sweep_llm_no_routing",
    "sweep_llm_no_dvfs",
    "sweep_llm_no_tp",
    "sweep_llm_no_capacity",
    "sweep_llm",          # full SWEEP-LLM as the last group, normalised to 1
]
ABLATION_LABELS = {
    "sweep_llm_no_routing":  "D+T+C",
    "sweep_llm_no_dvfs":     "R+T+C",
    "sweep_llm_no_tp":       "R+D+C",
    "sweep_llm_no_capacity": "R+D+T",
    "sweep_llm":             "R+D+T+C\n(Full)",
}

ROOT = Path(__file__).resolve().parent
PAPER_FIGURES_DIR = ROOT / "paper" / "figures" / "section45"


def load_summary(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "PASS":
                continue
            try:
                rows.append({
                    "trace": row["trace"],
                    "strategy": row["strategy"],
                    "slo_ttft_ms": int(row["slo_ttft_ms"]),
                    "tau_kv_us": float(row["tau_kv_us_per_tok"]),
                    "energy_j": float(row["total_energy_j"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def lookup_one(rows, **filters):
    for r in rows:
        if all(r[k] == v for k, v in filters.items()):
            return r
    return None


SLO_LABELS = {200: "Tight (200/100 ms)",
              500: "Moderate (500/200 ms)",
              1000: "Loose (1000/400 ms)"}


def _energy_ratio(abl_rows, full_rows, slo_ms, tau_kv_us, traces):
    """Return {trace: {strategy: energy / full_sweep_llm_energy}}.

    Full SWEEP-LLM is sourced from ``full_rows`` (the §4.2 sweep) and
    normalises to 1.0. Other strategies come from ``abl_rows``.
    """
    out: Dict[str, Dict[str, float]] = {t: {} for t in traces}
    for trace in traces:
        full = lookup_one(full_rows, trace=trace, strategy="sweep_llm",
                          slo_ttft_ms=slo_ms, tau_kv_us=tau_kv_us)
        if full is None or full["energy_j"] <= 0:
            continue
        full_e = full["energy_j"]
        out[trace]["sweep_llm"] = 1.0
        for strat in ABLATION_ORDER:
            if strat == "sweep_llm":
                continue
            row = lookup_one(abl_rows, trace=trace, strategy=strat,
                             slo_ttft_ms=slo_ms, tau_kv_us=tau_kv_us)
            if row is None:
                continue
            out[trace][strat] = row["energy_j"] / full_e
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-summary", default="results/paper/section45_ablation/summary.csv")
    ap.add_argument("--full-summary", default="results/paper/section42_sweep/summary.csv")
    ap.add_argument("--slos", default="200,500,1000",
                    help="Comma-separated SLO levels (ms TTFT) to plot, one panel each.")
    ap.add_argument("--tau-kv-us", type=float, default=16.0)
    ap.add_argument("--output-dir", default="results/paper/section45_ablation/figures")
    args = ap.parse_args()

    abl_rows  = load_summary(Path(args.ablation_summary))
    full_rows = load_summary(Path(args.full_summary))
    if not abl_rows:
        print(f"WARNING: no ablation rows in {args.ablation_summary}", file=sys.stderr)
        sys.exit(1)

    slos = [int(s) for s in args.slos.split(",") if s.strip()]
    traces = sorted({r["trace"] for r in abl_rows})

    # Compute energy ratios (full SWEEP-LLM = 1) per SLO panel; share y-axis.
    panels: Dict[int, Dict[str, Dict[str, float]]] = {}
    for slo in slos:
        panels[slo] = _energy_ratio(abl_rows, full_rows, slo, args.tau_kv_us, traces)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_slo = len(slos)
    fig, axes = plt.subplots(1, n_slo, figsize=(3.4 * n_slo, 2.5),
                             sharey=True, squeeze=False)
    n_traces = len(traces)
    bar_width = 0.78 / max(n_traces, 1)
    x = np.arange(len(ABLATION_ORDER))

    # Pre-compute global y-range with headroom for labels
    all_vals = [v for slo in slos for d in panels[slo].values() for v in d.values()]
    if all_vals:
        ymax = max(all_vals)
        ymax = ymax * 1.15
    else:
        ymax = 1.5

    for j, slo in enumerate(slos):
        ax = axes[0][j]
        ratios = panels[slo]
        for i, trace in enumerate(traces):
            ys = [ratios.get(trace, {}).get(s, np.nan) for s in ABLATION_ORDER]
            offset = (i - n_traces / 2.0 + 0.5) * bar_width
            ax.bar(x + offset, ys, bar_width,
                   label=trace, color=TRACE_COLORS.get(trace, "#444"),
                   edgecolor="none")
            for xi, yi in zip(x + offset, ys):
                if np.isnan(yi):
                    continue
                # Skip labels for ratios essentially equal to 1 — the
                # dashed reference line already conveys the baseline,
                # and overlapping "1.00" labels are unreadable.
                if abs(yi - 1.0) < 0.02:
                    continue
                ax.text(xi, yi + ymax * 0.012, f"{yi:.2f}",
                        ha="center", va="bottom", rotation=90,
                        fontsize=6.5, color="black")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_LABELS[s] for s in ABLATION_ORDER],
                           rotation=0, ha="center", fontsize=7.5)
        ax.set_title(SLO_LABELS.get(slo, f"SLO = {slo} ms"))
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if j == 0:
            ax.set_ylabel("Normalized energy")

    # Shared legend at the top
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(traces),
                   bbox_to_anchor=(0.5, 1.05), frameon=False)

    fig.tight_layout()
    out_stem = out_dir / "ablation_energy"
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_stem.with_suffix('.pdf')}")

    for ext in (".pdf", ".png"):
        shutil.copy2(out_stem.with_suffix(ext),
                     PAPER_FIGURES_DIR / f"ablation_energy{ext}")
    print(f"  mirrored to {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
