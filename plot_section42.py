#!/usr/bin/env python3
"""
plot_section42.py — Figures for SWEEP-LLM evaluation Section 4.2.

Reads results/paper/section42_sweep/summary.csv (produced by
run_section42_sweep.py) and writes:

  results/paper/section42_sweep/figures/
    - energy_by_strategy_per_trace.{pdf,png}
        2x2 grid (T1..T4); per-panel grouped bars: x = strategy,
        hue = SLO. Shows the headline energy comparison at one fixed tau_kv.
        Bars are annotated above with their SLO violation rate when > 0.5%,
        which preserves per-SLO detail that an averaged heatmap would lose.
    - tau_kv_sensitivity.{pdf,png}
        SWEEP-LLM total energy vs tau_kv across the four traces at one
        fixed SLO. Shows interconnect-assumption robustness.

Tolerant of partial summary.csv (use NaN for missing combinations) so it can
be invoked while the sweep is still running.

Usage:
  python plot_section42.py
  python plot_section42.py --tau-kv-fixed 5 --slo-fixed 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# After saving figures under results/, mirror the paper-bound versions
# into paper/figures/section42/ so LaTeX includes always see the latest.
_PAPER_FIGURES_SECTION42 = (
    Path(__file__).resolve().parent / "paper" / "figures" / "section42"
)


def _mirror_to_paper(stem: Path) -> None:
    _PAPER_FIGURES_SECTION42.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        src = stem.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, _PAPER_FIGURES_SECTION42 / src.name)

# Match the rcParams used by the motivation figures (paper/scripts/plot_motivation.py)
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


STRATEGY_ORDER = ["static_disagg", "greenllm", "dynamollm", "sweep_llm"]
STRATEGY_LABELS = {
    "static_disagg": "Static-disagg",
    "greenllm":      "GreenLLM",
    "dynamollm":     "DynamoLLM",
    "sweep_llm":     "SWEEP-LLM",
}
STRATEGY_COLORS = {
    "static_disagg": "#888888",
    "greenllm":      "#41a060",
    "dynamollm":     "#d57733",
    "sweep_llm":     "#3070b8",
}
# ColorBrewer Blues 3-class: light->dark indicates tight->loose SLO,
# matching the natural ordering of SLO permissiveness.
SLO_COLORS = {200: "#9ecae1", 500: "#3182bd", 1000: "#08519c"}
# Named SLO regimes from the experimental setup (paper Section 4.1).
SLO_NAMES = {200: "Tight", 500: "Moderate", 1000: "Loose"}
TRACE_MARKERS = {"T1": "o", "T2": "s", "T3": "^", "T4": "D"}
# Distinct trace colors for tau_kv-sensitivity panels (Tableau-style).
TRACE_COLORS = {"T1": "#1f77b4", "T2": "#ff7f0e", "T3": "#2ca02c", "T4": "#d62728"}


def load_summary(path: Path) -> List[dict]:
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
                    "violation_pct": float(row["slo_violation_rate_pct"]),
                    "mean_power_w": float(row["mean_power_w"]),
                    "wall_clock_s": float(row["wall_clock_s"]) if row.get("wall_clock_s") else 0.0,
                })
            except (KeyError, ValueError):
                continue
    return rows


def lookup(rows: List[dict], **filters):
    """Return the unique row matching all filters, or None."""
    matches = [r for r in rows if all(r[k] == v for k, v in filters.items())]
    if not matches:
        return None
    if len(matches) > 1:
        # Most-recent wins (last appended). Useful when the harness was rerun.
        return matches[-1]
    return matches[0]


def plot_energy_per_trace(rows: List[dict], traces: List[str], strategies: List[str],
                          slos: List[int], tau_kv_fixed: float, out_stem: Path) -> None:
    n_traces = len(traces)
    # 1 x N row layout
    fig, axes = plt.subplots(1, n_traces, figsize=(3.6 * n_traces, 2.6),
                             squeeze=False, sharey=True)

    n_slo = len(slos)
    bar_total = 0.78
    bar_width = bar_total / max(n_slo, 1)
    x = np.arange(len(strategies))

    # Pre-compute global y-limit so violation labels have headroom
    all_energies = []
    for t in traces:
        for slo in slos:
            for s in strategies:
                row = lookup(rows, trace=t, strategy=s,
                             slo_ttft_ms=slo, tau_kv_us=tau_kv_fixed)
                if row is not None:
                    all_energies.append(row["energy_j"] / 1000.0)
    ymax = max(all_energies) * 1.15 if all_energies else 1.0
    label_offset = ymax * 0.012

    for i, trace in enumerate(traces):
        ax = axes[0][i]
        any_data = False
        for j, slo in enumerate(slos):
            energies = []
            violations = []
            for s in strategies:
                row = lookup(rows, trace=trace, strategy=s,
                             slo_ttft_ms=slo, tau_kv_us=tau_kv_fixed)
                if row is None:
                    energies.append(np.nan)
                    violations.append(np.nan)
                else:
                    energies.append(row["energy_j"] / 1000.0)
                    violations.append(row["violation_pct"])
                    any_data = True
            offset = (j - n_slo / 2.0 + 0.5) * bar_width
            bars = ax.bar(x + offset, energies, bar_width,
                          label=SLO_NAMES.get(slo, f"SLO={slo}"),
                          color=SLO_COLORS.get(slo, "gray"),
                          edgecolor="none")
            for bar, viol in zip(bars, violations):
                if not np.isnan(viol) and viol > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + label_offset,
                            f"{viol:.0f}%", ha="center", va="bottom",
                            fontsize=8, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=15, ha="right")
        if i == 0:
            ax.set_ylabel("Total energy (kJ)")
        ax.set_title(f"{trace}")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if not any_data:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    # Legend inside the first panel; saves the vertical space that a
    # figure-level legend or a suptitle would consume.
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(loc="upper right", frameon=True, fontsize=7,
                          framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"  wrote {out_stem.with_suffix('.pdf')}")
    plt.close(fig)
    _mirror_to_paper(out_stem)


def plot_tau_kv_sensitivity(rows: List[dict], traces: List[str], tau_kvs: List[float],
                            slos: List[int], out_stem: Path) -> None:
    """1xN row of panels (one per SLO). Each panel shows SWEEP-LLM total energy
    as a function of tau_kv across the four traces.

    The figure answers two questions in one view:
      - Is SWEEP-LLM robust to the assumed interconnect cost?
      - Does that robustness vary with the SLO regime?
    """
    n_slo = len(slos)
    fig, axes = plt.subplots(1, n_slo, figsize=(3.4 * n_slo, 3.4),
                             squeeze=False, sharey=True)

    # Pre-compute global y-range so panels share a scale and visual
    # comparison across SLO is fair.
    all_energies = []
    for trace in traces:
        for slo in slos:
            for kv in tau_kvs:
                row = lookup(rows, trace=trace, strategy="sweep_llm",
                             slo_ttft_ms=slo, tau_kv_us=kv)
                if row is not None:
                    all_energies.append(row["energy_j"] / 1000.0)
    ymax = max(all_energies) * 1.10 if all_energies else 1.0
    ymin = max(0.0, min(all_energies) * 0.85) if all_energies else 0.0

    any_data_overall = False
    for i, slo in enumerate(slos):
        ax = axes[0][i]
        any_data = False
        for trace in traces:
            ys = []
            for kv in tau_kvs:
                row = lookup(rows, trace=trace, strategy="sweep_llm",
                             slo_ttft_ms=slo, tau_kv_us=kv)
                ys.append(row["energy_j"] / 1000.0 if row else np.nan)
                if row:
                    any_data = True
                    any_data_overall = True
            ax.plot(tau_kvs, ys,
                    marker=TRACE_MARKERS.get(trace, "o"),
                    color=TRACE_COLORS.get(trace, "#444444"),
                    label=trace, linewidth=1.8, markersize=7)

        ax.set_xlabel(r"$\tau_{kv}$ (µs / token)")
        if i == 0:
            ax.set_ylabel("Total energy (kJ)")
        regime = {200: "tight", 500: "moderate", 1000: "loose"}.get(slo, "")
        title = f"SLO = {slo} ms" + (f" ({regime})" if regime else "")
        ax.set_title(title)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(tau_kvs)
        ax.grid(alpha=0.3, linewidth=0.5)
        if not any_data:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    # Single shared legend at the top
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(traces),
                   bbox_to_anchor=(0.5, 1.04), frameon=False)

    if not any_data_overall:
        pass

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"  wrote {out_stem.with_suffix('.pdf')}")
    plt.close(fig)
    _mirror_to_paper(out_stem)


def plot_violation_heatmap(rows: List[dict], traces: List[str], strategies: List[str],
                           out_stem: Path) -> None:
    """Mean violation rate per (trace, strategy) across all (SLO, tau_kv) combos."""
    grid = np.full((len(strategies), len(traces)), np.nan)
    for i, s in enumerate(strategies):
        for j, t in enumerate(traces):
            sub = [r["violation_pct"] for r in rows
                   if r["strategy"] == s and r["trace"] == t]
            if sub:
                grid[i, j] = float(np.mean(sub))

    if np.all(np.isnan(grid)) or np.nanmax(grid) <= 0.5:
        return  # nothing interesting to plot

    fig, ax = plt.subplots(figsize=(1.4 * len(traces) + 2, 0.5 * len(strategies) + 2))
    im = ax.imshow(grid, cmap="OrRd", vmin=0, vmax=max(np.nanmax(grid), 1.0),
                   aspect="auto")
    ax.set_xticks(np.arange(len(traces)))
    ax.set_xticklabels(traces)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in strategies])
    for i in range(len(strategies)):
        for j in range(len(traces)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9,
                        color="white" if v > 5 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean SLO violation rate (%)")
    ax.set_title("SLO violations averaged over (SLO, $\\tau_{kv}$) settings")

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=160)
    print(f"  wrote {out_stem.with_suffix('.pdf')}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results/paper/section42_sweep/summary.csv")
    ap.add_argument("--output-dir", default="results/paper/section42_sweep/figures")
    ap.add_argument("--tau-kv-fixed", type=float, default=16.0,
                    help="τ_kv used for the per-trace energy figure (µs/tok)")
    ap.add_argument("--slo-fixed", type=int, default=500,
                    help="SLO used for the τ_kv sensitivity figure (ms)")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    rows = load_summary(summary_path)
    if not rows:
        print(f"WARNING: no PASS rows in {summary_path}; nothing to plot.",
              file=sys.stderr)
        sys.exit(0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces_present = sorted({r["trace"] for r in rows})
    strategies = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in rows)]
    slos = sorted({r["slo_ttft_ms"] for r in rows})
    tau_kvs = sorted({r["tau_kv_us"] for r in rows})

    print(f"Loaded {len(rows)} PASS rows from {summary_path}")
    print(f"  traces:     {traces_present}")
    print(f"  strategies: {strategies}")
    print(f"  SLOs:       {slos}")
    print(f"  τ_kv:        {tau_kvs}")
    print()

    plot_energy_per_trace(rows, traces_present, strategies, slos,
                          args.tau_kv_fixed,
                          out_dir / "energy_by_strategy_per_trace")
    plot_tau_kv_sensitivity(rows, traces_present, tau_kvs, slos,
                            out_dir / "tau_kv_sensitivity")


if __name__ == "__main__":
    main()
