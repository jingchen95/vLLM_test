#!/usr/bin/env python3
"""
Generate Phase B figures for the JSEP paper from simulation results.

Figures:
  5. Bar chart: energy savings per strategy across traces
  6. Time series: power over time for diurnal trace, all strategies
  7. HP/LP phase breakdown: stacked bar showing phase distribution

Usage:
    python plot_jsep.py
    python plot_jsep.py --slo 500
    python plot_jsep.py --summary jsep_results_summary.csv --detail jsep_results_detail.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from paths import PAPER_RESULTS_ANALYSES_DIR, PAPER_RESULTS_FIGURES_DIR

# ---------- style (matches plot_motivation.py) ----------
plt.rcParams.update({
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
})

FIG_DIR = PAPER_RESULTS_FIGURES_DIR

STRATEGY_ORDER = ["naive", "dvfs_only", "routing_only", "dynamollm", "jsep"]
STRATEGY_LABELS = {
    "naive": "Naive\n(All Max)",
    "dvfs_only": "DVFS\nOnly",
    "routing_only": "Routing\nOnly",
    "dynamollm": "DynamoLLM\n(Decoupled)",
    "jsep": "JSEP\n(Joint)",
}
STRATEGY_COLORS = {
    "naive": "#BDBDBD",
    "dvfs_only": "#90CAF9",
    "routing_only": "#A5D6A7",
    "dynamollm": "#FFCC80",
    "jsep": "#EF5350",
}
TRACE_LABELS = {"steady": "Steady", "bursty": "Bursty", "diurnal": "Diurnal"}


# ================================================================
# Figure 5: Energy savings bar chart — strategies × traces
# ================================================================
def fig5_energy_savings(df_summary, slo):
    """Grouped bar chart: energy savings vs naive for each strategy × trace."""
    sel = df_summary[df_summary["slo_ms"] == slo].copy()
    traces = [t for t in ["steady", "bursty", "diurnal"] if t in sel["trace"].values]
    strategies = [s for s in STRATEGY_ORDER if s != "naive" and s in sel["strategy"].values]

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(traces))
    n_strats = len(strategies)
    width = 0.8 / n_strats

    for i, strat in enumerate(strategies):
        savings = []
        for trace in traces:
            row = sel[(sel["strategy"] == strat) & (sel["trace"] == trace)]
            savings.append(row["saving_vs_naive_pct"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + (i - n_strats / 2 + 0.5) * width, savings, width,
                      color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                      edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, savings):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([TRACE_LABELS[t] for t in traces])
    ax.set_ylabel("Energy Saving vs Naive (%)")
    ax.set_title(f"Energy Savings by Strategy (SLO = {slo} ms)")
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(sel["saving_vs_naive_pct"].max() * 1.15, 10))

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_energy_savings_slo{slo}.pdf")
    fig.savefig(f"{FIG_DIR}/fig5_energy_savings_slo{slo}.png", dpi=300)
    plt.close(fig)
    print(f"  [OK] Figure 5: Energy savings (SLO={slo}ms)")


# ================================================================
# Figure 6: Power time series — diurnal trace, all strategies
# ================================================================
def fig6_power_timeseries(df_detail, slo, trace="diurnal"):
    """Power over time for a single trace, overlaying all strategies."""
    sel = df_detail[(df_detail["slo_ms"] == slo) & (df_detail["trace"] == trace)].copy()
    if sel.empty:
        print(f"  [SKIP] Figure 6: No data for {trace} at SLO={slo}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # --- Power over time ---
    line_styles = {
        "naive": ("-", 1.0, 0.4),
        "dvfs_only": ("-", 1.2, 0.6),
        "routing_only": ("-", 1.2, 0.7),
        "dynamollm": ("--", 1.5, 0.8),
        "jsep": ("-", 2.0, 1.0),
    }

    for strat in STRATEGY_ORDER:
        sub = sel[sel["strategy"] == strat].sort_values("window_start")
        ls, lw, alpha = line_styles.get(strat, ("-", 1.0, 0.7))
        ax1.plot(sub["window_start"] / 60, sub["power_w"],
                 color=STRATEGY_COLORS[strat], linestyle=ls, linewidth=lw,
                 alpha=alpha, label=STRATEGY_LABELS[strat].replace("\n", " "))

    ax1.set_ylabel("Cluster Power (W)")
    ax1.set_title(f"Power Over Time — {TRACE_LABELS[trace]} Trace (SLO = {slo} ms)")
    ax1.legend(loc="upper right", frameon=False, fontsize=7, ncol=3)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Request rate over time ---
    naive_sub = sel[sel["strategy"] == "naive"].sort_values("window_start")
    ax2.fill_between(naive_sub["window_start"] / 60, naive_sub["rate"],
                     alpha=0.3, color="#42A5F5")
    ax2.plot(naive_sub["window_start"] / 60, naive_sub["rate"],
             color="#1565C0", linewidth=1.0)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Rate (req/s)")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig6_power_timeseries_{trace}_slo{slo}.pdf")
    fig.savefig(f"{FIG_DIR}/fig6_power_timeseries_{trace}_slo{slo}.png", dpi=300)
    plt.close(fig)
    print(f"  [OK] Figure 6: Power time series ({trace}, SLO={slo}ms)")


# ================================================================
# Figure 7: HP/LP phase breakdown — stacked bars per strategy
# ================================================================
def fig7_phase_breakdown(df_summary, slo):
    """Stacked bar: fraction of windows in HP vs LP phase per strategy."""
    sel = df_summary[(df_summary["slo_ms"] == slo)].copy()
    traces = [t for t in ["steady", "bursty", "diurnal"] if t in sel["trace"].values]

    fig, axes = plt.subplots(1, len(traces), figsize=(2.5 * len(traces), 2.8),
                              sharey=True)
    if len(traces) == 1:
        axes = [axes]

    for ax, trace in zip(axes, traces):
        sub = sel[sel["trace"] == trace]
        strategies = [s for s in STRATEGY_ORDER if s in sub["strategy"].values]

        hp_fracs = []
        lp_fracs = []
        for strat in strategies:
            row = sub[sub["strategy"] == strat]
            hp = row["hp_frac_pct"].values[0] if len(row) > 0 else 0
            hp_fracs.append(hp)
            lp_fracs.append(100 - hp)

        x = np.arange(len(strategies))
        ax.bar(x, hp_fracs, color="#EF5350", label="HP", edgecolor="white", linewidth=0.5)
        ax.bar(x, lp_fracs, bottom=hp_fracs, color="#42A5F5", label="LP",
               edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in strategies],
                           fontsize=6, rotation=0)
        ax.set_title(TRACE_LABELS[trace], fontsize=9)
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Windows (%)")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle(f"HP/LP Phase Distribution (SLO = {slo} ms)", fontsize=10, y=1.02)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig7_phase_breakdown_slo{slo}.pdf")
    fig.savefig(f"{FIG_DIR}/fig7_phase_breakdown_slo{slo}.png", dpi=300)
    plt.close(fig)
    print(f"  [OK] Figure 7: Phase breakdown (SLO={slo}ms)")


# ================================================================
# Figure 8: Energy breakdown by GPU type (L40S vs L4)
# ================================================================
def fig8_gpu_energy_breakdown(df_detail, slo):
    """Stacked bar: energy contribution from L40S vs L4 per strategy."""
    sel = df_detail[df_detail["slo_ms"] == slo].copy()
    if "l40s_power" not in sel.columns or "l4_power" not in sel.columns:
        print(f"  [SKIP] Figure 8: No per-GPU power columns")
        return

    traces = [t for t in ["steady", "bursty", "diurnal"] if t in sel["trace"].values]
    strategies = [s for s in STRATEGY_ORDER if s in sel["strategy"].values]

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(strategies))
    width = 0.25

    for ti, trace in enumerate(traces):
        l40s_energy = []
        l4_energy = []
        for strat in strategies:
            sub = sel[(sel["trace"] == trace) & (sel["strategy"] == strat)]
            # Energy per window = power * 5s, sum up
            l40s_e = (sub["l40s_power"].fillna(0) * 5.0).sum() / 1000  # kJ
            l4_e = (sub["l4_power"].fillna(0) * 5.0).sum() / 1000
            l40s_energy.append(l40s_e)
            l4_energy.append(l4_e)

        offset = (ti - 1) * width
        ax.bar(x + offset, l40s_energy, width * 0.9, color="#EF5350",
               alpha=0.4 + 0.2 * ti, edgecolor="white", linewidth=0.5)
        ax.bar(x + offset, l4_energy, width * 0.9, bottom=l40s_energy,
               color="#42A5F5", alpha=0.4 + 0.2 * ti, edgecolor="white", linewidth=0.5,
               label=f"{TRACE_LABELS[trace]}" if ti == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=7)
    ax.set_ylabel("Energy (kJ)")
    ax.set_title(f"Energy by GPU Type (SLO = {slo} ms)")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EF5350", label="L40S"),
        Patch(facecolor="#42A5F5", label="L4"),
    ]
    for ti, trace in enumerate(traces):
        legend_elements.append(
            Patch(facecolor="gray", alpha=0.4 + 0.2 * ti,
                  label=TRACE_LABELS[trace])
        )
    ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig8_gpu_energy_slo{slo}.pdf")
    fig.savefig(f"{FIG_DIR}/fig8_gpu_energy_slo{slo}.png", dpi=300)
    plt.close(fig)
    print(f"  [OK] Figure 8: GPU energy breakdown (SLO={slo}ms)")


# ================================================================
# Summary table to console
# ================================================================
def print_summary_table(df_summary, slo):
    """Print a formatted summary table."""
    sel = df_summary[df_summary["slo_ms"] == slo]
    traces = [t for t in ["steady", "bursty", "diurnal"] if t in sel["trace"].values]

    print(f"\n{'='*70}")
    print(f"  SLO = {slo} ms")
    print(f"{'='*70}")
    print(f"{'Strategy':<16} ", end="")
    for t in traces:
        print(f"| {'Energy(kJ)':>10} {'Save%':>6} {'Viol%':>6} ", end="")
    print()
    print("-" * (18 + 26 * len(traces)))

    for strat in STRATEGY_ORDER:
        print(f"{strat:<16} ", end="")
        for trace in traces:
            row = sel[(sel["strategy"] == strat) & (sel["trace"] == trace)]
            if not row.empty:
                r = row.iloc[0]
                print(f"| {r['total_energy_kj']:>10.1f} "
                      f"{r['saving_vs_naive_pct']:>+5.1f}% "
                      f"{r['slo_violation_rate']:>5.1f}% ", end="")
            else:
                print(f"| {'N/A':>10} {'':>6} {'':>6} ", end="")
        print()


# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot JSEP Phase B results")
    parser.add_argument("--summary", default=str(PAPER_RESULTS_ANALYSES_DIR / "jsep_results_summary.csv"))
    parser.add_argument("--detail", default=str(PAPER_RESULTS_ANALYSES_DIR / "jsep_results_detail.csv"))
    parser.add_argument("--slo", type=int, action="append", default=None,
                        help="SLO threshold(s) in ms (can specify multiple)")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    df_summary = pd.read_csv(args.summary)
    df_detail = pd.read_csv(args.detail)

    slos = args.slo if args.slo else sorted(df_summary["slo_ms"].unique())

    print("Generating JSEP Phase B figures...")
    for slo in slos:
        print(f"\n--- SLO = {slo} ms ---")
        fig5_energy_savings(df_summary, slo)
        fig6_power_timeseries(df_detail, slo, trace="diurnal")
        fig7_phase_breakdown(df_summary, slo)
        fig8_gpu_energy_breakdown(df_detail, slo)
        print_summary_table(df_summary, slo)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
