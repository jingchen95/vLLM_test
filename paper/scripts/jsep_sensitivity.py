"""
JSEP Sensitivity Analysis (C) and Energy-Delay Product Analysis (D).

C: Sweep HP threshold to show how phase classification affects savings.
D: Compute EDP (Energy × Delay) for each strategy to quantify trade-offs.

Usage:
    python jsep_sensitivity.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import PAPER_RESULTS_ANALYSES_DIR, PAPER_RESULTS_FIGURES_DIR, paper_model_dir
from jsep_traces import generate_steady, generate_bursty, generate_diurnal
from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import create_all_strategies, JSEPStrategy
from jsep_simulator import simulate_trace, summarize_windows, WINDOW_S

FIG_DIR = str(PAPER_RESULTS_FIGURES_DIR)

# Consistent styling
COLORS = {
    "naive": "#999999",
    "dvfs_only": "#5DA5DA",
    "routing_only": "#FAA43A",
    "dynamollm": "#60BD68",
    "jsep": "#F15854",
}
LABELS = {
    "naive": "Naive",
    "dvfs_only": "DVFS-Only",
    "routing_only": "Routing-Only",
    "dynamollm": "DynamoLLM",
    "jsep": "JSEP",
}


def run_threshold_sweep(slo=500, duration_s=600.0, seed=42):
    """Sweep HP threshold for JSEP and measure energy savings."""
    traces = {
        "steady": generate_steady(duration_s, rate_rps=10.0, seed=seed),
        "bursty": generate_bursty(duration_s, seed=seed),
        "diurnal": generate_diurnal(duration_s, seed=seed),
    }

    strategies = create_all_strategies(
        str(paper_model_dir("models_l40s")),
        str(paper_model_dir("models_l4")),
    )

    # Get naive energy for normalization
    naive_energy = {}
    for trace_name, trace in traces.items():
        windows = simulate_trace(trace, strategies["naive"], "naive", slo)
        summary = summarize_windows(windows, "naive", trace_name, slo)
        naive_energy[trace_name] = summary["total_energy_kj"]

    # Sweep HP thresholds
    hp_thresholds = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40]
    results = []

    for hp_th in hp_thresholds:
        lp_th = hp_th * 0.5  # Keep ratio constant
        print(f"  HP threshold = {hp_th} req/s (LP = {lp_th})...")

        # Override JSEP thresholds by temporarily replacing _get_thresholds
        jsep = strategies["jsep"]
        jsep._override_thresholds = (hp_th, lp_th)

        for trace_name, trace in traces.items():
            jsep.current_phase = "LP"
            jsep.last_reconfig_time = -999.0
            jsep.prev_tp = {"l40s": 1, "l4": 1}

            windows = simulate_trace(trace, jsep, "jsep", slo)
            summary = summarize_windows(windows, "jsep", trace_name, slo)
            saving = (1 - summary["total_energy_kj"] / naive_energy[trace_name]) * 100

            results.append({
                "hp_threshold": hp_th,
                "trace": trace_name,
                "saving_pct": round(saving, 1),
                "slo_violation_rate": summary["slo_violation_rate"],
                "hp_frac_pct": summary["hp_frac_pct"],
                "total_energy_kj": summary["total_energy_kj"],
            })

    return pd.DataFrame(results)


def plot_threshold_sensitivity(df, slo):
    """Plot energy savings vs HP threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    trace_colors = {"steady": "#5DA5DA", "bursty": "#FAA43A", "diurnal": "#60BD68"}
    trace_markers = {"steady": "o", "bursty": "s", "diurnal": "^"}

    for trace_name in ["steady", "bursty", "diurnal"]:
        subset = df[df["trace"] == trace_name]
        ax1.plot(subset["hp_threshold"], subset["saving_pct"],
                marker=trace_markers[trace_name], color=trace_colors[trace_name],
                label=trace_name.capitalize(), linewidth=2, markersize=6)
        ax2.plot(subset["hp_threshold"], subset["slo_violation_rate"],
                marker=trace_markers[trace_name], color=trace_colors[trace_name],
                label=trace_name.capitalize(), linewidth=2, markersize=6)

    ax1.set_xlabel("HP Threshold (req/s)", fontsize=11)
    ax1.set_ylabel("Energy Savings vs Naive (%)", fontsize=11)
    ax1.set_title(f"(a) Energy Savings (SLO={slo}ms)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 70)

    ax2.set_xlabel("HP Threshold (req/s)", fontsize=11)
    ax2.set_ylabel("SLO Violation Rate (%)", fontsize=11)
    ax2.set_title(f"(b) SLO Violations (SLO={slo}ms)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"fig9_threshold_sensitivity_slo{slo}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 9: Threshold sensitivity (SLO={slo}ms)")


def compute_edp(summary_csv=str(PAPER_RESULTS_ANALYSES_DIR / "jsep_results_summary.csv")):
    """Compute Energy-Delay Product for each strategy.

    EDP = Energy × avg_latency_proxy
    We use violation rate as a latency proxy: strategies with more violations
    have higher effective delay. Specifically:
    EDP = E × (1 + violation_rate)
    This penalizes strategies that save energy by violating SLOs.
    """
    df = pd.read_csv(summary_csv)
    # EDP normalized: Energy (kJ) × (1 + violation_rate as fraction)
    df["edp"] = df["total_energy_kj"] * (1 + df["slo_violation_rate"] / 100.0)
    return df


def plot_edp(df):
    """Plot EDP comparison across strategies."""
    slos = sorted(df["slo_ms"].unique())

    fig, axes = plt.subplots(1, len(slos), figsize=(4 * len(slos), 4), sharey=False)
    if len(slos) == 1:
        axes = [axes]

    strategy_order = ["naive", "dvfs_only", "routing_only", "dynamollm", "jsep"]
    x = np.arange(3)  # 3 traces
    width = 0.15
    traces = ["steady", "bursty", "diurnal"]

    for ax, slo in zip(axes, slos):
        for i, strat in enumerate(strategy_order):
            edp_vals = []
            for trace in traces:
                row = df[(df["strategy"] == strat) &
                        (df["trace"] == trace) &
                        (df["slo_ms"] == slo)]
                if not row.empty:
                    edp_vals.append(row.iloc[0]["edp"])
                else:
                    edp_vals.append(0)

            ax.bar(x + i * width, edp_vals, width,
                  label=LABELS[strat], color=COLORS[strat],
                  edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Workload Trace", fontsize=11)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([t.capitalize() for t in traces])
        ax.set_title(f"SLO = {slo}ms", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("EDP (kJ × penalty)", fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig10_edp_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 10: EDP comparison")


def plot_edp_normalized(df):
    """Plot EDP normalized to Naive baseline."""
    slos = sorted(df["slo_ms"].unique())
    strategy_order = ["dvfs_only", "routing_only", "dynamollm", "jsep"]
    traces = ["steady", "bursty", "diurnal"]

    fig, axes = plt.subplots(1, len(slos), figsize=(4 * len(slos), 4), sharey=True)
    if len(slos) == 1:
        axes = [axes]

    x = np.arange(3)
    width = 0.18

    for ax, slo in zip(axes, slos):
        # Get naive EDP for normalization
        naive_edp = {}
        for trace in traces:
            row = df[(df["strategy"] == "naive") &
                    (df["trace"] == trace) &
                    (df["slo_ms"] == slo)]
            naive_edp[trace] = row.iloc[0]["edp"] if not row.empty else 1.0

        for i, strat in enumerate(strategy_order):
            norm_vals = []
            for trace in traces:
                row = df[(df["strategy"] == strat) &
                        (df["trace"] == trace) &
                        (df["slo_ms"] == slo)]
                if not row.empty:
                    norm_vals.append(row.iloc[0]["edp"] / naive_edp[trace])
                else:
                    norm_vals.append(1.0)

            ax.bar(x + i * width, norm_vals, width,
                  label=LABELS[strat], color=COLORS[strat],
                  edgecolor="white", linewidth=0.5)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Workload Trace", fontsize=11)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([t.capitalize() for t in traces])
        ax.set_title(f"SLO = {slo}ms", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Normalized EDP (lower = better)", fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig10_edp_normalized.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 10: EDP normalized comparison")


def print_edp_table(df):
    """Print EDP table for paper."""
    slos = sorted(df["slo_ms"].unique())
    strategies = ["naive", "dvfs_only", "routing_only", "dynamollm", "jsep"]
    traces = ["steady", "bursty", "diurnal"]

    for slo in slos:
        print(f"\n  SLO = {slo}ms:")
        print(f"  {'Strategy':<16} ", end="")
        for trace in traces:
            print(f"| {'EDP':>8} {'Norm':>6} ", end="")
        print(f"| {'Avg Norm':>8}")
        print("  " + "-" * 70)

        naive_edp = {}
        for trace in traces:
            row = df[(df["strategy"] == "naive") &
                    (df["trace"] == trace) &
                    (df["slo_ms"] == slo)]
            naive_edp[trace] = row.iloc[0]["edp"]

        for strat in strategies:
            print(f"  {LABELS[strat]:<16} ", end="")
            norms = []
            for trace in traces:
                row = df[(df["strategy"] == strat) &
                        (df["trace"] == trace) &
                        (df["slo_ms"] == slo)]
                edp = row.iloc[0]["edp"]
                norm = edp / naive_edp[trace]
                norms.append(norm)
                print(f"| {edp:8.1f} {norm:5.2f}x ", end="")
            avg_norm = np.mean(norms)
            print(f"| {avg_norm:7.2f}x")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    slo = 500  # Primary SLO for sensitivity

    # --- C: Threshold sensitivity ---
    print("=== C: HP Threshold Sensitivity Analysis ===")
    df_sweep = run_threshold_sweep(slo=slo)
    df_sweep.to_csv(PAPER_RESULTS_ANALYSES_DIR / "jsep_threshold_sweep.csv", index=False)
    plot_threshold_sensitivity(df_sweep, slo)

    # Print summary
    print(f"\n  Threshold sweep results (SLO={slo}ms):")
    for hp_th in sorted(df_sweep["hp_threshold"].unique()):
        subset = df_sweep[df_sweep["hp_threshold"] == hp_th]
        avg_saving = subset["saving_pct"].mean()
        avg_viol = subset["slo_violation_rate"].mean()
        print(f"    HP={hp_th:3.0f} → saving={avg_saving:5.1f}%, violations={avg_viol:5.1f}%")

    # --- D: EDP analysis ---
    print("\n=== D: Energy-Delay Product Analysis ===")
    df_edp = compute_edp()
    plot_edp(df_edp)
    plot_edp_normalized(df_edp)
    print_edp_table(df_edp)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
