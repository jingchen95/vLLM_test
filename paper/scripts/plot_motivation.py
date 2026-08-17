#!/usr/bin/env python3
"""
Generate motivation figures for the JSEP paper from Phase 2 characterization data.

Figures:
  1. Power heatmap across (input_len, output_len) at rate=1, TP1
  2. TTFT vs input_len at rate=1, 10, 50 (prefill bottleneck under load)
  3. TTFT and TPOT grouped bars for TP1/TP2/TP4 at rate=1
  4. Total power vs TP degree with per-token energy overlay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from paths import PAPER_RESULTS_ANALYSES_DIR, PAPER_RESULTS_FIGURES_DIR, REPO_ROOT

# ---------- style ----------
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

DATA_DIR = REPO_ROOT / "Phase2_Results_L40S"
L4_DATA = PAPER_RESULTS_ANALYSES_DIR / "master_results_l4.csv"
FIG_DIR = PAPER_RESULTS_FIGURES_DIR

# ---------- load data ----------
df_master = pd.read_csv(DATA_DIR / "master_results.csv")
df_2a = df_master[df_master["step"] == "step2a"].copy()
df_2b = df_master[df_master["step"] == "step2b"].copy()

# L4 data
df_l4 = pd.read_csv(L4_DATA)

# Average over runs
def avg_runs(df):
    group_cols = ["step", "gpu_type", "gpu_freq_mhz", "tp_degree",
                  "input_len", "output_len", "request_rate"]
    num_cols = ["mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
                "avg_power_w", "max_power_w",
                "request_throughput_rps", "output_token_throughput_tps",
                "benchmark_duration_s", "avg_gpu_util_pct"]
    return df.groupby(group_cols, as_index=False)[num_cols].mean()

avg_2a = avg_runs(df_2a)
avg_2b = avg_runs(df_2b)
avg_l4_2a = avg_runs(df_l4[df_l4["step"] == "step2a"])


# ================================================================
# Figure 1: Power heatmap — (input_len x output_len) at rate=1, TP1
#            Side-by-side: (a) L40S, (b) L4
# ================================================================
def fig1_power_heatmap():
    # L40S data
    sel_l40s = avg_2a[(avg_2a["request_rate"] == 1) & (avg_2a["tp_degree"] == 1)]
    piv_l40s = sel_l40s.pivot(index="input_len", columns="output_len", values="avg_power_w")
    piv_l40s = piv_l40s.sort_index(ascending=True)
    piv_l40s = piv_l40s[sorted(piv_l40s.columns)]

    # L4 data
    sel_l4 = avg_l4_2a[(avg_l4_2a["request_rate"] == 1) & (avg_l4_2a["tp_degree"] == 1)]
    piv_l4 = sel_l4.pivot(index="input_len", columns="output_len", values="avg_power_w")
    piv_l4 = piv_l4.sort_index(ascending=True)
    piv_l4 = piv_l4[sorted(piv_l4.columns)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # --- (a) L40S ---
    im1 = ax1.imshow(piv_l40s.values, cmap="YlOrRd", aspect="auto",
                     vmin=150, vmax=340)
    ax1.set_xticks(range(len(piv_l40s.columns)))
    ax1.set_xticklabels(piv_l40s.columns.astype(int))
    ax1.set_yticks(range(len(piv_l40s.index)))
    ax1.set_yticklabels(piv_l40s.index.astype(int))
    ax1.set_xlabel("Output length")
    ax1.set_ylabel("Input length")
    ax1.set_title("(a) L40S")

    for i in range(len(piv_l40s.index)):
        for j in range(len(piv_l40s.columns)):
            val = piv_l40s.values[i, j]
            color = "white" if val > 280 else "black"
            ax1.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=7, color=color)

    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
    cbar1.set_label("Power (W)", fontsize=8)

    # --- (b) L4 ---
    im2 = ax2.imshow(piv_l4.values, cmap="YlOrRd", aspect="auto",
                     vmin=60, vmax=75)
    ax2.set_xticks(range(len(piv_l4.columns)))
    ax2.set_xticklabels(piv_l4.columns.astype(int))
    ax2.set_yticks(range(len(piv_l4.index)))
    ax2.set_yticklabels(piv_l4.index.astype(int))
    ax2.set_xlabel("Output length")
    ax2.set_ylabel("Input length")
    ax2.set_title("(b) L4")

    for i in range(len(piv_l4.index)):
        for j in range(len(piv_l4.columns)):
            val = piv_l4.values[i, j]
            color = "white" if val > 70 else "black"
            ax2.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=7, color=color)

    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cbar2.set_label("Power (W)", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_power_heatmap.pdf")
    fig.savefig(f"{FIG_DIR}/fig1_power_heatmap.png", dpi=300)
    plt.close(fig)
    print("  [OK] Figure 1: Power heatmap (L40S + L4)")


# ================================================================
# Figure 2: TTFT vs input_len at rate=1, 10, 50  (output_len=128)
# ================================================================
def fig2_ttft_scaling():
    sel = avg_2a[(avg_2a["tp_degree"] == 1) & (avg_2a["output_len"] == 128)]
    sel = sel.sort_values("input_len")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    markers = {1: "o", 10: "s", 50: "^"}
    colors = {1: "#2196F3", 10: "#FF9800", 50: "#F44336"}

    for rate in [1, 10, 50]:
        sub = sel[sel["request_rate"] == rate]
        ax.plot(sub["input_len"], sub["mean_ttft_ms"],
                marker=markers[rate], color=colors[rate],
                label=f"rate={rate} req/s", linewidth=1.5, markersize=5)

    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.savefig(f"{FIG_DIR}/fig2_ttft_scaling.pdf")
    fig.savefig(f"{FIG_DIR}/fig2_ttft_scaling.png", dpi=300)
    plt.close(fig)
    print("  [OK] Figure 2: TTFT scaling")


# ================================================================
# Figure 3: TTFT and TPOT for TP1/TP2/TP4 — grouped bars at rate=1
# ================================================================
def fig3_tp_latency():
    sel = avg_2b[avg_2b["request_rate"] == 1].copy()
    configs = [(2048, 32), (1024, 128), (512, 512), (128, 1024)]
    config_labels = ["2048/32", "1024/128", "512/512", "128/1024"]
    tp_degrees = [1, 2, 4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5), sharey=False)

    x = np.arange(len(configs))
    width = 0.25
    colors_tp = {1: "#4CAF50", 2: "#2196F3", 4: "#9C27B0"}

    # --- TTFT ---
    for i, tp in enumerate(tp_degrees):
        ttft_vals = []
        for (il, ol) in configs:
            row = sel[(sel["tp_degree"] == tp) &
                      (sel["input_len"] == il) &
                      (sel["output_len"] == ol)]
            ttft_vals.append(row["mean_ttft_ms"].values[0] if len(row) > 0 else 0)
        ax1.bar(x + (i - 1) * width, ttft_vals, width,
                color=colors_tp[tp], label=f"TP{tp}")

    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels, fontsize=7)
    ax1.set_xlabel("Input/Output length")
    ax1.set_ylabel("Mean TTFT (ms)")
    ax1.legend(frameon=False, fontsize=7)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_title("(a) Prefill latency (TTFT)")

    # --- TPOT ---
    for i, tp in enumerate(tp_degrees):
        tpot_vals = []
        for (il, ol) in configs:
            row = sel[(sel["tp_degree"] == tp) &
                      (sel["input_len"] == il) &
                      (sel["output_len"] == ol)]
            tpot_vals.append(row["mean_tpot_ms"].values[0] if len(row) > 0 else 0)
        ax2.bar(x + (i - 1) * width, tpot_vals, width,
                color=colors_tp[tp], label=f"TP{tp}")

    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels, fontsize=7)
    ax2.set_xlabel("Input/Output length")
    ax2.set_ylabel("Mean TPOT (ms)")
    ax2.legend(frameon=False, fontsize=7)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_title("(b) Decode latency (TPOT)")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig3_tp_latency.pdf")
    fig.savefig(f"{FIG_DIR}/fig3_tp_latency.png", dpi=300)
    plt.close(fig)
    print("  [OK] Figure 3: TP latency bars")


# ================================================================
# Figure 4: 2x2 grid — (a) Power vs TP, (b-d) Power vs Freq at TP1/2/4
# ================================================================
def fig4_power_energy_tradeoff():
    # --- step2c data for frequency sweep ---
    avg_2c = avg_runs(df_master[df_master["step"] == "step2c"])

    sel_tp = avg_2b[avg_2b["request_rate"] == 1].copy()

    configs = [(2048, 32), (1024, 128), (512, 512), (128, 1024)]
    config_labels = ["2048/32", "1024/128", "512/512", "128/1024"]
    tp_degrees = [1, 2, 4]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.2))
    ax_tp = axes[0, 0]
    ax_f1 = axes[0, 1]
    ax_f2 = axes[1, 0]
    ax_f4 = axes[1, 1]

    markers = ["o", "s", "D", "^"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    # --- (a) Power vs TP ---
    ax_tp2 = ax_tp.twinx()
    for ci, (il, ol) in enumerate(configs):
        powers = []
        ept_vals = []
        for tp in tp_degrees:
            row = sel_tp[(sel_tp["tp_degree"] == tp) &
                         (sel_tp["input_len"] == il) &
                         (sel_tp["output_len"] == ol)]
            if len(row) > 0:
                pwr = row["avg_power_w"].values[0]
                tps = row["output_token_throughput_tps"].values[0]
                powers.append(pwr)
                ept_vals.append(pwr / tps if tps > 0 else 0)
            else:
                powers.append(0)
                ept_vals.append(0)

        ax_tp.plot(tp_degrees, powers, marker=markers[ci], color=colors[ci],
                   linewidth=1.5, markersize=5, label=config_labels[ci])
        ax_tp2.plot(tp_degrees, ept_vals, marker=markers[ci], color=colors[ci],
                    linewidth=1.5, markersize=5, linestyle="--", alpha=0.6)

    ax_tp.set_xlabel("Tensor Parallelism Degree")
    ax_tp.set_ylabel("Avg Total Power (W)")
    ax_tp.set_xticks(tp_degrees)
    ax_tp.grid(axis="y", alpha=0.3, linestyle="--")
    ax_tp.legend(frameon=False, fontsize=7, title="In/Out", title_fontsize=7,
                 loc="upper left")
    ax_tp2.set_ylabel("Energy/Token (J/tok)", color="gray", fontsize=8)
    ax_tp2.tick_params(axis="y", labelcolor="gray")
    ax_tp.set_title("(a) Power vs. TP degree")
    ax_tp.annotate("solid = power, dashed = energy/tok",
                   xy=(0.98, 0.02), xycoords="axes fraction",
                   ha="right", va="bottom", fontsize=6, color="gray")

    # --- Helper to plot one freq panel ---
    # Note: TP1 data at rate=1, TP2/TP4 data at rate=10
    sel_freq_all = avg_2c.copy()
    freqs = sorted(sel_freq_all["gpu_freq_mhz"].unique())

    def plot_freq_panel(ax, tp, title):
        ax2 = ax.twinx()
        plotted = False
        # Pick available rate for this TP degree
        tp_data = sel_freq_all[sel_freq_all["tp_degree"] == tp]
        avail_rates = sorted(tp_data["request_rate"].unique())
        rate = avail_rates[0] if len(avail_rates) > 0 else 1
        for ci, (il, ol) in enumerate(configs):
            subset = tp_data[(tp_data["input_len"] == il) &
                              (tp_data["output_len"] == ol) &
                              (tp_data["request_rate"] == rate)]
            if len(subset) == 0:
                continue
            plotted = True
            powers = []
            ept_vals = []
            for freq in freqs:
                row = subset[subset["gpu_freq_mhz"] == freq]
                if len(row) > 0:
                    pwr = row["avg_power_w"].values[0]
                    tps = row["output_token_throughput_tps"].values[0]
                    powers.append(pwr)
                    ept_vals.append(pwr / tps if tps > 0 else 0)
                else:
                    powers.append(np.nan)
                    ept_vals.append(np.nan)

            ax.plot(freqs, powers, marker=markers[ci], color=colors[ci],
                    linewidth=1.5, markersize=4, label=config_labels[ci])
            ax2.plot(freqs, ept_vals, marker=markers[ci], color=colors[ci],
                     linewidth=1.0, markersize=3, linestyle="--", alpha=0.4)

        ax.set_xlabel("GPU Frequency (MHz)")
        ax.set_ylabel("Avg Power (W)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax2.set_ylabel("Energy/Token (J/tok)", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray")
        ax.set_title(title)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")

    plot_freq_panel(ax_f1, 1, "(b) Freq sweep, TP=1 (rate=1)")
    plot_freq_panel(ax_f2, 2, "(c) Freq sweep, TP=2 (rate=10)")
    plot_freq_panel(ax_f4, 4, "(d) Freq sweep, TP=4 (rate=10)")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_power_energy.pdf")
    fig.savefig(f"{FIG_DIR}/fig4_power_energy.png", dpi=300)
    plt.close(fig)
    print("  [OK] Figure 4: Power-energy tradeoff (2x2 grid)")


# ================================================================
# Figure 5: Workload variation in evaluation traces
# ================================================================
def fig5_trace_variation():
    from jsep_traces import generate_all_traces

    traces = generate_all_traces(duration_s=600.0, seed=42)
    trace_names = ["steady", "bursty", "diurnal"]
    trace_colors = {"steady": "#2196F3", "bursty": "#FF9800", "diurnal": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))

    # --- (a) Arrival rate over time ---
    ax = axes[0]
    window = 10  # seconds
    for name in trace_names:
        trace = traces[name]
        times = [r.arrival_time for r in trace]
        bins = np.arange(0, 600, window)
        counts, _ = np.histogram(times, bins=bins)
        rates = counts / window
        ax.plot(bins[:-1], rates, color=trace_colors[name],
                linewidth=1.0, label=name, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arrival rate (req/s)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title("(a) Arrival rate")

    # --- (b) Input length distribution ---
    ax = axes[1]
    for name in trace_names:
        trace = traces[name]
        ils = [r.input_len for r in trace]
        il_vals = sorted(set(ils))
        il_counts = [ils.count(v) / len(ils) for v in il_vals]
        ax.bar([il_vals[i] + trace_names.index(name) * 40 - 40
                for i in range(len(il_vals))],
               il_counts, width=38, color=trace_colors[name],
               alpha=0.75, label=name)
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("Fraction of requests")
    ax.set_xticks([32, 128, 512, 1024, 2048])
    ax.set_xticklabels(["32", "128", "512", "1024", "2048"], fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title("(b) Input length dist.")

    # --- (c) Output length distribution ---
    ax = axes[2]
    for name in trace_names:
        trace = traces[name]
        ols = [r.output_len for r in trace]
        ol_vals = sorted(set(ols))
        ol_counts = [ols.count(v) / len(ols) for v in ol_vals]
        ax.bar([ol_vals[i] + trace_names.index(name) * 20 - 20
                for i in range(len(ol_vals))],
               ol_counts, width=18, color=trace_colors[name],
               alpha=0.75, label=name)
    ax.set_xlabel("Output length (tokens)")
    ax.set_ylabel("Fraction of requests")
    ax.set_xticks([32, 128, 512, 1024])
    ax.set_xticklabels(["32", "128", "512", "1024"], fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title("(c) Output length dist.")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_trace_variation.pdf")
    fig.savefig(f"{FIG_DIR}/fig5_trace_variation.png", dpi=300)
    plt.close(fig)
    print("  [OK] Figure 5: Trace workload variation")


# ================================================================
if __name__ == "__main__":
    print("Generating motivation figures...")
    fig1_power_heatmap()
    fig2_ttft_scaling()
    fig3_tp_latency()
    fig4_power_energy_tradeoff()
    fig5_trace_variation()
    print("All figures saved to", FIG_DIR)
