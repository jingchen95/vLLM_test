#!/usr/bin/env python3
"""
Plot synthetic trace behavior for debugging and paper preparation.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import synthetic_traces_path

# Where LaTeX includes the paper-bound copies. After saving each figure
# under results/.../plots/, mirror the trace-overview file here so the
# paper/evaluation_rewrite.tex include path stays in sync.
_PAPER_FIGURES_SECTION42 = (
    Path(__file__).resolve().parents[1] / "figures" / "section42"
)


def _mirror_to_paper(stem: Path, paper_name: str | None = None) -> None:
    """Copy <stem>.pdf and <stem>.png into paper/figures/section42/.
    paper_name overrides the destination basename if provided."""
    import shutil
    dest_dir = _PAPER_FIGURES_SECTION42
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = paper_name or stem.name
    for ext in (".pdf", ".png"):
        src = stem.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, dest_dir / f"{base}{ext}")

# Paper-figure rcParams (matches paper/scripts/plot_motivation.py).
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


STATE_COLORS = {
    "PREFILL_HEAVY": "#4C78A8",  # blue
    "DECODE_HEAVY":  "#d62728",  # red — distinct from the warm oranges used for the long-* request classes
    "BOTH_LOW":      "#54A24B",  # green
    "BOTH_HEAVY":    "#B279A2",  # purple
}

CLASS_COLORS = {
    "short_short": "#9ecae9",
    "short_long": "#fdd0a2",
    "long_short": "#6baed6",
    "long_long": "#fd8d3c",
}

CLASS_ORDER = ("short_short", "short_long", "long_short", "long_long")


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_trace(trace_rows: list[dict], window_rows: list[dict], output_prefix: Path, window_s: float) -> None:
    trace_name = trace_rows[0]["trace_name"] if trace_rows else output_prefix.name

    fig, axes = plt.subplots(
        3, 1, figsize=(6.8, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.1, 0.9]},
    )

    # Arrival rate over time.
    xs = [float(row["window_start_s"]) for row in window_rows]
    rate = [float(row["arrival_rate_rps"]) for row in window_rows]
    axes[0].plot(xs, rate, color="#4C78A8", linewidth=1.8)
    axes[0].set_ylabel("Rate (req/s)")
    axes[0].set_title(trace_name, fontsize=10)

    # Request-class composition over time.
    per_window = defaultdict(Counter)
    for row in trace_rows:
        window_id = int(float(row["arrival_time_s"]) // window_s)
        per_window[window_id][row["request_class"]] += 1
    class_order = ["short_short", "short_long", "long_short", "long_long"]
    bottoms = [0.0] * len(window_rows)
    for class_id in class_order:
        vals = []
        for row in window_rows:
            counts = per_window[int(row["window_id"])]
            total = sum(counts.values())
            vals.append((counts[class_id] / total) if total else 0.0)
        axes[1].bar(
            xs,
            vals,
            bottom=bottoms,
            width=window_s * 0.9,
            align="edge",
            color=CLASS_COLORS[class_id],
            label=class_id.replace("_", "-"),
            edgecolor="none",
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    axes[1].set_ylabel("Class fraction")
    axes[1].legend(loc="upper right", fontsize=7, frameon=False, ncol=2)

    # Assigned state over time.
    state_to_y = {state: idx for idx, state in enumerate(STATE_COLORS)}
    state_y = [state_to_y[row["assigned_state"]] for row in window_rows]
    colors = [STATE_COLORS[row["assigned_state"]] for row in window_rows]
    axes[2].bar(xs, [1] * len(xs), width=window_s * 0.9, align="edge", color=colors, edgecolor="none")
    axes[2].set_yticks([0.5])
    axes[2].set_yticklabels(["State"])
    axes[2].set_xlabel("Time (s)")

    # State labels as a compact legend inside the last panel.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in STATE_COLORS.values()
    ]
    axes[2].legend(handles, list(STATE_COLORS.keys()), loc="upper right", fontsize=7, frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(pdf_path)
    print(png_path)


def plot_combined_overview(
    trace_data: dict,
    output_prefix: Path,
    window_s: float,
) -> None:
    """One paper-ready figure: rows = {rate, class fraction, state},
    cols = one per trace. trace_data: {trace_name: (trace_rows, window_rows)}.

    Layout choices:
      - shared x-axis per column (time)
      - shared y-axis per row (so traces are visually comparable)
      - the state row is shorter (height_ratio < 1) since it's categorical
      - two stacked legends at the bottom (request class, window state)
    """
    trace_names = list(trace_data.keys())
    n_cols = len(trace_names)
    if n_cols == 0:
        return

    # Global rate range to share Row 0 y-axis
    max_rate = 0.0
    for name in trace_names:
        _, window_rows = trace_data[name]
        for row in window_rows:
            max_rate = max(max_rate, float(row["arrival_rate_rps"]))
    if max_rate <= 0:
        max_rate = 1.0

    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(3.2 * n_cols, 4.6),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [1.1, 1.1, 0.45]},
        squeeze=False,
    )

    for col, trace_name in enumerate(trace_names):
        trace_rows, window_rows = trace_data[trace_name]
        xs = [float(row["window_start_s"]) for row in window_rows]

        # Row 0: arrival rate
        ax_rate = axes[0][col]
        rates = [float(row["arrival_rate_rps"]) for row in window_rows]
        ax_rate.plot(xs, rates, color="#4C78A8", linewidth=1.6)
        ax_rate.set_title(trace_name)
        ax_rate.set_ylim(0, max_rate * 1.10)
        ax_rate.grid(axis="y", alpha=0.3, linewidth=0.5)
        if col == 0:
            ax_rate.set_ylabel("Rate (req/s)")

        # Row 1: class fraction (stacked)
        ax_class = axes[1][col]
        per_window = defaultdict(Counter)
        for row in trace_rows:
            window_id = int(float(row["arrival_time_s"]) // window_s)
            per_window[window_id][row["request_class"]] += 1
        bottoms = [0.0] * len(window_rows)
        for class_id in CLASS_ORDER:
            vals = []
            for row in window_rows:
                counts = per_window[int(row["window_id"])]
                total = sum(counts.values())
                vals.append((counts[class_id] / total) if total else 0.0)
            ax_class.bar(
                xs,
                vals,
                bottom=bottoms,
                width=window_s,
                align="edge",
                color=CLASS_COLORS[class_id],
                edgecolor="none",
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax_class.set_ylim(0, 1)
        if col == 0:
            ax_class.set_ylabel("Class fraction")

        # Row 2: assigned state
        ax_state = axes[2][col]
        colors = [STATE_COLORS[row["assigned_state"]] for row in window_rows]
        ax_state.bar(xs, [1] * len(xs), width=window_s, align="edge",
                     color=colors, edgecolor="none")
        ax_state.set_ylim(0, 1)
        ax_state.set_yticks([])
        ax_state.set_xlabel("Time (s)")
        if col == 0:
            ax_state.set_ylabel("State")

    # Spine cleanup
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Stacked legends at the bottom
    class_handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASS_ORDER]
    class_labels = [c.replace("_", "-") for c in CLASS_ORDER]
    state_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS.values()]
    state_labels = list(STATE_COLORS.keys())

    # Side-by-side legends sharing one horizontal row at the bottom.
    leg_class = fig.legend(
        class_handles, class_labels,
        loc="upper center", bbox_to_anchor=(0.27, -0.02),
        ncol=len(CLASS_ORDER), frameon=False,
        title="Request class", title_fontsize=8,
    )
    fig.add_artist(leg_class)
    fig.legend(
        state_handles, state_labels,
        loc="upper center", bbox_to_anchor=(0.75, -0.02),
        ncol=len(STATE_COLORS), frameon=False,
        title="Window state", title_fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output_prefix.with_suffix(".pdf"))
    print(output_prefix.with_suffix(".png"))
    _mirror_to_paper(output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=str(synthetic_traces_path()))
    parser.add_argument("--windows-csv", default=str(synthetic_traces_path("state_coverage_windows.csv")))
    parser.add_argument("--traces", default="T1,T2,T3,T4")
    parser.add_argument("--output-dir", default=str(synthetic_traces_path("plots")))
    parser.add_argument("--window", type=float, default=5.0)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    windows = load_csv(Path(args.windows_csv))
    trace_names = [item.strip() for item in args.traces.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_data = {}
    for trace_name in trace_names:
        trace_rows = load_csv(trace_dir / f"{trace_name}.csv")
        window_rows = [row for row in windows if row["trace_name"] == trace_name]
        plot_trace(trace_rows, window_rows, output_dir / f"{trace_name}_debug", args.window)
        trace_data[trace_name] = (trace_rows, window_rows)

    plot_combined_overview(trace_data, output_dir / "synthetic_trace_overview", args.window)


if __name__ == "__main__":
    main()
