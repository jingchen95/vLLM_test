#!/usr/bin/env python3
"""
Generate paper-facing figures and tables for the Observation 2 motivation study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import figures_results_path


RESTRICTED_ORDER = [
    "static",
    "route_only",
    "route_dvfs",
    "route_dvfs_tp",
    "full_joint",
]

SEQUENTIAL_ORDER = [
    "sequential",
    "full_joint",
]

POLICY_LABELS = {
    "static": "Static-disagg",
    "route_only": "Route-only",
    "route_dvfs": "Route+DVFS",
    "route_dvfs_tp": "Route+DVFS+TP",
    "full_joint": "Full joint",
    "sequential": "Sequential",
}

STATE_LABELS = {
    "prefill-heavy": "Prefill-heavy",
    "decode-heavy": "Decode-heavy",
    "both-low": "Light-load",
}

COLORS = {
    "static": "#7f8c8d",
    "route_only": "#4c78a8",
    "route_dvfs": "#72b7b2",
    "route_dvfs_tp": "#f58518",
    "full_joint": "#54a24b",
    "sequential": "#e45756",
}

STATE_COLORS = {
    "prefill-heavy": "#8ec127",
    "decode-heavy": "#f47835",
}

STATE_TABLE_COLUMNS = [
    "Policy",
    "Route",
    "f_A",
    "f_B",
    "TP_A",
    "TP_B",
    "Act_A",
    "Act_B",
    "Mgn.",
]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def restricted_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, group in df.groupby("state"):
        group = group.set_index("policy")
        if "full_joint" not in group.index:
            continue
        full_energy = group.loc["full_joint", "energy_j"]
        prev_energy = None
        for policy in RESTRICTED_ORDER:
            if policy not in group.index:
                continue
            row = group.loc[policy]
            delta_prev_pct = None
            if prev_energy is not None and prev_energy > 0:
                delta_prev_pct = round((prev_energy - row["energy_j"]) / prev_energy * 100.0, 1)
            delta_to_full_pct = None
            if row["energy_j"] > 0:
                delta_to_full_pct = round((row["energy_j"] - full_energy) / row["energy_j"] * 100.0, 1)
            rows.append(
                {
                    "state": state,
                    "policy": policy,
                    "feasible": row["feasible"],
                    "selection_status": row["selection_status"],
                    "energy_j": row["energy_j"],
                    "worst_margin_ms": row["worst_margin_ms"],
                    "delta_vs_prev_pct": delta_prev_pct,
                    "delta_to_full_pct": delta_to_full_pct,
                    "sequential_gap_to_full_pct": None,
                }
            )
            prev_energy = row["energy_j"]
        if "sequential" in group.index:
            seq = group.loc["sequential"]
            sequential_gap = None
            if seq["energy_j"] > 0:
                sequential_gap = round((seq["energy_j"] - full_energy) / seq["energy_j"] * 100.0, 1)
            rows.append(
                {
                    "state": state,
                    "policy": "sequential",
                    "feasible": seq["feasible"],
                    "selection_status": seq["selection_status"],
                    "energy_j": seq["energy_j"],
                    "worst_margin_ms": seq["worst_margin_ms"],
                    "delta_vs_prev_pct": None,
                    "delta_to_full_pct": None,
                    "sequential_gap_to_full_pct": sequential_gap,
                }
            )
    return pd.DataFrame(rows)


def export_config_table(df: pd.DataFrame, output_prefix: Path) -> Path:
    cols = [
        "state",
        "policy",
        "feasible",
        "selection_status",
        "route_map_json",
        "l40s_freq_mhz",
        "l4_freq_mhz",
        "l40s_tp",
        "l4_tp",
        "l40s_active_gpus",
        "l4_active_gpus",
        "energy_j",
        "worst_margin_ms",
    ]
    table = df[cols].copy()
    path = output_prefix.with_name(output_prefix.name + "_config_table.csv")
    table.to_csv(path, index=False)
    return path


def export_full_joint_table(df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    table_df = build_full_joint_table_df(df)
    csv_path = output_prefix.with_name(output_prefix.name + "_full_joint_table.csv")
    table_df.to_csv(csv_path, index=False)

    diff_cols = [
        col for col in table_df.columns[1:]
        if str(table_df.iloc[0][col]) != str(table_df.iloc[1][col])
    ]

    def fmt_value(col: str, value: object) -> str:
        rendered = str(value)
        if col in diff_cols:
            return f"\\textbf{{{rendered}}}"
        return rendered

    header = "State & Route & $f_{\\mathrm{L40S}}$ & $f_{\\mathrm{L4}}$ & $TP_{\\mathrm{L40S}}$ & $TP_{\\mathrm{L4}}$ & $Act_{\\mathrm{L40S}}$ & $Act_{\\mathrm{L4}}$ \\\\"
    body = []
    for _, row in table_df.iterrows():
        body.append(
            " & ".join(
                [
                    str(row["State"]),
                    fmt_value("Route", row["Route"]),
                    fmt_value("f_L40S", row["f_L40S"]),
                    fmt_value("f_L4", row["f_L4"]),
                    fmt_value("TP_L40S", row["TP_L40S"]),
                    fmt_value("TP_L4", row["TP_L4"]),
                    fmt_value("Act_L40S", row["Act_L40S"]),
                    fmt_value("Act_L4", row["Act_L4"]),
                ]
            )
            + " \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{tabular}{lccccccc}",
            "\\toprule",
            header,
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    tex_path = output_prefix.with_name(output_prefix.name + "_full_joint_table.tex")
    tex_path.write_text(tex)
    return csv_path, tex_path


def build_full_joint_table_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ["prefill-heavy", "decode-heavy"]:
        group = _state_panel_rows(df, state)
        row = group.loc["full_joint"]
        rows.append(
            {
                "State": STATE_LABELS[state],
                "Route": _route_label(row["route_map_json"]),
                "f_L40S": int(row["l40s_freq_mhz"]),
                "f_L4": int(row["l4_freq_mhz"]),
                "TP_L40S": int(row["l40s_tp"]),
                "TP_L4": int(row["l4_tp"]),
                "Act_L40S": int(row["l40s_active_gpus"]),
                "Act_L4": int(row["l4_active_gpus"]),
            }
        )
    return pd.DataFrame(rows)


def build_config_matrix_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ["prefill-heavy", "decode-heavy"]:
        group = _state_panel_rows(df, state)
        for policy in ["static", "route_only", "route_dvfs", "route_dvfs_tp", "sequential", "full_joint"]:
            row = group.loc[policy]
            rows.append(
                {
                    "Policy": POLICY_LABELS[policy],
                    "Feas.": "Y" if bool(row["feasible"]) else "N",
                    "Route": _route_cell(row["route_map_json"]),
                    "f_A": int(row["l40s_freq_mhz"]),
                    "f_B": int(row["l4_freq_mhz"]),
                    "TP_A": int(row["l40s_tp"]),
                    "TP_B": int(row["l4_tp"]),
                    "Act_A": int(row["l40s_active_gpus"]),
                    "Act_B": int(row["l4_active_gpus"]),
                    "Mgn.": f"{float(row['worst_margin_ms']):.1f}",
                    "_state_key": state,
                    "_policy_key": policy,
                    "_energy_j": float(row["energy_j"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_policy_bars(ax, group: pd.DataFrame, policies: list[str], title: str):
    x = range(len(policies))
    vals = [group.loc[policy, "energy_j"] for policy in policies]
    feasible = [bool(group.loc[policy, "feasible"]) for policy in policies]
    margins = [float(group.loc[policy, "worst_margin_ms"]) for policy in policies]
    bars = ax.bar(
        x,
        vals,
        color=[COLORS[p] for p in policies],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, ok, margin in zip(bars, feasible, margins):
        if not ok:
            bar.set_hatch("//")
            bar.set_alpha(0.55)
        _annotate_bar(ax, bar, ok, margin)
    ax.set_xticks(list(x))
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], rotation=30, ha="right")
    ax.set_ylabel("Predicted energy (J)")
    ax.set_title(title)


def _route_label(route_map_json: str) -> str:
    route_map = json.loads(route_map_json)
    unique_routes = sorted(set(route_map.values()))
    if len(unique_routes) == 1:
        return unique_routes[0]
    return "mixed"


def _route_cell(route_map_json: str) -> str:
    # Class ids are 2-letter input/output codes: input,output in {S=128/512/2048 etc}
    # ordered S<M<L on each axis (SS, SM, SL, MS, MM, ML, LS, LM, LL).
    route_map = json.loads(route_map_json)
    legacy = {"long_long": "LL", "long_short": "LS", "short_long": "SL", "short_short": "SS"}
    canon = [a + b for a in "SML" for b in "SML"]
    def lab(k):
        return legacy.get(k, k)
    ordered_keys = sorted(route_map, key=lambda k: canon.index(lab(k)) if lab(k) in canon else 99)
    return " | ".join(f"{lab(k)}:{route_map[k]}" for k in ordered_keys)


def _state_panel_rows(df: pd.DataFrame, state: str) -> pd.DataFrame:
    return df[df["state"] == state].set_index("policy")


def generate_main_figure(df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    states = ["prefill-heavy", "decode-heavy"]
    present_states = set(df["state"])
    missing_states = [state for state in states if state not in present_states]
    if missing_states:
        raise ValueError(
            "Main Observation 2 figure requires both core states "
            f"{states}, but missing: {missing_states}. "
            "If this was an intentional subset run, do not call "
            "generate_main_figure() on that selected CSV."
        )

    missing_policy_msgs = []
    for state in states:
        state_policies = set(df.loc[df["state"] == state, "policy"])
        missing_restricted = [p for p in RESTRICTED_ORDER if p not in state_policies]
        missing_seq = [p for p in SEQUENTIAL_ORDER if p not in state_policies]
        if missing_restricted:
            missing_policy_msgs.append(
                f"{state}: missing restricted-search policies {missing_restricted}"
            )
        if missing_seq:
            missing_policy_msgs.append(
                f"{state}: missing sequential-comparison policies {missing_seq}"
            )
    if missing_policy_msgs:
        raise ValueError(
            "Selected CSV does not contain the full policy set required for the "
            "main Observation 2 figure: " + "; ".join(missing_policy_msgs)
        )

    fig = plt.figure(figsize=(14.0, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.0, 1.0])
    ladder_ax = fig.add_subplot(gs[0, 0])
    prefill_ax = fig.add_subplot(gs[0, 1])
    decode_ax = fig.add_subplot(gs[0, 2])

    # (a) Policy ladder, shown as absolute energy with reduction vs Static-disagg annotated.
    all_policies = ["static", "route_only", "route_dvfs", "route_dvfs_tp", "sequential", "full_joint"]
    width = 0.31
    xs = list(range(len(all_policies)))
    ladder_groups = {state: _state_panel_rows(df, state) for state in states}
    for offset, state in [(-width / 2, "prefill-heavy"), (width / 2, "decode-heavy")]:
        group = ladder_groups[state]
        vals = [float(group.loc[policy, "energy_j"]) for policy in all_policies]
        feas = [bool(group.loc[policy, "feasible"]) for policy in all_policies]
        # Reductions are reported relative to the first feasible configuration
        # (Route-only); Static-disagg is infeasible at TP=1 and is not a valid
        # denominator. If Static-disagg is feasible, use it as the reference.
        static_feas = bool(group.loc["static", "feasible"])
        ref_policy = "static" if static_feas else "route_only"
        ref_energy = float(group.loc[ref_policy, "energy_j"])
        bars = ladder_ax.bar(
            [x + offset for x in xs],
            vals,
            width=width,
            color=STATE_COLORS[state],
            edgecolor="black",
            linewidth=0,
            label=STATE_LABELS[state],
        )
        for idx, (bar, ok) in enumerate(zip(bars, feas)):
            policy = all_policies[idx]
            if policy == "static":
                if not static_feas:  # only mark infeasible static; feasible -> solid bar
                    bar.set_hatch("//")
                    bar.set_alpha(0.55)
                    ladder_ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 70.0,
                        "infeas.",
                        ha="center", va="bottom", fontsize=6.2, rotation=90,
                    )
                continue
            if policy == ref_policy:
                continue
            reduction_pct = max(0.0, (ref_energy - vals[idx]) / ref_energy * 100.0)
            ladder_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 70.0,
                f"-{reduction_pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90,
            )
    ladder_ax.set_xticks(xs)
    ladder_ax.set_xticklabels([POLICY_LABELS[p] for p in all_policies], rotation=15, ha="right", fontsize=7.0)
    ladder_ax.set_ylabel("Energy (J)", fontsize=8.0)
    # ladder_ax.set_title("Energy across policies")
    ladder_ax.legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="upper right",
        borderaxespad=0.0,
    )
    ladder_ax.tick_params(axis="y", labelsize=7.0)
    ladder_ax.tick_params(axis="x", pad=2)
    ladder_ax.set_ylim(0, max(6200.0, ladder_ax.get_ylim()[1]))
    ladder_ax.set_box_aspect(0.76)

    table_df = build_config_matrix_df(df)

    def draw_state_table(ax, state_key: str, base_facecolor: str):
        ax.axis("off")
        state_df = table_df[table_df["_state_key"] == state_key].reset_index(drop=True)
        display_df = state_df[STATE_TABLE_COLUMNS]
        cell_text = [[str(row[col]) for col in STATE_TABLE_COLUMNS] for _, row in display_df.iterrows()]
        table = ax.table(
            cellText=cell_text,
            colLabels=STATE_TABLE_COLUMNS,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        # The Route column holds the longest text. Render it at a smaller font and
        # with tighter horizontal padding BEFORE measuring column widths, so the
        # column is sized to its actual content and carries no extra margin.
        route_col = STATE_TABLE_COLUMNS.index("Route")
        for (r, c), cell in table.get_celld().items():
            if c == route_col:
                cell.PAD = 0.02
                if r != 0:
                    cell.get_text().set_fontsize(6.6)
        # Let columns shrink to fit their text so the table is as narrow as possible.
        table.auto_set_column_width(col=list(range(len(STATE_TABLE_COLUMNS))))
        # Stretch rows vertically so the table fills a similar height as the ladder axes.
        table.scale(1.0, 1.35)

        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.35)
            if row_idx == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f0f0f0")
                continue
            policy_key = state_df.iloc[row_idx - 1]["_policy_key"]
            # cell.set_facecolor(base_facecolor)
            if policy_key == "full_joint":
                cell.set_facecolor("#e8f5e9")

    draw_state_table(prefill_ax, "prefill-heavy", "#f7fbff")
    draw_state_table(decode_ax, "decode-heavy", "#fff5eb")

    fig.canvas.draw()
    ladder_box = ladder_ax.get_position()
    prefill_box = prefill_ax.get_position()
    decode_box = decode_ax.get_position()
    title_y = max(ladder_box.y1, prefill_box.y1, decode_box.y1) -0.13
    fig.text((ladder_box.x0 + ladder_box.x1) / 2, title_y, "(a) Energy Comparisons", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    fig.text((prefill_box.x0 + prefill_box.x1) / 2, title_y, "(b) Prefill-heavy", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    fig.text((decode_box.x0 + decode_box.x1) / 2, title_y, "(c) Decode-heavy", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    pdf_path = output_prefix.with_name(output_prefix.name + "_obs2_main.pdf")
    png_path = output_prefix.with_name(output_prefix.name + "_obs2_main.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def generate_lightload_figure(df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path] | tuple[None, None]:
    if "both-low" not in set(df["state"]):
        return None, None
    group = df[df["state"] == "both-low"].set_index("policy")
    policies = [policy for policy in RESTRICTED_ORDER if policy in group.index]
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), constrained_layout=True)
    _plot_policy_bars(ax, group, policies, STATE_LABELS["both-low"])
    pdf_path = output_prefix.with_name(output_prefix.name + "_obs2_lightload.pdf")
    png_path = output_prefix.with_name(output_prefix.name + "_obs2_lightload.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=figures_results_path("motivation_joint"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.selected_csv)
    if df.empty:
        raise ValueError(f"No rows found in {args.selected_csv}")

    summary_df = restricted_delta_table(df)
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    config_table_path = export_config_table(df, args.output_prefix)
    full_joint_table_csv, full_joint_table_tex = export_full_joint_table(df, args.output_prefix)
    main_pdf, main_png = generate_main_figure(df, args.output_prefix)
    light_pdf, light_png = generate_lightload_figure(df, args.output_prefix)

    print(f"summary_csv: {summary_path}")
    print(f"config_table_csv: {config_table_path}")
    print(f"full_joint_table_csv: {full_joint_table_csv}")
    print(f"full_joint_table_tex: {full_joint_table_tex}")
    print(f"obs2_main_pdf: {main_pdf}")
    print(f"obs2_main_png: {main_png}")
    if light_pdf is not None:
        print(f"obs2_lightload_pdf: {light_pdf}")
        print(f"obs2_lightload_png: {light_png}")


if __name__ == "__main__":
    main()
