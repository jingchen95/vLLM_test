#!/usr/bin/env python3
"""
Search for scientifically strong Observation 2 regimes using the existing
model-based motivation harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from paths import figures_results_path, paper_model_dir
from motivation_joint_core import (
    DEFAULT_POLICIES,
    create_backend,
    default_reference_config,
    default_search_space,
    evaluate_policy,
    make_state_specs,
    write_outputs,
    write_replay_trace,
    _selected_config_record,
)
from figures.generate_fig_motivation_joint import (
    export_config_table,
    generate_lightload_figure,
    generate_main_figure,
    restricted_delta_table,
)


PRIMARY_STATES = ("prefill-heavy", "decode-heavy")
PRIMARY_MIXES = {
    "mix50": {
        "prefill-heavy": {"long_short": 0.50, "long_long": 0.25, "short_short": 0.25},
        "decode-heavy": {"short_long": 0.50, "long_long": 0.25, "short_short": 0.25},
    }
}
SECONDARY_MIXES = {
    "mix60": {
        "prefill-heavy": {"long_short": 0.60, "long_long": 0.20, "short_short": 0.20},
        "decode-heavy": {"short_long": 0.60, "long_long": 0.20, "short_short": 0.20},
    },
    "mix70": {
        "prefill-heavy": {"long_short": 0.70, "long_long": 0.15, "short_short": 0.15},
        "decode-heavy": {"short_long": 0.70, "long_long": 0.15, "short_short": 0.15},
    },
}

POLICY_ORDER = [policy.name for policy in DEFAULT_POLICIES]
RESTRICTED_ORDER = ["static", "route_only", "route_dvfs", "route_dvfs_tp", "full_joint"]


def parse_int_list(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def monotonic_nonincreasing(values: Sequence[float], tol: float = 1e-9) -> bool:
    return all(values[idx + 1] <= values[idx] + tol for idx in range(len(values) - 1))


def pct_gain(prev_energy: float, next_energy: float) -> float:
    if prev_energy <= 0:
        return 0.0
    return round((prev_energy - next_energy) / prev_energy * 100.0, 1)


def state_metrics(group: pd.DataFrame) -> Dict[str, object]:
    group = group.set_index("policy")
    metrics: Dict[str, object] = {}
    energies = {}
    feas = {}
    for policy in POLICY_ORDER:
        if policy not in group.index:
            continue
        energies[policy] = float(group.loc[policy, "energy_j"])
        feas[policy] = bool(group.loc[policy, "feasible"])
        metrics[f"{policy}_energy_j"] = energies[policy]
        metrics[f"{policy}_feasible"] = feas[policy]
        metrics[f"{policy}_selection_status"] = group.loc[policy, "selection_status"]
        metrics[f"{policy}_worst_margin_ms"] = float(group.loc[policy, "worst_margin_ms"])

    restricted_energies = [energies[policy] for policy in RESTRICTED_ORDER if policy in energies]
    metrics["restricted_monotonic"] = monotonic_nonincreasing(restricted_energies)

    step_pairs = list(zip(RESTRICTED_ORDER[:-1], RESTRICTED_ORDER[1:]))
    meaningful_steps = 0
    for prev_policy, next_policy in step_pairs:
        if prev_policy in energies and next_policy in energies:
            gain = pct_gain(energies[prev_policy], energies[next_policy])
            metrics[f"{prev_policy}_to_{next_policy}_gain_pct"] = gain
            if gain >= 2.0:
                meaningful_steps += 1
    metrics["meaningful_restricted_steps"] = meaningful_steps

    static = energies.get("static")
    route_only = energies.get("route_only")
    full = energies.get("full_joint")
    sequential = energies.get("sequential")

    metrics["static_restored_by_route"] = (
        not feas.get("static", False) and feas.get("route_only", False)
    )
    metrics["static_to_full_gap_pct"] = (
        pct_gain(static, full) if static is not None and full is not None else 0.0
    )
    metrics["sequential_to_full_gap_pct"] = (
        round((sequential - full) / sequential * 100.0, 1)
        if sequential is not None and full is not None and sequential > 0
        else 0.0
    )
    return metrics


def full_joint_config_signature(group: pd.DataFrame) -> Dict[str, object]:
    row = group[group["policy"] == "full_joint"].iloc[0]
    return {
        "route_map_json": row["route_map_json"],
        "l40s_freq_mhz": int(row["l40s_freq_mhz"]),
        "l4_freq_mhz": int(row["l4_freq_mhz"]),
        "l40s_tp": int(row["l40s_tp"]),
        "l4_tp": int(row["l4_tp"]),
        "l40s_active_gpus": int(row["l40s_active_gpus"]),
        "l4_active_gpus": int(row["l4_active_gpus"]),
    }


def config_diff_count(sig_a: Dict[str, object], sig_b: Dict[str, object]) -> int:
    return sum(1 for key in sig_a if sig_a[key] != sig_b[key])


def score_regime(selected_df: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    per_state_metrics = {}
    for state, group in selected_df.groupby("state"):
        per_state_metrics[state] = state_metrics(group)

    prefill = per_state_metrics["prefill-heavy"]
    decode = per_state_metrics["decode-heavy"]
    sig_prefill = full_joint_config_signature(selected_df[selected_df["state"] == "prefill-heavy"])
    sig_decode = full_joint_config_signature(selected_df[selected_df["state"] == "decode-heavy"])
    diff_count = config_diff_count(sig_prefill, sig_decode)

    score = 0.0
    reasons: List[str] = []

    static_weakness = 0.0
    for label, state_metrics_row in [("prefill", prefill), ("decode", decode)]:
        if state_metrics_row["static_restored_by_route"]:
            static_weakness += 2.0
            reasons.append(f"{label}:static_infeasible_route_restores")
        elif state_metrics_row["static_to_full_gap_pct"] >= 20.0:
            static_weakness += 1.0
            reasons.append(f"{label}:static_gap_{state_metrics_row['static_to_full_gap_pct']:.1f}")
    score += static_weakness

    restricted_clarity = 0.0
    for label, state_metrics_row in [("prefill", prefill), ("decode", decode)]:
        if state_metrics_row["restricted_monotonic"]:
            restricted_clarity += 1.0
            reasons.append(f"{label}:monotonic")
        restricted_clarity += min(float(state_metrics_row["meaningful_restricted_steps"]) * 0.75, 2.25)
    score += restricted_clarity

    state_dependence = 0.0
    if diff_count >= 3:
        state_dependence = 2.0
        reasons.append(f"state_diff_{diff_count}")
    elif diff_count >= 1:
        state_dependence = 1.0
        reasons.append(f"state_diff_{diff_count}")
    score += state_dependence

    sequential_score = 0.0
    if float(prefill["sequential_to_full_gap_pct"]) >= 2.0:
        sequential_score += 1.5
        reasons.append(f"prefill:seq_gap_{prefill['sequential_to_full_gap_pct']:.1f}")
    if float(decode["sequential_to_full_gap_pct"]) >= 2.0:
        sequential_score += 1.5
        reasons.append(f"decode:seq_gap_{decode['sequential_to_full_gap_pct']:.1f}")
    score += sequential_score

    penalty = 0.0
    if diff_count == 0:
        penalty += 1.5
        reasons.append("penalty:no_state_diff")
    if (
        float(prefill.get("route_only_to_route_dvfs_gain_pct", 0.0)) < 1.0 and
        float(prefill.get("route_dvfs_to_route_dvfs_tp_gain_pct", 0.0)) < 1.0 and
        float(prefill.get("route_dvfs_tp_to_full_joint_gain_pct", 0.0)) < 1.0 and
        float(decode.get("route_only_to_route_dvfs_gain_pct", 0.0)) < 1.0 and
        float(decode.get("route_dvfs_to_route_dvfs_tp_gain_pct", 0.0)) < 1.0 and
        float(decode.get("route_dvfs_tp_to_full_joint_gain_pct", 0.0)) < 1.0
    ):
        penalty += 2.0
        reasons.append("penalty:flat_ladder")
    score -= penalty

    summary = {
        "obs2_score": round(score, 3),
        "reason_codes": "|".join(reasons),
        "full_joint_config_diff_count": diff_count,
    }
    summary.update({f"prefill_{k}": v for k, v in prefill.items()})
    summary.update({f"decode_{k}": v for k, v in decode.items()})
    return summary, selected_df


def regime_id(rate: int, ttft_slo_ms: int, tpot_slo_ms: int, mix_variant: str) -> str:
    return f"r{rate}_ttft{ttft_slo_ms}_tpot{tpot_slo_ms}_{mix_variant}"


def run_regime(
    backend,
    search_space,
    reference,
    rate: int,
    ttft_slo_ms: int,
    tpot_slo_ms: int,
    mix_variant: str,
    state_mixes: Dict[str, Dict[str, float]],
    trace_duration_s: float,
    trace_seed: int,
    output_dir: Path,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    states = make_state_specs(
        rate_rps=rate,
        low_rate_rps=2.0,
        window_s=5.0,
        trace_duration_s=trace_duration_s,
        trace_seed=trace_seed,
        short_il=128,
        long_il=1024,
        short_ol=128,
        long_ol=1024,
        states=PRIMARY_STATES,
        state_mixes=state_mixes,
    )

    selected_rows: List[Dict[str, object]] = []
    evaluation_rows: List[Dict[str, object]] = []
    replay_records: List[Dict[str, object]] = []
    trace_dir = output_dir / f"{regime_id(rate, ttft_slo_ms, tpot_slo_ms, mix_variant)}_traces"

    for state in states:
        trace_path = write_replay_trace(trace_dir, state)
        for policy in DEFAULT_POLICIES:
            rows, selected = evaluate_policy(
                backend,
                state,
                policy,
                search_space,
                ttft_slo_ms,
                tpot_slo_ms,
                reference,
            )
            evaluation_rows.extend(rows)
            selected_rows.append(selected)
            replay_records.append(
                _selected_config_record(
                    state,
                    selected,
                    trace_path,
                    ttft_slo_ms,
                    tpot_slo_ms,
                    reference,
                )
            )

    regime_name = regime_id(rate, ttft_slo_ms, tpot_slo_ms, mix_variant)
    regime_prefix = output_dir / regime_name
    write_outputs(
        output_prefix=regime_prefix,
        experiment_meta={
            "rate": rate,
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
            "mix_variant": mix_variant,
        },
        evaluation_rows=evaluation_rows,
        selected_rows=selected_rows,
        replay_records=replay_records,
    )

    selected_df = pd.DataFrame(selected_rows)
    summary, _ = score_regime(selected_df)
    summary.update(
        {
            "regime_id": regime_name,
            "rate": rate,
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
            "mix_variant": mix_variant,
            "selected_csv": str(regime_prefix.with_name(regime_prefix.name + "_selected.csv")),
        }
    )
    return summary, selected_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=str, default="3,4,5,6,7,8")
    parser.add_argument("--ttft-grid", type=str, default="300,500,800")
    parser.add_argument("--tpot-grid", type=str, default="80,120,160,200")
    parser.add_argument("--include-secondary-mixes", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--figure-top-k", type=int, default=3)
    parser.add_argument("--trace-duration", type=float, default=60.0)
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument("--model-dir-l40s", type=str, default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", type=str, default=str(paper_model_dir("models_l4")))
    parser.add_argument("--freq-l40s", type=str, default="990,1245,1500,1755,2010,2520")
    parser.add_argument("--freq-l4", type=str, default="780,990,1200,1410,1620,2040")
    parser.add_argument("--tp-l40s", type=str, default="1,2,4")
    parser.add_argument("--tp-l4", type=str, default="1,2,4")
    parser.add_argument("--active-l40s", type=str, default="0,1,2,4")
    parser.add_argument("--active-l4", type=str, default="0,2,4,8")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=figures_results_path("obs2_regime_search"),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = create_backend(args.model_dir_l40s, args.model_dir_l4, 5.0)
    reference = default_reference_config()
    search_space = default_search_space(backend).__class__(
        freq_l40s=parse_int_list(args.freq_l40s),
        freq_l4=parse_int_list(args.freq_l4),
        tp_l40s=parse_int_list(args.tp_l40s),
        tp_l4=parse_int_list(args.tp_l4),
        active_l40s=parse_int_list(args.active_l40s),
        active_l4=parse_int_list(args.active_l4),
    )

    mix_variants = dict(PRIMARY_MIXES)
    if args.include_secondary_mixes:
        mix_variants.update(SECONDARY_MIXES)

    ranked_rows: List[Dict[str, object]] = []
    top_config_rows: List[pd.DataFrame] = []

    rates = parse_int_list(args.rates)
    ttft_grid = parse_int_list(args.ttft_grid)
    tpot_grid = parse_int_list(args.tpot_grid)

    selected_by_regime: Dict[str, pd.DataFrame] = {}
    for mix_variant, state_mixes in mix_variants.items():
        for rate in rates:
            for ttft_slo_ms in ttft_grid:
                for tpot_slo_ms in tpot_grid:
                    summary, selected_df = run_regime(
                        backend=backend,
                        search_space=search_space,
                        reference=reference,
                        rate=rate,
                        ttft_slo_ms=ttft_slo_ms,
                        tpot_slo_ms=tpot_slo_ms,
                        mix_variant=mix_variant,
                        state_mixes=state_mixes,
                        trace_duration_s=args.trace_duration,
                        trace_seed=args.trace_seed,
                        output_dir=output_dir,
                    )
                    ranked_rows.append(summary)
                    selected_by_regime[summary["regime_id"]] = selected_df.copy()

    ranked_df = pd.DataFrame(ranked_rows).sort_values(
        by=["obs2_score", "decode_sequential_to_full_gap_pct", "prefill_sequential_to_full_gap_pct"],
        ascending=[False, False, False],
    )
    ranked_csv = output_dir / "obs2_regime_ranked.csv"
    ranked_df.to_csv(ranked_csv, index=False)

    top_regimes = ranked_df.head(args.top_k)
    top_config_frames = []
    for _, row in top_regimes.iterrows():
        regime_df = selected_by_regime[row["regime_id"]].copy()
        if "regime_id" in regime_df.columns:
            regime_df["regime_id"] = row["regime_id"]
        else:
            regime_df.insert(0, "regime_id", row["regime_id"])
        for idx, (col, value) in enumerate(
            [
                ("mix_variant", row["mix_variant"]),
                ("rate", row["rate"]),
                ("ttft_slo_ms", row["ttft_slo_ms"]),
                ("tpot_slo_ms", row["tpot_slo_ms"]),
            ],
            start=1,
        ):
            if col in regime_df.columns:
                regime_df[col] = value
            else:
                regime_df.insert(idx, col, value)
        top_config_frames.append(regime_df)
    top_configs_df = pd.concat(top_config_frames, ignore_index=True)
    top_configs_csv = output_dir / "obs2_regime_top_configs.csv"
    top_configs_df.to_csv(top_configs_csv, index=False)

    # Generate figures for top few regimes.
    for _, row in ranked_df.head(args.figure_top_k).iterrows():
        regime_id_value = row["regime_id"]
        regime_df = selected_by_regime[regime_id_value]
        regime_prefix = output_dir / regime_id_value
        summary_df = restricted_delta_table(regime_df)
        summary_df.to_csv(regime_prefix.with_name(regime_prefix.name + "_summary.csv"), index=False)
        export_config_table(regime_df, regime_prefix)
        generate_main_figure(regime_df, regime_prefix)
        generate_lightload_figure(regime_df, regime_prefix)

    best = ranked_df.iloc[0]
    backup = ranked_df.iloc[1] if len(ranked_df) > 1 else None
    recommendation = {
        "best_regime": best.to_dict(),
        "backup_regime": None if backup is None else backup.to_dict(),
    }
    recommendation_path = output_dir / "obs2_recommendation.json"
    recommendation_path.write_text(json.dumps(recommendation, indent=2))

    print(f"ranked_csv: {ranked_csv}")
    print(f"top_configs_csv: {top_configs_csv}")
    print(f"recommendation_json: {recommendation_path}")
    print("Top regimes:")
    cols = [
        "regime_id",
        "obs2_score",
        "reason_codes",
        "prefill_static_feasible",
        "decode_static_feasible",
        "prefill_sequential_to_full_gap_pct",
        "decode_sequential_to_full_gap_pct",
        "full_joint_config_diff_count",
    ]
    print(ranked_df.head(min(args.top_k, 10))[cols].to_string(index=False))


if __name__ == "__main__":
    main()
