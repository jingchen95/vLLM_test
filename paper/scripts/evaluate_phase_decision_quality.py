#!/usr/bin/env python3
"""
Offline phase-level decision-quality evaluation for the current scheduler stack.

For each measured workload (il, ol, rate), compare the model-chosen candidate
against the measured best feasible candidate for the same phase and SLO.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from model_training_common import GPU_CONFIGS, prepare_data  # noqa: E402
from paths import analysis_prefix, paper_model_dir  # noqa: E402
from scheduler import EnergyScheduler  # noqa: E402


PHASE_TARGET = {
    "prefill": "p99_ttft",
    "decode": "p99_tpot",
}


def load_actual_data(gpu_type: str) -> pd.DataFrame:
    cfg = GPU_CONFIGS[gpu_type]
    data = prepare_data(cfg["data_file"], cfg["anomaly_filter"], cfg["max_freq"])
    data["total_power"] = data["power_per_gpu"] * data["tp"]
    return data


def evaluate_gpu_phase(gpu_type: str, scheduler: EnergyScheduler, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = load_actual_data(gpu_type)
    target_col = PHASE_TARGET[phase]
    detail_rows: List[Dict[str, object]] = []

    workloads = actual[["il", "ol", "rate"]].drop_duplicates().sort_values(["il", "ol", "rate"])
    for slo in scheduler.SLO_THRESHOLDS:
        for _, wl in workloads.iterrows():
            il, ol, rate = int(wl["il"]), int(wl["ol"]), int(wl["rate"])
            candidates = actual[
                (actual["il"] == il) &
                (actual["ol"] == ol) &
                (actual["rate"] == rate)
            ].copy()
            if candidates.empty:
                continue

            candidates["actual_feasible"] = candidates[target_col] <= slo
            oracle = candidates[candidates["actual_feasible"]].sort_values(["total_power", "tp", "freq"]).head(1)
            oracle_exists = not oracle.empty
            oracle_row = oracle.iloc[0] if oracle_exists else None

            preds = []
            for _, row in candidates.iterrows():
                pred = scheduler.predict_config(
                    int(row["il"]),
                    int(row["ol"]),
                    int(row["tp"]),
                    int(row["freq"]),
                    float(row["rate"]),
                    int(slo),
                    phase=phase,
                )
                preds.append({
                    "tp": int(row["tp"]),
                    "freq": int(row["freq"]),
                    "pred_safe": bool(pred["is_safe"]),
                    "pred_power": float(pred["total_power_w"]),
                    "pred_latency": float(pred["p99_ttft_ms"] if phase == "prefill" else pred["p99_tpot_ms"]),
                })
            pred_df = pd.DataFrame(preds)
            merged = candidates.merge(pred_df, on=["tp", "freq"], how="left")
            pred_safe = merged[merged["pred_safe"]].sort_values(["pred_power", "tp", "freq"]).reset_index(drop=True)
            chosen_exists = not pred_safe.empty
            chosen_row = pred_safe.iloc[0] if chosen_exists else None

            oracle_key = None
            if oracle_exists:
                oracle_key = (int(oracle_row["tp"]), int(oracle_row["freq"]))
            chosen_key = None
            if chosen_exists:
                chosen_key = (int(chosen_row["tp"]), int(chosen_row["freq"]))

            oracle_rank = None
            top1_match = False
            top3_recall = False
            regret_pct = None
            chosen_actual_feasible = None
            chosen_actual_power = None
            oracle_actual_power = None
            if oracle_exists:
                oracle_actual_power = float(oracle_row["total_power"])
                oracle_matches = pred_safe[(pred_safe["tp"] == oracle_row["tp"]) & (pred_safe["freq"] == oracle_row["freq"])]
                if not oracle_matches.empty:
                    oracle_rank = int(oracle_matches.index[0]) + 1
                    top3_recall = oracle_rank <= 3
            if oracle_exists and chosen_exists:
                top1_match = chosen_key == oracle_key
                chosen_actual_feasible = bool(chosen_row["actual_feasible"])
                chosen_actual_power = float(chosen_row["total_power"])
                regret_pct = ((chosen_actual_power - oracle_actual_power) / oracle_actual_power * 100.0)
            elif chosen_exists:
                chosen_actual_feasible = bool(chosen_row["actual_feasible"])
                chosen_actual_power = float(chosen_row["total_power"])

            detail_rows.append({
                "gpu_type": gpu_type,
                "phase": phase,
                "slo_ms": int(slo),
                "il": il,
                "ol": ol,
                "rate": rate,
                "num_candidates": int(len(merged)),
                "num_pred_safe": int(len(pred_safe)),
                "num_actual_feasible": int(merged["actual_feasible"].sum()),
                "pruning_ratio_pct": round((1.0 - len(pred_safe) / max(1, len(merged))) * 100.0, 2),
                "oracle_exists": oracle_exists,
                "chosen_exists": chosen_exists,
                "oracle_tp": int(oracle_row["tp"]) if oracle_exists else None,
                "oracle_freq": int(oracle_row["freq"]) if oracle_exists else None,
                "oracle_actual_power": round(oracle_actual_power, 3) if oracle_actual_power is not None else None,
                "chosen_tp": int(chosen_row["tp"]) if chosen_exists else None,
                "chosen_freq": int(chosen_row["freq"]) if chosen_exists else None,
                "chosen_pred_power": round(float(chosen_row["pred_power"]), 3) if chosen_exists else None,
                "chosen_actual_power": round(chosen_actual_power, 3) if chosen_actual_power is not None else None,
                "chosen_actual_feasible": chosen_actual_feasible,
                "oracle_rank_in_predicted_safe": oracle_rank,
                "top1_match": top1_match,
                "top3_recall": top3_recall,
                "regret_pct": round(regret_pct, 3) if regret_pct is not None else None,
                "unsafe_admission": bool(chosen_exists and chosen_actual_feasible is False),
                "oracle_rejected_by_model": bool(oracle_exists and oracle_rank is None),
            })

    detail_df = pd.DataFrame(detail_rows)

    summary_rows = []
    group_cols = ["gpu_type", "phase", "slo_ms"]
    for keys, group in detail_df.groupby(group_cols):
        oracle_group = group[group["oracle_exists"]].copy()
        chosen_group = oracle_group[oracle_group["chosen_exists"]].copy()
        summary_rows.append({
            "gpu_type": keys[0],
            "phase": keys[1],
            "slo_ms": int(keys[2]),
            "workloads": int(len(group)),
            "oracle_exists_workloads": int(len(oracle_group)),
            "chosen_exists_workloads": int(len(group[group["chosen_exists"]])),
            "top1_match_rate_pct": round(chosen_group["top1_match"].mean() * 100.0, 2) if not chosen_group.empty else None,
            "top3_recall_rate_pct": round(oracle_group["top3_recall"].mean() * 100.0, 2) if not oracle_group.empty else None,
            "mean_regret_pct": round(chosen_group["regret_pct"].mean(), 3) if not chosen_group.empty else None,
            "median_regret_pct": round(chosen_group["regret_pct"].median(), 3) if not chosen_group.empty else None,
            "p90_regret_pct": round(chosen_group["regret_pct"].quantile(0.9), 3) if not chosen_group.empty else None,
            "unsafe_admission_rate_pct": round(chosen_group["unsafe_admission"].mean() * 100.0, 2) if not chosen_group.empty else None,
            "oracle_rejected_rate_pct": round(oracle_group["oracle_rejected_by_model"].mean() * 100.0, 2) if not oracle_group.empty else None,
            "mean_pruning_ratio_pct": round(group["pruning_ratio_pct"].mean(), 2),
            "mean_pred_safe_count": round(group["num_pred_safe"].mean(), 2),
        })

    return detail_df, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir-l40s", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", default=str(paper_model_dir("models_l4")))
    parser.add_argument("--output-prefix", default=str(analysis_prefix("phase_decision_quality")))
    args = parser.parse_args()

    schedulers = {
        "l40s": EnergyScheduler(args.model_dir_l40s),
        "l4": EnergyScheduler(args.model_dir_l4),
    }

    details = []
    summaries = []
    for gpu_type, scheduler in schedulers.items():
        for phase in ("prefill", "decode"):
            detail_df, summary_df = evaluate_gpu_phase(gpu_type, scheduler, phase)
            details.append(detail_df)
            summaries.append(summary_df)

    detail_df = pd.concat(details, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(f"{prefix}_detail.csv", index=False)
    summary_df.to_csv(f"{prefix}_summary.csv", index=False)

    with open(f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
        for _, row in summary_df.sort_values(["gpu_type", "phase", "slo_ms"]).iterrows():
            f.write(
                f"{row['gpu_type']}/{row['phase']}/{int(row['slo_ms'])}ms: "
                f"top1={row['top1_match_rate_pct']}%, top3={row['top3_recall_rate_pct']}%, "
                f"mean_regret={row['mean_regret_pct']}%, unsafe_admit={row['unsafe_admission_rate_pct']}%, "
                f"oracle_rejected={row['oracle_rejected_rate_pct']}%, pruning={row['mean_pruning_ratio_pct']}%\n"
            )


if __name__ == "__main__":
    main()
