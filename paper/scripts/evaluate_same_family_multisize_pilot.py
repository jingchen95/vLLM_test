#!/usr/bin/env python3
"""
Evaluate a same-family multi-size phase-only pilot under three modes:

  Mode A: zero-shot reuse of the frozen existing-size models
  Mode B: new-size oracle trained on the pilot data
  Mode C: lightweight adaptation via threshold-only recalibration

The script stays intentionally lightweight and reuses the current phase-aware
scheduler runtime plus the existing training runner for oracle export.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from model_fitting_runner import (  # noqa: E402
    MIN_UNSAFE_SUPPORT_FOR_RETUNE,
    _evaluate_guard_metrics,
    _select_guard_threshold,
    run_training_pipeline,
)
from model_training_common import passthrough_filter  # noqa: E402
from paths import analysis_prefix, paper_model_dir  # noqa: E402
from scheduler import EnergyScheduler  # noqa: E402


PHASE_TARGET = {
    "prefill": "p99_ttft_ms",
    "decode": "p99_tpot_ms",
}


def build_dataset_metadata(df: pd.DataFrame, gpu_type: str) -> dict:
    row = df[df["gpu_type"] == gpu_type].iloc[0]
    return {
        "dataset_id": f"pilot_{gpu_type}_{row['model_id']}",
        "csv_path": "generated_by_evaluate_same_family_multisize_pilot.py",
        "gpu_type": gpu_type,
        "model_id": row["model_id"],
        "hf_name": row["hf_name"],
        "model_family": row.get("model_family", "unknown"),
        "param_count_b": float(row.get("param_count_b", np.nan)),
        "serving_stack": "vllm_phase_only_cd_pilot",
        "notes": "Same-family multi-size pilot summary CSV.",
    }


def _predict_row(scheduler: EnergyScheduler, row: pd.Series) -> dict:
    phase = row["phase"]
    slo_values = scheduler.SLO_THRESHOLDS
    predictions = []
    for slo in slo_values:
        if phase == "prefill":
            pred = scheduler.predict_prefill_config(
                int(row["input_len"]),
                int(row["tp_degree"]),
                int(row["gpu_freq_mhz"]),
                float(row["request_rate"]),
                int(slo),
                proxy_ol=int(row["output_len"]),
            )
            pred_latency = float(pred["p99_ttft_ms"])
        else:
            pred = scheduler.predict_decode_config(
                int(row["output_len"]),
                int(row["tp_degree"]),
                int(row["gpu_freq_mhz"]),
                float(row["request_rate"]),
                int(slo),
                proxy_il=int(row["input_len"]),
            )
            pred_latency = float(pred["p99_tpot_ms"])
        predictions.append(
            {
                "slo_ms": int(slo),
                "pred_safe": bool(pred["is_safe"]),
                "pred_p_violate": float(pred["p_violate"]),
                "pred_rho": float(pred["rho"]),
                "pred_latency_ms": pred_latency,
                "pred_power_w": float(pred["total_power_w"]),
            }
        )
    return {
        "row_id": int(row["row_id"]),
        "predictions": predictions,
    }


def run_predictions(df: pd.DataFrame,
                    schedulers: Dict[str, EnergyScheduler],
                    mode: str) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df.iterrows():
        scheduler = schedulers[row["gpu_type"]]
        for pred in _predict_row(scheduler, row)["predictions"]:
            actual_latency = float(row[PHASE_TARGET[row["phase"]]])
            actual_feasible = actual_latency <= pred["slo_ms"]
            rows.append(
                {
                    "mode": mode,
                    "row_id": int(row["row_id"]),
                    "gpu_type": row["gpu_type"],
                    "phase": row["phase"],
                    "slo_ms": int(pred["slo_ms"]),
                    "input_len": int(row["input_len"]),
                    "output_len": int(row["output_len"]),
                    "request_rate": int(row["request_rate"]),
                    "tp_degree": int(row["tp_degree"]),
                    "gpu_freq_mhz": int(row["gpu_freq_mhz"]),
                    "actual_latency_ms": actual_latency,
                    "actual_power_w": float(row["avg_power_w"]),
                    "actual_energy_j": float(row["cluster_energy_j"]) if pd.notna(row["cluster_energy_j"]) else np.nan,
                    "actual_j_per_req": float(row["j_per_req"]) if pd.notna(row["j_per_req"]) else np.nan,
                    "actual_feasible": bool(actual_feasible),
                    "pred_safe": bool(pred["pred_safe"]),
                    "pred_p_violate": float(pred["pred_p_violate"]),
                    "pred_rho": float(pred["pred_rho"]),
                    "pred_latency_ms": float(pred["pred_latency_ms"]),
                    "pred_power_w": float(pred["pred_power_w"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_feasibility(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail_df.groupby(["mode", "gpu_type", "phase", "slo_ms"]):
        actual_safe = group["actual_feasible"].astype(bool).values
        pred_safe = group["pred_safe"].astype(bool).values
        tp = int((actual_safe & pred_safe).sum())
        fp = int((~actual_safe & pred_safe).sum())
        tn = int((~actual_safe & ~pred_safe).sum())
        fn = int((actual_safe & ~pred_safe).sum())
        rows.append(
            {
                "mode": keys[0],
                "gpu_type": keys[1],
                "phase": keys[2],
                "slo_ms": int(keys[3]),
                "support": int(len(group)),
                "unsafe_support": int((~actual_safe).sum()),
                "safe_support": int(actual_safe.sum()),
                "false_safe": fp,
                "false_unsafe": fn,
                "true_safe": tp,
                "true_unsafe": tn,
                "false_safe_rate_pct": round(fp / max(1, (~actual_safe).sum()) * 100.0, 2),
                "false_unsafe_rate_pct": round(fn / max(1, actual_safe.sum()) * 100.0, 2),
                "admit_rate_pct": round(pred_safe.mean() * 100.0, 2),
                "safe_precision_pct": round(tp / max(1, tp + fp) * 100.0, 2),
            }
        )
    return pd.DataFrame(rows)


def summarize_prediction(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail_df.groupby(["mode", "gpu_type", "phase"]):
        base_group = group.sort_values("slo_ms").drop_duplicates(["row_id"])
        latency_true = base_group["actual_latency_ms"].values
        latency_pred = base_group["pred_latency_ms"].values
        power_true = base_group["actual_power_w"].values
        power_pred = base_group["pred_power_w"].values
        latency_mape = np.mean(np.abs((latency_pred - latency_true) / np.maximum(latency_true, 1e-6))) * 100.0
        power_mape = np.mean(np.abs((power_pred - power_true) / np.maximum(power_true, 1e-6))) * 100.0
        rows.append(
            {
                "mode": keys[0],
                "gpu_type": keys[1],
                "phase": keys[2],
                "rows": int(len(base_group)),
                "latency_mape_pct": round(float(latency_mape), 2),
                "power_mape_pct": round(float(power_mape), 2),
                "latency_mae_ms": round(float(np.mean(np.abs(latency_pred - latency_true))), 3),
                "power_mae_w": round(float(np.mean(np.abs(power_pred - power_true))), 3),
            }
        )
    return pd.DataFrame(rows)


def summarize_decision_quality(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["mode", "gpu_type", "phase", "slo_ms", "input_len", "output_len", "request_rate"]
    workload_rows: List[dict] = []
    for keys, group in detail_df.groupby(group_cols):
        feasible = group[group["actual_feasible"]].copy()
        oracle = feasible.sort_values(["actual_j_per_req", "actual_power_w", "tp_degree", "gpu_freq_mhz"]).head(1)
        pred_safe = group[group["pred_safe"]].sort_values(["pred_power_w", "tp_degree", "gpu_freq_mhz"]).reset_index(drop=True)
        oracle_exists = not oracle.empty
        chosen_exists = not pred_safe.empty
        oracle_row = oracle.iloc[0] if oracle_exists else None
        chosen_row = pred_safe.iloc[0] if chosen_exists else None
        oracle_rank = None
        top1 = False
        top3 = False
        regret_pct = None
        unsafe_admission = False
        oracle_rejected = False
        if oracle_exists:
            oracle_match = pred_safe[
                (pred_safe["tp_degree"] == int(oracle_row["tp_degree"])) &
                (pred_safe["gpu_freq_mhz"] == int(oracle_row["gpu_freq_mhz"]))
            ]
            if not oracle_match.empty:
                oracle_rank = int(oracle_match.index[0]) + 1
                top3 = oracle_rank <= 3
            else:
                oracle_rejected = True
        if oracle_exists and chosen_exists:
            top1 = (
                int(chosen_row["tp_degree"]) == int(oracle_row["tp_degree"]) and
                int(chosen_row["gpu_freq_mhz"]) == int(oracle_row["gpu_freq_mhz"])
            )
            regret_pct = (
                (float(chosen_row["actual_j_per_req"]) - float(oracle_row["actual_j_per_req"])) /
                max(float(oracle_row["actual_j_per_req"]), 1e-9)
            ) * 100.0
            unsafe_admission = not bool(chosen_row["actual_feasible"])
        elif chosen_exists:
            unsafe_admission = not bool(chosen_row["actual_feasible"])

        workload_rows.append(
            {
                "mode": keys[0],
                "gpu_type": keys[1],
                "phase": keys[2],
                "slo_ms": int(keys[3]),
                "input_len": int(keys[4]),
                "output_len": int(keys[5]),
                "request_rate": int(keys[6]),
                "num_candidates": int(len(group)),
                "num_pred_safe": int(len(pred_safe)),
                "oracle_exists": oracle_exists,
                "chosen_exists": chosen_exists,
                "oracle_rank_in_pred_safe": oracle_rank,
                "top1_match": top1,
                "top3_recall": top3,
                "energy_regret_pct": round(float(regret_pct), 3) if regret_pct is not None else None,
                "unsafe_admission": unsafe_admission,
                "oracle_rejected_by_model": oracle_rejected,
                "pruning_ratio_pct": round((1.0 - len(pred_safe) / max(1, len(group))) * 100.0, 2),
            }
        )

    workload_df = pd.DataFrame(workload_rows)
    for keys, group in workload_df.groupby(["mode", "gpu_type", "phase", "slo_ms"]):
        oracle_group = group[group["oracle_exists"]]
        chosen_group = oracle_group[oracle_group["chosen_exists"]]
        rows.append(
            {
                "mode": keys[0],
                "gpu_type": keys[1],
                "phase": keys[2],
                "slo_ms": int(keys[3]),
                "workloads": int(len(group)),
                "oracle_exists_workloads": int(len(oracle_group)),
                "chosen_exists_workloads": int(len(group[group["chosen_exists"]])),
                "top1_match_rate_pct": round(chosen_group["top1_match"].mean() * 100.0, 2) if not chosen_group.empty else None,
                "top3_recall_rate_pct": round(oracle_group["top3_recall"].mean() * 100.0, 2) if not oracle_group.empty else None,
                "mean_energy_regret_pct": round(chosen_group["energy_regret_pct"].mean(), 3) if not chosen_group.empty else None,
                "median_energy_regret_pct": round(chosen_group["energy_regret_pct"].median(), 3) if not chosen_group.empty else None,
                "unsafe_admission_rate_pct": round(chosen_group["unsafe_admission"].mean() * 100.0, 2) if not chosen_group.empty else None,
                "oracle_rejected_rate_pct": round(oracle_group["oracle_rejected_by_model"].mean() * 100.0, 2) if not oracle_group.empty else None,
                "mean_pruning_ratio_pct": round(group["pruning_ratio_pct"].mean(), 2),
            }
        )
    return pd.DataFrame(rows), workload_df


def select_adaptation_thresholds(zero_shot_df: pd.DataFrame,
                                 base_guard_lookup: Dict[tuple[str, str, int], dict]) -> pd.DataFrame:
    rows = []
    for keys, group in zero_shot_df.groupby(["gpu_type", "phase", "slo_ms"]):
        y_true_unsafe = (~group["actual_feasible"].astype(bool)).astype(int).values
        p_vals = group["pred_p_violate"].values
        rho_vals = group["pred_rho"].values

        p_grid = sorted(
            set([0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01] + list(np.unique(np.round(p_vals, 4)))),
            reverse=True,
        )
        rho_grid = sorted(
            set([1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10] + list(np.round(np.clip(rho_vals, 0.0, 2.0), 4))),
            reverse=True,
        )

        sweep_rows = []
        current_guard = base_guard_lookup[(keys[0], keys[1], int(keys[2]))]
        for p_th in p_grid:
            for rho_th in rho_grid:
                metrics = _evaluate_guard_metrics(y_true_unsafe, p_vals, rho_vals, p_th, rho_th)
                sweep_row = {
                    "gpu_type": keys[0],
                    "phase": keys[1],
                    "slo_ms": int(keys[2]),
                    "p_th": float(p_th),
                    "rho_th": float(rho_th),
                    **metrics,
                }
                sweep_rows.append(sweep_row)

        sweep_df = pd.DataFrame(sweep_rows)
        base_row = sweep_df[
            np.isclose(sweep_df["p_th"], current_guard["p_th"]) &
            np.isclose(sweep_df["rho_th"], current_guard["rho_th"])
        ]
        if base_row.empty:
            base_metrics = _evaluate_guard_metrics(
                y_true_unsafe,
                p_vals,
                rho_vals,
                current_guard["p_th"],
                current_guard["rho_th"],
            )
            base_row = pd.DataFrame([{
                "p_th": current_guard["p_th"],
                "rho_th": current_guard["rho_th"],
                **base_metrics,
            }])
            sweep_df = pd.concat([sweep_df, base_row], ignore_index=True)

        chosen_row, reason = _select_guard_threshold(
            sweep_df,
            base_p_th=current_guard["p_th"],
            base_rho_th=current_guard["rho_th"],
            unsafe_support=int((~group["actual_feasible"].astype(bool)).sum()),
        )
        rows.append(
            {
                "gpu_type": keys[0],
                "phase": keys[1],
                "slo_ms": int(keys[2]),
                "base_p_th": float(current_guard["p_th"]),
                "base_rho_th": float(current_guard["rho_th"]),
                "chosen_p_th": float(chosen_row["p_th"]),
                "chosen_rho_th": float(chosen_row["rho_th"]),
                "selection_reason": reason,
                "unsafe_support": int((~group["actual_feasible"].astype(bool)).sum()),
                "retuned": (
                    float(chosen_row["p_th"]) != float(current_guard["p_th"]) or
                    float(chosen_row["rho_th"]) != float(current_guard["rho_th"])
                ),
            }
        )
    return pd.DataFrame(rows)


def apply_adaptation(zero_shot_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    thresholds = threshold_df.set_index(["gpu_type", "phase", "slo_ms"])[["chosen_p_th", "chosen_rho_th"]].to_dict("index")
    adapted = zero_shot_df.copy()
    adapted["mode"] = "threshold_adapt"

    def adapted_safe(row: pd.Series) -> bool:
        cfg = thresholds[(row["gpu_type"], row["phase"], int(row["slo_ms"]))]
        return (row["pred_p_violate"] < cfg["chosen_p_th"]) and (row["pred_rho"] < cfg["chosen_rho_th"])

    adapted["pred_safe"] = adapted.apply(adapted_safe, axis=1)
    return adapted


def export_oracle_models(pilot_df: pd.DataFrame, model_root: Path) -> dict[str, str]:
    model_root.mkdir(parents=True, exist_ok=True)
    csv_root = model_root / "pilot_csvs"
    csv_root.mkdir(parents=True, exist_ok=True)
    model_dirs = {}
    for gpu_type in ("l40s", "l4"):
        gpu_df = pilot_df[pilot_df["gpu_type"] == gpu_type].copy()
        if gpu_df.empty:
            continue
        csv_path = csv_root / f"{gpu_type}_pilot.csv"
        gpu_df.to_csv(csv_path, index=False)
        out_dir = model_root / gpu_type
        run_training_pipeline(
            gpu_type,
            data_file=str(csv_path),
            model_dir=str(out_dir),
            dataset_metadata=build_dataset_metadata(gpu_df, gpu_type),
            anomaly_filter=passthrough_filter,
        )
        model_dirs[gpu_type] = str(out_dir)
    return model_dirs


def export_adapted_models(base_model_dirs: dict[str, str],
                          threshold_df: pd.DataFrame,
                          output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    per_gpu = {}
    for gpu_type, src in base_model_dirs.items():
        dst = output_root / gpu_type
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        config = joblib.load(dst / "config.pkl")
        phase_guard = config.get("PHASE_GUARD_SETTINGS", {})
        for _, row in threshold_df[threshold_df["gpu_type"] == gpu_type].iterrows():
            phase_guard.setdefault(row["phase"], {})
            phase_guard[row["phase"]][int(row["slo_ms"])] = {
                "p_th": float(row["chosen_p_th"]),
                "rho_th": float(row["chosen_rho_th"]),
            }
        config["PHASE_GUARD_SETTINGS"] = phase_guard
        joblib.dump(config, dst / "config.pkl")
        per_gpu[gpu_type] = str(dst)
    return per_gpu


def write_mode_comparison(feas_df: pd.DataFrame,
                          pred_df: pd.DataFrame,
                          dec_df: pd.DataFrame,
                          prefix: Path) -> None:
    merge_keys = ["mode", "gpu_type", "phase"]
    merged = pred_df.merge(
        dec_df.drop(columns=["workloads", "oracle_exists_workloads", "chosen_exists_workloads"]),
        on=merge_keys,
        how="left",
    )
    merged = merged.merge(
        feas_df.drop(columns=["support", "safe_support", "true_safe", "true_unsafe"]),
        on=merge_keys + ["slo_ms"],
        how="left",
    )
    merged.to_csv(f"{prefix}_mode_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-csv", required=True)
    parser.add_argument("--model-dir-l40s", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", default=str(paper_model_dir("models_l4")))
    parser.add_argument("--oracle-model-root", default="")
    parser.add_argument("--adapted-model-root", default="")
    parser.add_argument("--output-prefix", default=str(analysis_prefix("same_family_multisize_eval")))
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    pilot_df = pd.read_csv(args.pilot_csv).copy()
    pilot_df["row_id"] = np.arange(len(pilot_df))

    base_model_dirs = {
        "l40s": args.model_dir_l40s,
        "l4": args.model_dir_l4,
    }
    zero_shot_schedulers = {
        "l40s": EnergyScheduler(args.model_dir_l40s),
        "l4": EnergyScheduler(args.model_dir_l4),
    }

    zero_shot_df = run_predictions(pilot_df, zero_shot_schedulers, mode="zero_shot")

    with tempfile.TemporaryDirectory(prefix="sweep_multisize_oracle_") as tmpdir:
        oracle_root = Path(args.oracle_model_root) if args.oracle_model_root else Path(tmpdir) / "oracle_models"
        oracle_model_dirs = export_oracle_models(pilot_df, oracle_root)
        oracle_schedulers = {
            gpu_type: EnergyScheduler(model_dir)
            for gpu_type, model_dir in oracle_model_dirs.items()
        }
        oracle_df = run_predictions(pilot_df, oracle_schedulers, mode="oracle")

        base_guard_lookup = {}
        for gpu_type, scheduler in zero_shot_schedulers.items():
            for phase in ("prefill", "decode"):
                for slo in scheduler.SLO_THRESHOLDS:
                    base_guard_lookup[(gpu_type, phase, int(slo))] = scheduler._phase_guard_setting(phase, int(slo))
        threshold_df = select_adaptation_thresholds(zero_shot_df, base_guard_lookup)
        threshold_df.to_csv(f"{prefix}_adaptation_thresholds.csv", index=False)
        adapted_df = apply_adaptation(zero_shot_df, threshold_df)

        if args.adapted_model_root:
            export_adapted_models(base_model_dirs, threshold_df, Path(args.adapted_model_root))

        detail_df = pd.concat([zero_shot_df, oracle_df, adapted_df], ignore_index=True)
        detail_df.to_csv(f"{prefix}_mode_detail.csv", index=False)

        feas_df = summarize_feasibility(detail_df)
        feas_df.to_csv(f"{prefix}_mode_feasibility_summary.csv", index=False)

        pred_df = summarize_prediction(detail_df)
        pred_df.to_csv(f"{prefix}_mode_prediction_summary.csv", index=False)

        decision_df, workload_df = summarize_decision_quality(detail_df)
        decision_df.to_csv(f"{prefix}_mode_decision_summary.csv", index=False)
        workload_df.to_csv(f"{prefix}_mode_decision_detail.csv", index=False)

        write_mode_comparison(feas_df, pred_df, decision_df, prefix)

        diagnosis_rows = []
        for gpu_type, phase, slo in [
            ("l4", "decode", 500),
            ("l40s", "decode", 200),
        ]:
            zs = feas_df[
                (feas_df["mode"] == "zero_shot") &
                (feas_df["gpu_type"] == gpu_type) &
                (feas_df["phase"] == phase) &
                (feas_df["slo_ms"] == slo)
            ]
            if zs.empty:
                continue
            oracle = feas_df[
                (feas_df["mode"] == "oracle") &
                (feas_df["gpu_type"] == gpu_type) &
                (feas_df["phase"] == phase) &
                (feas_df["slo_ms"] == slo)
            ]
            adapt = feas_df[
                (feas_df["mode"] == "threshold_adapt") &
                (feas_df["gpu_type"] == gpu_type) &
                (feas_df["phase"] == phase) &
                (feas_df["slo_ms"] == slo)
            ]
            diagnosis_rows.append(
                {
                    "gpu_type": gpu_type,
                    "phase": phase,
                    "slo_ms": slo,
                    "zero_shot_false_safe_rate_pct": float(zs.iloc[0]["false_safe_rate_pct"]),
                    "oracle_false_safe_rate_pct": float(oracle.iloc[0]["false_safe_rate_pct"]) if not oracle.empty else np.nan,
                    "adapt_false_safe_rate_pct": float(adapt.iloc[0]["false_safe_rate_pct"]) if not adapt.empty else np.nan,
                }
            )
        pd.DataFrame(diagnosis_rows).to_csv(f"{prefix}_hard_bin_summary.csv", index=False)

        summary_txt = []
        for _, row in decision_df.sort_values(["mode", "gpu_type", "phase", "slo_ms"]).iterrows():
            summary_txt.append(
                f"{row['mode']}/{row['gpu_type']}/{row['phase']}/{int(row['slo_ms'])}ms: "
                f"top1={row['top1_match_rate_pct']}%, "
                f"mean_energy_regret={row['mean_energy_regret_pct']}%, "
                f"unsafe_admit={row['unsafe_admission_rate_pct']}%, "
                f"oracle_rejected={row['oracle_rejected_rate_pct']}%, "
                f"pruning={row['mean_pruning_ratio_pct']}%"
            )
        Path(f"{prefix}_summary.txt").write_text("\n".join(summary_txt) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
