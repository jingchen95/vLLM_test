#!/usr/bin/env python3
"""
Analyze the remaining hard decode bins for the current phase-specific stack.

Outputs:
  - per-case detailed OOF rows with candidate ids and current guard decisions
  - per-case false-safe rows
  - per-case summary statistics and a coarse diagnosis label
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from model_training_common import (  # noqa: E402
    DECODE_CLF_FEATURES,
    GPU_CONFIGS,
    build_group_key,
    compute_features,
    prepare_data,
    train_capacity_model,
)
from paths import analysis_prefix, paper_model_dir  # noqa: E402


DEFAULT_CASES = [
    {"gpu_type": "l4", "phase": "decode", "slo_ms": 500},
    {"gpu_type": "l40s", "phase": "decode", "slo_ms": 200},
]


def _load_case_data(gpu_type: str) -> pd.DataFrame:
    cfg = GPU_CONFIGS[gpu_type]
    data = prepare_data(cfg["data_file"], cfg["anomaly_filter"], cfg["max_freq"])
    cap_model, cap_groups, _ = train_capacity_model(data, cfg["max_freq"])
    data = compute_features(data, cap_model, cap_groups, cfg["max_freq"])
    data["group_full"] = build_group_key(data)
    data["gpu_type"] = gpu_type
    return data


def _phase_guard_setting(model_dir: Path, phase: str, slo_ms: int) -> Dict[str, float]:
    cfg = joblib.load(model_dir / "config.pkl")
    phase_cfg = cfg.get("PHASE_GUARD_SETTINGS", {}).get(phase, {})
    chosen = phase_cfg.get(slo_ms, phase_cfg.get(str(slo_ms)))
    if chosen is not None:
        return {"p_th": float(chosen["p_th"]), "rho_th": float(chosen["rho_th"])}
    legacy = cfg["GUARD_SETTINGS"][slo_ms]
    return {"p_th": float(legacy["p_th"]), "rho_th": float(legacy["rho_th"])}


def _compute_decode_oof(data: pd.DataFrame, slo_ms: int) -> pd.DataFrame:
    groups = data["group_full"].values
    y_true = (data["p99_tpot"] > slo_ms).astype(int).values
    oof_prob = np.full(len(data), np.nan)

    gkf = GroupKFold(n_splits=5)
    cfg = GPU_CONFIGS[data["gpu_type"].iloc[0]]
    for train_idx, test_idx in gkf.split(data, groups=groups):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        cap_fold, cap_fold_df, _ = train_capacity_model(train, cfg["max_freq"])
        train = compute_features(train, cap_fold, cap_fold_df, cfg["max_freq"])
        test = compute_features(test, cap_fold, cap_fold_df, cfg["max_freq"])
        y_train = (train["p99_tpot"] > slo_ms).astype(int).values
        sw = np.where(y_train == 1, 5.0, 1.0)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[DECODE_CLF_FEATURES].values)
        x_test = scaler.transform(test[DECODE_CLF_FEATURES].values)
        if len(np.unique(y_train)) < 2:
            prob = np.full(len(test), float(y_train[0]))
        else:
            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.1,
                min_samples_leaf=5,
                random_state=42,
            )
            clf.fit(x_train, y_train, sample_weight=sw)
            prob = clf.predict_proba(x_test)[:, 1]
        oof_prob[test_idx] = prob

    detail = data.copy()
    detail["actual_unsafe"] = y_true
    detail["p_violate_oof"] = oof_prob
    detail["rho_phase_oof"] = detail["rho_decode"]
    return detail


def _quantiles(series: pd.Series) -> tuple[float, float, float]:
    if len(series) == 0:
        return (float("nan"), float("nan"), float("nan"))
    q = series.quantile([0.1, 0.5, 0.9]).tolist()
    return tuple(round(float(v), 4) for v in q)


def _diagnose(summary: Dict[str, object]) -> str:
    unsafe_support = int(summary["unsafe_support"])
    if unsafe_support < 8:
        return "low_support_artifact"
    prob_auc = float(summary["prob_auc"])
    rho_auc = float(summary["rho_auc"])
    false_safe_rate = float(summary["false_safe_rate_pct"])
    if prob_auc < 0.65 and rho_auc < 0.65:
        return "weak_feature_separability_or_intrinsic_overlap"
    if false_safe_rate > 20 and prob_auc >= 0.70:
        return "classifier_boundary_or_guard_band_issue"
    if prob_auc < 0.75 and rho_auc < 0.75:
        return "mixed_overlap_and_threshold_limit"
    return "moderate_boundary_issue"


def analyze_case(model_dir: Path, case: Dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    gpu_type = str(case["gpu_type"])
    slo_ms = int(case["slo_ms"])
    data = _load_case_data(gpu_type)
    detail = _compute_decode_oof(data, slo_ms)
    guard = _phase_guard_setting(model_dir, "decode", slo_ms)
    detail["chosen_p_th"] = guard["p_th"]
    detail["chosen_rho_th"] = guard["rho_th"]
    detail["pred_safe_current"] = (
        (detail["p_violate_oof"] < guard["p_th"]) &
        (detail["rho_phase_oof"] < guard["rho_th"])
    )
    detail["false_safe_current"] = detail["pred_safe_current"] & (detail["actual_unsafe"] == 1)
    detail["false_unsafe_current"] = (~detail["pred_safe_current"]) & (detail["actual_unsafe"] == 0)
    detail["case_id"] = f"{gpu_type}_decode_{slo_ms}"

    safe = detail[detail["actual_unsafe"] == 0]
    unsafe = detail[detail["actual_unsafe"] == 1]
    prob_auc = roc_auc_score(detail["actual_unsafe"], detail["p_violate_oof"]) if unsafe.shape[0] > 0 else float("nan")
    rho_auc = roc_auc_score(detail["actual_unsafe"], detail["rho_phase_oof"]) if unsafe.shape[0] > 0 else float("nan")
    summary = {
        "case_id": f"{gpu_type}_decode_{slo_ms}",
        "gpu_type": gpu_type,
        "phase": "decode",
        "slo_ms": slo_ms,
        "unsafe_support": int(unsafe.shape[0]),
        "safe_support": int(safe.shape[0]),
        "chosen_p_th": guard["p_th"],
        "chosen_rho_th": guard["rho_th"],
        "false_safe": int(detail["false_safe_current"].sum()),
        "false_unsafe": int(detail["false_unsafe_current"].sum()),
        "admit_rate_pct": round(float(detail["pred_safe_current"].mean() * 100), 2),
        "safe_precision_pct": round(
            float((~detail["actual_unsafe"].astype(bool) & detail["pred_safe_current"]).sum() /
                  max(1, detail["pred_safe_current"].sum()) * 100), 2
        ),
        "false_safe_rate_pct": round(float(detail["false_safe_current"].sum() / max(1, unsafe.shape[0]) * 100), 2),
        "prob_auc": round(float(prob_auc), 4),
        "rho_auc": round(float(rho_auc), 4),
        "safe_prob_q10": _quantiles(safe["p_violate_oof"])[0],
        "safe_prob_q50": _quantiles(safe["p_violate_oof"])[1],
        "safe_prob_q90": _quantiles(safe["p_violate_oof"])[2],
        "unsafe_prob_q10": _quantiles(unsafe["p_violate_oof"])[0],
        "unsafe_prob_q50": _quantiles(unsafe["p_violate_oof"])[1],
        "unsafe_prob_q90": _quantiles(unsafe["p_violate_oof"])[2],
        "safe_rho_q10": _quantiles(safe["rho_phase_oof"])[0],
        "safe_rho_q50": _quantiles(safe["rho_phase_oof"])[1],
        "safe_rho_q90": _quantiles(safe["rho_phase_oof"])[2],
        "unsafe_rho_q10": _quantiles(unsafe["rho_phase_oof"])[0],
        "unsafe_rho_q50": _quantiles(unsafe["rho_phase_oof"])[1],
        "unsafe_rho_q90": _quantiles(unsafe["rho_phase_oof"])[2],
    }
    summary["diagnosis"] = _diagnose(summary)

    detail_cols = [
        "case_id", "gpu_type", "il", "ol", "rate", "tp", "freq",
        "p99_tpot", "power_per_gpu", "rho_decode", "rho", "actual_unsafe",
        "p_violate_oof", "rho_phase_oof", "chosen_p_th", "chosen_rho_th",
        "pred_safe_current", "false_safe_current", "false_unsafe_current",
        "log_il", "log_ol", "log_rate", "log_d_prefill", "log_d_decode",
        "decode_frac", "rho_decode_overflow", "log_rho_decode",
    ]
    return detail[detail_cols].copy(), detail[detail["false_safe_current"]].copy(), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir-l40s", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", default=str(paper_model_dir("models_l4")))
    parser.add_argument("--output-prefix", default=str(analysis_prefix("hard_decode_bins")))
    args = parser.parse_args()

    outputs_detail: List[pd.DataFrame] = []
    outputs_false_safe: List[pd.DataFrame] = []
    outputs_summary: List[Dict[str, object]] = []

    for case in DEFAULT_CASES:
        model_dir = Path(args.model_dir_l4 if case["gpu_type"] == "l4" else args.model_dir_l40s)
        detail, false_safe, summary = analyze_case(model_dir, case)
        outputs_detail.append(detail)
        outputs_false_safe.append(false_safe)
        outputs_summary.append(summary)

    summary_df = pd.DataFrame(outputs_summary)
    detail_df = pd.concat(outputs_detail, ignore_index=True)
    false_safe_df = pd.concat(outputs_false_safe, ignore_index=True)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(f"{prefix}_summary.csv", index=False)
    detail_df.to_csv(f"{prefix}_detail.csv", index=False)
    false_safe_df.to_csv(f"{prefix}_false_safe.csv", index=False)

    txt_lines = []
    for _, row in summary_df.iterrows():
        txt_lines.append(
            f"{row['case_id']}: diagnosis={row['diagnosis']}, "
            f"false-safe={int(row['false_safe'])}/{int(row['unsafe_support'])}, "
            f"admit={row['admit_rate_pct']:.1f}%, prob_auc={row['prob_auc']:.3f}, rho_auc={row['rho_auc']:.3f}"
        )
    Path(f"{prefix}_summary.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
