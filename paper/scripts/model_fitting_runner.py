#!/usr/bin/env python3
"""
Shared training runner for SWEEP-LLM single-pool models.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

from model_training_common import (
    CAP_FEATURES,
    CLF_FEATURES,
    DECODE_CLF_FEATURES,
    DECODE_POWER_FEATURES,
    DECODE_REG_FEATURES,
    ETA,
    GAMMA,
    GPU_CONFIGS,
    POWER_FEATURES,
    PREFILL_CLF_FEATURES,
    PREFILL_POWER_FEATURES,
    PREFILL_REG_FEATURES,
    REG_FEATURES,
    build_group_key,
    check_plateau,
    compute_features,
    prepare_data,
    train_capacity_model,
)
from dataset_manifest import get_dataset_manifest_for_gpu


PHASE_MODEL_SPECS = {
    "prefill": {
        "latency_target": "p99_ttft",
        "latency_label": "P99 TTFT",
        "clf_features": PREFILL_CLF_FEATURES,
        "reg_features": PREFILL_REG_FEATURES,
        "power_features": PREFILL_POWER_FEATURES,
        "rho_col": "rho_prefill",
    },
    "decode": {
        "latency_target": "p99_tpot",
        "latency_label": "P99 TPOT",
        "clf_features": DECODE_CLF_FEATURES,
        "reg_features": DECODE_REG_FEATURES,
        "power_features": DECODE_POWER_FEATURES,
        "rho_col": "rho_decode",
    },
}

GUARD_P_GRID = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
GUARD_RHO_GRID = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
MIN_UNSAFE_SUPPORT_FOR_RETUNE = 8
LOW_SUPPORT_MAX_FALSE_SAFE_TO_IGNORE = 1


def _print_key_configs(all_data: pd.DataFrame, gpu_type: str, max_freq: int) -> None:
    if gpu_type == "l40s":
        print(f"\n  2048:32 configs at f={max_freq}:")
        sub = all_data[(all_data["il"] == 2048) &
                       (all_data["ol"] == 32) &
                       (all_data["freq"] == max_freq)]
        for _, r in sub.drop_duplicates(["tp", "rate"]).sort_values(["tp", "rate"]).iterrows():
            print(f"    TP={int(r['tp'])} r={int(r['rate']):>2d}: C_hat={r['C_hat']:.0f} "
                  f"({'plateau' if r['plateau_confirmed'] else f'η={ETA}'}), "
                  f"rho={r['rho']:.3f}, p99={r['p99_ttft']:.0f}ms")
    else:
        print(f"\n  Key configs at f={max_freq}:")
        sub = all_data[all_data["freq"] == max_freq]
        cols = ["il", "ol", "tp", "rate"]
        for _, r in sub.drop_duplicates(cols).sort_values(cols).iterrows():
            print(f"    {int(r['il']):>5d}:{int(r['ol']):<5d} TP={int(r['tp'])} "
                  f"r={int(r['rate']):>2d}: C_hat={r['C_hat']:.0f} "
                  f"({'plateau' if r['plateau_confirmed'] else f'η={ETA}'}), "
                  f"rho={r['rho']:.3f}, p99={r['p99_ttft']:.0f}ms")


def _summary_text(gpu_type: str, max_freq: int, n_plateau: int, n_total: int,
                  cap_mape: float, rho_cutoff: float, power_cv_mape: float) -> str:
    if gpu_type == "l40s":
        return f"""
Model Architecture (4 stages):

  Stage 1: Capacity — plateau-confirmed max TPS
    - {n_plateau}/{n_total} groups plateau-confirmed (TPS plateaus or TPS << demand)
    - Shallow GBR (depth 3, 200 trees) trained on confirmed groups only
    - Uncertainty margin η={ETA} applied to non-plateau C_hat estimates
    - In-sample MAPE: {cap_mape:.1f}% (held-out CV reported in training log + metadata)

  Stage 2: Utilization surrogate
    - Token-weighted demand: D = r*(il + {GAMMA}*ol)
      γ={GAMMA}: prefill is more expensive than decode per token
    - Split features: log(D_prefill), log(D_decode), prefill_frac
    - Conservative rho: ρ = D / C_hat (with η margin)

  Stage 3: SLO classifier (PRIMARY OUTPUT)
    - Cost-sensitive GBC: 5x penalty on missed violations
    - 12 features including split demand and rho derivatives
    - Guard band: safe only if p(violate) < p_th AND ρ < ρ_th
    - With guard band: 0 false-safe predictions (100% safe-precision)

  Stage 4: Latency regression
    - Monotonic HistGradientBoosting on log-latency (freq down, load up)
    - Trained only on unsaturated region (ρ < {rho_cutoff})
    - CV MAPE ~20-30% for latency, ~6% for power

  Power model: standalone shallow GBR, {power_cv_mape:.1f}% CV MAPE

Key contribution:
  SLO-aware scheduling via classification + guard band.
  The model explicitly addresses the saturation phase transition
  through a utilization surrogate (ρ = demand/capacity), and
  achieves operationally safe SLO decisions with a conservative
  guard band that eliminates false-safe predictions.
"""
    return f"""
Model Architecture (4 stages) — same as L40S:

  Stage 1: Capacity — plateau-confirmed max TPS
    - {n_plateau}/{n_total} groups plateau-confirmed (TPS plateaus or TPS << demand)
    - Shallow GBR (depth 3, 200 trees) trained on confirmed groups only
    - Uncertainty margin η={ETA} applied to non-plateau C_hat estimates
    - In-sample MAPE: {cap_mape:.1f}% (held-out CV reported in training log + metadata)

  Stage 2: Utilization surrogate
    - Token-weighted demand: D = r*(il + {GAMMA}*ol)
      γ={GAMMA}: prefill is more expensive than decode per token
    - Split features: log(D_prefill), log(D_decode), prefill_frac
    - Conservative rho: ρ = D / C_hat (with η margin)

  Stage 3: SLO classifier (PRIMARY OUTPUT)
    - Cost-sensitive GBC: 5x penalty on missed violations
    - 12 features including split demand and rho derivatives
    - Guard band: safe only if p(violate) < p_th AND ρ < ρ_th

  Stage 4: Latency regression
    - Monotonic HistGradientBoosting on log-latency (freq down, load up)
    - Trained only on unsaturated region (ρ < {rho_cutoff})

  Power model: standalone shallow GBR, {power_cv_mape:.1f}% CV MAPE

Key differences from L40S:
  - L4 is severely capacity-limited (most TP=1 configs saturate)
  - MAX_FREQ = {max_freq} MHz (vs 2520 MHz for L40S)
  - No anomaly exclusion needed
  - Higher proportion of saturated configs expected
"""


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100


# --- Latency regressor (monotonic gradient boosting) -------------------------
# Replaces the Ridge+poly2+log+expm1 latency model, which underfits the
# nonlinear latency surface (deg1) or extrapolates explosively (deg2: expm1 of a
# poly-Ridge excursion -> million-percent errors).  A monotonic HistGBM recovers
# the physical priors the linear model was chosen for (latency falls with
# frequency, rises with rate/demand/utilization) while fitting the nonlinearity:
# median APE ~7-13% (at the measurement-noise floor), no blow-ups.
def _latency_monotonic_cst(feature_cols, phase: str):
    inc = {"log_rate", "log_rho", "log_rho_total", "rho_overflow", "rho_total_overflow"}
    if phase in ("prefill", "unified"):
        inc |= {"log_il", "log_d_prefill", "log_rho_prefill", "rho_prefill_overflow"}
    if phase in ("decode", "unified"):
        inc |= {"log_d_decode", "log_rho_decode", "rho_decode_overflow"}
    dec = {"freq_norm"}
    return [(-1 if c in dec else (1 if c in inc else 0)) for c in feature_cols]


def _new_latency_regressor(feature_cols, phase: str):
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, min_samples_leaf=3,
        monotonic_cst=_latency_monotonic_cst(feature_cols, phase), random_state=42)


def fit_latency_model(train_df, target: str, feature_cols, phase: str, *, use_log: bool = True):
    """Fit the latency model and return (scaler, poly, model) matching the saved
    artifact interface so scheduler.py is unchanged.  `poly` is an identity
    passthrough (trees need no polynomial expansion); the scaler is kept (it is
    monotone per-feature, so monotonic_cst signs are preserved)."""
    scaler = StandardScaler().fit(train_df[feature_cols].values)
    Xs = scaler.transform(train_df[feature_cols].values)
    y = np.log1p(train_df[target].values) if use_log else train_df[target].values
    model = _new_latency_regressor(feature_cols, phase)
    model.fit(Xs, y)
    poly = FunctionTransformer()  # identity: poly.transform(scaler.transform(X)) == scaler.transform(X)
    return scaler, poly, model


def _predict_latency_model(scaler, poly, model, X_raw, use_log: bool = True):
    pred = model.predict(poly.transform(scaler.transform(X_raw)))
    return np.expm1(pred) if use_log else pred


def _evaluate_guard_metrics(y_true_unsafe: np.ndarray,
                            pred_prob: np.ndarray,
                            rho_vals: np.ndarray,
                            p_th: float,
                            rho_th: float) -> dict:
    pred_safe = (pred_prob < p_th) & (rho_vals < rho_th)
    actual_safe = (y_true_unsafe == 0)
    tp = int((actual_safe & pred_safe).sum())
    fp = int((~actual_safe & pred_safe).sum())
    tn = int((~actual_safe & ~pred_safe).sum())
    fn = int((actual_safe & ~pred_safe).sum())
    unsafe_support = int((~actual_safe).sum())
    safe_support = int(actual_safe.sum())
    admit_count = int(pred_safe.sum())
    total = len(pred_safe)
    return {
        "false_safe": fp,
        "false_unsafe": fn,
        "true_safe": tp,
        "true_unsafe": tn,
        "unsafe_support": unsafe_support,
        "safe_support": safe_support,
        "false_safe_rate_pct": round(fp / max(1, unsafe_support) * 100, 2),
        "false_unsafe_rate_pct": round(fn / max(1, safe_support) * 100, 2),
        "safe_precision_pct": round(tp / max(1, tp + fp) * 100, 2),
        "admit_rate_pct": round(admit_count / max(1, total) * 100, 2),
    }


def _select_guard_threshold(sweep_df: pd.DataFrame,
                            *,
                            base_p_th: float,
                            base_rho_th: float,
                            unsafe_support: int) -> tuple[dict, str]:
    base_match = sweep_df[
        np.isclose(sweep_df["p_th"], base_p_th) &
        np.isclose(sweep_df["rho_th"], base_rho_th)
    ]
    if base_match.empty:
        raise ValueError("Base guard threshold not found in sweep grid")
    base_row = base_match.iloc[0]

    if unsafe_support < MIN_UNSAFE_SUPPORT_FOR_RETUNE and int(base_row["false_safe"]) <= LOW_SUPPORT_MAX_FALSE_SAFE_TO_IGNORE:
        return base_row.to_dict(), "low_support_keep_base"

    min_false_safe = int(sweep_df["false_safe"].min())
    if unsafe_support < MIN_UNSAFE_SUPPORT_FOR_RETUNE and min_false_safe >= int(base_row["false_safe"]):
        return base_row.to_dict(), "low_support_no_better_point"

    frontier = sweep_df[sweep_df["false_safe"] == min_false_safe].copy()
    frontier = frontier.sort_values(
        ["admit_rate_pct", "safe_precision_pct", "rho_th", "p_th"],
        ascending=[False, False, False, False],
    )
    chosen = frontier.iloc[0].to_dict()

    if min_false_safe == int(base_row["false_safe"]) and float(chosen["admit_rate_pct"]) <= float(base_row["admit_rate_pct"]):
        return base_row.to_dict(), "base_dominates_frontier"
    return chosen, "retuned_min_false_safe_max_admit"


def run_training_pipeline(gpu_type: str,
                          *,
                          data_file: str | None = None,
                          model_dir: str | None = None,
                          dataset_metadata: dict | None = None,
                          anomaly_filter=None) -> None:
    cfg = dict(GPU_CONFIGS[gpu_type])
    dataset_manifest = get_dataset_manifest_for_gpu(gpu_type) if dataset_metadata is None else None
    if data_file is not None:
        cfg["data_file"] = data_file
    if anomaly_filter is not None:
        cfg["anomaly_filter"] = anomaly_filter
    max_freq = cfg["max_freq"]
    slo_thresholds = cfg["slo_thresholds"]
    guard_settings = cfg["guard_settings"]
    model_dir = model_dir or cfg["model_dir"]

    raw_df = pd.read_csv(cfg["data_file"])
    print(f"Total rows: {len(raw_df)}")
    anomaly_mask = ~cfg["anomaly_filter"](raw_df)
    print(f"Excluding {anomaly_mask.sum()} anomalous rows")
    all_data = prepare_data(cfg["data_file"], cfg["anomaly_filter"], max_freq)
    print(f"\nTotal averaged data points: {len(all_data)}")
    print(f"  Step 2a: {(all_data['step'] == 'step2a').sum()}")
    print(f"  Step 2b: {(all_data['step'] == 'step2b').sum()}")
    print(f"  Step 2c: {(all_data['step'] == 'step2c').sum()}")

    print("\n" + "=" * 70)
    print("STAGE 1: Capacity model (plateau-confirmed, conservative)")
    print("=" * 70)

    gbr_cap, cap_groups, cap_train = train_capacity_model(all_data, max_freq)
    pred_cap = np.exp(gbr_cap.predict(cap_train[CAP_FEATURES]))
    cap_mape = mean_absolute_percentage_error(cap_train["max_tps"], pred_cap) * 100
    cap_r2 = r2_score(cap_train["max_tps"], pred_cap)

    # Held-out CV (the in-sample MAPE above is optimistic; capacity is queried
    # off the measured grid by the scheduler).  Report two regimes:
    #   interp    = held-out CELLS (il,ol,tp,freq) -> typical scheduler query
    #   new-shape = held-out (il,ol,tp) families    -> extrapolation upper bound
    def _cap_cv(group_vals):
        ns = min(5, len(set(group_vals)))
        if ns < 2:
            return float("nan")
        oof = np.full(len(cap_train), np.nan)
        for tr, te in GroupKFold(n_splits=ns).split(cap_train, groups=group_vals):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                                          min_samples_leaf=3, random_state=42)
            m.fit(cap_train.iloc[tr][CAP_FEATURES], np.log(cap_train.iloc[tr]["max_tps"]))
            oof[te] = np.exp(m.predict(cap_train.iloc[te][CAP_FEATURES]))
        return mean_absolute_percentage_error(cap_train["max_tps"], oof) * 100
    _s = cap_train["il"].astype(str) + "_" + cap_train["ol"].astype(str) + "_" + cap_train["tp"].astype(str)
    cap_cv_mape = _cap_cv((_s + "_" + cap_train["freq"].astype(str)).values)
    cap_cv_mape_newshape = _cap_cv(_s.values)

    n_plateau = int(cap_groups["plateau_confirmed"].sum())
    n_total = len(cap_groups)
    print(f"\n  Capacity groups: {n_total}")
    print(f"  Plateau-confirmed: {n_plateau} ({n_plateau / n_total * 100:.0f}%)")
    print(f"  Uncertain (lower-bound only): {n_total - n_plateau}")
    print(f"  Training on {len(cap_train)} plateau-confirmed groups")
    print(f"  Capacity model: in-sample MAPE={cap_mape:.1f}% (R²={cap_r2:.4f}); "
          f"held-out CV MAPE={cap_cv_mape:.1f}% (interp), "
          f"{cap_cv_mape_newshape:.1f}% (new-shape extrap)")
    imp = sorted(zip(CAP_FEATURES, gbr_cap.feature_importances_), key=lambda x: -x[1])
    print(f"  Features: {', '.join(f'{k}={v:.2f}' for k, v in imp)}")

    print(f"\n  {'Config':>15s} {'TP':>3s} {'freq':>5s} {'maxTPS':>8s} {'C_hat':>8s} {'plat':>5s}")
    for _, r in cap_groups.sort_values(["il", "ol", "tp", "freq"]).iterrows():
        if r["freq"] == max_freq:
            plat = "YES" if r["plateau_confirmed"] else "no"
            print(f"  {int(r['il']):>5d}:{int(r['ol']):<5d}  {int(r['tp']):>3d} {int(r['freq']):>5d} "
                  f"{r['max_tps']:>8.0f} {r['C_hat']:>8.0f} {plat:>5s}")

    cap_train_pred = np.exp(gbr_cap.predict(cap_train[CAP_FEATURES]))
    cap_residuals = (cap_train_pred - cap_train["max_tps"].values) / cap_train["max_tps"].values
    print(f"\n  Capacity prediction residual (C_hat - C_true)/C_true on plateau groups:")
    for q in [5, 25, 50, 75, 90, 95]:
        print(f"    {q}th percentile: {np.percentile(cap_residuals, q) * 100:+.1f}%")

    q90_over = np.percentile(cap_residuals, 90)
    eta_derived = 1.0 / (1.0 + max(0, q90_over))
    print(f"\n  90th percentile overestimation: {q90_over * 100:+.1f}%")
    print(f"  Derived η = 1/(1+q90) = {eta_derived:.3f}")
    print(f"  Using η = {ETA} (configured)")

    print(f"\n  Capacity diagnostic (C_hat vs observed max TPS at f={max_freq}):")
    print(f"  {'Config':>15s} {'TP':>3s} {'maxTPS':>8s} {'C_hat':>8s} {'err%':>6s} {'status':>12s}")
    for _, r in cap_groups.sort_values(["il", "ol", "tp", "freq"]).iterrows():
        if r["freq"] == max_freq:
            err = (r["C_hat"] - r["max_tps"]) / r["max_tps"] * 100
            status = "plateau" if r["plateau_confirmed"] else f"uncertain(×{ETA})"
            print(f"  {int(r['il']):>5d}:{int(r['ol']):<5d}  {int(r['tp']):>3d} "
                  f"{r['max_tps']:>8.0f} {r['C_hat']:>8.0f} {err:>+5.1f}% {status:>12s}")

    print(f"\n  Capacity uncertainty margin η={ETA} for non-plateau groups")
    print(f"  Conservative C_hat = η * C_hat for uncertain capacity estimates")

    print("\n\n" + "=" * 70)
    print(f"STAGE 2: Utilization surrogate (γ={GAMMA}, η={ETA})")
    print("=" * 70)

    all_data = compute_features(all_data, gbr_cap, cap_groups, max_freq)
    all_data["group_full"] = build_group_key(all_data)
    print(f"\n  rho statistics (with η={ETA} margin on uncertain groups):")
    print(f"    min={all_data['rho'].min():.3f}, median={all_data['rho'].median():.3f}, "
          f"max={all_data['rho'].max():.1f}")
    print(f"    rho < 0.5: {(all_data['rho'] < 0.5).sum()}")
    print(f"    0.5 <= rho < 1.0: {((all_data['rho'] >= 0.5) & (all_data['rho'] < 1.0)).sum()}")
    print(f"    rho >= 1.0: {(all_data['rho'] >= 1.0).sum()}")
    _print_key_configs(all_data, gpu_type, max_freq)

    def fit_cap_model_fold(fold_data: pd.DataFrame):
        gbr, cap_fold, _ = train_capacity_model(fold_data, max_freq)
        return gbr, cap_fold

    def compute_features_fold(data: pd.DataFrame, cap_model, cap_fold_df: pd.DataFrame) -> pd.DataFrame:
        return compute_features(data, cap_model, cap_fold_df, max_freq)

    print("\n\n" + "=" * 70)
    print("STAGE 3: SLO classifier (cost-sensitive + guard band)")
    print("=" * 70)
    print(f"\n  Classifier features ({len(CLF_FEATURES)}):")
    for f in CLF_FEATURES:
        print(f"    {f}")

    for slo in slo_thresholds:
        print(f"\n  === SLO = {slo} ms (P99 TTFT) ===")
        all_data[f"violated_{slo}"] = (all_data["p99_ttft"] > slo).astype(int)
        n_v = int(all_data[f"violated_{slo}"].sum())
        n_ok = len(all_data) - n_v
        print(f"  Balance: {n_ok} OK, {n_v} violated ({n_v / len(all_data) * 100:.0f}%)")

        gkf = GroupKFold(n_splits=5)
        groups = all_data["group_full"].values
        oof_pred = np.full(len(all_data), np.nan)
        oof_prob = np.full(len(all_data), np.nan)
        oof_rho = np.full(len(all_data), np.nan)

        for train_idx, test_idx in gkf.split(all_data, groups=groups):
            train_f = all_data.iloc[train_idx].copy()
            test_f = all_data.iloc[test_idx].copy()
            cap_fold, cap_fold_df = fit_cap_model_fold(train_f)
            train_f = compute_features_fold(train_f, cap_fold, cap_fold_df)
            test_f = compute_features_fold(test_f, cap_fold, cap_fold_df)

            y_train = (train_f["p99_ttft"] > slo).astype(int).values
            sw = np.where(y_train == 1, 5.0, 1.0)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_f[CLF_FEATURES].values)
            X_te = scaler.transform(test_f[CLF_FEATURES].values)
            if len(np.unique(y_train)) < 2:
                oof_pred[test_idx] = float(y_train[0])
                oof_prob[test_idx] = float(y_train[0])
            else:
                gbc = GradientBoostingClassifier(
                    n_estimators=100, max_depth=2, learning_rate=0.1,
                    min_samples_leaf=5, random_state=42)
                gbc.fit(X_tr, y_train, sample_weight=sw)
                oof_pred[test_idx] = gbc.predict(X_te)
                oof_prob[test_idx] = gbc.predict_proba(X_te)[:, 1]
            oof_rho[test_idx] = test_f["rho"].values

        valid = ~np.isnan(oof_pred)
        y_true = all_data[f"violated_{slo}"].values[valid]
        safe_true = 1 - y_true
        safe_pred = 1 - oof_pred[valid].astype(int)
        tp = ((safe_true == 1) & (safe_pred == 1)).sum()
        fp = ((safe_true == 0) & (safe_pred == 1)).sum()
        tn = ((safe_true == 0) & (safe_pred == 0)).sum()
        fn = ((safe_true == 1) & (safe_pred == 0)).sum()
        acc = (tp + tn) / len(y_true) * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        print(f"\n  [Classifier only — no guard band]")
        print(f"    Safe={tp} FP(danger)={fp} Violated={tn} Conserv={fn}")
        print(f"    Acc={acc:.1f}%  Safe-Precision={prec:.1f}%")

        if 0 < fp <= 15:
            fp_idx = np.where(valid)[0][(safe_true == 0) & (safe_pred == 1)]
            for idx in fp_idx:
                r = all_data.iloc[idx]
                print(f"      FP: {int(r['il']):>5d}:{int(r['ol']):<5d} TP={int(r['tp'])} "
                      f"r={int(r['rate'])} f={int(r['freq'])}: p99={r['p99_ttft']:.0f}ms, "
                      f"rho={oof_rho[idx]:.2f}, p(viol)={oof_prob[idx]:.2f}")

        n_total_configs = len(safe_true)
        n_actually_safe = int((safe_true == 1).sum())
        print(f"\n  Pareto trade-off (guard band operating points):")
        print(f"  Total configs: {n_total_configs}, actually safe: {n_actually_safe}")
        print(f"  {'p_th':>5s} {'rho_th':>6s} | {'FP':>3s} {'FN':>3s} {'Acc':>5s} {'Prec':>5s} "
              f"| {'FP%':>5s} {'FN%':>5s} {'Admit':>5s}")

        pareto_points = []
        for p_th in [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05]:
            for rho_th in [1.0, 0.9, 0.8, 0.7]:
                pred_safe_gb = (oof_prob[valid] < p_th) & (oof_rho[valid] < rho_th)
                sp = pred_safe_gb.astype(int)
                fp_g = ((safe_true == 0) & (sp == 1)).sum()
                fn_g = ((safe_true == 1) & (sp == 0)).sum()
                tp_g = ((safe_true == 1) & (sp == 1)).sum()
                tn_g = ((safe_true == 0) & (sp == 0)).sum()
                acc_g = (tp_g + tn_g) / len(safe_true) * 100
                prec_g = tp_g / (tp_g + fp_g) * 100 if (tp_g + fp_g) > 0 else 100.0
                fp_pct = fp_g / max(1, (safe_true == 0).sum()) * 100
                fn_pct = fn_g / max(1, (safe_true == 1).sum()) * 100
                admitted = (sp == 1).sum()
                admit_pct = admitted / n_total_configs * 100
                pareto_points.append((p_th, rho_th, fp_g, fn_g, acc_g, prec_g,
                                      fp_pct, fn_pct, admitted, admit_pct))

        shown = set()
        for pt in sorted(pareto_points, key=lambda x: (x[2], x[3])):
            p_th, rho_th, fp_g, fn_g, acc_g, prec_g, fp_pct, fn_pct, admitted, admit_pct = pt
            key = (fp_g, fn_g)
            if key not in shown:
                shown.add(key)
                marker = " <<<" if fp_g == 0 else ""
                print(f"  {p_th:>5.2f} {rho_th:>6.1f} | {fp_g:>3d} {fn_g:>3d} "
                      f"{acc_g:>4.1f}% {prec_g:>4.1f}% | {fp_pct:>4.1f}% "
                      f"{fn_pct:>4.1f}% {admit_pct:>4.1f}%{marker}")
                if len(shown) > 12:
                    break

    print("\n\n" + "=" * 70)
    print("STAGE 4: Safe-region regression (Ridge + polynomial interactions)")
    print("=" * 70)

    reg_targets = [
        ("p99_ttft", "P99 TTFT", True),
        ("ttft", "Mean TTFT", True),
        ("p99_tpot", "P99 TPOT", True),
        ("tpot", "Mean TPOT", True),
        ("power_per_gpu", "Power/GPU", False),
    ]
    rho_cutoff = 1.5
    unsaturated = all_data[all_data["rho"] < rho_cutoff].copy()
    print(f"\n  Unsaturated (rho < {rho_cutoff}): {len(unsaturated)} pts")
    print(f"  Saturated (rho >= {rho_cutoff}): {len(all_data) - len(unsaturated)} pts")

    for target, label, use_log in reg_targets:
        reg_data = all_data.copy() if target == "power_per_gpu" else unsaturated.copy()
        reg_data = reg_data[reg_data[target].notna()].copy()
        reg_label = "all data" if target == "power_per_gpu" else f"rho<{rho_cutoff}"
        groups_reg = build_group_key(reg_data).values
        n_unique = len(set(groups_reg))
        if n_unique < 3:
            if gpu_type == "l4":
                print(f"  {label} ({reg_label}): SKIPPED (only {n_unique} unique groups)")
            continue

        gkf = GroupKFold(n_splits=min(5, n_unique))
        oof = np.full(len(reg_data), np.nan)
        for train_idx, test_idx in gkf.split(reg_data, groups=groups_reg):
            train_f = reg_data.iloc[train_idx].copy()
            test_f = reg_data.iloc[test_idx].copy()
            cap_fold, cap_fold_df = fit_cap_model_fold(train_f)
            train_f = compute_features_fold(train_f, cap_fold, cap_fold_df)
            test_f = compute_features_fold(test_f, cap_fold, cap_fold_df)

            if use_log:  # latency targets -> monotonic HistGBM
                scaler, poly, model = fit_latency_model(
                    train_f, target, REG_FEATURES, "unified", use_log=True)
                oof[test_idx] = _predict_latency_model(
                    scaler, poly, model, test_f[REG_FEATURES].values, use_log=True)
            else:        # power (raw) keeps the linear baseline here
                scaler = StandardScaler()
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_tr = poly.fit_transform(scaler.fit_transform(train_f[REG_FEATURES].values))
                X_te = poly.transform(scaler.transform(test_f[REG_FEATURES].values))
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr, y_tr := train_f[target].values)
                oof[test_idx] = ridge.predict(X_te)

        valid = ~np.isnan(oof)
        cv_mape = mean_absolute_percentage_error(reg_data[target].values[valid], oof[valid]) * 100
        cv_r2 = r2_score(reg_data[target].values[valid], oof[valid])
        spear_r, spear_p = spearmanr(reg_data[target].values[valid], oof[valid])
        print(f"  {label} ({reg_label}): CV MAPE={cv_mape:.1f}%, R²={cv_r2:.4f}, "
              f"Spearman ρ={spear_r:.3f} (p={spear_p:.2e})")

    print("\n\n" + "=" * 70)
    print("STAGE 5: Full pipeline (classify + guard band + regress)")
    print("=" * 70)
    print("""
  Pipeline for each candidate config:
    1. Predict p(violate) and rho from classifier
    2. Apply guard band: SAFE only if p(viol) < p_th AND rho < rho_th
    3. If SAFE: regress p99 TTFT for ranking
    4. If UNSAFE: clamp to >SLO (reject / need more resources)
""")

    gkf = GroupKFold(n_splits=5)
    groups = all_data["group_full"].values
    for slo in slo_thresholds:
        gs = guard_settings[slo]
        p_th, rho_th = gs["p_th"], gs["rho_th"]
        print(f"\n  --- SLO={slo}ms (guard: p<{p_th}, rho<{rho_th}) ---")
        oof_no_guard = np.full(len(all_data), np.nan)
        oof_guarded = np.full(len(all_data), np.nan)

        for train_idx, test_idx in gkf.split(all_data, groups=groups):
            train_f = all_data.iloc[train_idx].copy()
            test_f = all_data.iloc[test_idx].copy()
            cap_fold, cap_fold_df = fit_cap_model_fold(train_f)
            train_f = compute_features_fold(train_f, cap_fold, cap_fold_df)
            test_f = compute_features_fold(test_f, cap_fold, cap_fold_df)

            y_clf = (train_f["p99_ttft"] > slo).astype(int).values
            sw = np.where(y_clf == 1, 5.0, 1.0)
            scaler_c = StandardScaler()
            X_tr_c = scaler_c.fit_transform(train_f[CLF_FEATURES].values)
            X_te_c = scaler_c.transform(test_f[CLF_FEATURES].values)
            if len(np.unique(y_clf)) < 2:
                pred_violated = np.full(len(test_f), float(y_clf[0]))
                pred_prob = np.full(len(test_f), float(y_clf[0]))
            else:
                gbc = GradientBoostingClassifier(
                    n_estimators=100, max_depth=2, learning_rate=0.1,
                    min_samples_leaf=5, random_state=42)
                gbc.fit(X_tr_c, y_clf, sample_weight=sw)
                pred_violated = gbc.predict(X_te_c)
                pred_prob = gbc.predict_proba(X_te_c)[:, 1]
            test_rho = test_f["rho"].values
            pred_safe_gb = (pred_prob < p_th) & (test_rho < rho_th)

            safe_train = train_f[train_f["p99_ttft"] <= slo * 2]
            if len(safe_train) >= 10:
                scaler_r = StandardScaler()
                poly_r = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_tr_r = poly_r.fit_transform(scaler_r.fit_transform(safe_train[REG_FEATURES].values))
                X_te_r = poly_r.transform(scaler_r.transform(test_f[REG_FEATURES].values))
                y_reg = np.log1p(safe_train["p99_ttft"].values)
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr_r, y_reg)
                pred_reg = np.expm1(ridge.predict(X_te_r))
            else:
                pred_reg = np.full(len(test_f), slo * 2)

            oof_no_guard[test_idx] = np.where(pred_violated == 1, slo * 2, pred_reg)
            oof_guarded[test_idx] = np.where(pred_safe_gb, pred_reg, slo * 2)

        for pipe_name, pipe_pred in [("Classifier only", oof_no_guard),
                                     (f"Guard(p<{p_th},rho<{rho_th})", oof_guarded)]:
            valid = ~np.isnan(pipe_pred)
            actual = all_data["p99_ttft"].values[valid]
            pred = pipe_pred[valid]
            actual_ok = actual <= slo
            pred_ok = pred <= slo
            tp = (actual_ok & pred_ok).sum()
            fp = (~actual_ok & pred_ok).sum()
            tn = (~actual_ok & ~pred_ok).sum()
            fn = (actual_ok & ~pred_ok).sum()
            acc = (tp + tn) / len(actual) * 100
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 100.0
            safe_mask = pred_ok
            admitted = int(safe_mask.sum())
            admit_pct = admitted / len(actual) * 100
            if safe_mask.sum() > 0 and (actual[safe_mask] > 0).all():
                safe_mape = mean_absolute_percentage_error(actual[safe_mask], pred[safe_mask]) * 100
                if safe_mask.sum() >= 5:
                    sp_r, _ = spearmanr(actual[safe_mask], pred[safe_mask])
                    sp_str = f" Spearman={sp_r:.3f}"
                else:
                    sp_str = ""
            else:
                safe_mape = float("nan")
                sp_str = ""

            print(f"\n    [{pipe_name}]")
            print(f"    Acc={acc:.1f}% Safe-Prec={prec:.1f}% FP={fp} FN={fn} "
                  f"SafeMAPE={safe_mape:.1f}%{sp_str}")
            print(f"    Admitted: {admitted}/{len(actual)} ({admit_pct:.1f}%)")
            if 0 < fp <= 10:
                fp_idx = np.where(valid)[0][(~actual_ok) & pred_ok]
                for idx in fp_idx:
                    r = all_data.iloc[idx]
                    print(f"      FP: {int(r['il']):>5d}:{int(r['ol']):<5d} TP={int(r['tp'])} "
                          f"r={int(r['rate'])} f={int(r['freq'])}: "
                          f"actual={r['p99_ttft']:.0f}ms, pred={pipe_pred[idx]:.0f}ms")

    print("\n\n" + "=" * 70)
    print("Power model")
    print("=" * 70)
    gkf = GroupKFold(n_splits=5)
    power_data = all_data[all_data["power_per_gpu"].notna()].copy()
    oof_power = np.full(len(power_data), np.nan)
    for train_idx, test_idx in gkf.split(power_data, groups=power_data["group_full"].values):
        train_f = power_data.iloc[train_idx]
        test_f = power_data.iloc[test_idx]
        gbr_p = GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.1,
            min_samples_leaf=4, random_state=42)
        gbr_p.fit(train_f[POWER_FEATURES], train_f["power_per_gpu"])
        oof_power[test_idx] = gbr_p.predict(test_f[POWER_FEATURES])
    valid = ~np.isnan(oof_power)
    power_cv_mape = mean_absolute_percentage_error(
        power_data["power_per_gpu"].values[valid], oof_power[valid]) * 100
    print(f"  Power/GPU CV MAPE (GroupKFold): {power_cv_mape:.1f}%")

    print("\n\n" + "=" * 70)
    print("PHASE-SPECIFIC MODEL STACK")
    print("=" * 70)

    phase_validation_rows = []
    phase_guard_sweep_rows = []
    phase_guard_before_after_rows = []
    phase_guard_oof_rows = []
    phase_artifacts = {}
    phase_guard_settings = {phase: {} for phase in PHASE_MODEL_SPECS}

    def fit_phase_classifier_cv(phase_name: str,
                                target_col: str,
                                feature_cols,
                                rho_col: str,
                                slo: int):
        phase_data = all_data[all_data[target_col].notna()].copy()
        if phase_data.empty:
            return {
                "phase": phase_name,
                "metric_type": "feasibility",
                "target": target_col,
                "slo_ms": int(slo),
                "false_safe": 0,
                "false_unsafe": 0,
                "true_safe": 0,
                "true_unsafe": 0,
                "unsafe_support": 0,
                "safe_support": 0,
                "false_safe_rate_pct": float("nan"),
                "false_unsafe_rate_pct": float("nan"),
                "safe_precision_pct": float("nan"),
                "admit_rate_pct": float("nan"),
                "oof_prob": np.array([]),
                "oof_rho": np.array([]),
                "y_true_unsafe": np.array([]),
            }
        gkf_local = GroupKFold(n_splits=5)
        groups_local = phase_data["group_full"].values
        oof_prob = np.full(len(phase_data), np.nan)
        oof_safe = np.full(len(phase_data), np.nan)
        y_true_full = (phase_data[target_col] > slo).astype(int).values

        for train_idx, test_idx in gkf_local.split(phase_data, groups=groups_local):
            train_f = phase_data.iloc[train_idx].copy()
            test_f = phase_data.iloc[test_idx].copy()
            cap_fold, cap_fold_df = fit_cap_model_fold(train_f)
            train_f = compute_features_fold(train_f, cap_fold, cap_fold_df)
            test_f = compute_features_fold(test_f, cap_fold, cap_fold_df)

            y_train = (train_f[target_col] > slo).astype(int).values
            sw = np.where(y_train == 1, 5.0, 1.0)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_f[feature_cols].values)
            X_te = scaler.transform(test_f[feature_cols].values)
            if len(np.unique(y_train)) < 2:
                pred_prob = np.full(len(test_f), float(y_train[0]))
            else:
                gbc = GradientBoostingClassifier(
                    n_estimators=100, max_depth=2, learning_rate=0.1,
                    min_samples_leaf=5, random_state=42)
                gbc.fit(X_tr, y_train, sample_weight=sw)
                pred_prob = gbc.predict_proba(X_te)[:, 1]
            gs = guard_settings[slo]
            pred_safe = (pred_prob < gs["p_th"]) & (test_f[rho_col].values < gs["rho_th"])
            oof_prob[test_idx] = pred_prob
            oof_safe[test_idx] = pred_safe.astype(float)

        valid_mask = ~np.isnan(oof_safe)
        pred_prob_valid = oof_prob[valid_mask]
        rho_valid = phase_data.loc[valid_mask, rho_col].values
        y_true_valid = y_true_full[valid_mask]
        base_metrics = _evaluate_guard_metrics(
            y_true_valid,
            pred_prob_valid,
            rho_valid,
            guard_settings[slo]["p_th"],
            guard_settings[slo]["rho_th"],
        )
        return {
            "phase": phase_name,
            "metric_type": "feasibility",
            "target": target_col,
            "slo_ms": int(slo),
            **base_metrics,
            "oof_prob": pred_prob_valid,
            "oof_rho": rho_valid,
            "y_true_unsafe": y_true_valid,
        }

    def fit_phase_regression_cv(phase_name: str,
                                target_col: str,
                                feature_cols,
                                *,
                                use_log: bool,
                                filter_df: pd.DataFrame,
                                metric_type: str):
        reg_data = filter_df[filter_df[target_col].notna()].copy()
        groups_reg = build_group_key(reg_data).values
        n_unique = len(set(groups_reg))
        if n_unique < 3:
            return {
                "phase": phase_name,
                "metric_type": metric_type,
                "target": target_col,
                "n_rows": int(len(reg_data)),
                "cv_mape_pct": float("nan"),
                "cv_r2": float("nan"),
                "spearman_r": float("nan"),
            }

        gkf_local = GroupKFold(n_splits=min(5, n_unique))
        oof = np.full(len(reg_data), np.nan)
        for train_idx, test_idx in gkf_local.split(reg_data, groups=groups_reg):
            train_f = reg_data.iloc[train_idx].copy()
            test_f = reg_data.iloc[test_idx].copy()
            cap_fold, cap_fold_df = fit_cap_model_fold(train_f)
            train_f = compute_features_fold(train_f, cap_fold, cap_fold_df)
            test_f = compute_features_fold(test_f, cap_fold, cap_fold_df)

            if metric_type == "latency":
                scaler, poly, model = fit_latency_model(
                    train_f, target_col, feature_cols, phase_name, use_log=use_log)
                oof[test_idx] = _predict_latency_model(
                    scaler, poly, model, test_f[feature_cols].values, use_log=use_log)
            else:
                y_tr = np.log1p(train_f[target_col].values) if use_log else train_f[target_col].values
                scaler = StandardScaler()
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_tr = poly.fit_transform(scaler.fit_transform(train_f[feature_cols].values))
                X_te = poly.transform(scaler.transform(test_f[feature_cols].values))
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr, y_tr)
                oof[test_idx] = np.expm1(ridge.predict(X_te)) if use_log else ridge.predict(X_te)

        valid_mask = ~np.isnan(oof)
        y_true = reg_data[target_col].values[valid_mask]
        y_pred = oof[valid_mask]
        spear_r, _ = spearmanr(y_true, y_pred)
        return {
            "phase": phase_name,
            "metric_type": metric_type,
            "target": target_col,
            "n_rows": int(len(reg_data)),
            "cv_mape_pct": round(_safe_mape(y_true, y_pred), 2),
            "cv_r2": round(float(r2_score(y_true, y_pred)), 4),
            "spearman_r": round(float(spear_r), 4),
        }

    for phase_name, spec in PHASE_MODEL_SPECS.items():
        print(f"\n  [{phase_name}]")
        phase_validation_rows.append(
            fit_phase_regression_cv(
                phase_name,
                spec["latency_target"],
                spec["reg_features"],
                use_log=True,
                filter_df=all_data[all_data["rho"] < rho_cutoff].copy(),
                metric_type="latency",
            )
        )
        phase_validation_rows.append(
            fit_phase_regression_cv(
                phase_name,
                "power_per_gpu",
                spec["power_features"],
                use_log=False,
                filter_df=all_data.copy(),
                metric_type="power",
            )
        )
        for slo in slo_thresholds:
            clf_result = fit_phase_classifier_cv(
                phase_name,
                spec["latency_target"],
                spec["clf_features"],
                spec["rho_col"],
                slo,
            )
            oof_prob = clf_result.pop("oof_prob")
            oof_rho = clf_result.pop("oof_rho")
            y_true_unsafe = clf_result.pop("y_true_unsafe")
            phase_validation_rows.append(clf_result)

            base_p = guard_settings[slo]["p_th"]
            base_rho = guard_settings[slo]["rho_th"]
            sweep_rows = []
            p_grid = sorted(set(GUARD_P_GRID + [base_p]), reverse=True)
            rho_grid = sorted(set(GUARD_RHO_GRID + [base_rho]), reverse=True)
            for p_th in p_grid:
                for rho_th in rho_grid:
                    metrics = _evaluate_guard_metrics(y_true_unsafe, oof_prob, oof_rho, p_th, rho_th)
                    sweep_row = {
                        "gpu_type": gpu_type,
                        "phase": phase_name,
                        "slo_ms": int(slo),
                        "target": spec["latency_target"],
                        "p_th": float(p_th),
                        "rho_th": float(rho_th),
                        **metrics,
                    }
                    sweep_rows.append(sweep_row)
                    phase_guard_sweep_rows.append(sweep_row)

            sweep_df = pd.DataFrame(sweep_rows)
            chosen_row, selection_reason = _select_guard_threshold(
                sweep_df,
                base_p_th=base_p,
                base_rho_th=base_rho,
                unsafe_support=int(clf_result["unsafe_support"]),
            )
            phase_guard_settings[phase_name][int(slo)] = {
                "p_th": float(chosen_row["p_th"]),
                "rho_th": float(chosen_row["rho_th"]),
            }
            phase_guard_before_after_rows.append({
                "gpu_type": gpu_type,
                "phase": phase_name,
                "slo_ms": int(slo),
                "unsafe_support": int(clf_result["unsafe_support"]),
                "safe_support": int(clf_result["safe_support"]),
                "base_p_th": float(base_p),
                "base_rho_th": float(base_rho),
                "chosen_p_th": float(chosen_row["p_th"]),
                "chosen_rho_th": float(chosen_row["rho_th"]),
                "selection_reason": selection_reason,
                "before_false_safe": int(clf_result["false_safe"]),
                "before_false_unsafe": int(clf_result["false_unsafe"]),
                "before_true_safe": int(clf_result["true_safe"]),
                "before_true_unsafe": int(clf_result["true_unsafe"]),
                "before_false_safe_rate_pct": float(clf_result["false_safe_rate_pct"]),
                "before_false_unsafe_rate_pct": float(clf_result["false_unsafe_rate_pct"]),
                "before_safe_precision_pct": float(clf_result["safe_precision_pct"]),
                "before_admit_rate_pct": float(clf_result["admit_rate_pct"]),
                "after_false_safe": int(chosen_row["false_safe"]),
                "after_false_unsafe": int(chosen_row["false_unsafe"]),
                "after_true_safe": int(chosen_row["true_safe"]),
                "after_true_unsafe": int(chosen_row["true_unsafe"]),
                "after_false_safe_rate_pct": float(chosen_row["false_safe_rate_pct"]),
                "after_false_unsafe_rate_pct": float(chosen_row["false_unsafe_rate_pct"]),
                "after_safe_precision_pct": float(chosen_row["safe_precision_pct"]),
                "after_admit_rate_pct": float(chosen_row["admit_rate_pct"]),
                "retuned": (float(chosen_row["p_th"]) != float(base_p)) or (float(chosen_row["rho_th"]) != float(base_rho)),
                "low_support_flag": int(clf_result["unsafe_support"]) < MIN_UNSAFE_SUPPORT_FOR_RETUNE,
            })
            for idx, (prob, rho_val, y_unsafe) in enumerate(zip(oof_prob, oof_rho, y_true_unsafe)):
                phase_guard_oof_rows.append({
                    "gpu_type": gpu_type,
                    "phase": phase_name,
                    "slo_ms": int(slo),
                    "sample_idx": idx,
                    "target": spec["latency_target"],
                    "p_violate_oof": float(prob),
                    "rho_phase_oof": float(rho_val),
                    "actual_unsafe": int(y_unsafe),
                })

        print(f"    latency target: {spec['latency_target']}")
        print(f"    power target: power_per_gpu")
        print(f"    feasibility target: {spec['latency_target']} <= SLO")

        for row in [r for r in phase_validation_rows if r["phase"] == phase_name]:
            if row["metric_type"] == "feasibility":
                calib_row = next(
                    r for r in phase_guard_before_after_rows
                    if r["gpu_type"] == gpu_type and r["phase"] == phase_name and r["slo_ms"] == row["slo_ms"]
                )
                print(
                    f"    SLO {row['slo_ms']:>4d}ms: false-safe={row['false_safe_rate_pct']:>5.1f}% "
                    f"false-unsafe={row['false_unsafe_rate_pct']:>5.1f}% "
                    f"safe-precision={row['safe_precision_pct']:>5.1f}% "
                    f"-> guard(p<{calib_row['chosen_p_th']:.2f},rho<{calib_row['chosen_rho_th']:.2f})"
                )
            else:
                print(
                    f"    {row['metric_type']:>7s}: MAPE={row['cv_mape_pct']:>5.1f}% "
                    f"R^2={row['cv_r2']:>6.3f} Spearman={row['spearman_r']:>6.3f}"
                )

        # Final exported artifacts.
        classifiers = {}
        scalers = {}
        for slo in slo_thresholds:
            phase_fit_data = all_data[all_data[spec["latency_target"]].notna()].copy()
            y = (phase_fit_data[spec["latency_target"]] > slo).astype(int).values
            sw = np.where(y == 1, 5.0, 1.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(phase_fit_data[spec["clf_features"]].values)
            if len(np.unique(y)) < 2:
                gbc = DummyClassifier(strategy="constant", constant=int(y[0]))
                gbc.fit(X_scaled, y)
            else:
                gbc = GradientBoostingClassifier(
                    n_estimators=100, max_depth=2, learning_rate=0.1,
                    min_samples_leaf=5, random_state=42)
                gbc.fit(X_scaled, y, sample_weight=sw)
            classifiers[slo] = gbc
            scalers[slo] = scaler

        reg_train = all_data[(all_data["rho"] < rho_cutoff) & all_data[spec["latency_target"]].notna()].copy()
        scaler_r_phase, poly_r_phase, ridge_phase = fit_latency_model(
            reg_train, spec["latency_target"], spec["reg_features"], phase_name, use_log=True)

        power_train = all_data[all_data["power_per_gpu"].notna()].copy()
        power_model_phase = GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.1,
            min_samples_leaf=4, random_state=42)
        power_model_phase.fit(power_train[spec["power_features"]], power_train["power_per_gpu"])

        phase_artifacts[phase_name] = {
            "classifiers": classifiers,
            "scalers": scalers,
            "latency_model": ridge_phase,
            "latency_scaler": scaler_r_phase,
            "latency_poly": poly_r_phase,
            "power_model": power_model_phase,
        }

    print("\n\n" + "=" * 70)
    summary_title = "FINAL SUMMARY — v8" if gpu_type == "l40s" else "FINAL SUMMARY — L4 GPU v8"
    print(summary_title)
    print("=" * 70)
    print(_summary_text(gpu_type, max_freq, n_plateau, n_total, cap_mape, rho_cutoff, power_cv_mape))

    print("\n" + "=" * 70)
    print("Exporting trained models for scheduler")
    print("=" * 70)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(gbr_cap, f"{model_dir}/capacity_model.pkl")
    joblib.dump(CAP_FEATURES, f"{model_dir}/capacity_features.pkl")
    joblib.dump(cap_groups, f"{model_dir}/cap_groups.pkl")
    print(f"  Saved capacity model ({len(cap_train)} plateau-confirmed groups)")

    for slo in slo_thresholds:
        y = (all_data["p99_ttft"] > slo).astype(int).values
        sw = np.where(y == 1, 5.0, 1.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(all_data[CLF_FEATURES].values)
        if len(np.unique(y)) < 2:
            gbc = DummyClassifier(strategy="constant", constant=int(y[0]))
            gbc.fit(X_scaled, y)
        else:
            gbc = GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.1,
                min_samples_leaf=5, random_state=42)
            gbc.fit(X_scaled, y, sample_weight=sw)
        joblib.dump(gbc, f"{model_dir}/slo_classifier_{slo}.pkl")
        joblib.dump(scaler, f"{model_dir}/slo_scaler_{slo}.pkl")
        print(f"  Saved SLO classifier for {slo}ms")
    joblib.dump(CLF_FEATURES, f"{model_dir}/clf_features.pkl")

    reg_unsaturated = all_data[all_data["rho"] < rho_cutoff].copy()
    scaler_r, poly_r, ridge_final = fit_latency_model(
        reg_unsaturated, "p99_ttft", REG_FEATURES, "unified", use_log=True)
    joblib.dump(ridge_final, f"{model_dir}/regressor_p99.pkl")
    joblib.dump(scaler_r, f"{model_dir}/regressor_scaler.pkl")
    joblib.dump(poly_r, f"{model_dir}/regressor_poly.pkl")
    joblib.dump(REG_FEATURES, f"{model_dir}/reg_features.pkl")
    print(f"  Saved regressor ({len(reg_unsaturated)} unsaturated points)")

    power_train = all_data[all_data["power_per_gpu"].notna()].copy()
    gbr_power_final = GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.1,
        min_samples_leaf=4, random_state=42)
    gbr_power_final.fit(power_train[POWER_FEATURES], power_train["power_per_gpu"])
    joblib.dump(gbr_power_final, f"{model_dir}/power_model.pkl")
    joblib.dump(POWER_FEATURES, f"{model_dir}/power_features.pkl")
    print(f"  Saved power model ({len(all_data)} points)")

    for phase_name, spec in PHASE_MODEL_SPECS.items():
        artifacts = phase_artifacts[phase_name]
        for slo in slo_thresholds:
            joblib.dump(
                artifacts["classifiers"][slo],
                f"{model_dir}/{phase_name}_slo_classifier_{slo}.pkl",
            )
            joblib.dump(
                artifacts["scalers"][slo],
                f"{model_dir}/{phase_name}_slo_scaler_{slo}.pkl",
            )
        joblib.dump(spec["clf_features"], f"{model_dir}/{phase_name}_clf_features.pkl")
        joblib.dump(artifacts["latency_model"], f"{model_dir}/{phase_name}_latency_regressor_p99.pkl")
        joblib.dump(artifacts["latency_scaler"], f"{model_dir}/{phase_name}_latency_scaler.pkl")
        joblib.dump(artifacts["latency_poly"], f"{model_dir}/{phase_name}_latency_poly.pkl")
        joblib.dump(spec["reg_features"], f"{model_dir}/{phase_name}_reg_features.pkl")
        joblib.dump(artifacts["power_model"], f"{model_dir}/{phase_name}_power_model.pkl")
        joblib.dump(spec["power_features"], f"{model_dir}/{phase_name}_power_features.pkl")
        print(f"  Saved {phase_name}-specific feasibility/latency/power models")

    config = {
        "GAMMA": GAMMA,
        "ETA": ETA,
        "MAX_FREQ": max_freq,
        "GUARD_SETTINGS": guard_settings,
        "PHASE_GUARD_SETTINGS": phase_guard_settings,
        "SLO_THRESHOLDS": slo_thresholds,
        "TP_DEGREES": sorted(all_data["tp"].unique().tolist()),
        "FREQUENCIES": sorted(all_data["freq"].unique().tolist()),
        "MODEL_STACK_VERSION": "phase_specific_v1",
        "PHASE_MODEL_TARGETS": {
            phase_name: {
                "latency_target": spec["latency_target"],
                "rho_col": spec["rho_col"],
            }
            for phase_name, spec in PHASE_MODEL_SPECS.items()
        },
    }
    joblib.dump(config, f"{model_dir}/config.pkl")

    phase_validation_df = pd.DataFrame(phase_validation_rows)
    phase_validation_df.to_csv(f"{model_dir}/phase_model_validation.csv", index=False)
    phase_guard_sweep_df = pd.DataFrame(phase_guard_sweep_rows)
    phase_guard_sweep_df.to_csv(f"{model_dir}/phase_guard_sweep.csv", index=False)
    phase_guard_before_after_df = pd.DataFrame(phase_guard_before_after_rows)
    phase_guard_before_after_df.to_csv(f"{model_dir}/phase_guard_before_after.csv", index=False)
    phase_guard_oof_df = pd.DataFrame(phase_guard_oof_rows)
    phase_guard_oof_df.to_csv(f"{model_dir}/phase_guard_oof_details.csv", index=False)

    summary_lines = []
    for _, row in phase_guard_before_after_df.sort_values(["phase", "slo_ms"]).iterrows():
        status = "low-support" if row["low_support_flag"] else "supported"
        changed = "retuned" if row["retuned"] else "kept"
        summary_lines.append(
            f"{row['gpu_type']}/{row['phase']}/{int(row['slo_ms'])}ms: {changed}, {status}, "
            f"false-safe {int(row['before_false_safe'])}->{int(row['after_false_safe'])}, "
            f"admit {row['before_admit_rate_pct']:.1f}%->{row['after_admit_rate_pct']:.1f}%, "
            f"guard (p<{row['chosen_p_th']:.2f}, rho<{row['chosen_rho_th']:.2f})"
        )
    with open(f"{model_dir}/phase_guard_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + ("\n" if summary_lines else ""))

    training_metadata = {
        "gpu_type": gpu_type,
        "source_data_file": cfg["data_file"],
        "dataset_manifest": dataset_metadata if dataset_metadata is not None else dataset_manifest.to_metadata_dict(),
        "n_rows_raw": int(len(raw_df)),
        "n_rows_averaged": int(len(all_data)),
        "n_anomaly_rows_excluded": int(anomaly_mask.sum()),
        "step_counts": {
            "step2a": int((all_data["step"] == "step2a").sum()),
            "step2b": int((all_data["step"] == "step2b").sum()),
            "step2c": int((all_data["step"] == "step2c").sum()),
        },
        "n_capacity_groups": int(n_total),
        "n_plateau_confirmed": int(n_plateau),
        "capacity_train_mape_pct": round(float(cap_mape), 2),
        "capacity_cv_mape_pct": round(float(cap_cv_mape), 2),
        "capacity_cv_mape_newshape_pct": round(float(cap_cv_mape_newshape), 2),
        "power_cv_mape_pct": round(float(power_cv_mape), 2),
        "guard_settings": guard_settings,
        "slo_thresholds": list(config["SLO_THRESHOLDS"]),
        "model_stack_version": config["MODEL_STACK_VERSION"],
        "phase_model_targets": config["PHASE_MODEL_TARGETS"],
        "phase_validation_rows": json.loads(phase_validation_df.to_json(orient="records")),
        "phase_guard_settings": phase_guard_settings,
        "phase_guard_before_after": json.loads(phase_guard_before_after_df.to_json(orient="records")),
    }
    with open(f"{model_dir}/training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2, sort_keys=True)

    print(f"\nAll models saved to {model_dir}/")
    print(f"  Config space: TP={config['TP_DEGREES']}, Freq={config['FREQUENCIES']}")
