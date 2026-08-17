"""
fit_b5_models.py — Fit Ridge regressors to B5 calibration data.

Produces models for P99 TTFT, P99 TPOT, and avg_power_w per (gpu_type, exp_type).
These replace the raw LUT nearest-neighbor queries in Experiment F.

Usage:
    python fit_b5_models.py --b5-csv ../b5_lookup_table.csv --output-dir models_b5
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_percentage_error

from paths import paper_model_dir


# Features for regression
BASE_FEATURES = ["log_il", "log_ol", "tp", "freq_norm", "log_rate"]
TARGETS = ["p99_ttft_ms", "p99_tpot_ms", "avg_power_w"]

# GPU max frequencies for normalization
MAX_FREQ = {"l40s": 2520, "l4": 2040}


def build_features(df: pd.DataFrame, gpu_type: str) -> pd.DataFrame:
    """Add derived feature columns to dataframe."""
    df = df.copy()
    df["log_il"] = np.log1p(df["input_len"])
    df["log_ol"] = np.log1p(df["output_len"])
    df["freq_norm"] = df["freq_mhz"] / MAX_FREQ[gpu_type]
    df["log_rate"] = np.log1p(df["rate_rps"])
    # Group identifier for cross-validation
    df["group"] = (df["input_len"].astype(str) + "_" +
                   df["output_len"].astype(str) + "_" +
                   df["rate_rps"].astype(str))
    return df


def fit_model(X, y, alpha=10.0):
    """Fit Ridge with polynomial degree=2 interactions."""
    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_scaled = scaler.fit_transform(X)
    X_poly = poly.fit_transform(X_scaled)
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    return model, scaler, poly


def predict(model, scaler, poly, X):
    """Predict from fitted model."""
    X_scaled = scaler.transform(X)
    X_poly = poly.transform(X_scaled)
    return model.predict(X_poly)


def cross_validate(df, features, target, alpha=10.0):
    """Leave-one-group-out cross-validation, returns MAPE."""
    X = df[features].values
    # Log-transform targets for TTFT/TPOT (wide dynamic range)
    if "ttft" in target or "tpot" in target:
        y = np.log1p(df[target].values)
        log_target = True
    else:
        y = df[target].values
        log_target = False

    groups = df["group"].values
    logo = LeaveOneGroupOut()

    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in logo.split(X, y, groups):
        model, scaler, poly = fit_model(X[train_idx], y[train_idx], alpha)
        preds[test_idx] = predict(model, scaler, poly, X[test_idx])

    if log_target:
        actuals = np.expm1(y)
        predictions = np.expm1(preds)
    else:
        actuals = y
        predictions = preds

    # Filter valid
    mask = (actuals > 0) & np.isfinite(predictions) & np.isfinite(actuals)
    if mask.sum() == 0:
        return float("nan"), actuals, predictions
    mape = mean_absolute_percentage_error(actuals[mask], predictions[mask])
    return mape, actuals, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b5-csv", default="../b5_lookup_table.csv")
    parser.add_argument("--output-dir", default=str(paper_model_dir("models_b5")))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load B5 data
    df = pd.read_csv(args.b5_csv)
    df = df[df["successful_requests"] > 0].copy()
    print(f"Loaded {len(df)} valid B5 records")

    # Model groups: (gpu_type, exp_type) pairs
    model_groups = [
        ("l40s", "A"),  # Monolithic L40S: TTFT + TPOT + Power
        ("l40s", "C"),  # Prefill-only L40S: TTFT + Power
        ("l4", "B"),    # Monolithic L4: TTFT + TPOT + Power
        ("l4", "D"),    # Decode-only L4: TPOT + Power
    ]

    all_results = []

    for gpu_type, exp_type in model_groups:
        subset = df[(df["gpu_type"] == gpu_type) & (df["exp_type"] == exp_type)].copy()
        if subset.empty:
            print(f"\n  [{gpu_type}/{exp_type}] No data — skipping")
            continue

        subset = build_features(subset, gpu_type)
        print(f"\n{'='*60}")
        print(f"  [{gpu_type}/{exp_type}] {len(subset)} rows")
        print(f"  Freq: {sorted(subset['freq_mhz'].unique())}")
        print(f"  TP:   {sorted(subset['tp'].unique())}")
        print(f"  IL:   {sorted(subset['input_len'].unique())}")
        print(f"  OL:   {sorted(subset['output_len'].unique())}")
        print(f"  Rate: {sorted(subset['rate_rps'].unique())}")

        for target in TARGETS:
            if target not in subset.columns:
                continue
            valid = subset[subset[target].notna() & (subset[target] > 0)]
            if len(valid) < 5:
                print(f"  {target}: too few valid rows ({len(valid)}) — skipping")
                continue

            # Cross-validate
            mape, actuals, predictions = cross_validate(valid, BASE_FEATURES, target)

            # Fit final model on all data
            X = valid[BASE_FEATURES].values
            if "ttft" in target or "tpot" in target:
                y = np.log1p(valid[target].values)
                log_target = True
            else:
                y = valid[target].values
                log_target = False

            model, scaler, poly = fit_model(X, y)

            # Save model
            model_name = f"{gpu_type}_{exp_type}_{target}"
            model_path = os.path.join(args.output_dir, f"{model_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": model,
                    "scaler": scaler,
                    "poly": poly,
                    "features": BASE_FEATURES,
                    "target": target,
                    "gpu_type": gpu_type,
                    "exp_type": exp_type,
                    "log_target": log_target,
                    "max_freq": MAX_FREQ[gpu_type],
                    "n_train": len(valid),
                    "cv_mape": mape,
                }, f)

            print(f"  {target:20s}: CV MAPE = {mape*100:6.1f}%  "
                  f"(n={len(valid)}, range=[{actuals.min():.1f}, {actuals.max():.1f}])")

            all_results.append({
                "gpu_type": gpu_type, "exp_type": exp_type,
                "target": target, "cv_mape": mape, "n_train": len(valid),
            })

    # Summary
    print(f"\n{'='*60}")
    print("Summary: Cross-validation MAPE")
    summary = pd.DataFrame(all_results)
    summary["cv_mape_pct"] = (summary["cv_mape"] * 100).round(1)
    print(summary[["gpu_type", "exp_type", "target", "cv_mape_pct", "n_train"]].to_string(index=False))
    print(f"\nModels saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
