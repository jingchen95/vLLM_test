#!/usr/bin/env python3
"""
Shared utilities for SWEEP-LLM model training.
"""

from __future__ import annotations

import os
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from paths import REPO_ROOT, paper_model_dir


GAMMA = 0.3
ETA = 0.8


def passthrough_filter(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


# Keep for backward compatibility with any callers that reference it by name.
l40s_anomaly_filter = passthrough_filter


def drop_warmup_run1(df: pd.DataFrame, ratio_threshold: float = 3.0) -> pd.DataFrame:
    """Drop run=1 rows whose p99_ttft is implausibly high vs runs 2 and 3.

    After each set_gpu_freq call the first benchmark run executes on a
    cold GPU clock, producing latencies 3–100x higher than steady-state.
    We detect this by comparing run=1 p99_ttft to the median of runs 2+3
    within each (step, freq, tp, il, ol, rate) group and drop the run=1
    row when the ratio exceeds ratio_threshold.  Groups that have no
    runs 2 or 3 are left untouched so no data is lost for single-run configs.
    """
    group_cols = ["step", "gpu_freq_mhz", "tp_degree",
                  "input_len", "output_len", "request_rate"]

    later = (df[df["run"] > 1]
             .groupby(group_cols)["p99_ttft_ms"]
             .median()
             .rename("med_later"))

    merged = df.merge(later.reset_index(), on=group_cols, how="left")

    bad_run1 = (
        (merged["run"] == 1) &
        merged["med_later"].notna() &
        (merged["med_later"] > 0) &
        (merged["p99_ttft_ms"] > ratio_threshold * merged["med_later"])
    )
    return df[~bad_run1.values].copy()


GPU_CONFIGS: Dict[str, Dict[str, object]] = {
    "l40s": {
        "max_freq": 2520,
        "data_file": str(REPO_ROOT / "Phase2_Results_L40S" / "master_results.csv"),
        "guard_settings": {
            200: {"p_th": 0.40, "rho_th": 1.0},
            500: {"p_th": 0.05, "rho_th": 1.0},
            1000: {"p_th": 0.10, "rho_th": 0.7},
        },
        "slo_thresholds": [200, 500, 1000],
        "anomaly_filter": l40s_anomaly_filter,
        "model_dir": str(paper_model_dir("models_l40s")),
    },
    "l4": {
        "max_freq": 2040,
        "data_file": str(REPO_ROOT / "Phase2_Results_L4" / "master_results.csv"),
        "guard_settings": {
            200: {"p_th": 0.05, "rho_th": 1.0},
            500: {"p_th": 0.05, "rho_th": 0.6},
            1000: {"p_th": 0.10, "rho_th": 0.8},
            2000: {"p_th": 0.05, "rho_th": 1.0},
        },
        "slo_thresholds": [200, 500, 1000, 2000],
        "anomaly_filter": passthrough_filter,
        "model_dir": str(paper_model_dir("models_l4")),
    },
}

CAP_FEATURES = ["log_il", "log_ol", "tp", "freq_norm", "decode_frac"]
CLF_FEATURES = ["log_il", "log_ol", "tp", "freq_norm", "decode_frac",
                "log_rho", "rho_overflow", "rho_sq", "log_rate",
                "log_d_prefill", "log_d_decode", "prefill_frac"]
REG_FEATURES = ["log_il", "log_ol", "tp", "freq_norm", "decode_frac",
                "log_rho", "rho_overflow", "log_rate",
                "log_d_prefill", "log_d_decode", "prefill_frac"]
POWER_FEATURES = ["log_il", "log_ol", "tp", "freq_norm", "decode_frac", "log_rho"]

# Phase-specific features for the scheduler-facing model stack.
# These keep a shared lightweight capacity/load surrogate, but route it through
# explicit prefill/decode models instead of one unified TTFT-centric path.
PREFILL_CLF_FEATURES = [
    "log_il", "log_ol", "tp", "freq_norm", "log_rate",
    "log_d_prefill", "log_d_decode", "prefill_frac",
    "log_rho_prefill", "rho_prefill_overflow", "rho_prefill_sq",
    "log_rho_total", "rho_total_overflow",
]
PREFILL_REG_FEATURES = [
    "log_il", "log_ol", "tp", "freq_norm", "log_rate",
    "log_d_prefill", "prefill_frac", "log_rho_prefill",
    "rho_prefill_overflow", "log_rho_total",
]
PREFILL_POWER_FEATURES = [
    "log_il", "log_ol", "tp", "freq_norm", "log_rate",
    "log_d_prefill", "log_rho_prefill", "log_rho_total",
]

DECODE_CLF_FEATURES = [
    "log_il", "log_ol", "tp", "freq_norm", "log_rate",
    "log_d_prefill", "log_d_decode", "decode_frac",
    "log_kv_pressure",
    "log_rho_decode", "rho_decode_overflow", "rho_decode_sq",
    "log_rho_total", "rho_total_overflow",
]
DECODE_REG_FEATURES = [
    "log_il", "log_ol", "tp", "freq_norm", "log_rate",
    "log_d_decode", "decode_frac", "log_kv_pressure", "log_rho_decode",
    "rho_decode_overflow", "log_rho_total",
]
DECODE_POWER_FEATURES = [
    "log_il", "log_ol", "tp", "freq_norm", "log_rate",
    "log_d_decode", "log_rho_decode", "log_rho_total",
]


def prepare_data(data_file: str,
                 anomaly_filter: Callable[[pd.DataFrame], pd.Series],
                 max_freq: int,
                 run_filter=None) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    dup_key = ["step", "gpu_freq_mhz", "tp_degree",
               "input_len", "output_len", "request_rate", "run"]
    df = df.drop_duplicates(subset=dup_key, keep="last").copy()
    df = drop_warmup_run1(df)
    df = df[anomaly_filter(df)].copy()
    if run_filter is not None:
        df = df[df["run"].isin(run_filter)].copy()

    group_cols = ["step", "gpu_type", "gpu_freq_mhz", "tp_degree",
                  "input_len", "output_len", "request_rate"]
    num_cols = ["mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
                "avg_power_w", "max_power_w",
                "request_throughput_rps", "output_token_throughput_tps",
                "total_token_throughput_tps",
                "benchmark_duration_s", "avg_gpu_util_pct"]
    avg = df.groupby(group_cols, as_index=False)[num_cols].mean()

    avg["freq"] = avg["gpu_freq_mhz"]
    avg["il"] = avg["input_len"]
    avg["ol"] = avg["output_len"]
    avg["rate"] = avg["request_rate"]
    avg["tp"] = avg["tp_degree"]
    avg["ttft"] = avg["mean_ttft_ms"]
    avg["tpot"] = avg["mean_tpot_ms"]
    avg["p99_ttft"] = avg["p99_ttft_ms"]
    avg["p99_tpot"] = avg["p99_tpot_ms"]
    avg["power"] = avg["avg_power_w"]
    avg["power_per_gpu"] = avg["power"] / avg["tp"]
    avg["tps"] = avg["total_token_throughput_tps"]
    avg["decode_frac"] = avg["ol"] / (avg["il"] + avg["ol"])
    avg["log_il"] = np.log1p(avg["il"])
    avg["log_ol"] = np.log1p(avg["ol"])
    avg["log_rate"] = np.log1p(avg["rate"])
    avg["freq_norm"] = avg["freq"] / max_freq

    return avg


def check_plateau(group: pd.DataFrame) -> bool:
    if len(group) < 2:
        return False
    sorted_g = group.sort_values("rate")
    tps_vals = sorted_g["tps"].values
    ratio = tps_vals[-1] / tps_vals[-2] if tps_vals[-2] > 0 else 999
    return ratio < 1.03


def build_group_key(df: pd.DataFrame) -> pd.Series:
    return (df["il"].astype(str) + "_" + df["ol"].astype(str) + "_" +
            df["tp"].astype(str) + "_" + df["freq"].astype(str))


def train_capacity_model(data: pd.DataFrame, max_freq: int):
    cap_groups = data.groupby(["il", "ol", "tp", "freq"]).agg(
        max_tps=("tps", "max"),
        max_demand=("rate", "max"),
        n_rates=("rate", "nunique"),
    ).reset_index()

    plateau_flags = data.groupby(["il", "ol", "tp", "freq"]).apply(check_plateau)
    plateau_flags.name = "is_plateau"
    cap_groups = cap_groups.merge(plateau_flags.reset_index(),
                                  on=["il", "ol", "tp", "freq"], how="left")
    cap_groups["max_demand_tokens"] = cap_groups["max_demand"] * (
        cap_groups["il"] + cap_groups["ol"])
    cap_groups["is_saturated"] = cap_groups["max_tps"] < cap_groups["max_demand_tokens"] * 0.9
    cap_groups["plateau_confirmed"] = cap_groups["is_plateau"] | cap_groups["is_saturated"]

    cap_train = cap_groups[cap_groups["plateau_confirmed"]].copy()
    if len(cap_train) < 5:
        cap_train = cap_groups.copy()

    for c in CAP_FEATURES:
        if c == "log_il":
            cap_train[c] = np.log1p(cap_train["il"])
        elif c == "log_ol":
            cap_train[c] = np.log1p(cap_train["ol"])
        elif c == "freq_norm":
            cap_train[c] = cap_train["freq"] / max_freq
        elif c == "decode_frac":
            cap_train[c] = cap_train["ol"] / (cap_train["il"] + cap_train["ol"])
    cap_train["log_cap"] = np.log(cap_train["max_tps"])

    gbr_cap = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        min_samples_leaf=3, random_state=42)
    gbr_cap.fit(cap_train[CAP_FEATURES], cap_train["log_cap"])

    for c in CAP_FEATURES:
        if c == "log_il":
            cap_groups[c] = np.log1p(cap_groups["il"])
        elif c == "log_ol":
            cap_groups[c] = np.log1p(cap_groups["ol"])
        elif c == "freq_norm":
            cap_groups[c] = cap_groups["freq"] / max_freq
        elif c == "decode_frac":
            cap_groups[c] = cap_groups["ol"] / (cap_groups["il"] + cap_groups["ol"])
    cap_groups["C_hat"] = np.exp(gbr_cap.predict(cap_groups[CAP_FEATURES]))
    return gbr_cap, cap_groups, cap_train


def compute_features(data: pd.DataFrame, cap_model, cap_groups_df: pd.DataFrame,
                     max_freq: int, gamma: float = GAMMA, eta: float = ETA) -> pd.DataFrame:
    del max_freq  # normalization is already materialized in the input rows
    d = data.copy()
    d["C_hat_raw"] = np.exp(cap_model.predict(d[CAP_FEATURES]))

    if cap_groups_df is not None:
        plateau_map = cap_groups_df.set_index(
            ["il", "ol", "tp", "freq"])["plateau_confirmed"].to_dict()
        d["plateau_confirmed"] = [plateau_map.get((il, ol, tp, freq), False)
                                  for il, ol, tp, freq in zip(d["il"], d["ol"], d["tp"], d["freq"])]
    else:
        d["plateau_confirmed"] = False

    d["C_hat"] = np.where(d["plateau_confirmed"], d["C_hat_raw"], eta * d["C_hat_raw"])
    d["D_prefill"] = d["rate"] * d["il"]
    d["D_decode"] = d["rate"] * d["ol"]
    d["demand"] = d["D_prefill"] + gamma * d["D_decode"]
    d["rho"] = d["demand"] / d["C_hat"]
    d["rho_prefill"] = d["D_prefill"] / d["C_hat"]
    d["rho_decode"] = (gamma * d["D_decode"]) / d["C_hat"]
    d["log_rho"] = np.log1p(d["rho"])
    d["log_rho_total"] = d["log_rho"]
    d["rho_sq"] = d["rho"] ** 2
    d["rho_overflow"] = np.maximum(0, d["rho"] - 1.0)
    d["log_rho_prefill"] = np.log1p(d["rho_prefill"])
    d["log_rho_decode"] = np.log1p(d["rho_decode"])
    d["rho_prefill_sq"] = d["rho_prefill"] ** 2
    d["rho_decode_sq"] = d["rho_decode"] ** 2
    d["rho_prefill_overflow"] = np.maximum(0, d["rho_prefill"] - 1.0)
    d["rho_decode_overflow"] = np.maximum(0, d["rho_decode"] - 1.0)
    d["rho_total_overflow"] = d["rho_overflow"]
    d["log_d_prefill"] = np.log1p(d["D_prefill"])
    d["log_d_decode"] = np.log1p(d["D_decode"])
    d["log_kv_pressure"] = np.log1p(d["rate"] * d["il"] * d["ol"])
    d["prefill_frac"] = d["D_prefill"] / (d["D_prefill"] + gamma * d["D_decode"])
    d["decode_frac"] = 1.0 - d["prefill_frac"]
    return d
