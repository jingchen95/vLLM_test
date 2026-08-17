#!/usr/bin/env python3
"""
Cross-Model Validation: breaks model-on-model circularity.

Trains two independent model sets on different data splits:
  - Model-A: trained on runs {1, 2}
  - Model-B: trained on run {3}

Then runs JSEP simulation where:
  - Scheduler uses Model-A for decide() calls
  - Evaluator checks SLO compliance and computes power using Model-B

If JSEP still outperforms baselines under cross-model evaluation,
the gains are less likely to be artifacts of model self-consistency.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from paths import PAPER_MODELS_DIR, PAPER_RESULTS_ANALYSES_DIR
from jsep_traces import generate_steady, generate_bursty, generate_diurnal
from jsep_cluster import ClusterState
from jsep_scheduler import create_all_strategies
from scheduler import EnergyScheduler
from model_training_common import (
    GAMMA,
    ETA,
    GPU_CONFIGS,
    CAP_FEATURES,
    CLF_FEATURES,
    REG_FEATURES,
    POWER_FEATURES,
    prepare_data,
    train_capacity_model,
    compute_features,
)


def train_full_pipeline(gpu_type, run_filter, output_dir):
    """Train all 4 stages and save to output_dir."""
    cfg = GPU_CONFIGS[gpu_type]
    max_freq = cfg["max_freq"]
    guard_settings = cfg["guard_settings"]
    slo_thresholds = cfg["slo_thresholds"]

    print(f"  Loading data for {gpu_type} (runs={run_filter})...")
    data = prepare_data(cfg["data_file"], cfg["anomaly_filter"],
                        max_freq, run_filter)
    print(f"    {len(data)} averaged data points")

    # Stage 1: Capacity
    gbr_cap, cap_groups, _ = train_capacity_model(data, max_freq)
    n_plateau = cap_groups["plateau_confirmed"].sum()
    print(f"    Capacity: {n_plateau}/{len(cap_groups)} plateau-confirmed")

    # Stage 2: Features
    data = compute_features(data, gbr_cap, cap_groups, max_freq)

    # Stage 3: SLO classifiers
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(gbr_cap, f"{output_dir}/capacity_model.pkl")
    joblib.dump(CAP_FEATURES, f"{output_dir}/capacity_features.pkl")
    joblib.dump(cap_groups, f"{output_dir}/cap_groups.pkl")

    for slo in slo_thresholds:
        y = (data["p99_ttft"] > slo).astype(int).values
        sw = np.where(y == 1, 5.0, 1.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[CLF_FEATURES].values)
        gbc = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.1,
            min_samples_leaf=5, random_state=42)
        gbc.fit(X_scaled, y, sample_weight=sw)
        joblib.dump(gbc, f"{output_dir}/slo_classifier_{slo}.pkl")
        joblib.dump(scaler, f"{output_dir}/slo_scaler_{slo}.pkl")

    joblib.dump(CLF_FEATURES, f"{output_dir}/clf_features.pkl")

    # Stage 4: Regressor
    rho_cutoff = 1.5
    reg_data = data[data["rho"] < rho_cutoff].copy()
    if len(reg_data) < 10:
        reg_data = data.copy()
    scaler_r = StandardScaler()
    poly_r = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_reg = poly_r.fit_transform(scaler_r.fit_transform(reg_data[REG_FEATURES].values))
    y_reg = np.log1p(reg_data["p99_ttft"].values)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_reg, y_reg)
    joblib.dump(ridge, f"{output_dir}/regressor_p99.pkl")
    joblib.dump(scaler_r, f"{output_dir}/regressor_scaler.pkl")
    joblib.dump(poly_r, f"{output_dir}/regressor_poly.pkl")
    joblib.dump(REG_FEATURES, f"{output_dir}/reg_features.pkl")

    # Power model
    gbr_power = GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.1,
        min_samples_leaf=4, random_state=42)
    gbr_power.fit(data[POWER_FEATURES], data["power_per_gpu"])
    joblib.dump(gbr_power, f"{output_dir}/power_model.pkl")
    joblib.dump(POWER_FEATURES, f"{output_dir}/power_features.pkl")

    # Config
    config = {
        "GAMMA": GAMMA,
        "ETA": ETA,
        "MAX_FREQ": max_freq,
        "GUARD_SETTINGS": guard_settings,
        "SLO_THRESHOLDS": slo_thresholds,
        "TP_DEGREES": sorted(data["tp"].unique().tolist()),
        "FREQUENCIES": sorted(data["freq"].unique().tolist()),
    }
    joblib.dump(config, f"{output_dir}/config.pkl")
    print(f"    Models saved to {output_dir}/")
    return config


# ============================================================
# Cross-model simulation
# ============================================================

WINDOW_S = 5.0
DURATION = 600.0
SNAP_IL = [32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
SNAP_OL = [32, 128, 512, 1024]


def snap_to_nearest(val, grid):
    return min(grid, key=lambda x: abs(x - val))


def run_cross_model_simulation(slo_list=[200, 500, 1000]):
    """Run simulation with Model-A scheduling, Model-B evaluation."""

    # Step 1: Train split models
    print("=" * 60)
    print("STEP 1: Training split models")
    print("=" * 60)

    base = os.fspath(PAPER_MODELS_DIR)
    for gpu_type in ["l40s", "l4"]:
        print(f"\n  --- {gpu_type.upper()} ---")
        train_full_pipeline(gpu_type, run_filter=[1, 2],
                           output_dir=os.path.join(base, f"models_{gpu_type}_A"))
        train_full_pipeline(gpu_type, run_filter=[3],
                           output_dir=os.path.join(base, f"models_{gpu_type}_B"))

    # Step 2: Load schedulers
    print(f"\n{'=' * 60}")
    print("STEP 2: Loading Model-A (scheduler) and Model-B (evaluator)")
    print("=" * 60)

    strategies_A = create_all_strategies(
        model_dir_l40s=os.path.join(base, "models_l40s_A"),
        model_dir_l4=os.path.join(base, "models_l4_A"))
    strategy_names = list(strategies_A.keys())
    print(f"  Scheduler strategies (Model-A): {strategy_names}")

    eval_l40s = EnergyScheduler(model_dir=os.path.join(base, "models_l40s_B"))
    eval_l4 = EnergyScheduler(model_dir=os.path.join(base, "models_l4_B"))
    print(f"  Evaluator loaded (Model-B)")

    # Step 3: Generate traces
    print(f"\n{'=' * 60}")
    print("STEP 3: Generating traces")
    print("=" * 60)
    traces = {
        "steady": generate_steady(DURATION, rate_rps=10.0),
        "bursty": generate_bursty(DURATION),
        "diurnal": generate_diurnal(DURATION),
    }
    for name, reqs in traces.items():
        print(f"  {name}: {len(reqs)} requests")

    # Step 4: Run simulation
    print(f"\n{'=' * 60}")
    print("STEP 4: Cross-model simulation")
    print("=" * 60)

    rows = []
    for slo in slo_list:
        print(f"\n  SLO = {slo} ms")
        for trace_name, requests in traces.items():
            for strat_name in strategy_names:
                strat = strategies_A[strat_name]
                # Reset strategy state
                if hasattr(strat, 'current_phase'):
                    strat.current_phase = "LP"
                if hasattr(strat, 'last_reconfig_time'):
                    strat.last_reconfig_time = -999.0
                if hasattr(strat, 'prev_tp'):
                    strat.prev_tp = {"l40s": 1, "l4": 1}

                cluster = ClusterState()
                # Scheduler energy (Model-A evaluates itself)
                total_energy_A = 0.0
                # Cross-model energy (Model-B evaluates)
                total_energy_B = 0.0
                n_windows = 0
                n_violations_A = 0
                n_violations_B = 0

                t = 0.0
                while t < DURATION:
                    t_end = t + WINDOW_S
                    window_reqs = [r for r in requests
                                   if t <= r.arrival_time < t_end]

                    if len(window_reqs) == 0:
                        rate = 0.1
                        il, ol = 128, 128
                    else:
                        rate = len(window_reqs) / WINDOW_S
                        ils = [r.input_len for r in window_reqs]
                        ols = [r.output_len for r in window_reqs]
                        il = snap_to_nearest(np.median(ils), SNAP_IL)
                        ol = snap_to_nearest(np.percentile(ols, 75), SNAP_OL)

                    try:
                        result = strat.decide(il, ol, rate, slo, cluster,
                                              current_time=t)
                    except Exception:
                        result = strategies_A["jsep"].decide(
                            il, ol, rate, slo, cluster, current_time=t)

                    # Model-A evaluation (same as co-trained)
                    energy_A = result.total_power_w * WINDOW_S
                    total_energy_A += energy_A
                    if not result.slo_met:
                        n_violations_A += 1

                    # Model-B evaluation (cross-model)
                    # Re-evaluate the chosen config with Model-B
                    from jsep_cluster import GPU_SPECS
                    power_B = 0.0
                    slo_met_B = True

                    for pool_name in ["l40s", "l4"]:
                        pool_cfg = result.config.get(pool_name, {})
                        evaluator = eval_l40s if pool_name == "l40s" else eval_l4
                        tp_p = pool_cfg.get("tp", 1)
                        freq_p = pool_cfg.get("freq_mhz", 0)
                        n_active = pool_cfg.get("active_instances", 0)
                        rate_per_inst = pool_cfg.get("rate_per_instance", 0.0)
                        total_gpus = GPU_SPECS[pool_name]["total_gpus"]
                        idle_power = GPU_SPECS[pool_name]["idle_power_w"]

                        if n_active > 0 and freq_p > 0 and rate_per_inst > 0:
                            pred = evaluator.predict_config(
                                il, ol, tp_p, freq_p, rate_per_inst, slo)
                            power_B += n_active * pred["power_per_gpu_w"] * tp_p
                            # Idle GPUs in pool
                            idle_gpus = total_gpus - n_active * tp_p
                            if idle_gpus > 0:
                                power_B += idle_gpus * idle_power
                            if not pred["is_safe"]:
                                slo_met_B = False
                        else:
                            # Entire pool idle
                            power_B += total_gpus * idle_power

                    energy_B = power_B * WINDOW_S
                    total_energy_B += energy_B
                    if not slo_met_B:
                        n_violations_B += 1

                    n_windows += 1
                    t = t_end

                total_energy_A_kj = total_energy_A / 1000.0
                total_energy_B_kj = total_energy_B / 1000.0
                avg_power_A = total_energy_A / (n_windows * WINDOW_S)
                avg_power_B = total_energy_B / (n_windows * WINDOW_S)
                viol_rate_A = 100.0 * n_violations_A / n_windows
                viol_rate_B = 100.0 * n_violations_B / n_windows

                rows.append({
                    "trace": trace_name,
                    "strategy": strat_name,
                    "slo_ms": slo,
                    "energy_kj_comodel": round(total_energy_A_kj, 2),
                    "energy_kj_crossmodel": round(total_energy_B_kj, 2),
                    "avg_power_comodel": round(avg_power_A, 1),
                    "avg_power_crossmodel": round(avg_power_B, 1),
                    "viol_pct_comodel": round(viol_rate_A, 1),
                    "viol_pct_crossmodel": round(viol_rate_B, 1),
                })

                print(f"    {trace_name:8s} {strat_name:15s}: "
                      f"co={total_energy_A_kj:7.1f}kJ "
                      f"cross={total_energy_B_kj:7.1f}kJ "
                      f"viol_co={viol_rate_A:5.1f}% "
                      f"viol_cross={viol_rate_B:5.1f}%",
                      flush=True)

    df = pd.DataFrame(rows)

    # Step 5: Summary comparison
    print(f"\n{'=' * 60}")
    print("STEP 5: CROSS-MODEL VALIDATION SUMMARY")
    print("=" * 60)

    for slo in slo_list:
        slo_df = df[df["slo_ms"] == slo]
        print(f"\n  SLO = {slo} ms:")

        # Get naive energy as baseline
        naive_co = {}
        naive_cross = {}
        for trace in traces:
            naive_row = slo_df[(slo_df["strategy"] == "naive") &
                               (slo_df["trace"] == trace)]
            if len(naive_row) > 0:
                naive_co[trace] = naive_row["energy_kj_comodel"].values[0]
                naive_cross[trace] = naive_row["energy_kj_crossmodel"].values[0]

        print(f"  {'Strategy':15s} | {'Co-model savings':>18s} | "
              f"{'Cross-model savings':>20s} | {'Delta':>8s}")
        print("  " + "-" * 75)

        for strat_name in strategy_names:
            if strat_name == "naive":
                continue
            savings_co = []
            savings_cross = []
            for trace in traces:
                row = slo_df[(slo_df["strategy"] == strat_name) &
                             (slo_df["trace"] == trace)]
                if len(row) > 0 and trace in naive_co:
                    s_co = (1 - row["energy_kj_comodel"].values[0] /
                            naive_co[trace]) * 100
                    s_cross = (1 - row["energy_kj_crossmodel"].values[0] /
                               naive_cross[trace]) * 100
                    savings_co.append(s_co)
                    savings_cross.append(s_cross)

            if savings_co:
                avg_co = np.mean(savings_co)
                avg_cross = np.mean(savings_cross)
                delta = avg_cross - avg_co
                print(f"  {strat_name:15s} | {avg_co:>16.1f}% | "
                      f"{avg_cross:>18.1f}% | {delta:>+7.1f}pp")

    # Policy ordering check
    print(f"\n  POLICY ORDERING CHECK:")
    for slo in slo_list:
        slo_df = df[df["slo_ms"] == slo]
        ordering_co = []
        ordering_cross = []
        for strat_name in strategy_names:
            if strat_name == "naive":
                continue
            avg_co = slo_df[slo_df["strategy"] == strat_name][
                "energy_kj_comodel"].mean()
            avg_cross = slo_df[slo_df["strategy"] == strat_name][
                "energy_kj_crossmodel"].mean()
            ordering_co.append((strat_name, avg_co))
            ordering_cross.append((strat_name, avg_cross))

        ordering_co.sort(key=lambda x: x[1])
        ordering_cross.sort(key=lambda x: x[1])
        co_order = [x[0] for x in ordering_co]
        cross_order = [x[0] for x in ordering_cross]
        match = "MATCH" if co_order == cross_order else "DIFFERS"
        print(f"    SLO={slo}: co-model={co_order}, "
              f"cross-model={cross_order} [{match}]")

    out_file = os.fspath(PAPER_RESULTS_ANALYSES_DIR / "cross_model_results.csv")
    df.to_csv(out_file, index=False)
    print(f"\n  Results saved to {out_file}")
    return df


if __name__ == "__main__":
    run_cross_model_simulation(slo_list=[200, 500, 1000])
