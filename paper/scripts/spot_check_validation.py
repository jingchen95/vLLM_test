#!/usr/bin/env python3
"""
Spot-Check Validation: selects decision-critical configs for hardware testing.

Analyzes simulation results and model predictions to identify configs that
should be validated on real hardware. Selects ~20 configs per GPU type,
stratified across 5 categories:

1. Accepted by guard band — configs the classifier predicts safe
2. Rejected by guard band — configs predicted unsafe (verify conservatism)
3. Near threshold — configs within ±10% of rho_th or p_th boundary
4. Chosen by JSEP in simulation — most frequently selected configs
5. Chosen by baseline but not JSEP — configs that differ between strategies
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from collections import Counter

from paths import PAPER_RESULTS_ANALYSES_DIR, paper_model_dir
from scheduler import EnergyScheduler
from jsep_traces import IL_VALUES, OL_VALUES


def load_simulation_results():
    """Load JSEP simulation detail results."""
    detail_file = PAPER_RESULTS_ANALYSES_DIR / "jsep_results_detail.csv"
    if not os.path.exists(detail_file):
        print(f"ERROR: {detail_file} not found. Run jsep_simulator.py first.")
        sys.exit(1)
    return pd.read_csv(detail_file)


def get_jsep_chosen_configs(detail_df, gpu_type):
    """Extract most frequently chosen configs by JSEP across all traces/SLOs."""
    jsep_df = detail_df[detail_df["strategy"] == "jsep"].copy()
    tp_col = f"{gpu_type}_tp"
    freq_col = f"{gpu_type}_freq"
    active_col = f"{gpu_type}_active"

    # Filter to rows where this pool is active
    active = jsep_df[jsep_df[active_col] > 0].copy()
    if len(active) == 0:
        return []

    configs = list(zip(
        active["il"], active["ol"], active["rate"],
        active[tp_col], active[freq_col], active["slo_ms"]
    ))
    counter = Counter(configs)
    # Return top 10 most frequent
    return [(cfg, count) for cfg, count in counter.most_common(10)]


def get_baseline_only_configs(detail_df, gpu_type):
    """Find configs chosen by baselines but NOT by JSEP."""
    tp_col = f"{gpu_type}_tp"
    freq_col = f"{gpu_type}_freq"

    jsep_df = detail_df[detail_df["strategy"] == "jsep"]
    jsep_configs = set(zip(jsep_df[tp_col], jsep_df[freq_col]))

    baseline_only = []
    for strat in ["naive", "dvfs_only", "routing_only", "dynamollm"]:
        strat_df = detail_df[detail_df["strategy"] == strat]
        strat_configs = set(zip(strat_df[tp_col], strat_df[freq_col]))
        diff = strat_configs - jsep_configs
        for tp, freq in diff:
            if pd.notna(tp) and pd.notna(freq) and tp > 0 and freq > 0:
                baseline_only.append((strat, int(tp), int(freq)))

    return baseline_only[:5]


def select_spot_check_configs(gpu_type, model_dir, detail_df, n_target=20):
    """Select decision-critical configs for one GPU type."""
    scheduler = EnergyScheduler(model_dir=model_dir)
    configs = []
    seen = set()

    def add_config(il, ol, rate, tp, freq, slo, category, pred):
        key = (il, ol, rate, tp, freq, slo)
        if key in seen:
            return
        seen.add(key)
        configs.append({
            "gpu_type": gpu_type,
            "il": il, "ol": ol, "rate": rate,
            "tp": tp, "freq_mhz": freq, "slo_ms": slo,
            "category": category,
            "predicted_safe": pred.get("is_safe", False),
            "p_violate": pred.get("p_violate", 0.0),
            "rho": pred.get("rho", 0.0),
            "predicted_p99_ttft_ms": pred.get("p99_ttft_ms", 0.0),
            "predicted_power_w": pred.get("total_power_w", 0.0),
        })

    # Category 1 & 2 & 3: Sweep guard-band boundary
    print(f"  Scanning guard-band boundary for {gpu_type}...")
    test_rates = [1, 2, 5, 10]
    for slo in scheduler.GUARD_SETTINGS:
        gs = scheduler.GUARD_SETTINGS[slo]
        for il in IL_VALUES:
            for ol in OL_VALUES:
                for rate in test_rates:
                    for tp in scheduler.TP_DEGREES:
                        for freq in scheduler.FREQUENCIES:
                            pred = scheduler.predict_config(
                                il, ol, tp, freq, rate, slo)

                            # Category 3: near threshold
                            p_near = abs(pred["p_violate"] - gs["p_th"]) / max(gs["p_th"], 0.01) < 0.10
                            rho_near = abs(pred["rho"] - gs["rho_th"]) / max(gs["rho_th"], 0.01) < 0.10
                            if p_near or rho_near:
                                add_config(il, ol, rate, tp, freq, slo,
                                          "near_threshold", pred)

    # From near-threshold, also pick some safe and unsafe
    for slo in scheduler.GUARD_SETTINGS:
        for il in [128, 512, 1024]:
            for ol in [128, 512]:
                for rate in [2, 5, 10]:
                    # Find the lowest-power safe config
                    rec = scheduler.recommend(il, ol, rate, slo)
                    if rec.get("status") != "NO_SAFE_CONFIG":
                        best = rec["recommended"]
                        pred = scheduler.predict_config(
                            il, ol, best["tp"], best["freq_mhz"],
                            rate, slo)
                        add_config(il, ol, rate, best["tp"],
                                  best["freq_mhz"], slo,
                                  "accepted_safe", pred)

                    # Pick one rejected config (unsafe)
                    for tp in scheduler.TP_DEGREES[:1]:
                        low_freq = scheduler.FREQUENCIES[0]
                        pred = scheduler.predict_config(
                            il, ol, tp, low_freq, rate, slo)
                        if not pred["is_safe"]:
                            add_config(il, ol, rate, tp, low_freq, slo,
                                      "rejected_unsafe", pred)
                            break

    # Category 4: Chosen by JSEP in simulation
    print(f"  Extracting JSEP-chosen configs...")
    jsep_chosen = get_jsep_chosen_configs(detail_df, gpu_type)
    for (il, ol, rate, tp, freq, slo), count in jsep_chosen:
        tp, freq = int(tp), int(freq)
        pred = scheduler.predict_config(il, ol, tp, freq, rate, slo)
        add_config(il, ol, rate, tp, freq, slo,
                  f"jsep_chosen(n={count})", pred)

    # Category 5: Baseline-only configs
    print(f"  Finding baseline-only configs...")
    baseline_only = get_baseline_only_configs(detail_df, gpu_type)
    for strat, tp, freq in baseline_only:
        for il in [512]:
            for ol in [128]:
                for rate in [5]:
                    for slo in [500]:
                        pred = scheduler.predict_config(
                            il, ol, tp, freq, rate, slo)
                        add_config(il, ol, rate, tp, freq, slo,
                                  f"baseline_only({strat})", pred)

    df = pd.DataFrame(configs)

    # Stratified selection: aim for ~n_target total
    if len(df) > n_target:
        selected = []
        # Priority: jsep_chosen configs (keep all, up to 5)
        jsep_mask = df["category"].str.contains("jsep_chosen")
        selected.append(df[jsep_mask].head(5))

        # Near threshold: sample stratified by safe/unsafe
        near_mask = df["category"].str.contains("near_threshold")
        near_df = df[near_mask]
        near_safe = near_df[near_df["predicted_safe"]]
        near_unsafe = near_df[~near_df["predicted_safe"]]
        n_near = max(0, n_target - 8)  # reserve slots for other categories
        n_safe = min(len(near_safe), n_near // 2)
        n_unsafe = min(len(near_unsafe), n_near - n_safe)
        if n_safe > 0:
            selected.append(near_safe.sample(n=n_safe, random_state=42))
        if n_unsafe > 0:
            selected.append(near_unsafe.sample(n=n_unsafe, random_state=42))

        # Other categories: keep remaining
        other_mask = ~jsep_mask & ~near_mask
        remaining = n_target - sum(len(s) for s in selected)
        if remaining > 0:
            selected.append(df[other_mask].head(remaining))

        df = pd.concat(selected).drop_duplicates().reset_index(drop=True)

    return df


def main():
    detail_df = load_simulation_results()

    print("=" * 60)
    print("SPOT-CHECK CONFIG SELECTION")
    print("=" * 60)

    all_configs = []
    for gpu_type, model_dir in [("l40s", paper_model_dir("models_l40s")), ("l4", paper_model_dir("models_l4"))]:
        full_model_dir = os.fspath(model_dir)
        if not os.path.exists(full_model_dir):
            print(f"  WARNING: {full_model_dir} not found, skipping {gpu_type}")
            continue

        print(f"\n--- {gpu_type.upper()} ---")
        df = select_spot_check_configs(
            gpu_type, full_model_dir, detail_df, n_target=20)
        all_configs.append(df)

        out_file = PAPER_RESULTS_ANALYSES_DIR / f"spot_check_configs_{gpu_type}.csv"
        df.to_csv(out_file, index=False)
        print(f"  Selected {len(df)} configs -> {out_file}")

        # Summary by category
        print(f"\n  Category breakdown:")
        for cat in df["category"].unique():
            n = len(df[df["category"] == cat])
            n_safe = df[df["category"] == cat]["predicted_safe"].sum()
            print(f"    {cat:30s}: {n:3d} configs ({n_safe} safe)")

    # Print sample configs
    if all_configs:
        combined = pd.concat(all_configs).reset_index(drop=True)
        print(f"\n{'=' * 60}")
        print(f"TOTAL: {len(combined)} configs selected for spot-check")
        print(f"{'=' * 60}")

        print(f"\nSample configs (first 10):")
        print(combined[["gpu_type", "il", "ol", "rate", "tp", "freq_mhz",
                        "slo_ms", "category", "predicted_safe",
                        "predicted_p99_ttft_ms", "predicted_power_w"
                        ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
