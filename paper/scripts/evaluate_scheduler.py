#!/usr/bin/env python3
"""
Evaluate scheduler recommendations against actual benchmark data.

For each workload the scheduler recommends a config for, we check:
  1. Does the recommended config actually meet the SLO? (safety validation)
  2. How much energy does it save? (value validation)
  3. How accurate are the predictions? (model validation)

Two baselines:
  - DVFS baseline: same TP as recommended, but at max frequency.
    Isolates the DVFS contribution: "how much does frequency scaling save?"
  - Oracle baseline: actual lowest-power config from measured data that meets SLO.
    Shows optimality gap: "how close to the best possible is the scheduler?"

Usage:
  python3 evaluate_scheduler.py --data ../Phase2_Results_L40S/master_results.csv
  python3 evaluate_scheduler.py --data ../Phase2_Results_L40S/master_results.csv --gen-validation
"""

import pandas as pd
import numpy as np
import argparse
from scheduler import EnergyScheduler
from paths import paper_model_dir


def load_actual_data(csv_file):
    """Load and average benchmark data."""
    df = pd.read_csv(csv_file)

    # Exclude same anomalies as model_fitting.py
    anomaly_mask = (
        ((df["tp_degree"] == 4) & (df["input_len"] == 2048) & (df["output_len"] == 32)
         & (df["request_rate"] == 10) & (df["run"] == 1)) |
        ((df["tp_degree"] == 4) & (df["input_len"] == 1024) & (df["output_len"] == 128)
         & (df["request_rate"] == 10) & (df["gpu_freq_mhz"] == 2520) & (df["run"] == 1)) |
        ((df["tp_degree"] == 2) & (df["input_len"] == 512) & (df["output_len"] == 512)
         & (df["request_rate"] == 10) & (df["run"] == 1))
    )
    df = df[~anomaly_mask].copy()

    group_cols = ["gpu_freq_mhz", "tp_degree", "input_len", "output_len", "request_rate"]
    num_cols = ["p99_ttft_ms", "avg_power_w", "total_token_throughput_tps"]
    avg = df.groupby(group_cols, as_index=False)[num_cols].mean()

    avg["freq"] = avg["gpu_freq_mhz"]
    avg["il"] = avg["input_len"]
    avg["ol"] = avg["output_len"]
    avg["rate"] = avg["request_rate"]
    avg["tp"] = avg["tp_degree"]
    avg["p99_ttft"] = avg["p99_ttft_ms"]
    avg["power_per_gpu"] = avg["avg_power_w"] / avg["tp"]
    avg["total_power"] = avg["avg_power_w"]
    return avg


def find_oracle(actual, il, ol, rate, slo):
    """Find the actual lowest-power config that meets SLO from measured data."""
    candidates = actual[
        (actual["il"] == il) & (actual["ol"] == ol) & (actual["rate"] == rate) &
        (actual["p99_ttft"] <= slo)
    ]
    if len(candidates) == 0:
        return None
    return candidates.loc[candidates["total_power"].idxmin()]


def find_dvfs_baseline(actual, il, ol, rate, tp, max_freq):
    """Find the same-TP config at max frequency (isolates DVFS contribution)."""
    match = actual[
        (actual["il"] == il) & (actual["ol"] == ol) &
        (actual["rate"] == rate) & (actual["tp"] == tp) &
        (actual["freq"] == max_freq)
    ]
    if len(match) == 0:
        return None
    return match.iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--gen-validation", action="store_true",
                        help="Generate list of untested configs for validation benchmarking")
    args = parser.parse_args()

    print("Loading actual data...")
    actual = load_actual_data(args.data)
    print(f"  {len(actual)} averaged data points")

    print("\nLoading scheduler...")
    scheduler = EnergyScheduler(model_dir=args.model_dir)

    workloads = actual[["il", "ol", "rate"]].drop_duplicates().sort_values(["il", "ol", "rate"])
    print(f"\n{len(workloads)} unique workloads in data")

    all_untested = []  # Collect untested configs across all SLOs

    for slo in scheduler.SLO_THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"SLO = {slo} ms")
        print(f"{'='*70}")

        results = []
        for _, wl in workloads.iterrows():
            il, ol, rate = int(wl["il"]), int(wl["ol"]), int(wl["rate"])
            rec = scheduler.recommend(il, ol, rate, slo)

            if rec["status"] != "OK":
                results.append({
                    "il": il, "ol": ol, "rate": rate,
                    "status": "NO_SAFE_CONFIG",
                })
                continue

            cfg = rec["recommended"]

            # Look up actual data for recommended config
            match = actual[
                (actual["il"] == il) & (actual["ol"] == ol) &
                (actual["rate"] == rate) & (actual["tp"] == cfg["tp"]) &
                (actual["freq"] == cfg["freq_mhz"])
            ]

            if len(match) == 0:
                results.append({
                    "il": il, "ol": ol, "rate": rate,
                    "rec_tp": cfg["tp"], "rec_freq": cfg["freq_mhz"],
                    "pred_p99": cfg["p99_ttft_ms"],
                    "pred_power": cfg["total_power_w"],
                    "status": "CONFIG_NOT_IN_DATA",
                })
                all_untested.append({
                    "il": il, "ol": ol, "rate": rate,
                    "tp": cfg["tp"], "freq": cfg["freq_mhz"],
                    "slo": slo, "pred_p99": cfg["p99_ttft_ms"],
                    "pred_power": cfg["total_power_w"],
                })
                continue

            m = match.iloc[0]
            actual_meets_slo = m["p99_ttft"] <= slo

            # Baseline 1: DVFS baseline — same TP, max frequency
            dvfs_bl = find_dvfs_baseline(actual, il, ol, rate, cfg["tp"], scheduler.MAX_FREQ)
            if dvfs_bl is not None:
                dvfs_saving = (dvfs_bl["total_power"] - m["total_power"]) / dvfs_bl["total_power"] * 100
                dvfs_bl_power = dvfs_bl["total_power"]
            else:
                dvfs_saving = None
                dvfs_bl_power = None

            # Baseline 2: Oracle — actual lowest-power config that meets SLO
            oracle = find_oracle(actual, il, ol, rate, slo)
            if oracle is not None:
                oracle_power = oracle["total_power"]
                oracle_tp = int(oracle["tp"])
                oracle_freq = int(oracle["freq"])
                # How much more power does the scheduler use vs oracle?
                oracle_gap = (m["total_power"] - oracle_power) / oracle_power * 100
            else:
                oracle_power = None
                oracle_tp = None
                oracle_freq = None
                oracle_gap = None

            results.append({
                "il": il, "ol": ol, "rate": rate,
                "rec_tp": cfg["tp"], "rec_freq": cfg["freq_mhz"],
                "pred_p99": cfg["p99_ttft_ms"],
                "actual_p99": round(m["p99_ttft"], 1),
                "actual_meets_slo": actual_meets_slo,
                "pred_power": cfg["total_power_w"],
                "actual_power": round(m["total_power"], 1),
                # DVFS baseline
                "dvfs_bl_power": round(dvfs_bl_power, 1) if dvfs_bl_power else None,
                "dvfs_saving_pct": round(dvfs_saving, 1) if dvfs_saving is not None else None,
                # Oracle baseline
                "oracle_tp": oracle_tp,
                "oracle_freq": oracle_freq,
                "oracle_power": round(oracle_power, 1) if oracle_power else None,
                "oracle_gap_pct": round(oracle_gap, 1) if oracle_gap is not None else None,
                "status": "OK",
            })

        df = pd.DataFrame(results)
        ok = df[df["status"] == "OK"]
        no_safe = df[df["status"] == "NO_SAFE_CONFIG"]
        not_in_data = df[df["status"] == "CONFIG_NOT_IN_DATA"]

        # --- Coverage ---
        print(f"\n  Coverage:")
        print(f"    Safe config found: {len(ok) + len(not_in_data)} / {len(df)} "
              f"({(len(ok)+len(not_in_data))/len(df)*100:.0f}%)")
        print(f"    No safe config: {len(no_safe)} / {len(df)}")
        if len(not_in_data) > 0:
            print(f"    Config not in data (untested): {len(not_in_data)}")
            for _, r in not_in_data.iterrows():
                print(f"      {int(r['il']):>5d}:{int(r['ol']):<5d} r={int(r['rate'])} "
                      f"→ TP={int(r['rec_tp'])} f={int(r['rec_freq'])} (no benchmark data)")

        if len(ok) == 0:
            print("  No evaluable configs (all recommended configs missing from data)")
            continue

        # --- Safety ---
        violations = ok[~ok["actual_meets_slo"]]
        print(f"\n  Safety (the critical metric):")
        print(f"    SLO violations: {len(violations)} / {len(ok)} "
              f"({len(violations)/len(ok)*100:.1f}%)")
        if len(violations) > 0:
            for _, r in violations.iterrows():
                print(f"      VIOLATION: {int(r['il']):>5d}:{int(r['ol']):<5d} r={int(r['rate'])} "
                      f"TP={int(r['rec_tp'])} f={int(r['rec_freq'])}: "
                      f"actual={r['actual_p99']:.0f}ms > {slo}ms (pred={r['pred_p99']:.0f}ms)")

        # --- DVFS savings (Baseline 1) ---
        has_dvfs = ok[ok["dvfs_saving_pct"].notna()]
        if len(has_dvfs) > 0:
            print(f"\n  DVFS savings (same TP, max-freq → recommended freq):")
            print(f"    Mean:   {has_dvfs['dvfs_saving_pct'].mean():.1f}%")
            print(f"    Median: {has_dvfs['dvfs_saving_pct'].median():.1f}%")
            print(f"    Max:    {has_dvfs['dvfs_saving_pct'].max():.1f}%")
            print(f"    Min:    {has_dvfs['dvfs_saving_pct'].min():.1f}%")

        # --- Oracle gap (Baseline 2) ---
        has_oracle = ok[ok["oracle_gap_pct"].notna()]
        if len(has_oracle) > 0:
            print(f"\n  Oracle gap (scheduler power vs best measured config):")
            print(f"    Mean gap:   {has_oracle['oracle_gap_pct'].mean():.1f}% above oracle")
            print(f"    Median gap: {has_oracle['oracle_gap_pct'].median():.1f}%")
            n_optimal = (has_oracle["oracle_gap_pct"] <= 1.0).sum()
            print(f"    At or near oracle (<1% gap): {n_optimal} / {len(has_oracle)}")

        # --- Detailed table ---
        print(f"\n  {'il':>5s} {'ol':>5s} {'r':>2s} | {'TP':>2s} {'freq':>5s} | "
              f"{'pred_p99':>8s} {'act_p99':>8s} {'safe':>4s} | "
              f"{'DVFS_sav':>8s} | {'oracle':>10s} {'gap':>5s}")
        for _, r in ok.sort_values("dvfs_saving_pct", ascending=False, na_position="last").iterrows():
            safe_mark = "OK" if r["actual_meets_slo"] else "FAIL"
            dvfs_str = f"{r['dvfs_saving_pct']:>7.1f}%" if pd.notna(r.get("dvfs_saving_pct")) else "    N/A"
            if pd.notna(r.get("oracle_tp")):
                oracle_str = f"TP={int(r['oracle_tp'])} f={int(r['oracle_freq'])}"
                gap_str = f"{r['oracle_gap_pct']:>4.1f}%"
            else:
                oracle_str = "N/A"
                gap_str = "N/A"
            print(f"  {int(r['il']):>5d} {int(r['ol']):>5d} {int(r['rate']):>2d} | "
                  f"{int(r['rec_tp']):>2d} {int(r['rec_freq']):>5d} | "
                  f"{r['pred_p99']:>7.0f}ms {r['actual_p99']:>7.0f}ms {safe_mark:>4s} | "
                  f"{dvfs_str} | {oracle_str:>10s} {gap_str:>5s}")

        # --- Prediction accuracy ---
        print(f"\n  Prediction accuracy:")
        p99_errors = np.abs(ok["pred_p99"] - ok["actual_p99"]) / ok["actual_p99"] * 100
        power_errors = np.abs(ok["pred_power"] - ok["actual_power"]) / ok["actual_power"] * 100
        print(f"    P99 TTFT MAPE: {p99_errors.mean():.1f}%")
        print(f"    Power MAPE:    {power_errors.mean():.1f}%")

    # Save evaluation results
    output_file = "scheduler_evaluation.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")

    # --- Generate validation configs ---
    if args.gen_validation and all_untested:
        print(f"\n{'='*70}")
        print(f"VALIDATION CONFIGS (untested scheduler recommendations)")
        print(f"{'='*70}")

        val_df = pd.DataFrame(all_untested)
        # Deduplicate: same (il, ol, rate, tp, freq) across SLOs
        unique_configs = val_df.drop_duplicates(subset=["il", "ol", "rate", "tp", "freq"])
        unique_configs = unique_configs.sort_values(["tp", "freq", "il", "ol", "rate"])

        print(f"\n  Total untested recommendations: {len(val_df)}")
        print(f"  Unique (il, ol, rate, tp, freq) configs: {len(unique_configs)}")

        print(f"\n  {'il':>5s} {'ol':>5s} {'rate':>4s} | {'TP':>2s} {'freq':>5s} | "
              f"{'pred_p99':>8s} {'pred_pwr':>8s} | SLOs")
        for _, r in unique_configs.iterrows():
            # Find all SLOs this config was recommended for
            matching_slos = val_df[
                (val_df["il"] == r["il"]) & (val_df["ol"] == r["ol"]) &
                (val_df["rate"] == r["rate"]) & (val_df["tp"] == r["tp"]) &
                (val_df["freq"] == r["freq"])
            ]["slo"].tolist()
            slo_str = ",".join(str(s) for s in sorted(matching_slos))
            print(f"  {int(r['il']):>5d} {int(r['ol']):>5d} {int(r['rate']):>4d} | "
                  f"{int(r['tp']):>2d} {int(r['freq']):>5d} | "
                  f"{r['pred_p99']:>7.0f}ms {r['pred_power']:>7.0f}W | {slo_str}")

        # Save as CSV for easy benchmarking
        val_file = "validation_configs.csv"
        unique_configs[["il", "ol", "rate", "tp", "freq"]].to_csv(val_file, index=False)
        print(f"\n  Configs saved to {val_file}")
        print(f"  Run these {len(unique_configs)} configs × 3 runs = "
              f"{len(unique_configs) * 3} benchmark runs")


if __name__ == "__main__":
    main()
