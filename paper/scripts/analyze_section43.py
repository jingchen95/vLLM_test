#!/usr/bin/env python3
"""Analyze section43 production trace results and print paper-ready text."""

import pandas as pd
import numpy as np
import sys
import os

CSV_PATH = os.path.join(os.path.dirname(__file__),
                         "../../results/paper/section43_production/summary.csv")


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"Traces: {sorted(df['trace'].unique())}")
    print(f"Strategies: {sorted(df['strategy'].unique())}")
    print()

    # Use tau_kv=16 as reference
    if 'tau_kv_us' in df.columns:
        df = df[df['tau_kv_us'] == 16.0].copy()
    print(f"After tau_kv=16 filter: {len(df)} rows")
    print()

    df['energy_kj'] = df['total_energy_j'] / 1000.0

    STRATS = ['static_disagg', 'greenllm', 'dynamollm', 'sweep_llm']
    SLOS = [200, 500, 1000]
    TRACES = sorted(df['trace'].unique())

    # ----------------------------------------------------------------
    # Table: energy (kJ) and violation rate per (trace, slo, strategy)
    # ----------------------------------------------------------------
    print("=" * 80)
    print("ENERGY (kJ) BY TRACE × SLO × STRATEGY")
    print("=" * 80)
    slo_col = 'slo_ttft_ms' if 'slo_ttft_ms' in df.columns else 'slo_ms'
    for trace in TRACES:
        for slo in SLOS:
            subset = df[(df['trace'] == trace) & (df[slo_col] == slo)]
            if subset.empty:
                continue
            row_str = f"{trace:12s} SLO={slo:4d}ms |"
            for s in STRATS:
                r = subset[subset['strategy'] == s]
                if r.empty:
                    row_str += f"  {'N/A':>12}"
                else:
                    e = r['energy_kj'].values[0]
                    v = r['slo_violation_rate_pct'].values[0]
                    row_str += f"  {e:7.0f}kJ {v:5.1f}%viol"
            print(row_str)
        print()

    # ----------------------------------------------------------------
    # SWEEP-LLM savings vs static_disagg (baseline)
    # ----------------------------------------------------------------
    print("=" * 80)
    print("SWEEP-LLM SAVINGS vs STATIC-DISAGG (at tau_kv=16)")
    print("=" * 80)
    for slo in SLOS:
        print(f"\n  SLO = {slo} ms:")
        for trace in TRACES:
            sub = df[(df['trace'] == trace) & (df[slo_col] == slo)]
            ref = sub[sub['strategy'] == 'static_disagg']['energy_kj'].values
            jsep = sub[sub['strategy'] == 'sweep_llm']['energy_kj'].values
            if len(ref) > 0 and len(jsep) > 0:
                saving = (1 - jsep[0] / ref[0]) * 100
                viol_ref = sub[sub['strategy'] == 'static_disagg']['slo_violation_rate_pct'].values[0]
                viol_js = sub[sub['strategy'] == 'sweep_llm']['slo_violation_rate_pct'].values[0]
                print(f"    {trace:15s}: {ref[0]:.0f} -> {jsep[0]:.0f} kJ  saving={saving:.1f}%  "
                      f"viol_ref={viol_ref:.1f}%  viol_sweep={viol_js:.1f}%")

    # ----------------------------------------------------------------
    # SWEEP-LLM vs DynamoLLM
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SWEEP-LLM SAVINGS vs DYNAMOLLM (at tau_kv=16)")
    print("=" * 80)
    for slo in SLOS:
        print(f"\n  SLO = {slo} ms:")
        for trace in TRACES:
            sub = df[(df['trace'] == trace) & (df[slo_col] == slo)]
            dynamo = sub[sub['strategy'] == 'dynamollm']['energy_kj'].values
            jsep = sub[sub['strategy'] == 'sweep_llm']['energy_kj'].values
            if len(dynamo) > 0 and len(jsep) > 0:
                saving = (1 - jsep[0] / dynamo[0]) * 100
                print(f"    {trace:15s}: {dynamo[0]:.0f} -> {jsep[0]:.0f} kJ  saving={saving:.1f}%")

    # ----------------------------------------------------------------
    # Summary: mean savings by trace type (conv vs code) and util level
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: MEAN SAVINGS OVER SLO=500,1000 at tau_kv=16")
    print("=" * 80)
    for trace in TRACES:
        sub = df[(df['trace'] == trace) & (df[slo_col].isin([500, 1000]))]
        ref_e = sub[sub['strategy'] == 'static_disagg']['energy_kj'].mean()
        sw_e = sub[sub['strategy'] == 'sweep_llm']['energy_kj'].mean()
        dy_e = sub[sub['strategy'] == 'dynamollm']['energy_kj'].mean()
        gr_e = sub[sub['strategy'] == 'greenllm']['energy_kj'].mean()
        sw_viol = sub[sub['strategy'] == 'sweep_llm']['slo_violation_rate_pct'].mean()
        dy_viol = sub[sub['strategy'] == 'dynamollm']['slo_violation_rate_pct'].mean()
        if ref_e > 0:
            print(f"  {trace:15s}: static={ref_e:.0f}  green={gr_e:.0f} ({(1-gr_e/ref_e)*100:.0f}% off)  "
                  f"dynamo={dy_e:.0f} ({(1-dy_e/ref_e)*100:.0f}% off)  "
                  f"sweep={sw_e:.0f} ({(1-sw_e/ref_e)*100:.0f}% off)  "
                  f"sweep_viol={sw_viol:.1f}%  dynamo_viol={dy_viol:.1f}%")

    # ----------------------------------------------------------------
    # Overhead table update
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DECISION LATENCY (ms) ON PRODUCTION TRACES")
    print("=" * 80)
    print(f"{'Strategy':20s} | {'Mean':>8s} | {'p50':>8s} | {'p99':>8s} | {'Max':>8s}")
    print("-" * 60)
    for s in STRATS:
        sub = df[df['strategy'] == s]
        if sub.empty:
            continue
        m = sub['decision_ms_mean'].mean()
        p50 = sub['decision_ms_p50'].mean()
        p99 = sub['decision_ms_p99'].mean()
        mx = sub['decision_ms_max'].max()
        print(f"{s:20s} | {m:8.1f} | {p50:8.1f} | {p99:8.1f} | {mx:8.1f}")


if __name__ == "__main__":
    main()
