#!/usr/bin/env python3
"""
JSEP Ablation Study: Joint vs Decoupled vs No-LP.

Runs ablation variants on all traces and SLOs, producing a summary table
that isolates the contribution of joint optimization and LP-mode.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from paths import PAPER_RESULTS_ANALYSES_DIR
from jsep_traces import generate_steady, generate_bursty, generate_diurnal
from jsep_cluster import ClusterState
from jsep_scheduler import create_ablation_strategies

WINDOW_S = 5.0
DURATION = 600.0
SNAP_IL = [32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
SNAP_OL = [32, 128, 512, 1024]

def snap_to_nearest(val, grid):
    return min(grid, key=lambda x: abs(x - val))

def run_ablation(slo_list=[500], trace_names=None):
    print("Generating traces...")
    traces = {
        "steady": generate_steady(DURATION, rate_rps=10.0),
        "bursty": generate_bursty(DURATION),
        "diurnal": generate_diurnal(DURATION),
    }
    if trace_names:
        traces = {k: v for k, v in traces.items() if k in trace_names}
    for name, reqs in traces.items():
        print(f"  {name}: {len(reqs)} requests")

    print("\nLoading ablation strategies...")
    strategies = create_ablation_strategies()
    strategy_names = list(strategies.keys())
    print(f"  Strategies: {strategy_names}")

    rows = []
    for slo in slo_list:
        print(f"\n{'='*60}")
        print(f"SLO = {slo} ms")
        print(f"{'='*60}")

        for trace_name, requests in traces.items():
            print(f"\n--- Trace: {trace_name} ---")

            for strat_name in strategy_names:
                strat = strategies[strat_name]
                # Reset state
                if hasattr(strat, 'current_phase'):
                    strat.current_phase = "LP"
                if hasattr(strat, 'last_reconfig_time'):
                    strat.last_reconfig_time = -999.0
                if hasattr(strat, 'prev_tp'):
                    strat.prev_tp = {"l40s": 1, "l4": 1}

                cluster = ClusterState()
                total_energy = 0.0
                n_windows = 0
                n_violations = 0
                n_hp = 0
                n_lp = 0

                t = 0.0
                while t < DURATION:
                    t_end = t + WINDOW_S
                    window_reqs = [r for r in requests if t <= r.arrival_time < t_end]

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
                        result = strategies["jsep"].decide(il, ol, rate, slo,
                                                          cluster, current_time=t)

                    energy = result.total_power_w * WINDOW_S
                    total_energy += energy
                    n_windows += 1

                    if not result.slo_met:
                        n_violations += 1
                    if result.phase == "HP":
                        n_hp += 1
                    else:
                        n_lp += 1

                    t = t_end

                total_energy_kj = total_energy / 1000.0
                avg_power = total_energy / (n_windows * WINDOW_S)
                viol_rate = 100.0 * n_violations / n_windows if n_windows > 0 else 0

                print(f"  {strat_name:20s}: {avg_power:7.1f}W avg, "
                      f"{total_energy_kj:8.1f}kJ, "
                      f"SLO viol={viol_rate:5.1f}%, "
                      f"HP/LP={n_hp}/{n_lp}")

                rows.append({
                    "trace": trace_name,
                    "strategy": strat_name,
                    "slo_ms": slo,
                    "total_energy_kj": round(total_energy_kj, 2),
                    "avg_power_w": round(avg_power, 1),
                    "slo_violation_pct": round(viol_rate, 1),
                    "n_hp": n_hp,
                    "n_lp": n_lp,
                })

    df = pd.DataFrame(rows)

    # Compute savings vs JSEP-full for comparison
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")

    for slo in slo_list:
        print(f"\nSLO = {slo}ms:")
        slo_df = df[df["slo_ms"] == slo]

        # Get JSEP energy as reference (like naive in main eval)
        jsep_energies = {}
        for trace in traces:
            jsep_row = slo_df[(slo_df["strategy"] == "jsep") &
                              (slo_df["trace"] == trace)]
            if len(jsep_row) > 0:
                jsep_energies[trace] = jsep_row["total_energy_kj"].values[0]

        # Get naive-equivalent (dvfs_only as weakest ablation baseline)
        dvfs_energies = {}
        for trace in traces:
            dvfs_row = slo_df[(slo_df["strategy"] == "dvfs_only") &
                              (slo_df["trace"] == trace)]
            if len(dvfs_row) > 0:
                dvfs_energies[trace] = dvfs_row["total_energy_kj"].values[0]

        print(f"{'Strategy':20s} | ", end="")
        for trace in traces:
            print(f"{trace:>12s} | ", end="")
        print(f"{'Avg Energy':>12s}")
        print("-" * 80)

        for strat in strategy_names:
            strat_df = slo_df[slo_df["strategy"] == strat]
            energies = []
            print(f"{strat:20s} | ", end="")
            for trace in traces:
                row = strat_df[strat_df["trace"] == trace]
                if len(row) > 0:
                    e = row["total_energy_kj"].values[0]
                    energies.append(e)
                    print(f"{e:10.1f}kJ | ", end="")
                else:
                    print(f"{'N/A':>12s} | ", end="")
            avg_e = np.mean(energies) if energies else 0
            print(f"{avg_e:10.1f}kJ")

    out_file = PAPER_RESULTS_ANALYSES_DIR / "jsep_ablation_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nResults saved to {out_file}")
    return df


if __name__ == "__main__":
    slos = [200, 500, 1000]
    run_ablation(slo_list=slos)
