"""
JSEP Phase B: Discrete time-window simulation of heterogeneous GPU scheduling.

Replays workload traces through a cluster model (4×L40S + 8×L4),
comparing scheduling strategies on identical traces.

Usage:
    python jsep_simulator.py --slo 500
    python jsep_simulator.py --slo 500 --trace diurnal
    python jsep_simulator.py --slo 200 --slo 500 --slo 1000
    python jsep_simulator.py --slo 500 --disagg          # include disagg strategies
"""
from __future__ import annotations

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass, field

from paths import PAPER_RESULTS_ANALYSES_DIR, paper_model_dir
from jsep_traces import (Request, generate_steady, generate_bursty,
                         generate_diurnal, snap_to_nearest, IL_VALUES, OL_VALUES)
from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import (create_all_strategies, SchedulingResult,
                            JSEPStrategy)
from jsep_disagg_scheduler import (JSEPDisaggStrategy,
                                    create_disagg_strategies)
from sweep_llm_scheduler import (SweepLLMStrategy,
                                 create_sweep_llm_strategies)


WINDOW_S = 5.0  # scheduling window duration


@dataclass
class WindowMetrics:
    window_start: float
    phase: str
    rate: float
    il_rep: int
    ol_rep: int
    total_power_w: float
    energy_j: float
    slo_met: bool
    n_requests: int
    config: dict = field(default_factory=dict)
    search_stats: dict = field(default_factory=dict)


def _ttft_slo_value(slo) -> int:
    if isinstance(slo, dict):
        return int(slo.get("ttft_ms", slo.get("slo_ms", 500)))
    if isinstance(slo, tuple) and len(slo) == 2:
        return int(slo[0])
    return int(slo)


def _tpot_slo_value(slo) -> int:
    if isinstance(slo, dict):
        return int(slo.get("tpot_ms", 200))
    if isinstance(slo, tuple) and len(slo) == 2:
        return int(slo[1])
    return int(slo)


def simulate_trace(trace: List[Request], strategy, strategy_name: str,
                   slo, window_s: float = WINDOW_S) -> List[WindowMetrics]:
    """Run discrete time-window simulation on a trace."""
    if not trace:
        return []

    cluster = ClusterState()
    windows = []
    trace_end = trace[-1].arrival_time + window_s
    t = 0.0

    # Reset stateful strategies for each trace
    if isinstance(strategy, JSEPStrategy):
        strategy.current_phase = "LP"
        strategy.last_reconfig_time = -999.0
        strategy.prev_tp = {"l40s": 1, "l4": 1}
    if isinstance(strategy, JSEPDisaggStrategy):
        strategy.state_classifier.prev_rate = 0.0
        strategy.state_classifier.current_state = "BALANCED_LP"
        strategy.last_reconfig_time = -999.0
        strategy.prev_tp = {"l40s": 1, "l4": 1}
    if isinstance(strategy, SweepLLMStrategy):
        strategy.reset()

    while t < trace_end:
        # Collect requests in this window
        window_reqs = [r for r in trace if t <= r.arrival_time < t + window_s]

        if len(window_reqs) == 0:
            # No requests: all GPUs idle
            idle_power = cluster.idle_power()
            windows.append(WindowMetrics(
                window_start=t, phase="LP", rate=0.0,
                il_rep=0, ol_rep=0,
                total_power_w=idle_power, energy_j=idle_power * window_s,
                slo_met=True, n_requests=0,
            ))
            t += window_s
            continue

        rate = len(window_reqs) / window_s

        # Representative workload: median il, p75 ol (conservative)
        ils = [r.input_len for r in window_reqs]
        ols = [r.output_len for r in window_reqs]
        il_rep = snap_to_nearest(int(np.median(ils)), IL_VALUES)
        ol_rep = snap_to_nearest(int(np.percentile(ols, 75)), OL_VALUES)

        # Strategy decides
        if isinstance(strategy, SweepLLMStrategy):
            result = strategy.decide_window(
                window_reqs,
                slo,
                cluster,
                current_time=t,
            )
        elif isinstance(strategy, (JSEPStrategy, JSEPDisaggStrategy)):
            result = strategy.decide(il_rep, ol_rep, rate, _ttft_slo_value(slo), cluster,
                                     current_time=t)
        else:
            result = strategy.decide(il_rep, ol_rep, rate, _ttft_slo_value(slo), cluster)

        windows.append(WindowMetrics(
            window_start=t, phase=result.phase, rate=rate,
            il_rep=il_rep, ol_rep=ol_rep,
            total_power_w=result.total_power_w,
            energy_j=result.total_power_w * window_s,
            slo_met=result.slo_met, n_requests=len(window_reqs),
            config=result.config,
            search_stats=result.config.get("search_stats", {}),
        ))

        t += window_s

    return windows


def summarize_windows(windows: List[WindowMetrics], strategy_name: str,
                      trace_name: str, slo,
                      feasible_mask: List[bool] = None) -> dict:
    """Compute summary metrics from window-level results.

    Args:
        feasible_mask: boolean list, True for windows where at least one
                       strategy meets SLO. Used for feasible-window analysis.
    """
    if not windows:
        return {}

    total_energy = sum(w.energy_j for w in windows)
    total_duration = len(windows) * WINDOW_S
    avg_power = total_energy / total_duration if total_duration > 0 else 0

    active_windows = [w for w in windows if w.n_requests > 0]
    n_violations = sum(1 for w in active_windows if not w.slo_met)
    violation_rate = n_violations / len(active_windows) if active_windows else 0

    n_hp = sum(1 for w in windows if w.phase == "HP")
    n_lp = sum(1 for w in windows if w.phase == "LP")
    windows_with_stats = [w for w in windows if w.search_stats]

    avg_decision_s = (
        sum(w.search_stats.get("decision_time_s", 0.0) for w in windows_with_stats) / len(windows_with_stats)
        if windows_with_stats else 0.0
    )
    p95_decision_s = (
        float(np.percentile([w.search_stats.get("decision_time_s", 0.0) for w in windows_with_stats], 95))
        if windows_with_stats else 0.0
    )
    avg_ideal_s = (
        sum(w.search_stats.get("ideal", {}).get("time_s", 0.0) for w in windows_with_stats) / len(windows_with_stats)
        if windows_with_stats else 0.0
    )
    avg_fast_s = (
        sum(w.search_stats.get("fast", {}).get("time_s", 0.0) for w in windows_with_stats) / len(windows_with_stats)
        if windows_with_stats else 0.0
    )
    ideal_skipped_frac = (
        sum(1 for w in windows_with_stats if w.search_stats.get("ideal", {}).get("skipped", False)) / len(windows_with_stats)
        if windows_with_stats else 0.0
    )

    result = {
        "trace": trace_name,
        "strategy": strategy_name,
        "slo_ms": _ttft_slo_value(slo),
        "ttft_slo_ms": _ttft_slo_value(slo),
        "tpot_slo_ms": _tpot_slo_value(slo),
        "total_energy_kj": round(total_energy / 1000, 2),
        "avg_power_w": round(avg_power, 1),
        "duration_s": total_duration,
        "n_windows": len(windows),
        "n_requests": sum(w.n_requests for w in windows),
        "slo_violations": n_violations,
        "slo_violation_rate": round(violation_rate * 100, 1),
        "n_hp_windows": n_hp,
        "n_lp_windows": n_lp,
        "hp_frac_pct": round(n_hp / len(windows) * 100, 1) if windows else 0,
        "avg_decision_time_s": round(avg_decision_s, 4),
        "p95_decision_time_s": round(p95_decision_s, 4),
        "avg_ideal_search_time_s": round(avg_ideal_s, 4),
        "avg_fast_search_time_s": round(avg_fast_s, 4),
        "ideal_skipped_frac": round(ideal_skipped_frac * 100, 1),
    }

    # Feasible-window analysis: violations among windows where SLO is achievable
    if feasible_mask is not None:
        feasible_windows = [w for w, f in zip(windows, feasible_mask)
                           if f and w.n_requests > 0]
        n_feasible = len(feasible_windows)
        n_feasible_violations = sum(1 for w in feasible_windows if not w.slo_met)
        result["n_feasible_windows"] = n_feasible
        result["feasible_violations"] = n_feasible_violations
        result["feasible_violation_rate"] = (
            round(n_feasible_violations / n_feasible * 100, 1)
            if n_feasible > 0 else 0.0
        )

    return result


def run_simulation(slos: List[int], trace_names: List[str] = None,
                   duration_s: float = 600.0, seed: int = 42,
                   model_dir_l40s: str | None = None,
                   model_dir_l4: str | None = None,
                   include_disagg: bool = False):
    """Run full simulation across traces × strategies × SLOs."""
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4

    # Generate traces
    all_traces = {
        "steady": generate_steady(duration_s, rate_rps=10.0, seed=seed),
        "bursty": generate_bursty(duration_s, seed=seed),
        "diurnal": generate_diurnal(duration_s, seed=seed),
    }

    if trace_names:
        all_traces = {k: v for k, v in all_traces.items() if k in trace_names}

    print(f"\nTraces generated:")
    for name, trace in all_traces.items():
        print(f"  {name}: {len(trace)} requests over {duration_s}s")

    # Create strategies
    print(f"\nLoading Phase A models...")
    strategies = create_all_strategies(model_dir_l40s, model_dir_l4)
    strategy_order = ["naive", "dvfs_only", "routing_only", "dynamollm", "jsep"]

    if include_disagg:
        print(f"\nLoading Phase B disaggregated models...")
        disagg_strategies = create_disagg_strategies(model_dir_l40s, model_dir_l4)
        strategies.update(disagg_strategies)
        sweep_strategies = create_sweep_llm_strategies(
            model_dir_l40s,
            model_dir_l4,
            window_s=WINDOW_S,
        )
        strategies.update(sweep_strategies)
        strategy_order += [
            "jsep_disagg",
            "jsep_disagg_nospill",
            "sweep_llm",
            "sweep_llm_triggered",
            "sweep_llm_emergency",
            "sweep_llm_hybrid",
        ]

    # Run simulations
    all_summaries = []
    all_details = []

    for slo in slos:
        print(f"\n{'='*70}")
        print(f"SLO = {slo} ms")
        print(f"{'='*70}")

        for trace_name, trace in all_traces.items():
            print(f"\n--- Trace: {trace_name} ---")

            naive_energy = None

            # First pass: collect per-window results for all strategies
            all_windows = {}
            for strat_name in strategy_order:
                strategy = strategies[strat_name]
                all_windows[strat_name] = simulate_trace(
                    trace, strategy, strat_name, slo)

            # Compute feasible mask: window is feasible if ANY non-naive
            # strategy meets SLO for that window
            n_win = len(all_windows["naive"])
            feasible_mask = [False] * n_win
            for i in range(n_win):
                for sn in strategy_order:
                    if sn == "naive":
                        continue
                    if i < len(all_windows[sn]) and all_windows[sn][i].slo_met:
                        feasible_mask[i] = True
                        break

            for strat_name in strategy_order:
                windows = all_windows[strat_name]
                summary = summarize_windows(windows, strat_name, trace_name,
                                           slo, feasible_mask)

                if strat_name == "naive":
                    naive_energy = summary["total_energy_kj"]

                if naive_energy and naive_energy > 0:
                    saving = (1 - summary["total_energy_kj"] / naive_energy) * 100
                    summary["saving_vs_naive_pct"] = round(saving, 1)
                else:
                    summary["saving_vs_naive_pct"] = 0.0

                all_summaries.append(summary)

                # Collect per-window details
                for w in windows:
                    detail = {
                        "trace": trace_name, "strategy": strat_name,
                        "slo_ms": slo,
                        "window_start": w.window_start, "phase": w.phase,
                        "rate": round(w.rate, 1),
                        "il": w.il_rep, "ol": w.ol_rep,
                        "power_w": round(w.total_power_w, 1),
                        "energy_j": round(w.energy_j, 1),
                        "slo_met": w.slo_met, "n_requests": w.n_requests,
                    }
                    if w.search_stats:
                        detail["decision_time_s"] = w.search_stats.get("decision_time_s", 0.0)
                        detail["search_state"] = w.search_stats.get("state", "")
                        detail["search_burst"] = w.search_stats.get("burst", False)
                        detail["search_num_classes"] = w.search_stats.get("num_classes", 0)
                        detail["ideal_candidate_count"] = w.search_stats.get("ideal", {}).get("candidate_count", 0)
                        detail["ideal_bundle_pair_count"] = w.search_stats.get("ideal", {}).get("bundle_pair_count", 0)
                        detail["ideal_freq_pair_count"] = w.search_stats.get("ideal", {}).get("freq_pair_count", 0)
                        detail["ideal_candidates_visited"] = w.search_stats.get("ideal", {}).get("candidates_visited", 0)
                        detail["ideal_partial_evals"] = w.search_stats.get("ideal", {}).get("partial_route_evals", 0)
                        detail["ideal_complete_evals"] = w.search_stats.get("ideal", {}).get("complete_evals", 0)
                        detail["ideal_time_s"] = w.search_stats.get("ideal", {}).get("time_s", 0.0)
                        detail["ideal_skipped"] = w.search_stats.get("ideal", {}).get("skipped", False)
                        detail["ideal_trigger_reason"] = w.search_stats.get("ideal", {}).get("trigger_reason", "")
                        detail["fast_candidate_count"] = w.search_stats.get("fast", {}).get("candidate_count", 0)
                        detail["fast_bundle_pair_count"] = w.search_stats.get("fast", {}).get("bundle_pair_count", 0)
                        detail["fast_freq_pair_count"] = w.search_stats.get("fast", {}).get("freq_pair_count", 0)
                        detail["fast_candidates_visited"] = w.search_stats.get("fast", {}).get("candidates_visited", 0)
                        detail["fast_partial_evals"] = w.search_stats.get("fast", {}).get("partial_route_evals", 0)
                        detail["fast_complete_evals"] = w.search_stats.get("fast", {}).get("complete_evals", 0)
                        detail["fast_time_s"] = w.search_stats.get("fast", {}).get("time_s", 0.0)
                    # Add per-pool config
                    for gpu_type in ["l40s", "l4"]:
                        if gpu_type in w.config:
                            c = w.config[gpu_type]
                            detail[f"{gpu_type}_tp"] = c.get("tp", 0)
                            detail[f"{gpu_type}_freq"] = c.get("freq_mhz", 0)
                            detail[f"{gpu_type}_active"] = c.get("active_instances", 0)
                            detail[f"{gpu_type}_power"] = c.get("pool_power_w", 0)
                    all_details.append(detail)

                # Print progress
                fvr = summary.get('feasible_violation_rate', 0.0)
                print(f"  {strat_name:15s}: {summary['avg_power_w']:7.1f}W avg, "
                      f"{summary['total_energy_kj']:7.1f}kJ, "
                      f"SLO viol={summary['slo_violation_rate']:4.1f}% "
                      f"(feasible={fvr:4.1f}%), "
                      f"saving={summary['saving_vs_naive_pct']:+5.1f}%, "
                      f"decision={summary['avg_decision_time_s']:.3f}s")

    # Print final summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    df_summary = pd.DataFrame(all_summaries)

    for slo in slos:
        print(f"\nSLO = {slo}ms:")
        print(f"{'Strategy':<16} ", end="")
        for trace_name in all_traces:
            print(f"| {trace_name:>12s} ", end="")
        print(f"| {'Avg Saving':>12s}")
        print("-" * (18 + 16 * (len(all_traces) + 1)))

        for strat in strategy_order:
            print(f"{strat:<16} ", end="")
            savings = []
            for trace_name in all_traces:
                row = df_summary[(df_summary["strategy"] == strat) &
                                 (df_summary["trace"] == trace_name) &
                                 (df_summary["slo_ms"] == slo)]
                if not row.empty:
                    s = row.iloc[0]["saving_vs_naive_pct"]
                    savings.append(s)
                    print(f"| {s:+11.1f}% ", end="")
                else:
                    print(f"| {'N/A':>12s} ", end="")
            avg_s = np.mean(savings) if savings else 0
            print(f"| {avg_s:+11.1f}%")

    # Save results
    df_summary.to_csv(PAPER_RESULTS_ANALYSES_DIR / "jsep_results_summary.csv", index=False)
    df_detail = pd.DataFrame(all_details)
    df_detail.to_csv(PAPER_RESULTS_ANALYSES_DIR / "jsep_results_detail.csv", index=False)
    print(
        f"\nResults saved to "
        f"{PAPER_RESULTS_ANALYSES_DIR / 'jsep_results_summary.csv'} and "
        f"{PAPER_RESULTS_ANALYSES_DIR / 'jsep_results_detail.csv'}"
    )

    return df_summary, df_detail


def main():
    parser = argparse.ArgumentParser(description="JSEP Phase B Simulator")
    parser.add_argument("--slo", type=int, action="append", default=None,
                        help="SLO threshold(s) in ms (can specify multiple)")
    parser.add_argument("--trace", type=str, default=None,
                        choices=["steady", "bursty", "diurnal"],
                        help="Run only one trace (default: all)")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="Trace duration in seconds (default: 600)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir-l40s", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", default=str(paper_model_dir("models_l4")))
    parser.add_argument("--disagg", action="store_true", default=False,
                        help="Include Phase B disaggregated strategies in comparison")
    args = parser.parse_args()

    slos = args.slo or [500]
    trace_names = [args.trace] if args.trace else None

    run_simulation(slos, trace_names, args.duration, args.seed,
                   args.model_dir_l40s, args.model_dir_l4,
                   include_disagg=args.disagg)


if __name__ == "__main__":
    main()
