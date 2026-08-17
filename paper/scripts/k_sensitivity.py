#!/usr/bin/env python3
"""
Request-class granularity (K) sensitivity study.

Compares the deployed 4-class (2x2 short/long) request taxonomy against a finer
9-class (3x3 S/M/L) taxonomy, to show that SWEEP-LLM's energy savings and SLO
behavior are insensitive to K. Run on the length-diverse synthetic traces
(steady/bursty/diurnal use DEFAULT_MIX with il in {512,1024,2048}, so the third
length level is actually exercised; the T1-T4 controlled traces use only two
distinct lengths per axis and would make K=9 trivially equal to K=4).

Usage:
  python3 k_sensitivity.py [--duration 600] [--window 5] [--ttft-slo 500] [--tpot-slo 200]
"""
from __future__ import annotations
import argparse
import time

from paths import paper_model_dir
from schedulers.common import SweepLLMConfig
from sweep_llm_scheduler import create_sweep_llm_strategies
from jsep_simulator import simulate_trace, summarize_windows
from jsep_traces import generate_steady, generate_bursty, generate_diurnal


def build_traces(duration_s, seed):
    return {
        "steady": generate_steady(duration_s, rate_rps=10.0, seed=seed),
        "bursty": generate_bursty(
            duration_s, base_rate=2.0, burst_rate=30.0,
            burst_duration_s=15.0, burst_interval_s=60.0, seed=seed),
        "diurnal": generate_diurnal(duration_s, seed=seed),
    }


def run_one(trace, cfg, slo, window_s):
    strat = create_sweep_llm_strategies(config=cfg, window_s=window_s)["sweep_llm"]
    windows = simulate_trace(trace, strat, "sweep_llm", slo=slo, window_s=window_s)
    summary = summarize_windows(windows, "sweep_llm", "x", slo)
    # count distinct populated request classes across the run, for sanity
    return summary, strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--ttft-slo", type=int, default=500)
    ap.add_argument("--tpot-slo", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--traces", type=str, default="steady,bursty,diurnal",
                    help="comma-separated subset of steady,bursty,diurnal")
    args = ap.parse_args()

    slo = {"ttft_ms": args.ttft_slo, "tpot_ms": args.tpot_slo}
    traces = build_traces(args.duration, args.seed)
    wanted = [t.strip() for t in args.traces.split(",")]
    traces = {k: v for k, v in traces.items() if k in wanted}

    # K=4: deployed 2x2 taxonomy.  K=9: 3-level per axis aligned with the
    # motivation S/M/L levels (il: S<512<=M<1536<=L ; ol: S<512<=M<768<=L).
    schemes = {
        "K=4 (2x2)": SweepLLMConfig(ideal_every_window=True, print_search_stats=False),
        "K=9 (3x3)": SweepLLMConfig(ideal_every_window=True, print_search_stats=False,
                                    theta_il_hi=1536, theta_ol_hi=768),
    }

    print(f"\nSLO: TTFT<={args.ttft_slo}ms TPOT<={args.tpot_slo}ms | "
          f"duration={args.duration:.0f}s window={args.window:.0f}s seed={args.seed}\n")
    header = f"{'trace':<9} {'scheme':<11} {'energy_kJ':>10} {'viol%':>7} {'Δenergy':>9} {'wall_s':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for trace_name, trace in traces.items():
        base_energy = None
        for scheme_name, cfg in schemes.items():
            t0 = time.perf_counter()
            summary, _ = run_one(trace, cfg, slo, args.window)
            wall = time.perf_counter() - t0
            e = summary.get("total_energy_kj", 0.0)
            v = summary.get("slo_violation_rate", 0.0)  # already a percentage
            if base_energy is None:
                base_energy = e
                delta = "—"
            else:
                delta = f"{(e - base_energy) / base_energy * 100:+.2f}%"
            print(f"{trace_name:<9} {scheme_name:<11} {e:>10.2f} {v:>6.1f}% {delta:>9} {wall:>8.1f}",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
