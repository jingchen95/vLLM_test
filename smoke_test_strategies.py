#!/usr/bin/env python3
"""
smoke_test_strategies.py — Verify the simulator pipeline end-to-end with the
new StaticDisaggStrategy alongside SweepLLMStrategy.

For a slice of windows from a synthetic trace, runs both strategies through
their decide_window() interface and prints per-window:
  - request count and arrival rate
  - StaticDisagg: total power, slo_met, phase
  - SWEEP-LLM:   total power, slo_met, phase (state)

This catches integration issues (model loading, EnergyScheduler API drift,
config wiring, KV-transfer cost) before launching a full sweep.

Usage:
  python smoke_test_strategies.py
  python smoke_test_strategies.py --trace results/paper/synthetic_traces/T2.csv
  python smoke_test_strategies.py --slo 200 --num-windows 12 --tau-kv 0.005
"""

import argparse
import csv
import sys
from pathlib import Path

# Make paper/scripts importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

from jsep_cluster import ClusterState
from jsep_traces import Request
from sweep_llm_scheduler import (
    SweepLLMConfig,
    create_dynamollm_strategy,
    create_greenllm_strategy,
    create_static_disagg_strategy,
    create_sweep_llm_strategies,
)


def load_trace(csv_path):
    requests = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            requests.append(Request(
                arrival_time=float(row["arrival_time_s"]),
                input_len=int(row["input_len"]),
                output_len=int(row["output_len"]),
            ))
    requests.sort(key=lambda r: r.arrival_time)
    return requests


def windowize(requests, window_s):
    if not requests:
        return []
    duration = requests[-1].arrival_time
    n = int(duration / window_s) + 1
    windows = [[] for _ in range(n)]
    for r in requests:
        idx = int(r.arrival_time / window_s)
        if idx < n:
            windows[idx].append(r)
    return windows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="results/paper/synthetic_traces/T1.csv")
    ap.add_argument("--slo", type=int, default=500,
                    help="TTFT SLO in ms (tight=200, moderate=500, loose=1000)")
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--num-windows", type=int, default=8,
                    help="how many windows to run from the start of the trace")
    ap.add_argument("--start-window", type=int, default=0,
                    help="window index to start at (lets you sample mid-ramp)")
    ap.add_argument("--tau-kv", type=float, default=0.075,
                    help="KV transfer cost in ms/token (default 0.075 = 75 µs/tok)")
    ap.add_argument("--models-l40s-dir", default=None,
                    help="L40S model dir (default: paper_model_dir('models_l40s'))")
    ap.add_argument("--models-l4-dir", default=None,
                    help="L4 model dir (default: paper_model_dir('models_l4'))")
    args = ap.parse_args()

    trace_path = (ROOT / args.trace) if not Path(args.trace).is_absolute() else Path(args.trace)
    if not trace_path.exists():
        print(f"ERROR: trace not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading trace: {trace_path}")
    requests = load_trace(trace_path)
    print(f"  {len(requests)} requests, last arrival at t={requests[-1].arrival_time:.1f}s")

    windows = windowize(requests, args.window_s)
    print(f"  {len(windows)} windows of {args.window_s}s")

    end = min(args.start_window + args.num_windows, len(windows))
    print(f"\nBuilding strategies (τ_kv = {args.tau_kv * 1000:.1f} µs/tok) …")

    cfg_static = SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=False,
        print_search_stats=False,
        kv_transfer_ms_per_token=args.tau_kv,
    )
    cfg_sweep = SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=True,
        print_search_stats=False,
        ideal_refresh_windows=12,
        ideal_load_change_frac=0.50,
        ideal_on_class_mix_change=False,
        ideal_state_change_hold_windows=2,
        kv_transfer_ms_per_token=args.tau_kv,
    )

    cfg_green = SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=False,
        print_search_stats=False,
        kv_transfer_ms_per_token=args.tau_kv,
    )
    cfg_dyn = SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=True,
        print_search_stats=False,
        ideal_refresh_windows=12,
        ideal_load_change_frac=0.50,
        ideal_on_class_mix_change=False,
        ideal_state_change_hold_windows=2,
        kv_transfer_ms_per_token=args.tau_kv,
    )

    static = create_static_disagg_strategy(
        model_dir_l40s=args.models_l40s_dir,
        model_dir_l4=args.models_l4_dir,
        config=cfg_static,
        window_s=args.window_s,
    )
    green = create_greenllm_strategy(
        model_dir_l40s=args.models_l40s_dir,
        model_dir_l4=args.models_l4_dir,
        config=cfg_green,
        window_s=args.window_s,
    )
    dyn = create_dynamollm_strategy(
        model_dir_l40s=args.models_l40s_dir,
        model_dir_l4=args.models_l4_dir,
        config=cfg_dyn,
        window_s=args.window_s,
    )
    sweep = create_sweep_llm_strategies(
        model_dir_l40s=args.models_l40s_dir,
        model_dir_l4=args.models_l4_dir,
        config=cfg_sweep,
        window_s=args.window_s,
    )["sweep_llm"]

    static.reset()
    green.reset()
    dyn.reset()
    sweep.reset()

    cluster = ClusterState()

    # Map TTFT SLO to (TTFT, TPOT) pair as defined in paper §4.1.
    slo_pairs = {
        200:  {"ttft_ms": 200,  "tpot_ms": 100},
        500:  {"ttft_ms": 500,  "tpot_ms": 200},
        1000: {"ttft_ms": 1000, "tpot_ms": 400},
    }
    slo_arg = slo_pairs.get(args.slo, args.slo)
    print(f"Running {end - args.start_window} windows at SLO_TTFT = {args.slo} ms (TPOT = {slo_arg.get('tpot_ms', args.slo) if isinstance(slo_arg, dict) else args.slo} ms)")
    print()
    hdr = (f"{'win':>4} {'reqs':>5} {'rate':>7}  | "
           f"{'static':>9} {'ok':>5}  | "
           f"{'green':>9} {'ok':>5}  | "
           f"{'dynamo':>9} {'ok':>5}  | "
           f"{'sweep':>9} {'ok':>5}  {'sweep_phase':>15}")
    print(hdr)
    print("-" * len(hdr))

    for i in range(args.start_window, end):
        win_reqs = windows[i]
        rate = len(win_reqs) / args.window_s
        t_now = i * args.window_s

        rs = static.decide_window(win_reqs, slo_arg, cluster, current_time=t_now)
        rg = green.decide_window(win_reqs, slo_arg, cluster, current_time=t_now)
        rd = dyn.decide_window(win_reqs, slo_arg, cluster, current_time=t_now)
        rw = sweep.decide_window(win_reqs, slo_arg, cluster, current_time=t_now)
        print(f"{i:>4} {len(win_reqs):>5} {rate:>7.2f}  | "
              f"{rs.total_power_w:>9.1f} {str(rs.slo_met):>5}  | "
              f"{rg.total_power_w:>9.1f} {str(rg.slo_met):>5}  | "
              f"{rd.total_power_w:>9.1f} {str(rd.slo_met):>5}  | "
              f"{rw.total_power_w:>9.1f} {str(rw.slo_met):>5}  {rw.phase:>15}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
