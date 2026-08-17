#!/usr/bin/env python3
"""
Gate-sensitivity pilot.

Compares three common feasibility gates, applied identically to all strategies:
  rho            — utilization stability gate (rho<=1); the deployed default.
  classifier_rho — trained SLO guard-band classifier (is_safe) AND rho<=1.
  latency_rho    — predicted P99 TTFT/TPOT <= SLO AND rho<=1 (diagnostic).

For each gate it runs the six main strategies on the requested traces at a
representative setting (TTFT<=500, TPOT<=200, tau_kv=16 us/tok) and reports, per
(gate, trace): SWEEP energy + modeled-violation rate, the best feasible (0%-viol)
baseline, and whether the headline "SWEEP is the lowest-energy 0%-violation
scheduler" still holds.

Usage:
  python3 gate_sensitivity_pilot.py --traces T1,T2,T3,T4 --duration 300
  python3 gate_sensitivity_pilot.py --azure conv,code --util 0.70 --duration 300
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time as _time
from datetime import datetime, timezone
from dataclasses import replace as dc_replace

from paths import paper_model_dir
from sweep_llm_scheduler import (
    create_sweep_llm_strategies, create_dynamollm_strategy,
    create_dualscale_strategy, create_greenllm_strategy,
    create_static_disagg_strategy, create_hierarchical_disagg_strategy,
)
from jsep_simulator import simulate_trace, summarize_windows
from jsep_traces import (
    generate_T1_prefill_heavy, generate_T2_decode_heavy,
    generate_T3_phase_shift, generate_T4_overload_burst,
)

GATES = ["rho", "classifier_rho", "latency_rho"]
TAU_KV_MS = 0.016  # 16 us/tok


def build_strategies(window_s):
    return {
        "sweep_llm": create_sweep_llm_strategies(window_s=window_s)["sweep_llm"],
        "dynamollm": create_dynamollm_strategy(window_s=window_s),
        "dualscale": create_dualscale_strategy(window_s=window_s),
        "greenllm": create_greenllm_strategy(window_s=window_s),
        "static": create_static_disagg_strategy(window_s=window_s),
        "hierarchical": create_hierarchical_disagg_strategy(window_s=window_s),
    }


def build_traces(names, duration, seed):
    gen = {
        "T1": lambda: generate_T1_prefill_heavy(duration, seed=seed),
        "T2": lambda: generate_T2_decode_heavy(duration, seed=seed),
        "T3": lambda: generate_T3_phase_shift(duration, seed=seed),
        "T4": lambda: generate_T4_overload_burst(duration, seed=seed),
    }
    return {n: gen[n]() for n in names if n in gen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="T1,T2,T3,T4")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--ttft-slo", type=int, default=500)
    ap.add_argument("--tpot-slo", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="output dir for manifest.json + results CSV (paper-grade reproducibility)")
    args = ap.parse_args()

    slo = {"ttft_ms": args.ttft_slo, "tpot_ms": args.tpot_slo}
    trace_names = [t.strip() for t in args.traces.split(",")]
    traces = build_traces(trace_names, args.duration, args.seed)
    strategies = build_strategies(args.window)

    print(f"\nSLO TTFT<={args.ttft_slo} TPOT<={args.tpot_slo} | tau_kv=16us/tok | "
          f"dur={args.duration:.0f}s win={args.window:.0f}s seed={args.seed}", flush=True)

    # --- reproducibility manifest ---
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_rev = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or "not-a-git-repo"
    except Exception:
        git_rev = "unknown"
    manifest = {
        "experiment": "gate_sensitivity_pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": git_rev,
        "command": "python " + " ".join(sys.argv),
        "model_dir_l40s": str(paper_model_dir("models_l40s")),
        "model_dir_l4": str(paper_model_dir("models_l4")),
        "gates": GATES,
        "traces": trace_names,
        "trace_source": "synthetic generators in jsep_traces (generate_T{1..4}_*)",
        "duration_s": args.duration, "window_s": args.window, "seed": args.seed,
        "ttft_slo_ms": args.ttft_slo, "tpot_slo_ms": args.tpot_slo,
        "tau_kv_ms_per_tok": TAU_KV_MS,
        "strategies": list(strategies.keys()),
    }

    # results[gate][trace][strat] = (energy_kj, viol_pct)
    results = {g: {t: {} for t in traces} for g in GATES}
    for gate in GATES:
        for sname, strat in strategies.items():
            strat.cfg = dc_replace(strat.cfg, gate_mode=gate,
                                   kv_transfer_ms_per_token=TAU_KV_MS,
                                   print_search_stats=False)
            for tname, trace in traces.items():
                windows = simulate_trace(trace, strat, sname, slo=slo, window_s=args.window)
                s = summarize_windows(windows, sname, tname, slo)
                results[gate][tname][sname] = (
                    s.get("total_energy_kj", 0.0), s.get("slo_violation_rate", 0.0))
            print(f"  [{gate}] {sname} done", flush=True)

    # Report
    for gate in GATES:
        print(f"\n===== GATE: {gate} =====", flush=True)
        print(f"{'trace':<7}{'SWEEP kJ':>10}{'SWEEP v%':>9} | {'best feasible baseline':>26}{'kJ':>9} | headline", flush=True)
        print("-" * 80, flush=True)
        for tname in traces:
            row = results[gate][tname]
            sweep_e, sweep_v = row["sweep_llm"]
            feasible = {k: e for k, (e, v) in row.items() if v == 0.0}
            # best non-sweep feasible baseline
            base = {k: e for k, e in feasible.items() if k != "sweep_llm"}
            if base:
                bname = min(base, key=base.get); be = base[bname]
            else:
                bname, be = "(none)", float("nan")
            sweep_feasible = sweep_v == 0.0
            sweep_is_min = sweep_feasible and (not base or sweep_e <= min(base.values()) + 1e-6)
            verdict = "HOLDS" if sweep_is_min else ("SWEEP INFEASIBLE" if not sweep_feasible else "baseline lower")
            print(f"{tname:<7}{sweep_e:>10.2f}{sweep_v:>8.1f}% | {bname:>26}{be:>9.2f} | {verdict}", flush=True)

    # --- paper-grade outputs ---
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        csv_path = os.path.join(args.out, "gate_sensitivity_results.csv")
        write_header = not os.path.exists(csv_path)
        # append so T4 then T3 (separate invocations) accumulate into one table
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["gate", "trace", "strategy", "energy_kj", "viol_pct",
                            "ttft_slo_ms", "tpot_slo_ms", "tau_kv_ms_per_tok",
                            "duration_s", "seed", "git_rev"])
            for gate in GATES:
                for tname in traces:
                    for sname, (e, v) in results[gate][tname].items():
                        w.writerow([gate, tname, sname, f"{e:.2f}", f"{v:.1f}",
                                    args.ttft_slo, args.tpot_slo, TAU_KV_MS,
                                    args.duration, args.seed, manifest["git_rev"]])
        print(f"\nwrote {csv_path} and manifest.json", flush=True)


if __name__ == "__main__":
    main()
