#!/usr/bin/env python3
"""
Lightweight benchmark harness for SWEEP-LLM scheduler variants.

Runs synthetic traces through:
  * sweep_llm
  * sweep_llm_triggered
  * sweep_llm_emergency
  * sweep_llm_hybrid

and writes a compact CSV/JSON summary for quick iteration.
"""

import argparse
import csv
import json
from pathlib import Path

from paths import figures_results_path, paper_model_dir
from jsep_simulator import simulate_trace, summarize_windows
from jsep_traces import (
    generate_steady,
    generate_bursty,
    generate_diurnal,
    generate_T1_prefill_heavy,
    generate_T2_decode_heavy,
    generate_T3_phase_shift,
    generate_T4_overload_burst,
)
from sweep_llm_scheduler import create_sweep_llm_strategies


STRATEGY_ORDER = (
    "sweep_llm",
    "sweep_llm_triggered",
    "sweep_llm_emergency",
    "sweep_llm_hybrid",
)


def build_traces(duration_s: float, seed: int):
    return {
        "steady": generate_steady(duration_s, rate_rps=10.0, seed=seed),
        "bursty": generate_bursty(
            duration_s,
            base_rate=2.0,
            burst_rate=30.0,
            burst_duration_s=15.0,
            burst_interval_s=60.0,
            seed=seed,
        ),
        "diurnal": generate_diurnal(duration_s, seed=seed),
        "T1": generate_T1_prefill_heavy(duration_s, seed=seed),
        "T2": generate_T2_decode_heavy(duration_s, seed=seed),
        "phase_shift": generate_T3_phase_shift(duration_s, seed=seed),
        "T3": generate_T3_phase_shift(duration_s, seed=seed),
        "T4": generate_T4_overload_burst(duration_s, seed=seed),
    }


def strategy_config(name: str):
    if name == "sweep_llm":
        return {
            "ideal_every_window": True,
            "emergency_bundle_override": False,
        }
    if name == "sweep_llm_triggered":
        return {
            "ideal_every_window": False,
            "emergency_bundle_override": False,
            "ideal_refresh_windows": 12,
            "ideal_load_change_frac": 0.50,
            "ideal_on_class_mix_change": False,
            "ideal_state_change_hold_windows": 2,
        }
    if name == "sweep_llm_emergency":
        return {
            "ideal_every_window": True,
            "emergency_bundle_override": True,
        }
    if name == "sweep_llm_hybrid":
        return {
            "ideal_every_window": False,
            "emergency_bundle_override": True,
            "ideal_refresh_windows": 12,
            "ideal_load_change_frac": 0.50,
            "ideal_on_class_mix_change": False,
            "ideal_state_change_hold_windows": 2,
        }
    raise ValueError(f"Unsupported strategy: {name}")


def run_benchmark(args):
    traces = build_traces(args.duration, args.seed)
    selected_traces = args.traces.split(",") if args.traces else list(traces.keys())
    strategies = create_sweep_llm_strategies(
        model_dir_l40s=args.model_dir_l40s,
        model_dir_l4=args.model_dir_l4,
        window_s=args.window,
    )

    rows = []
    for trace_name in selected_traces:
        trace = traces[trace_name]
        for strategy_name in STRATEGY_ORDER:
            strategy = strategies[strategy_name]
            strategy.cfg = strategy.cfg.__class__(
                print_search_stats=False,
                **strategy_config(strategy_name),
            )
            windows = simulate_trace(
                trace,
                strategy,
                strategy_name,
                slo={"ttft_ms": args.ttft_slo, "tpot_ms": args.tpot_slo},
                window_s=args.window,
            )
            summary = summarize_windows(
                windows,
                strategy_name,
                trace_name,
                {"ttft_ms": args.ttft_slo, "tpot_ms": args.tpot_slo},
            )
            active = [w for w in windows if w.n_requests > 0]
            rows.append({
                "trace": trace_name,
                "strategy": strategy_name,
                "slo_ms": args.ttft_slo,
                "ttft_slo_ms": args.ttft_slo,
                "tpot_slo_ms": args.tpot_slo,
                "duration_s": args.duration,
                "window_s": args.window,
                "active_windows": len(active),
                "slo_violation_rate": summary.get("slo_violation_rate", 0.0),
                "avg_decision_time_s": summary.get("avg_decision_time_s", 0.0),
                "avg_ideal_search_time_s": summary.get("avg_ideal_search_time_s", 0.0),
                "avg_fast_search_time_s": summary.get("avg_fast_search_time_s", 0.0),
                "ideal_skipped_frac": summary.get("ideal_skipped_frac", 0.0),
                "emergency_overrides": sum(
                    1 for w in active if w.search_stats.get("emergency_override")
                ),
            })
    return rows


def write_outputs(rows, output_prefix: Path):
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--slo", type=int, default=None,
                        help="Legacy alias for TTFT SLO in ms.")
    parser.add_argument("--ttft-slo", type=int, default=None)
    parser.add_argument("--tpot-slo", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traces", type=str, default="steady")
    parser.add_argument("--model-dir-l40s", type=str, default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", type=str, default=str(paper_model_dir("models_l4")))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=figures_results_path("sweep_variant_benchmark"),
    )
    args = parser.parse_args()
    if args.ttft_slo is None:
        args.ttft_slo = args.slo if args.slo is not None else 500

    rows = run_benchmark(args)
    if rows:
        write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
