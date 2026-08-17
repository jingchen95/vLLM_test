#!/usr/bin/env python3
"""
Dual-SLO sweep for SWEEP-LLM variants.

Example:
  python paper/benchmark_sweep_slo_grid.py \
    --traces steady,bursty,phase_shift \
    --duration 60 \
    --ttft-grid 300,500,800 \
    --tpot-grid 100,150,200,300 \
    --strategies sweep_llm_hybrid \
    --output-prefix paper/figures/sweep_slo_grid
"""

import argparse
import csv
import json
from pathlib import Path

from paths import figures_results_path, paper_model_dir
from benchmark_sweep_variants import (
    STRATEGY_ORDER,
    build_traces,
    create_sweep_llm_strategies,
    strategy_config,
)
from jsep_simulator import simulate_trace, summarize_windows


def _parse_int_grid(spec: str):
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_name_list(spec: str, default):
    if not spec:
        return list(default)
    return [x.strip() for x in spec.split(",") if x.strip()]


def run_grid(args):
    traces = build_traces(args.duration, args.seed)
    trace_names = _parse_name_list(args.traces, traces.keys())
    strategy_names = _parse_name_list(args.strategies, STRATEGY_ORDER)
    ttft_grid = _parse_int_grid(args.ttft_grid)
    tpot_grid = _parse_int_grid(args.tpot_grid)

    strategies = create_sweep_llm_strategies(
        model_dir_l40s=args.model_dir_l40s,
        model_dir_l4=args.model_dir_l4,
        window_s=args.window,
    )

    rows = []
    for trace_name in trace_names:
        trace = traces[trace_name]
        for strategy_name in strategy_names:
            strategy = strategies[strategy_name]
            for ttft_slo in ttft_grid:
                for tpot_slo in tpot_grid:
                    strategy.reset()
                    strategy.cfg = strategy.cfg.__class__(
                        print_search_stats=False,
                        **strategy_config(strategy_name),
                    )
                    slo = {"ttft_ms": ttft_slo, "tpot_ms": tpot_slo}
                    windows = simulate_trace(
                        trace,
                        strategy,
                        strategy_name,
                        slo=slo,
                        window_s=args.window,
                    )
                    summary = summarize_windows(
                        windows,
                        strategy_name,
                        trace_name,
                        slo,
                    )
                    active = [w for w in windows if w.n_requests > 0]
                    rows.append({
                        "trace": trace_name,
                        "strategy": strategy_name,
                        "ttft_slo_ms": ttft_slo,
                        "tpot_slo_ms": tpot_slo,
                        "duration_s": args.duration,
                        "window_s": args.window,
                        "active_windows": len(active),
                        "slo_violation_rate": summary.get("slo_violation_rate", 0.0),
                        "avg_decision_time_s": summary.get("avg_decision_time_s", 0.0),
                        "p95_decision_time_s": summary.get("p95_decision_time_s", 0.0),
                        "avg_ideal_search_time_s": summary.get("avg_ideal_search_time_s", 0.0),
                        "avg_fast_search_time_s": summary.get("avg_fast_search_time_s", 0.0),
                        "ideal_skipped_frac": summary.get("ideal_skipped_frac", 0.0),
                        "emergency_overrides": sum(
                            1 for w in active if w.search_stats.get("emergency_override")
                        ),
                        "total_energy_kj": summary.get("total_energy_kj", 0.0),
                        "avg_power_w": summary.get("avg_power_w", 0.0),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traces", type=str, default="steady,bursty,phase_shift")
    parser.add_argument("--strategies", type=str, default="sweep_llm_hybrid")
    parser.add_argument("--ttft-grid", type=str, default="300,500,800")
    parser.add_argument("--tpot-grid", type=str, default="100,150,200,300")
    parser.add_argument("--model-dir-l40s", type=str, default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", type=str, default=str(paper_model_dir("models_l4")))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=figures_results_path("sweep_slo_grid"),
    )
    args = parser.parse_args()

    rows = run_grid(args)
    if rows:
        write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
