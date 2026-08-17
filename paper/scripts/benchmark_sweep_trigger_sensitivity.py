#!/usr/bin/env python3
"""
Trigger-sensitivity sweep for the SWEEP-LLM hybrid policy.

Example:
  python paper/benchmark_sweep_trigger_sensitivity.py \
    --traces steady,phase_shift,bursty \
    --duration 60 \
    --ttft-slo 500 \
    --tpot-slo 200 \
    --refresh-grid 6,12,20 \
    --load-grid 0.25,0.50,0.75 \
    --hold-grid 1,2,3 \
    --output-prefix paper/figures/sweep_trigger_sensitivity
"""

import argparse
import csv
import json
from pathlib import Path

from paths import figures_results_path, paper_model_dir
from benchmark_sweep_variants import build_traces, create_sweep_llm_strategies
from jsep_simulator import simulate_trace, summarize_windows


def _parse_int_grid(spec: str):
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_float_grid(spec: str):
    return [float(x) for x in spec.split(",") if x.strip()]


def _parse_name_list(spec: str, default):
    if not spec:
        return list(default)
    return [x.strip() for x in spec.split(",") if x.strip()]


def run_sweep(args):
    traces = build_traces(args.duration, args.seed)
    trace_names = _parse_name_list(args.traces, traces.keys())
    refresh_grid = _parse_int_grid(args.refresh_grid)
    load_grid = _parse_float_grid(args.load_grid)
    hold_grid = _parse_int_grid(args.hold_grid)

    strategies = create_sweep_llm_strategies(
        model_dir_l40s=args.model_dir_l40s,
        model_dir_l4=args.model_dir_l4,
        window_s=args.window,
    )
    base = strategies["sweep_llm_hybrid"]

    rows = []
    slo = {"ttft_ms": args.ttft_slo, "tpot_ms": args.tpot_slo}
    for trace_name in trace_names:
        trace = traces[trace_name]
        for refresh in refresh_grid:
            for load_frac in load_grid:
                for hold in hold_grid:
                    base.reset()
                    base.cfg = base.cfg.__class__(
                        print_search_stats=False,
                        ideal_every_window=False,
                        emergency_bundle_override=True,
                        ideal_refresh_windows=refresh,
                        ideal_load_change_frac=load_frac,
                        ideal_on_class_mix_change=False,
                        ideal_state_change_hold_windows=hold,
                    )
                    windows = simulate_trace(
                        trace,
                        base,
                        "sweep_llm_hybrid",
                        slo=slo,
                        window_s=args.window,
                    )
                    summary = summarize_windows(
                        windows,
                        "sweep_llm_hybrid",
                        trace_name,
                        slo,
                    )
                    active = [w for w in windows if w.n_requests > 0]
                    rows.append({
                        "trace": trace_name,
                        "ttft_slo_ms": args.ttft_slo,
                        "tpot_slo_ms": args.tpot_slo,
                        "refresh_windows": refresh,
                        "load_change_frac": load_frac,
                        "state_change_hold_windows": hold,
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
    parser.add_argument("--traces", type=str, default="steady,phase_shift,bursty")
    parser.add_argument("--ttft-slo", type=int, default=500)
    parser.add_argument("--tpot-slo", type=int, default=200)
    parser.add_argument("--refresh-grid", type=str, default="6,12,20")
    parser.add_argument("--load-grid", type=str, default="0.25,0.50,0.75")
    parser.add_argument("--hold-grid", type=str, default="1,2,3")
    parser.add_argument("--model-dir-l40s", type=str, default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", type=str, default=str(paper_model_dir("models_l4")))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=figures_results_path("sweep_trigger_sensitivity"),
    )
    args = parser.parse_args()

    rows = run_sweep(args)
    if rows:
        write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
