#!/usr/bin/env python3
"""
Observation 2 motivation harness: restricted-policy search over routing,
DVFS, TP, and active GPU count using the existing simulator/model backend.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from time import time
from typing import Dict, List

from paths import figures_results_path, paper_model_dir
from motivation_joint_core import (
    DEFAULT_POLICIES,
    create_backend,
    default_search_space,
    default_reference_config,
    evaluate_policy,
    iso_now,
    make_state_specs,
    reference_config_metadata,
    write_outputs,
    write_replay_trace,
    _selected_config_record,
)

VALID_STATES = ("prefill-heavy", "decode-heavy", "both-low")


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value:
        return tuple()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states",
        type=str,
        default="prefill-heavy,decode-heavy,both-low",
        help="Comma-separated workload states.",
    )
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Arrival rate for prefill-heavy and decode-heavy states.")
    parser.add_argument("--low-rate", type=float, default=2.0,
                        help="Arrival rate for the both-low state.")
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--trace-duration", type=float, default=60.0)
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument("--ttft-slo", type=int, default=500)
    parser.add_argument("--tpot-slo", type=int, default=120)
    parser.add_argument("--short-il", type=int, default=128)
    parser.add_argument("--long-il", type=int, default=1024)
    parser.add_argument("--short-ol", type=int, default=128)
    parser.add_argument("--long-ol", type=int, default=1024)
    parser.add_argument("--freq-l40s", type=str, default="")
    parser.add_argument("--freq-l4", type=str, default="")
    parser.add_argument("--tp-l40s", type=str, default="")
    parser.add_argument("--tp-l4", type=str, default="")
    parser.add_argument("--active-l40s", type=str, default="")
    parser.add_argument("--active-l4", type=str, default="")
    parser.add_argument(
        "--policies",
        type=str,
        default=",".join(policy.name for policy in DEFAULT_POLICIES),
        help="Comma-separated subset of policies to run.",
    )
    parser.add_argument("--model-dir-l40s", type=str, default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", type=str, default=str(paper_model_dir("models_l4")))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=figures_results_path("motivation_joint"),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    state_names = [item.strip() for item in args.states.split(",") if item.strip()]
    invalid_states = [state for state in state_names if state not in VALID_STATES]
    if invalid_states:
        parser.error(
            f"Unknown state(s): {', '.join(invalid_states)}. "
            f"Valid states are: {', '.join(VALID_STATES)}"
        )

    policy_map = {policy.name: policy for policy in DEFAULT_POLICIES}
    selected_policy_names = [item.strip() for item in args.policies.split(",") if item.strip()]
    invalid_policies = [name for name in selected_policy_names if name not in policy_map]
    if invalid_policies:
        parser.error(
            f"Unknown policy/policies: {', '.join(invalid_policies)}. "
            f"Valid policies are: {', '.join(policy_map)}"
        )
    selected_policies = [policy_map[name] for name in selected_policy_names]

    experiment_started_at = iso_now()
    experiment_started_unix_s = time()
    backend = create_backend(args.model_dir_l40s, args.model_dir_l4, args.window)
    search_space = default_search_space(backend)
    reference = default_reference_config()

    if args.freq_l40s:
        search_space = search_space.__class__(
            freq_l40s=_parse_int_list(args.freq_l40s),
            freq_l4=search_space.freq_l4,
            tp_l40s=search_space.tp_l40s,
            tp_l4=search_space.tp_l4,
            active_l40s=search_space.active_l40s,
            active_l4=search_space.active_l4,
        )
    if args.freq_l4:
        search_space = search_space.__class__(
            freq_l40s=search_space.freq_l40s,
            freq_l4=_parse_int_list(args.freq_l4),
            tp_l40s=search_space.tp_l40s,
            tp_l4=search_space.tp_l4,
            active_l40s=search_space.active_l40s,
            active_l4=search_space.active_l4,
        )
    if args.tp_l40s:
        search_space = search_space.__class__(
            freq_l40s=search_space.freq_l40s,
            freq_l4=search_space.freq_l4,
            tp_l40s=_parse_int_list(args.tp_l40s),
            tp_l4=search_space.tp_l4,
            active_l40s=search_space.active_l40s,
            active_l4=search_space.active_l4,
        )
    if args.tp_l4:
        search_space = search_space.__class__(
            freq_l40s=search_space.freq_l40s,
            freq_l4=search_space.freq_l4,
            tp_l40s=search_space.tp_l40s,
            tp_l4=_parse_int_list(args.tp_l4),
            active_l40s=search_space.active_l40s,
            active_l4=search_space.active_l4,
        )
    if args.active_l40s:
        search_space = search_space.__class__(
            freq_l40s=search_space.freq_l40s,
            freq_l4=search_space.freq_l4,
            tp_l40s=search_space.tp_l40s,
            tp_l4=search_space.tp_l4,
            active_l40s=_parse_int_list(args.active_l40s),
            active_l4=search_space.active_l4,
        )
    if args.active_l4:
        search_space = search_space.__class__(
            freq_l40s=search_space.freq_l40s,
            freq_l4=search_space.freq_l4,
            tp_l40s=search_space.tp_l40s,
            tp_l4=search_space.tp_l4,
            active_l40s=search_space.active_l40s,
            active_l4=_parse_int_list(args.active_l4),
        )

    state_specs = make_state_specs(
        rate_rps=args.rate,
        low_rate_rps=args.low_rate,
        window_s=args.window,
        trace_duration_s=args.trace_duration,
        trace_seed=args.trace_seed,
        short_il=args.short_il,
        long_il=args.long_il,
        short_ol=args.short_ol,
        long_ol=args.long_ol,
        states=state_names,
    )

    trace_dir = args.output_prefix.with_name(args.output_prefix.name + "_traces")
    evaluation_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    replay_records: List[Dict[str, object]] = []

    trace_paths = {}
    for state in state_specs:
        trace_paths[state.name] = write_replay_trace(trace_dir, state)
        for policy in selected_policies:
            rows, selected = evaluate_policy(
                strategy=backend,
                state=state,
                policy=policy,
                search_space=search_space,
                ttft_slo_ms=args.ttft_slo,
                tpot_slo_ms=args.tpot_slo,
                reference=reference,
            )
            evaluation_rows.extend(rows)
            selected_rows.append(selected)
            replay_records.append(
                _selected_config_record(
                    state=state,
                    selected_row=selected,
                    trace_path=trace_paths[state.name],
                    ttft_slo_ms=args.ttft_slo,
                    tpot_slo_ms=args.tpot_slo,
                    reference=reference,
                )
            )

    experiment_finished_at = iso_now()
    experiment_finished_unix_s = time()
    meta = {
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "experiment_started_unix_s": round(experiment_started_unix_s, 6),
        "experiment_finished_unix_s": round(experiment_finished_unix_s, 6),
        "states": [state.name for state in state_specs],
        "policies": [policy.name for policy in selected_policies],
        "argv": sys.argv,
        "command": shlex.join(sys.argv),
        "ttft_slo_ms": args.ttft_slo,
        "tpot_slo_ms": args.tpot_slo,
        "rate_rps": args.rate,
        "low_rate_rps": args.low_rate,
        "window_s": args.window,
        "trace_duration_s": args.trace_duration,
        "trace_seed": args.trace_seed,
        "search_space": {
            "freq_l40s": list(search_space.freq_l40s),
            "freq_l4": list(search_space.freq_l4),
            "tp_l40s": list(search_space.tp_l40s),
            "tp_l4": list(search_space.tp_l4),
            "active_l40s": list(search_space.active_l40s),
            "active_l4": list(search_space.active_l4),
        },
        "trace_paths": {key: str(value) for key, value in trace_paths.items()},
        "reference_config": reference_config_metadata(reference),
    }

    paths = write_outputs(
        output_prefix=args.output_prefix,
        experiment_meta=meta,
        evaluation_rows=evaluation_rows,
        selected_rows=selected_rows,
        replay_records=replay_records,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
