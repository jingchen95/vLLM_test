#!/usr/bin/env python3
"""
Generate the controlled synthetic trace suite for SWEEP-LLM evaluation.

Exports each trace as a CSV with one row per request plus a compact manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from paths import synthetic_traces_path
from jsep_traces import (
    CONTROLLED_CLASSES,
    CONTROLLED_T1_MIN_RPS,
    CONTROLLED_T1_PEAK_RPS,
    CONTROLLED_T2_MIN_RPS,
    CONTROLLED_T2_PEAK_RPS,
    CONTROLLED_T3_RATE_RPS,
    CONTROLLED_T3_TRANSITION_S,
    CONTROLLED_T4_BASE_RPS,
    CONTROLLED_T4_SPIKE_RPS,
    CONTROLLED_T4_SPIKE_DURATION_S,
    CONTROLLED_T4_SPIKE_PERIOD_S,
    generate_T1_prefill_heavy,
    generate_T2_decode_heavy,
    generate_T3_phase_shift,
    generate_T4_overload_burst,
)


TRACE_BUILDERS: Dict[str, Callable[[float, int], list]] = {
    "T1": generate_T1_prefill_heavy,
    "T2": generate_T2_decode_heavy,
    "T3": generate_T3_phase_shift,
    "T4": generate_T4_overload_burst,
}


CLASS_BY_LENGTH = {value: key for key, value in CONTROLLED_CLASSES.items()}


def parse_names(raw: str) -> List[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [name for name in names if name not in TRACE_BUILDERS]
    if invalid:
        raise ValueError(f"Unknown trace(s): {', '.join(invalid)}")
    return names


def nominal_rate_rps(trace_name: str, t: float, duration_s: float) -> float:
    if trace_name == "T1":
        phase = 0.5 * (1.0 + math.sin(2.0 * math.pi * t / duration_s - math.pi / 2.0))
        return CONTROLLED_T1_MIN_RPS + (CONTROLLED_T1_PEAK_RPS - CONTROLLED_T1_MIN_RPS) * phase
    if trace_name == "T2":
        phase = 0.5 * (1.0 + math.sin(2.0 * math.pi * t / duration_s - math.pi / 2.0))
        return CONTROLLED_T2_MIN_RPS + (CONTROLLED_T2_PEAK_RPS - CONTROLLED_T2_MIN_RPS) * phase
    if trace_name == "T3":
        return CONTROLLED_T3_RATE_RPS
    if trace_name == "T4":
        return CONTROLLED_T4_SPIKE_RPS if (t % CONTROLLED_T4_SPIKE_PERIOD_S) < CONTROLLED_T4_SPIKE_DURATION_S else CONTROLLED_T4_BASE_RPS
    raise ValueError(f"Unsupported trace: {trace_name}")


def phase_mix_label(trace_name: str, t: float, duration_s: float) -> str:
    if trace_name == "T1":
        return "prefill-heavy"
    if trace_name == "T2":
        return "decode-heavy"
    if trace_name == "T3":
        transition_start = duration_s / 2.0 - CONTROLLED_T3_TRANSITION_S / 2.0
        transition_end = duration_s / 2.0 + CONTROLLED_T3_TRANSITION_S / 2.0
        if t < transition_start:
            return "prefill-heavy"
        if t > transition_end:
            return "decode-heavy"
        return "transition"
    if trace_name == "T4":
        return "balanced"
    raise ValueError(f"Unsupported trace: {trace_name}")


def segment_id(trace_name: str, t: float, duration_s: float) -> str:
    if trace_name == "T3":
        transition_start = duration_s / 2.0 - CONTROLLED_T3_TRANSITION_S / 2.0
        transition_end = duration_s / 2.0 + CONTROLLED_T3_TRANSITION_S / 2.0
        if t < transition_start:
            return "prefill_segment"
        if t > transition_end:
            return "decode_segment"
        return "transition"
    if trace_name == "T4":
        return "spike" if (t % CONTROLLED_T4_SPIKE_PERIOD_S) < CONTROLLED_T4_SPIKE_DURATION_S else "baseline"
    return "full"


def write_trace_csv(path: Path, trace_name: str, requests: Iterable, duration_s: float) -> dict:
    fieldnames = [
        "request_id",
        "arrival_time_s",
        "input_len",
        "output_len",
        "request_class",
        "trace_name",
        "phase_mix_label",
        "nominal_rate_rps",
        "segment_id",
    ]
    requests = list(requests)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for request_id, req in enumerate(requests):
            writer.writerow(
                {
                    "request_id": request_id,
                    "arrival_time_s": round(float(req.arrival_time), 6),
                    "input_len": int(req.input_len),
                    "output_len": int(req.output_len),
                    "request_class": CLASS_BY_LENGTH[(int(req.input_len), int(req.output_len))],
                    "trace_name": trace_name,
                    "phase_mix_label": phase_mix_label(trace_name, float(req.arrival_time), duration_s),
                    "nominal_rate_rps": round(nominal_rate_rps(trace_name, float(req.arrival_time), duration_s), 4),
                    "segment_id": segment_id(trace_name, float(req.arrival_time), duration_s),
                }
            )

    return {
        "trace_name": trace_name,
        "path": path.as_posix(),
        "requests": len(requests),
        "duration_s": float(duration_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="T1,T2,T3,T4")
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(synthetic_traces_path()))
    args = parser.parse_args()

    trace_names = parse_names(args.traces)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "duration_s": float(args.duration),
        "seed": int(args.seed),
        "traces": [],
    }

    for trace_name in trace_names:
        requests = TRACE_BUILDERS[trace_name](args.duration, seed=args.seed)
        trace_path = output_dir / f"{trace_name}.csv"
        manifest["traces"].append(write_trace_csv(trace_path, trace_name, requests, args.duration))
        print(trace_path)

    manifest_path = output_dir / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
