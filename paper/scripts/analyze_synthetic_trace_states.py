#!/usr/bin/env python3
"""
Analyze window-level state coverage for the controlled synthetic trace suite.

This script reuses the current SWEEP-LLM window summarization and state
classifier logic from sweep_llm_scheduler.py.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from jsep_traces import Request
from paths import paper_model_dir, synthetic_traces_path
from sweep_llm_scheduler import SweepLLMConfig, create_sweep_llm_strategies


STATE_ORDER = ("PREFILL_HEAVY", "DECODE_HEAVY", "BOTH_LOW", "BOTH_HEAVY")


def load_trace_csv(path: Path) -> List[Request]:
    requests: List[Request] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            requests.append(
                Request(
                    arrival_time=float(row["arrival_time_s"]),
                    input_len=int(row["input_len"]),
                    output_len=int(row["output_len"]),
                )
            )
    requests.sort(key=lambda req: req.arrival_time)
    return requests


def discover_trace_paths(trace_dir: Path, traces: Sequence[str] | None) -> List[Path]:
    if traces:
        return [trace_dir / f"{name}.csv" for name in traces]
    return sorted(path for path in trace_dir.glob("T*.csv") if path.is_file())


def intended_role_match(trace_name: str,
                        fractions: Dict[str, float],
                        first_half_dominant: str,
                        second_half_dominant: str) -> bool:
    if trace_name == "T1":
        return fractions["PREFILL_HEAVY"] > max(fractions["DECODE_HEAVY"], fractions["BOTH_LOW"])
    if trace_name == "T2":
        return fractions["DECODE_HEAVY"] > max(fractions["PREFILL_HEAVY"], fractions["BOTH_LOW"])
    if trace_name == "T3":
        return first_half_dominant == "PREFILL_HEAVY" and second_half_dominant == "DECODE_HEAVY"
    if trace_name == "T4":
        return fractions["BOTH_LOW"] > 0.0 and fractions["BOTH_HEAVY"] > 0.0
    return False


def dominant_state(values: Iterable[str]) -> str:
    counts = Counter(values)
    if not counts:
        return "NONE"
    return counts.most_common(1)[0][0]


def analyze_requests(trace_name: str,
                     requests: List[Request],
                     strategy,
                     window_s: float,
                     duration_s: float | None) -> tuple[list[dict], dict]:
    if duration_s is None:
        duration_s = max((req.arrival_time for req in requests), default=0.0)
        duration_s = window_s * max(1, int(duration_s / window_s) + 1)

    strategy.reset()
    rows: List[dict] = []
    n_windows = int(round(duration_s / window_s))

    for window_id in range(n_windows):
        start = window_id * window_s
        end = start + window_s
        window_requests = [req for req in requests if start <= req.arrival_time < end]
        summary = strategy._summarize_requests(window_requests)
        d_pf = sum(cls.request_rate * cls.input_len for cls in summary.classes)
        d_dc = sum(cls.request_rate * cls.output_len for cls in summary.classes)
        x = d_pf / max(strategy.ref_capacities["prefill"], 1.0)
        y = d_dc / max(strategy.ref_capacities["decode"], 1.0)
        state, burst = strategy._classify_window(summary)

        rows.append(
            {
                "trace_name": trace_name,
                "window_id": window_id,
                "window_start_s": round(start, 3),
                "window_end_s": round(end, 3),
                "request_count": len(window_requests),
                "arrival_rate_rps": round(summary.arrival_rate, 4),
                "D_pf": round(d_pf, 4),
                "D_dc": round(d_dc, 4),
                "x": round(x, 6),
                "y": round(y, 6),
                "assigned_state": state,
                "burst": bool(burst),
            }
        )

    states = [row["assigned_state"] for row in rows]
    first_half = [row["assigned_state"] for row in rows if row["window_start_s"] < duration_s / 2.0]
    second_half = [row["assigned_state"] for row in rows if row["window_start_s"] >= duration_s / 2.0]
    fractions = {
        state: sum(1 for row in rows if row["assigned_state"] == state) / max(1, len(rows))
        for state in STATE_ORDER
    }
    summary_row = {
        "trace_name": trace_name,
        "windows": len(rows),
        "frac_prefill_heavy": round(fractions["PREFILL_HEAVY"], 4),
        "frac_decode_heavy": round(fractions["DECODE_HEAVY"], 4),
        "frac_both_low": round(fractions["BOTH_LOW"], 4),
        "frac_both_high": round(fractions["BOTH_HEAVY"], 4),
        "dominant_state": dominant_state(states),
        "first_half_dominant_state": dominant_state(first_half),
        "second_half_dominant_state": dominant_state(second_half),
        "intended_role_match": intended_role_match(
            trace_name,
            fractions,
            dominant_state(first_half),
            dominant_state(second_half),
        ),
    }
    return rows, summary_row


def analyze_trace(path: Path,
                  strategy,
                  window_s: float,
                  duration_s: float | None) -> tuple[list[dict], dict]:
    requests = load_trace_csv(path)
    return analyze_requests(path.stem, requests, strategy, window_s, duration_s)


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(path: Path, summaries: List[dict]) -> None:
    lines = []
    for row in summaries:
        lines.append(
            f"{row['trace_name']}: dominant={row['dominant_state']}, "
            f"prefill={row['frac_prefill_heavy']:.2f}, "
            f"decode={row['frac_decode_heavy']:.2f}, "
            f"both_low={row['frac_both_low']:.2f}, "
            f"both_high={row['frac_both_high']:.2f}, "
            f"match={row['intended_role_match']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=str(synthetic_traces_path()))
    parser.add_argument("--traces", default="T1,T2,T3,T4")
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--model-dir-l40s", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", default=str(paper_model_dir("models_l4")))
    parser.add_argument("--output-prefix", default=str(synthetic_traces_path("state_coverage")))
    args = parser.parse_args()

    trace_names = [item.strip() for item in args.traces.split(",") if item.strip()]
    trace_paths = discover_trace_paths(Path(args.trace_dir), trace_names)

    strategy = create_sweep_llm_strategies(
        model_dir_l40s=args.model_dir_l40s,
        model_dir_l4=args.model_dir_l4,
        window_s=args.window,
        config=SweepLLMConfig(print_search_stats=False),
    )["sweep_llm"]

    window_rows: List[dict] = []
    summary_rows: List[dict] = []
    for path in trace_paths:
        rows, summary_row = analyze_trace(path, strategy, args.window, args.duration)
        window_rows.extend(rows)
        summary_rows.append(summary_row)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    windows_path = output_prefix.with_name(output_prefix.name + "_windows.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    text_path = output_prefix.with_name(output_prefix.name + "_summary.txt")

    write_csv(
        windows_path,
        window_rows,
        [
            "trace_name", "window_id", "window_start_s", "window_end_s",
            "request_count", "arrival_rate_rps", "D_pf", "D_dc", "x", "y",
            "assigned_state", "burst",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "trace_name", "windows",
            "frac_prefill_heavy", "frac_decode_heavy",
            "frac_both_low", "frac_both_high",
            "dominant_state", "first_half_dominant_state",
            "second_half_dominant_state", "intended_role_match",
        ],
    )
    write_text_summary(text_path, summary_rows)

    print(windows_path)
    print(summary_path)
    print(text_path)


if __name__ == "__main__":
    main()
