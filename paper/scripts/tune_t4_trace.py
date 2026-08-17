#!/usr/bin/env python3
"""
Small offline tuning pass for T4 only.

Searches a compact grid of burst parameters and scores candidates by favoring
balanced-state coverage:
  - high BOTH_HIGH
  - meaningful BOTH_LOW
  - low PREFILL_HEAVY leakage
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_synthetic_trace_states import analyze_requests
from jsep_traces import CONTROLLED_MIX_T4, generate_T4_overload_burst
from paths import paper_model_dir, synthetic_traces_path
from sweep_llm_scheduler import SweepLLMConfig, create_sweep_llm_strategies


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def candidate_score(summary_row: dict) -> float:
    both_low = float(summary_row["frac_both_low"])
    both_high = float(summary_row["frac_both_high"])
    prefill = float(summary_row["frac_prefill_heavy"])
    decode = float(summary_row["frac_decode_heavy"])
    return round(2.0 * both_high + 0.75 * both_low - 1.5 * prefill - 0.5 * decode, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--base-rates", default="5,6,8,10")
    parser.add_argument("--spike-rates", default="80,100")
    parser.add_argument("--spike-durations", default="10,15,20")
    parser.add_argument("--spike-periods", default="60,75")
    parser.add_argument("--model-dir-l40s", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--model-dir-l4", default=str(paper_model_dir("models_l4")))
    parser.add_argument("--output-prefix", default=str(synthetic_traces_path("T4_tuning")))
    args = parser.parse_args()

    base_rates = parse_float_list(args.base_rates)
    spike_rates = parse_float_list(args.spike_rates)
    spike_durations = parse_float_list(args.spike_durations)
    spike_periods = parse_float_list(args.spike_periods)

    strategy = create_sweep_llm_strategies(
        model_dir_l40s=args.model_dir_l40s,
        model_dir_l4=args.model_dir_l4,
        window_s=args.window,
        config=SweepLLMConfig(print_search_stats=False),
    )["sweep_llm"]

    rows = []
    best = None
    for base_rate in base_rates:
        for spike_rate in spike_rates:
            for spike_duration in spike_durations:
                for spike_period in spike_periods:
                    requests = generate_T4_overload_burst(
                        duration_s=args.duration,
                        seed=args.seed,
                        base_rate=base_rate,
                        spike_rate=spike_rate,
                        spike_duration_s=spike_duration,
                        spike_period_s=spike_period,
                        mix=CONTROLLED_MIX_T4,
                    )
                    _, summary_row = analyze_requests(
                        "T4_candidate",
                        requests,
                        strategy,
                        args.window,
                        args.duration,
                    )
                    row = {
                        "base_rate_rps": base_rate,
                        "spike_rate_rps": spike_rate,
                        "spike_duration_s": spike_duration,
                        "spike_period_s": spike_period,
                        "mix_label": "balanced_25_25_25_25",
                        "frac_both_low": summary_row["frac_both_low"],
                        "frac_both_high": summary_row["frac_both_high"],
                        "frac_prefill_heavy": summary_row["frac_prefill_heavy"],
                        "frac_decode_heavy": summary_row["frac_decode_heavy"],
                    }
                    row["score"] = candidate_score(row)
                    rows.append(row)
                    if best is None or row["score"] > best["score"]:
                        best = row

    rows.sort(
        key=lambda row: (
            row["score"],
            row["frac_both_high"],
            row["frac_both_low"],
            -row["frac_prefill_heavy"],
        ),
        reverse=True,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(output_prefix.name + "_candidates.csv")
    txt_path = output_prefix.with_name(output_prefix.name + "_selected.txt")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_rate_rps",
                "spike_rate_rps",
                "spike_duration_s",
                "spike_period_s",
                "mix_label",
                "frac_both_low",
                "frac_both_high",
                "frac_prefill_heavy",
                "frac_decode_heavy",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    txt_path.write_text(
        (
            "Selected T4 variant\n"
            f"base_rate_rps={best['base_rate_rps']}\n"
            f"spike_rate_rps={best['spike_rate_rps']}\n"
            f"spike_duration_s={best['spike_duration_s']}\n"
            f"spike_period_s={best['spike_period_s']}\n"
            f"frac_both_low={best['frac_both_low']}\n"
            f"frac_both_high={best['frac_both_high']}\n"
            f"frac_prefill_heavy={best['frac_prefill_heavy']}\n"
            f"frac_decode_heavy={best['frac_decode_heavy']}\n"
            f"score={best['score']}\n"
        ),
        encoding="utf-8",
    )

    print(csv_path)
    print(txt_path)


if __name__ == "__main__":
    main()
