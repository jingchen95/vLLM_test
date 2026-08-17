"""
analyze_idle_sensitivity.py — Summarize idle-power sensitivity results.

Reads summary.csv files from four idle settings (zero/low/baseline/high)
and produces a comparison table: energy and saving vs DynamoLLM/Hier-Disagg
for each (trace, idle-setting) pair.

Usage:
    python analyze_idle_sensitivity.py \
        --result-dir results/paper \
        --settings zero,low,baseline,high \
        --traces T1,T2,T3,T4 \
        --slo 500 --tau-kv 16
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

IDLE_W = {
    "zero":     (0, 0),
    "low":      (45, 9),
    "baseline": (90, 18),
    "high":     (150, 30),
}


def load_summary(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", default="results/paper")
    ap.add_argument("--settings", default="zero,low,baseline,high")
    ap.add_argument("--traces", default="T1,T2,T3,T4")
    ap.add_argument("--slo", type=int, default=500)
    ap.add_argument("--tau-kv", type=float, default=16.0)
    ap.add_argument("--out-csv", default="results/paper/analyses/idle_sensitivity.csv")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    settings = args.settings.split(",")
    traces = args.traces.split(",")
    strategies = ["dynamollm", "hierarchical_disagg", "sweep_llm"]

    # Load all data: {setting -> {(trace, strategy) -> energy_j}}
    data: dict[str, dict[tuple, float]] = {}
    for setting in settings:
        summary_path = result_dir / f"idle_sensitivity_{setting}" / "summary.csv"
        rows = load_summary(summary_path)
        data[setting] = {}
        for row in rows:
            if (float(row.get("slo_ttft_ms", 0)) == args.slo and
                    float(row.get("tau_kv_us_per_tok", 0)) == args.tau_kv and
                    row["trace"] in traces and
                    row["strategy"] in strategies):
                key = (row["trace"], row["strategy"])
                data[setting][key] = float(row["total_energy_j"])

    # Build comparison table
    out_rows = []
    print(f"\n{'Setting':>10} {'L40S/L4 W':>10} {'Trace':>5} | "
          f"{'Dynamo':>10} {'Hier-Disagg':>12} {'SWEEP':>10} | "
          f"{'vs Hier%':>9} {'vs Dynamo%':>10}")
    print("-" * 90)

    for setting in settings:
        l40s_w, l4_w = IDLE_W[setting]
        for trace in traces:
            dynamo = data[setting].get((trace, "dynamollm"), float("nan"))
            hier = data[setting].get((trace, "hierarchical_disagg"), float("nan"))
            sweep = data[setting].get((trace, "sweep_llm"), float("nan"))

            vs_hier = 100 * (hier - sweep) / hier if hier > 0 else float("nan")
            vs_dynamo = 100 * (dynamo - sweep) / dynamo if dynamo > 0 else float("nan")

            label = f"L40S={l40s_w}W L4={l4_w}W"
            print(f"{setting:>10} {label:>10} {trace:>5} | "
                  f"{dynamo:>10.0f} {hier:>12.0f} {sweep:>10.0f} | "
                  f"{vs_hier:>8.1f}% {vs_dynamo:>9.1f}%")

            out_rows.append({
                "setting": setting,
                "idle_l40s_w": l40s_w,
                "idle_l4_w": l4_w,
                "trace": trace,
                "slo_ms": args.slo,
                "tau_kv_us": args.tau_kv,
                "dynamo_energy_j": round(dynamo, 1) if dynamo == dynamo else "",
                "hier_disagg_energy_j": round(hier, 1) if hier == hier else "",
                "sweep_energy_j": round(sweep, 1) if sweep == sweep else "",
                "sweep_saving_vs_hier_pct": round(vs_hier, 2) if vs_hier == vs_hier else "",
                "sweep_saving_vs_dynamo_pct": round(vs_dynamo, 2) if vs_dynamo == vs_dynamo else "",
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["setting", "idle_l40s_w", "idle_l4_w", "trace", "slo_ms", "tau_kv_us",
              "dynamo_energy_j", "hier_disagg_energy_j", "sweep_energy_j",
              "sweep_saving_vs_hier_pct", "sweep_saving_vs_dynamo_pct"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
