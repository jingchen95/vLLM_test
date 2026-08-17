#!/usr/bin/env python3
"""First-pass test of the scheduler's proxy assumptions on the existing Phase-2 data.

The scheduler predicts:
  * decode  from OL only  (decode_proxy_il = 32)  -> assumes decode metrics are IL-independent
  * prefill from IL only  (prefill_proxy_ol = 32)  -> assumes prefill metrics are OL-independent

We test both directly: hold everything fixed except the "ignored" dimension and
measure how much the phase metric moves. Decision rule (plan #2): if the metric
varies > ~10% with the ignored dimension, the proxy is invalid and that phase
model must keep the full (IL, OL).

Runs offline on Phase2_Results_{L4,L40S}/master_results.csv (no new hardware).
"""
from __future__ import annotations
import csv, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSVS = {"l4": ROOT / "Phase2_Results_L4" / "master_results.csv",
        "l40s": ROOT / "Phase2_Results_L40S" / "master_results.csv"}


def load_avg(csv_path):
    """Average the repeated runs -> one row per (il,ol,tp,freq,rate)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(csv_path)):
        try:
            key = (int(r["input_len"]), int(r["output_len"]), int(r["tp_degree"]),
                   int(r["gpu_freq_mhz"]), float(r["request_rate"]))
            acc[key]["ttft"].append(float(r["p99_ttft_ms"]))
            acc[key]["tpot"].append(float(r["p99_tpot_ms"]))
        except (KeyError, ValueError):
            continue
    return {k: {m: st.mean(v) for m, v in d.items()} for k, d in acc.items()}


def rel_spread(vals):
    lo, hi = min(vals), max(vals)
    return (hi - lo) / lo if lo > 0 else float("inf")


def analyze(rows, ignored: str):
    """ignored='il' -> test decode(TPOT) vs IL at fixed (OL,tp,f,rate).
       ignored='ol' -> test prefill(TTFT) vs OL at fixed (IL,tp,f,rate)."""
    metric = "tpot" if ignored == "il" else "ttft"
    groups = defaultdict(dict)
    for (il, ol, tp, f, rate), d in rows.items():
        if ignored == "il":
            groups[(ol, tp, f, rate)][il] = d[metric]
        else:
            groups[(il, tp, f, rate)][ol] = d[metric]

    spreads, monotone_up = [], 0
    n_groups = 0
    for key, series in groups.items():
        if len(series) < 2:
            continue
        n_groups += 1
        xs = sorted(series)
        vals = [series[x] for x in xs]
        spreads.append(rel_spread(vals))
        # does the metric increase with the "ignored" dimension? (e.g., IL -> KV size -> TPOT)
        if vals[-1] > vals[0] * 1.10:
            monotone_up += 1
    return {
        "metric": metric, "ignored": ignored, "n_groups": n_groups,
        "median_spread_pct": 100 * st.median(spreads) if spreads else 0.0,
        "p90_spread_pct": 100 * sorted(spreads)[int(0.9 * len(spreads))] if spreads else 0.0,
        "frac_over_10pct": sum(s > 0.10 for s in spreads) / len(spreads) if spreads else 0.0,
        "frac_increasing_with_ignored": monotone_up / n_groups if n_groups else 0.0,
    }


def main():
    print(f"{'gpu':5s} {'test':32s} {'n':>4s} {'med%':>7s} {'p90%':>8s} {'>10%':>6s} {'incr':>6s}")
    print("-" * 74)
    for gpu, path in CSVS.items():
        rows = load_avg(path)
        for ignored, label in [("il", "decode TPOT vs IL (proxy_il=32)"),
                               ("ol", "prefill TTFT vs OL (proxy_ol=32)")]:
            r = analyze(rows, ignored)
            print(f"{gpu:5s} {label:32s} {r['n_groups']:>4d} "
                  f"{r['median_spread_pct']:>6.1f} {r['p90_spread_pct']:>7.1f} "
                  f"{100*r['frac_over_10pct']:>5.0f}% {100*r['frac_increasing_with_ignored']:>5.0f}%")
    print("\nColumns: med%/p90% = within-group relative spread of the phase metric across the")
    print("ignored dimension; >10% = fraction of groups exceeding the 10% proxy-validity bar;")
    print("incr = fraction where the metric rises with the ignored dimension (e.g. KV-size effect).")


if __name__ == "__main__":
    main()
