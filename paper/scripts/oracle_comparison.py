#!/usr/bin/env python3
"""
Oracle comparison: exhaustive search vs SWEEP-LLM state-pruned search.

For a small set of representative windows (one per workload state), runs:
  1. SWEEP-LLM normal decision
  2. Exhaustive oracle (same scoring function, full candidate space)

Reports energy gap, SLO match, and decision time.

Usage:
  .venv/bin/python paper/scripts/oracle_comparison.py
  .venv/bin/python paper/scripts/oracle_comparison.py --slo 500 --tau-kv 16
"""

from __future__ import annotations
import argparse
import sys
import os
import time
import math
from itertools import product
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from jsep_cluster import GPU_SPECS, ClusterState              # type: ignore[attr-defined]
POOL_TO_GPU = {"A": "l40s", "B": "l4"}
from jsep_traces import Request, IL_VALUES, OL_VALUES           # type: ignore[attr-defined]
from scheduler import EnergyScheduler                            # type: ignore[attr-defined]
from sweep_llm_scheduler import (                                # type: ignore[attr-defined]
    ALL_ROUTES, ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB,
    PoolBundle, SearchCandidate, SearchResult,
    SweepLLMConfig, SweepLLMStrategy,
    WindowSummary, RequestClassSummary,
    snap_to_nearest,
)
from paths import paper_model_dir                                # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Load models (once)
# ---------------------------------------------------------------------------

def load_schedulers(models_l40s_dir=None, models_l4_dir=None):
    dir_l40s = models_l40s_dir or str(paper_model_dir("models_l40s"))
    dir_l4   = models_l4_dir   or str(paper_model_dir("models_l4"))
    return {
        "l40s": EnergyScheduler(dir_l40s),
        "l4":   EnergyScheduler(dir_l4),
    }


# ---------------------------------------------------------------------------
# Exhaustive oracle
# ---------------------------------------------------------------------------

def _all_bundles(pool_label: str) -> List[PoolBundle]:
    """Return ALL valid PoolBundle combos for the given pool (same logic as
    SweepLLMStrategy._enumerate_pool_bundles, including the off-bundle)."""
    gpu_type = POOL_TO_GPU[pool_label]
    spec = GPU_SPECS[gpu_type]
    bundles = {PoolBundle(tp=1, n_pf=0, n_dc=0)}
    for tp in spec["tp_degrees"]:
        values = list(range(0, spec["total_gpus"] + 1, tp))
        for n_pf in values:
            for n_dc in values:
                if n_pf + n_dc <= spec["total_gpus"]:
                    bundles.add(PoolBundle(tp=tp, n_pf=n_pf, n_dc=n_dc))
    return sorted(bundles, key=lambda b: (b.tp, b.n_pf, b.n_dc))


def _reduced_freqs(pool_label: str) -> List[int]:
    """5 representative frequency levels (min, 25%, 50%, 75%, max)."""
    freqs = GPU_SPECS[POOL_TO_GPU[pool_label]]["frequencies"]
    n = len(freqs)
    idxs = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    return [freqs[i] for i in idxs]


def exhaustive_oracle(strategy: SweepLLMStrategy, summary: WindowSummary,
                      slo: int, use_full_freqs: bool = False) -> Tuple[Optional[SearchResult], float, int]:
    """Run exhaustive search using SWEEP's own scoring function.

    Returns (best_result, elapsed_s, n_candidates_evaluated).
    Uses the SAME _evaluate_candidate method SWEEP uses, so results are
    directly comparable — only the pruning strategy differs.
    """
    bundles_a = _all_bundles("A")
    bundles_b = _all_bundles("B")
    freqs_a = GPU_SPECS["l40s"]["frequencies"] if use_full_freqs else _reduced_freqs("A")
    freqs_b = GPU_SPECS["l4"]["frequencies"]   if use_full_freqs else _reduced_freqs("B")

    # All routes: for simplicity, use one route for all classes (4 options).
    # This captures the dominant energy difference between route strategies.
    all_class_ids = [cls.class_id for cls in summary.classes]
    route_options = [
        {cid: r for cid in all_class_ids} for r in (ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB)
    ]

    best: Optional[SearchResult] = None
    n_eval = 0
    t0 = time.perf_counter()

    for ba, bb, fa, fb in product(bundles_a, bundles_b, freqs_a, freqs_b):
        candidate = SearchCandidate(bundle_a=ba, bundle_b=bb, freq_a=fa, freq_b=fb)
        for routes in route_options:
            n_eval += 1
            result = strategy._evaluate_candidate(
                summary, slo, candidate, routes,
                require_all_classes=True,
                require_safe=True,
            )
            if result is not None:
                if best is None or result.total_energy_j < best.total_energy_j:
                    best = result

    elapsed = time.perf_counter() - t0
    return best, elapsed, n_eval


# ---------------------------------------------------------------------------
# Representative windows
# ---------------------------------------------------------------------------

def _make_window(classes_spec: List[Tuple[str, int, int, float]],
                 total_rate: float, window_s: float = 5.0) -> WindowSummary:
    """Build a WindowSummary from (class_id, il, ol, frac) specs."""
    classes = []
    for cid, il, ol, frac in classes_spec:
        il_s = snap_to_nearest(il, IL_VALUES)
        ol_s = snap_to_nearest(ol, OL_VALUES)
        classes.append(RequestClassSummary(
            class_id=cid,
            fraction=frac,
            request_rate=round(total_rate * frac, 4),
            input_len=il_s,
            output_len=ol_s,
            count=int(total_rate * frac * window_s),
        ))
    total_rate = sum(cls.request_rate for cls in classes)
    return WindowSummary(arrival_rate=total_rate, classes=tuple(classes))


REPRESENTATIVE_WINDOWS = [
    # (name, classes_spec, rate, description)
    # ---- BOTH_LOW (light load, rate < ~14 rps for long_short workloads) ----
    ("BOTH_LOW_pf_bias",
     [("long_short",  1024, 128, 0.6), ("short_short", 32,  128, 0.4)],
     5.0, "BOTH_LOW: prefill-biased, light load"),
    ("BOTH_LOW_dc_bias",
     [("short_long",  32,  512, 0.6), ("short_short", 32,  128, 0.4)],
     5.0, "BOTH_LOW: decode-biased, light load"),
    ("BOTH_LOW_balanced",
     [("short_short", 32,  128, 0.25), ("short_long", 32,  512, 0.25),
      ("long_short",  1024, 128, 0.25), ("long_long", 1024, 512, 0.25)],
     6.0, "BOTH_LOW: balanced, light load"),
    # ---- PREFILL_HEAVY (rate >= ~14 rps with long inputs) ----
    # For long_short (IL=1024, OL=128): BOTH_LOW exits at ~13.9 rps
    ("PREFILL_HEAVY",
     [("long_short",  1024, 128, 0.7), ("short_short", 32,  128, 0.3)],
     20.0, "PREFILL_HEAVY: long inputs dominate (rate=20 rps)"),
    ("PREFILL_HEAVY_hi",
     [("long_short",  1024, 128, 0.5), ("long_long",  1024, 512, 0.5)],
     25.0, "PREFILL_HEAVY: all-long inputs (rate=25 rps)"),
    # ---- DECODE_HEAVY (rate >= ~22 rps for short_long workloads) ----
    # For short_long (IL=32, OL=512): BOTH_LOW exits at ~22.2 rps
    ("DECODE_HEAVY",
     [("short_long",  32,  512, 0.7), ("short_short", 32,  128, 0.3)],
     30.0, "DECODE_HEAVY: long outputs dominate (rate=30 rps)"),
    # ---- BOTH_HEAVY (balanced at high load) ----
    ("BOTH_HEAVY",
     [("short_short", 32,  128, 0.25), ("short_long", 32,  512, 0.25),
      ("long_short",  1024, 128, 0.25), ("long_long", 1024, 512, 0.25)],
     30.0, "BOTH_HEAVY: balanced, high load (rate=30 rps)"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slo",    type=int,   default=500,  help="TTFT SLO in ms")
    ap.add_argument("--tau-kv", type=float, default=16.0, help="KV transfer cost µs/tok")
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--full-freqs", action="store_true",
                    help="Use all 10 freq levels per pool (slower, more thorough)")
    ap.add_argument("--out-csv", default="results/paper/analyses/oracle_comparison.csv")
    args = ap.parse_args()

    print(f"Loading models ...")
    schedulers = load_schedulers()
    # tau_kv in SweepLLMConfig is kv_transfer_ms_per_token (ms per token)
    tau_kv_ms = args.tau_kv / 1000.0
    cfg = SweepLLMConfig(kv_transfer_ms_per_token=tau_kv_ms)
    strategy = SweepLLMStrategy(schedulers, config=cfg, window_s=args.window_s)

    cluster = ClusterState()
    slo_pair = (args.slo, args.slo // 2)  # (TTFT, TPOT) — standard pair

    rows = []
    header = ("window_name", "description", "rate_rps", "slo_ms", "tau_kv_us",
              "sweep_energy_j", "sweep_slo_met", "sweep_decision_ms",
              "oracle_energy_j", "oracle_slo_met", "oracle_decision_ms",
              "oracle_n_candidates", "energy_gap_pct",
              "sweep_route", "oracle_route")

    print(f"\nSLO={args.slo}ms  tau_kv={args.tau_kv}µs/tok  "
          f"freqs={'full' if args.full_freqs else 'reduced-5'}\n")
    print(f"{'Window':<24} {'Rate':>6} | {'SWEEP E':>10} {'Oracle E':>10} "
          f"{'Gap%':>7} | {'SwpT':>8} {'OrcT':>8} | {'SwpOK':>6} {'OrcOK':>6}")
    print("-" * 100)

    for name, classes_spec, rate, desc in REPRESENTATIVE_WINDOWS:
        summary = _make_window(classes_spec, rate, args.window_s)

        # --- SWEEP decision ---
        strategy.reset()
        reqs = []
        for cid, il, ol, frac in classes_spec:
            n = int(rate * frac * args.window_s)
            reqs += [Request(arrival_time=j / rate, input_len=il, output_len=ol)
                     for j in range(n)]

        t0 = time.perf_counter()
        sweep_result = strategy.decide_window(reqs, slo_pair, cluster)
        sweep_ms = (time.perf_counter() - t0) * 1000

        sweep_energy = sweep_result.total_power_w * args.window_s if sweep_result.total_power_w else None
        sweep_ok = sweep_result.slo_met

        # Extract sweep route (dominant route across classes)
        sweep_cfg = sweep_result.config or {}
        sweep_routes = sweep_cfg.get("routes", {}) if isinstance(sweep_cfg, dict) else {}
        # Summarize: which routes are used (and their traffic share)
        route_counts = {}
        for r in sweep_routes.values():
            route_counts[r] = route_counts.get(r, 0) + 1
        if route_counts:
            dominant_route = max(route_counts, key=route_counts.get)
        else:
            dominant_route = "?"

        # --- Oracle ---
        oracle_result, oracle_s, n_cand = exhaustive_oracle(
            strategy, summary, args.slo, use_full_freqs=args.full_freqs)
        oracle_ms = oracle_s * 1000

        oracle_energy = oracle_result.total_energy_j if oracle_result else None
        oracle_ok = oracle_result is not None

        # Route summary for oracle
        if oracle_result:
            oracle_route_counts = {}
            for r in oracle_result.routes.values():
                rstr = f"{r[0]}->{r[1]}"
                oracle_route_counts[rstr] = oracle_route_counts.get(rstr, 0) + 1
            oracle_route_str = max(oracle_route_counts, key=oracle_route_counts.get)
        else:
            oracle_route_str = "infeasible"

        # Gap
        if sweep_energy and oracle_energy:
            gap_pct = 100.0 * (sweep_energy - oracle_energy) / oracle_energy
        elif not sweep_ok and not oracle_ok:
            gap_pct = 0.0
        else:
            gap_pct = float("nan")

        se_str = f"{sweep_energy:.1f}" if sweep_energy else "infeasible"
        oe_str = f"{oracle_energy:.1f}" if oracle_energy else "infeasible"
        gap_str = f"{gap_pct:.1f}" if math.isfinite(gap_pct) else "—"

        print(f"{name:<24} {rate:>6.1f} | {se_str:>10} {oe_str:>10} "
              f"{gap_str:>7} | {sweep_ms:>8.1f} {oracle_ms:>8.1f} | "
              f"{'YES' if sweep_ok else 'NO':>6} {'YES' if oracle_ok else 'NO':>6}")

        rows.append({
            "window_name": name,
            "description": desc,
            "rate_rps": rate,
            "slo_ms": args.slo,
            "tau_kv_us": args.tau_kv,
            "sweep_energy_j": round(sweep_energy, 1) if sweep_energy else "",
            "sweep_slo_met": sweep_ok,
            "sweep_decision_ms": round(sweep_ms, 1),
            "oracle_energy_j": round(oracle_energy, 1) if oracle_energy else "",
            "oracle_slo_met": oracle_ok,
            "oracle_decision_ms": round(oracle_ms, 1),
            "oracle_n_candidates": n_cand,
            "energy_gap_pct": round(gap_pct, 2) if math.isfinite(gap_pct) else "",
            "sweep_route": dominant_route,
            "oracle_route": oracle_route_str,
        })

    # Write CSV
    import csv
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {out_path}")

    # Summary stats
    valid_gaps = [r["energy_gap_pct"] for r in rows if isinstance(r["energy_gap_pct"], float)]
    if valid_gaps:
        print(f"\nEnergy gap vs oracle:  mean={sum(valid_gaps)/len(valid_gaps):.1f}%  "
              f"max={max(valid_gaps):.1f}%  n={len(valid_gaps)}")


if __name__ == "__main__":
    main()
