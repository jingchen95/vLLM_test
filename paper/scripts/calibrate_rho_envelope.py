#!/usr/bin/env python3
"""Calibrate the SLO-conditioned safe-utilization envelope rho_max(gpu, phase, tp, freq, SLO).

Motivation
----------
The deployed scheduler admits a candidate when rho_pf <= 1 and rho_dc <= 1, where
rho is the scheduler's model-predicted phase utilization (demand / capacity).  A
fixed bound of 1 is SLO-independent: tight and loose SLOs select the same config.

This script replaces the fixed 1 with an *empirical* bound rho_max(., SLO): the
largest predicted utilization at which the *measured* P99 still meets the SLO.  It
is derived entirely from the existing Phase-2 profiling (no new hardware runs) and
is expressed in the scheduler's OWN rho units, so it drops straight into the gate:

    admit iff rho_pf <= rho_max_pf(g, tp, f, SLO_TTFT)  and
              rho_dc <= rho_max_dc(g, tp, f, SLO_TPOT).

Because the model's rho is used (not a re-derived one), the envelope absorbs
capacity-model error: it tells the scheduler "at your predicted rho=X, real P99
exceeds the SLO," which is exactly the guarantee we want on hardware.

Design choices (documented so they are easy to change)
------------------------------------------------------
* Aggregation is conservative and workload-agnostic: for each (gpu, phase, tp, f, SLO)
  we take the largest rho0 such that EVERY profiled point with rho <= rho0 met the
  SLO (a monotone safe frontier across il/ol).  This yields one bound per
  (gpu, phase, tp, f, SLO), matching rho_max(g, TP, f, SLO) in the paper.
  Set --per-shape to instead emit a separate bound per (il-bucket, ol-bucket).
* A guard margin k (default 1.0) can shrink the bound: rho_max <- k * rho_max.

Usage
-----
    python calibrate_rho_envelope.py \
        --l4-csv  ../../Phase2_Results_L4/master_results.csv \
        --l40s-csv ../../Phase2_Results_L40S/master_results.csv \
        --slos-ttft 200 500 1000 --slos-tpot 100 200 400 \
        --out ../../artifacts/paper/models/rho_envelope.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# The scheduler's capacity model lives in scheduler.py; we query its predicted rho
# so the envelope is expressed in the same units the runtime gate uses.
from scheduler import EnergyScheduler  # noqa: E402
from paths import paper_model_dir  # noqa: E402

POOL_BY_GPU = {"l4": "l4", "l40s": "l40s"}


def load_rows(csv_path: Path, gpu: str) -> List[dict]:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "gpu": gpu,
                    "tp": int(r["tp_degree"]),
                    "freq": int(r["gpu_freq_mhz"]),
                    "il": int(r["input_len"]),
                    "ol": int(r["output_len"]),
                    "rate": float(r["request_rate"]),
                    "p99_ttft": float(r["p99_ttft_ms"]),
                    "p99_tpot": float(r["p99_tpot_ms"]),
                })
            except (KeyError, ValueError):
                continue  # skip malformed / failed runs
    return rows


_RHO_CACHE: Dict[tuple, float] = {}


def predicted_rho(sched: EnergyScheduler, gpu: str, row: dict, phase: str) -> float:
    """Scheduler's model-predicted utilization for this config (runtime-consistent).

    Memoized on (gpu, il, ol, tp, freq, rate, phase): the profiling repeats each
    config 3x and shares configs across rows, so caching collapses ~7200 calls.
    """
    key = (gpu, row["il"], row["ol"], row["tp"], row["freq"], row["rate"], phase)
    if key not in _RHO_CACHE:
        # A large SLO is passed so predict_config does not gate; we only want rho.
        pred = sched.predict_config(
            il=row["il"], ol=row["ol"], tp=row["tp"], freq=row["freq"],
            rate=row["rate"], slo=10 ** 9, phase=phase,
        )
        _RHO_CACHE[key] = float(pred.get("rho", 0.0))
    return _RHO_CACHE[key]


def monotone_safe_frontier(points: List[Tuple[float, float]], slo: float) -> float:
    """Largest rho0 such that every point with rho <= rho0 has P99 <= slo.

    points: list of (rho, p99).  Returns 0.0 if even the lowest-rho point violates.
    """
    if not points:
        return 0.0
    pts = sorted(points, key=lambda t: t[0])
    frontier = 0.0
    for rho, p99 in pts:
        if p99 <= slo:
            frontier = rho
        else:
            break  # first violation ends the monotone-safe region
    return frontier


def calibrate(rows_by_gpu: Dict[str, List[dict]],
              slos_ttft: List[int], slos_tpot: List[int],
              guard_k: float, per_shape: bool) -> List[dict]:
    schedulers = {g: EnergyScheduler(model_dir=paper_model_dir(f"models_{g}"))
                  for g in rows_by_gpu}

    # key -> phase -> list of (rho, p99)
    def shape_bucket(il_or_ol: int) -> str:
        if not per_shape:
            return "all"
        return "long" if il_or_ol >= 512 else "short"

    buckets: Dict[Tuple, Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: {"prefill": [], "decode": []})

    for gpu, rows in rows_by_gpu.items():
        sched = schedulers[gpu]
        for row in rows:
            rho_pf = predicted_rho(sched, gpu, row, "prefill")
            rho_dc = predicted_rho(sched, gpu, row, "decode")
            k_pf = (gpu, row["tp"], row["freq"], shape_bucket(row["il"]))
            k_dc = (gpu, row["tp"], row["freq"], shape_bucket(row["ol"]))
            buckets[k_pf]["prefill"].append((rho_pf, row["p99_ttft"]))
            buckets[k_dc]["decode"].append((rho_dc, row["p99_tpot"]))

    out = []
    for key, phases in buckets.items():
        gpu, tp, freq, shape = key
        for slo in slos_ttft:
            out.append({
                "gpu": gpu, "phase": "prefill", "tp": tp, "freq_mhz": freq,
                "shape": shape, "slo_ms": slo,
                "rho_max": round(guard_k * monotone_safe_frontier(phases["prefill"], slo), 4),
                "n_points": len(phases["prefill"]),
            })
        for slo in slos_tpot:
            out.append({
                "gpu": gpu, "phase": "decode", "tp": tp, "freq_mhz": freq,
                "shape": shape, "slo_ms": slo,
                "rho_max": round(guard_k * monotone_safe_frontier(phases["decode"], slo), 4),
                "n_points": len(phases["decode"]),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l4-csv", required=True)
    ap.add_argument("--l40s-csv", required=True)
    ap.add_argument("--slos-ttft", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--slos-tpot", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--guard-k", type=float, default=1.0,
                    help="shrink factor on rho_max (e.g. 0.9 for a safety margin)")
    ap.add_argument("--per-shape", action="store_true",
                    help="emit separate bounds per short/long shape bucket")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows_by_gpu = {
        "l4": load_rows(Path(args.l4_csv), "l4"),
        "l40s": load_rows(Path(args.l40s_csv), "l40s"),
    }
    table = calibrate(rows_by_gpu, args.slos_ttft, args.slos_tpot,
                      args.guard_k, args.per_shape)

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(table, f, indent=2)

    # Quick sanity summary: how much does the envelope move across SLO?
    print(f"Wrote {len(table)} rows to {out}")
    for gpu in ("l40s", "l4"):
        for phase in ("prefill", "decode"):
            vals = [r["rho_max"] for r in table if r["gpu"] == gpu and r["phase"] == phase]
            if vals:
                print(f"  {gpu:5s} {phase:8s}: rho_max range {min(vals):.2f}--{max(vals):.2f}")


if __name__ == "__main__":
    main()
