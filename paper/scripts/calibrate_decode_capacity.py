#!/usr/bin/env python3
"""Rebuild the decode safe-capacity table from existing Phase-2 profiling.

Why
---
The old decode feasibility signal was `p99_tpot <= SLO`.  That is necessary but
NOT sufficient: TPOT measures inter-token latency *within* a request already in
decode, so a system whose decode queue is backing up can still report a healthy
TPOT while dropping far below the offered load.  Re-anchoring the decode capacity
denominator on p99_tpot therefore produced absurd utilization (rho up to 116 on
L40S / 634 on L4) at points that were in fact saturated.

This script replaces that with a *goodput-based* safe capacity that folds BOTH
conditions into a single number the runtime gate can use directly:

  * stability:   served/offered decode-token ratio >= kappa  (queue not growing)
  * SLO quality: p99_tpot <= S                               (tokens fast enough)

For each config c=(gpu, il, ol, tp, freq) we sweep the profiled request rates and
emit, per TPOT SLO s:

    C_dc_stable(c)    = max over STABLE rates of achieved decode tokens/s
    C_dc_SLO(c, s)    = max over (STABLE and p99_tpot<=s) rates of achieved dc tok/s

The runtime gate then becomes the clean, interpretable form

    rho_dc = D_dc / (eta * C_dc_SLO(c, s)) <= 1,   D_dc = rate * OL,

with eta a conservative margin (applied at runtime, not baked into the table).

Design choices (documented so they are easy to change)
------------------------------------------------------
* We use ACHIEVED decode tokens/s (output_token_throughput_tps), not completed
  requests/s: long-OL requests may not finish inside the benchmark horizon, which
  biases request-level goodput but not generated-token throughput.
* stability uses the token-served ratio r = achieved_dc_tps / (rate*OL) >= kappa.
  A request-level ratio r_req = achieved_rps / rate is emitted as a cross-check.
* C is the MAX achieved throughput over qualifying rates (the demonstrated
  sustainable load).  If the top qualifying rate is the highest rate tested, the
  true capacity may be higher -> the estimate is conservative (fine; we want that).
* Horizon-bias guard: `frontier_confirmed` is True only when at least one tested
  rate ABOVE the chosen capacity point was observed to be unstable (i.e. we saw
  the saturation knee).  If the config saturates at its very lowest tested rate,
  or is a single-rate point, capacity is UNCONFIRMED and the caller should fall
  back conservatively (borrow a lower-freq/TP bound; do NOT extrapolate upward).

No new hardware: reads Phase2_Results_{L4,L40S}/master_results.csv only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]


def load_sweeps(csv_path: Path, gpu: str) -> Dict[Tuple, Dict[float, dict]]:
    """(gpu,il,ol,tp,freq) -> {rate: averaged metrics over runs}."""
    acc: Dict[Tuple, Dict[float, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(csv_path)):
        try:
            cfg = (gpu, int(r["input_len"]), int(r["output_len"]),
                   int(r["tp_degree"]), int(r["gpu_freq_mhz"]))
            rate = float(r["request_rate"])
            acc[cfg][rate]["ach_tps"].append(float(r["output_token_throughput_tps"]))
            acc[cfg][rate]["ach_rps"].append(float(r["request_throughput_rps"]))
            acc[cfg][rate]["tpot"].append(float(r["p99_tpot_ms"]))
        except (KeyError, ValueError):
            continue
    out: Dict[Tuple, Dict[float, dict]] = {}
    for cfg, rates in acc.items():
        out[cfg] = {rate: {m: st.mean(v) for m, v in d.items()}
                    for rate, d in rates.items()}
    return out


def capacity_for_config(sweep: Dict[float, dict], ol: int, kappa: float,
                        slo_ms: float) -> dict:
    """Best sustainable decode throughput under stability (+ optional TPOT SLO).

    Returns achieved-throughput capacity and a frontier_confirmed flag.
    slo_ms=inf -> stability-only capacity (C_dc_stable).
    """
    rates = sorted(sweep)
    qualifying = []   # (rate, achieved_tps)
    any_unstable_rate = []
    for rate in rates:
        d = sweep[rate]
        offered = rate * ol
        r_served = d["ach_tps"] / offered if offered > 0 else 0.0
        stable = r_served >= kappa
        slo_ok = d["tpot"] <= slo_ms
        if stable and slo_ok:
            qualifying.append((rate, d["ach_tps"]))
        if not stable:
            any_unstable_rate.append(rate)
    if not qualifying:
        return {"capacity_tps": 0.0, "cap_rate": 0.0, "n_qual": 0,
                "n_rates": len(rates), "frontier_confirmed": False}
    cap_rate, cap_tps = max(qualifying, key=lambda t: t[1])
    # Confirmed iff we actually observed instability at some rate > cap_rate
    # (we saw the saturation knee), not just ran out of tested rates.
    frontier_confirmed = any(r > cap_rate for r in any_unstable_rate)
    return {"capacity_tps": round(cap_tps, 2), "cap_rate": cap_rate,
            "n_qual": len(qualifying), "n_rates": len(rates),
            "frontier_confirmed": frontier_confirmed}


def build_table(sweeps: Dict[Tuple, Dict[float, dict]], kappa: float,
                slos_tpot: List[float]) -> List[dict]:
    rows = []
    for (gpu, il, ol, tp, freq), sweep in sweeps.items():
        stable = capacity_for_config(sweep, ol, kappa, float("inf"))
        rec_base = {"gpu": gpu, "il": il, "ol": ol, "tp": tp, "freq_mhz": freq,
                    "C_dc_stable_tps": stable["capacity_tps"],
                    "stable_confirmed": stable["frontier_confirmed"],
                    "n_rates": stable["n_rates"]}
        for s in slos_tpot:
            c = capacity_for_config(sweep, ol, kappa, s)
            row = dict(rec_base)
            row["slo_tpot_ms"] = s
            row["C_dc_SLO_tps"] = c["capacity_tps"]
            row["slo_confirmed"] = c["frontier_confirmed"]
            row["n_qual"] = c["n_qual"]
            rows.append(row)
    _certify_freq_monotone(rows)
    return rows


def _certify_freq_monotone(rows: List[dict]) -> None:
    """Emit `freq_monotone_certified` per (gpu,il,ol,tp,slo) group, in place.

    The runtime `lower_freq_bound` fallback borrows `max_{f'<=f, confirmed} C_dc_SLO`
    as a lower bound on C(f). That is only SAFE if C_dc_SLO is non-decreasing in
    frequency (a slower clock never serves MORE than a faster one at fixed shape).
    We certify that offline over the confirmed points rather than letting the
    runtime assume it. Groups with <2 confirmed freqs are certified True vacuously
    (a single confirmed point is its own trivially-safe bound; borrowing still
    requires a confirmed source freq <= target).
    """
    from collections import defaultdict
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["gpu"], r["il"], r["ol"], r["tp"], r["slo_tpot_ms"])].append(r)
    for _, grp in groups.items():
        confirmed = sorted((g for g in grp if g["slo_confirmed"] and g["C_dc_SLO_tps"] > 0),
                           key=lambda g: g["freq_mhz"])
        caps = [g["C_dc_SLO_tps"] for g in confirmed]
        # non-decreasing with a small tolerance for measurement noise (2%)
        monotone = all(caps[i + 1] >= 0.98 * caps[i] for i in range(len(caps) - 1))
        for g in grp:
            g["freq_monotone_certified"] = bool(monotone)


def scale_report(sweeps: Dict[Tuple, Dict[float, dict]], table: List[dict],
                 kappa: float) -> None:
    """Sanity: does rho = offered/C_dc_stable return to an interpretable scale?"""
    cap = {(r["gpu"], r["il"], r["ol"], r["tp"], r["freq_mhz"]): r["C_dc_stable_tps"]
           for r in table}
    for gpu in ("l40s", "l4"):
        rhos = []
        confirmed = unconfirmed = 0
        for (g, il, ol, tp, freq), sweep in sweeps.items():
            if g != gpu:
                continue
            C = cap.get((g, il, ol, tp, freq), 0.0)
            if C <= 0:
                unconfirmed += 1
                continue
            confirmed += 1
            for rate in sweep:
                rhos.append((rate * ol) / C)
        rhos.sort()
        if not rhos:
            continue
        n = len(rhos)
        print(f"  {gpu:5s}: rho=offered/C_dc_stable  median={rhos[n//2]:.2f} "
              f"p90={rhos[int(.9*n)]:.2f} max={rhos[-1]:.1f}  "
              f"| configs with capacity={confirmed}, without={unconfirmed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l4-csv", default=str(ROOT / "Phase2_Results_L4" / "master_results.csv"))
    ap.add_argument("--l40s-csv", default=str(ROOT / "Phase2_Results_L40S" / "master_results.csv"))
    ap.add_argument("--kappa", type=float, default=0.95,
                    help="served/offered decode-token ratio to count a rate as stable")
    ap.add_argument("--slos-tpot", type=float, nargs="+", default=[50, 100, 200, 500])
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "paper" / "models" / "decode_capacity.csv"))
    args = ap.parse_args()

    sweeps = {}
    sweeps.update(load_sweeps(Path(args.l40s_csv), "l40s"))
    sweeps.update(load_sweeps(Path(args.l4_csv), "l4"))

    table = build_table(sweeps, args.kappa, args.slos_tpot)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(table, f, indent=2)

    print(f"Wrote {len(table)} rows (kappa={args.kappa}) to {out}")
    print("Re-normalized decode utilization scale check:")
    scale_report(sweeps, table, args.kappa)
    # Coverage: how many configs have a confirmed vs unconfirmed saturation knee.
    for gpu in ("l40s", "l4"):
        base = [r for r in table if r["gpu"] == gpu and r["slo_tpot_ms"] == args.slos_tpot[-1]]
        conf = sum(r["stable_confirmed"] for r in base)
        withcap = sum(r["C_dc_stable_tps"] > 0 for r in base)
        print(f"  {gpu:5s}: {len(base)} configs | with capacity={withcap} "
              f"| saturation-knee confirmed={conf} (rest are conservative lower bounds)")


if __name__ == "__main__":
    main()
