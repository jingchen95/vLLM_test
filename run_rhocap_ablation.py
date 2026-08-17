#!/usr/bin/env python3
"""
run_rhocap_ablation.py
Check 2: mono_consolidation_rho_cap ablation.

Runs DynamoLLMMono with rho_cap=0.0 (consolidation disabled) vs. default
rho_cap=0.85 for T1-T4, SLO=500ms, tau_kv=16 µs/tok.

Goal: determine whether final decisions are materially sensitive to the
rho-only monolithic consolidation gate.  If energy and violations are
identical (or near-identical), the rho-only path is a dead code path
under the corrected model.  If they diverge, we must explain why.

Output: printed comparison table + results saved to /tmp/rhocap_ablation.csv
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

from jsep_cluster import ClusterState, set_idle_power_w
from jsep_traces import Request, load_azure_trace
from scheduler import EnergyScheduler
from sweep_llm_scheduler import (
    DynamoLLMMono,
    SweepLLMConfig,
)

TRACE_DIR   = ROOT / "results" / "paper" / "synthetic_traces"
MODEL_L40S  = ROOT / "artifacts" / "paper" / "models" / "models_l40s"
MODEL_L4    = ROOT / "artifacts" / "paper" / "models" / "models_l4"
SLO_MS      = 500
TAU_KV_US   = 16.0
WINDOW_S    = 5.0
TRACES      = ["T1", "T2", "T3", "T4"]

SLO_PAIRS = {
    200:  {"ttft_ms": 200,  "tpot_ms": 100},
    500:  {"ttft_ms": 500,  "tpot_ms": 200},
    1000: {"ttft_ms": 1000, "tpot_ms": 400},
}

_DYNAMO_KWARGS = dict(
    ideal_every_window=False,
    emergency_bundle_override=True,
    print_search_stats=False,
    ideal_refresh_windows=12,
    ideal_load_change_frac=0.50,
    ideal_on_class_mix_change=False,
    ideal_state_change_hold_windows=2,
)


def load_trace(path: Path) -> List[Request]:
    requests = []
    with path.open() as f:
        for row in csv.DictReader(f):
            requests.append(Request(
                arrival_time=float(row["arrival_time_s"]),
                input_len=int(row["input_len"]),
                output_len=int(row["output_len"]),
            ))
    requests.sort(key=lambda r: r.arrival_time)
    return requests


def windowize(requests: List[Request], window_s: float) -> List[List[Request]]:
    if not requests:
        return []
    duration = requests[-1].arrival_time
    n = max(int(math.ceil(duration / window_s)), 1)
    windows = [[] for _ in range(n)]
    for r in requests:
        idx = min(int(r.arrival_time / window_s), n - 1)
        windows[idx].append(r)
    return windows


def run_trace(trace_name: str, requests: List[Request],
              schedulers: dict, rho_cap: float) -> dict:
    cfg = SweepLLMConfig(
        kv_transfer_ms_per_token=TAU_KV_US / 1000.0,
        mono_consolidation_rho_cap=rho_cap,
        **_DYNAMO_KWARGS,
    )
    strategy = DynamoLLMMono(schedulers, config=cfg, window_s=WINDOW_S)
    strategy.reset()
    cluster = ClusterState()
    windows = windowize(requests, WINDOW_S)
    slo_arg = SLO_PAIRS[SLO_MS]

    total_energy = 0.0
    slo_met = 0
    t0 = time.perf_counter()

    for i, win_reqs in enumerate(windows):
        t_start_s = i * WINDOW_S
        result = strategy.decide_window(win_reqs, slo_arg, cluster,
                                        current_time=t_start_s)
        power = float(result.total_power_w or 0.0)
        total_energy += power * WINDOW_S
        slo_met += int(bool(result.slo_met))

    elapsed = time.perf_counter() - t0
    n_win = len(windows)
    viol_pct = 100.0 * (1.0 - slo_met / n_win) if n_win else 0.0
    return {
        "trace": trace_name,
        "rho_cap": rho_cap,
        "n_windows": n_win,
        "total_energy_j": round(total_energy, 1),
        "slo_violation_pct": round(viol_pct, 2),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    print("Loading model stack …", flush=True)
    schedulers = {
        "l40s": EnergyScheduler(model_dir=MODEL_L40S),
        "l4":   EnergyScheduler(model_dir=MODEL_L4),
    }
    print(f"  L40S: {len(schedulers['l40s'].FREQUENCIES)} freqs, "
          f"L4: {len(schedulers['l4'].FREQUENCIES)} freqs")

    rows = []
    rho_caps = [0.85, 0.0]  # baseline first, then ablation

    for trace_name in TRACES:
        trace_path = TRACE_DIR / f"{trace_name}.csv"
        if not trace_path.exists():
            print(f"  SKIP {trace_name}: {trace_path} not found")
            continue
        requests = load_trace(trace_path)
        print(f"\n{trace_name}: {len(requests)} requests", flush=True)

        for rho_cap in rho_caps:
            label = "baseline" if rho_cap > 0 else "rho_cap=0 (ablation)"
            print(f"  Running DynamoLLM [{label}] …", end="", flush=True)
            row = run_trace(trace_name, requests, schedulers, rho_cap)
            rows.append(row)
            print(f" E={row['total_energy_j']:.0f}J  viol={row['slo_violation_pct']:.2f}%"
                  f"  ({row['elapsed_s']:.0f}s)", flush=True)

    # Print comparison table
    print("\n" + "=" * 72)
    print(f"{'Trace':>5} {'rho_cap':>8} | {'Energy(J)':>10} {'Viol%':>7} | "
          f"{'Delta E':>10} {'Delta V':>8}")
    print("-" * 72)
    by_trace = {}
    for r in rows:
        by_trace.setdefault(r["trace"], {})[r["rho_cap"]] = r

    for trace_name in TRACES:
        if trace_name not in by_trace:
            continue
        base = by_trace[trace_name].get(0.85)
        abl  = by_trace[trace_name].get(0.0)
        if base:
            print(f"{trace_name:>5} {'0.85':>8} | {base['total_energy_j']:>10.0f} "
                  f"{base['slo_violation_pct']:>7.2f}% | {'—':>10} {'—':>8}")
        if abl and base:
            delta_e = abl['total_energy_j'] - base['total_energy_j']
            delta_v = abl['slo_violation_pct'] - base['slo_violation_pct']
            print(f"{trace_name:>5} {'0.00':>8} | {abl['total_energy_j']:>10.0f} "
                  f"{abl['slo_violation_pct']:>7.2f}% | {delta_e:>+10.0f} {delta_v:>+8.2f}%")
        elif abl:
            print(f"{trace_name:>5} {'0.00':>8} | {abl['total_energy_j']:>10.0f} "
                  f"{abl['slo_violation_pct']:>7.2f}% |")

    # Save CSV
    out_path = Path("/tmp/rhocap_ablation.csv")
    fieldnames = ["trace", "rho_cap", "n_windows", "total_energy_j",
                  "slo_violation_pct", "elapsed_s"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
