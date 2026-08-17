#!/usr/bin/env python3
"""Instrumented replay: capture the decode/prefill capacity LOOKUPS SWEEP issues.

We monkey-patch FeasibilityTables so every lookup is LOGGED with its real arguments
and real coverage resolution, but returns a PERMISSIVE value so the (currently sparse)
gate does not reject-and-distort the search. The goal is the query distribution of a
*functioning* scheduler — i.e. which calibration points it actually needs — not the
degenerate behaviour under today's incomplete table.

Output: artifacts/paper/analyses/scheduler_query_log.csv  (one row per lookup)
Runs the SWEEP search over representative traces; heavy, so window counts are capped.
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
from scheduler import EnergyScheduler
from paths import PAPER_MODELS_DIR, paper_model_dir
from schedulers.sweep import SweepLLMStrategy
from schedulers.common import SweepLLMConfig
from schedulers import feasibility_tables as FT
from jsep_cluster import ClusterState
import jsep_traces as JT

OUT = ROOT / "artifacts" / "paper" / "analyses" / "scheduler_query_log.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

LOG = []  # see header row in main()

# ---- monkey-patch the two lookups: log real, return permissive -------------
_real_decode = FT.FeasibilityTables.decode_safe_capacity
_real_prefill = FT.FeasibilityTables.prefill_rho_max
_CTX = {"trace": "?", "slo": 0, "window": -1}


def _patched_decode(self, gpu, il, ol, tp, freq, slo_tpot_ms):
    lk = _real_decode(self, gpu, il, ol, tp, freq, slo_tpot_ms)  # real resolution
    # Log BOTH the requested TPOT budget (slo_tpot_ms) and the selected bucket
    # (lk.selected_slo_ms) plus the window id, so the analysis can (a) tell why a
    # lookup missed and (b) do window-weighted coverage.
    LOG.append(("decode", _CTX["trace"], _CTX["slo"], _CTX["window"], gpu, tp, freq,
                round(float(il)), round(float(ol)),
                round(float(slo_tpot_ms), 3), lk.selected_slo_ms,
                lk.source, lk.capacity_tps))
    # permissive: never reject on decode so candidate flow stays functional
    return FT.CapacityLookup(1e12, "measured", None, slo_tpot_ms, slo_tpot_ms)


def _patched_prefill(self, gpu, tp, freq, slo_ttft_ms):
    rho = _real_prefill(self, gpu, tp, freq, slo_ttft_ms)  # real (may be None)
    LOG.append(("prefill", _CTX["trace"], _CTX["slo"], _CTX["window"], gpu, tp, freq,
                None, None, round(float(slo_ttft_ms), 3), slo_ttft_ms,
                ("hit" if rho is not None else "miss"),
                rho if rho is not None else -1.0))
    return 1e9  # permissive: never reject on prefill envelope


FT.FeasibilityTables.decode_safe_capacity = _patched_decode
FT.FeasibilityTables.prefill_rho_max = _patched_prefill


def build_traces(dur=150.0, full=False):
    t = {}
    gens = [("T1_prefill_heavy", lambda: JT.generate_T1_prefill_heavy(duration_s=dur)),
            ("T2_decode_heavy", lambda: JT.generate_T2_decode_heavy(duration_s=dur)),
            ("T3_phase_shift", lambda: JT.generate_T3_phase_shift(duration_s=dur)),
            ("T4_overload_burst", lambda: JT.generate_T4_overload_burst(duration_s=dur))]
    if full:
        gens += [("steady", lambda: JT.generate_steady(dur, 8.0)),
                 ("bursty", lambda: JT.generate_bursty(dur)),
                 ("diurnal", lambda: JT.generate_diurnal(dur))]
    for name, fn in gens:
        try:
            t[name] = fn()
        except Exception as e:
            print(f"{name} gen issue:", e, flush=True)
    APF = ROOT / "traces" / "azure_production_full"
    for name, path in (("azure_conv", APF / "azure_conv_10min_arrival.csv"),
                       ("azure_code", APF / "azure_code_10min_arrival.csv")):
        if path.exists():
            try:
                reqs = JT.load_azure_trace(str(path))
                t[name] = [r for r in reqs if r.arrival_time < dur]
            except Exception as e:
                print(f"{name} load issue:", e, flush=True)
    return t


_WINDOW_SEQ = [0]  # global monotonic window id across the whole run


def replay(strategy, trace, slo, window_s=5.0, max_windows=30):
    cluster = ClusterState()
    if hasattr(strategy, "reset"):
        strategy.reset()
    if not trace:
        return 0
    end = max(r.arrival_time for r in trace)
    t = 0.0; nw = 0
    while t < end and nw < max_windows:
        wr = [r for r in trace if t <= r.arrival_time < t + window_s]
        if wr:
            _CTX["window"] = _WINDOW_SEQ[0]
            _WINDOW_SEQ[0] += 1
            strategy.decide_window(wr, slo, cluster, current_time=t)
            nw += 1
        t += window_s
    return nw


def main():
    scheds = {g: EnergyScheduler(model_dir=paper_model_dir(f"models_{g}"))
              for g in ("l40s", "l4")}
    traces = build_traces(full="--all" in sys.argv)
    if "--azure-only" in sys.argv:
        traces = {k: v for k, v in traces.items() if k.startswith("azure")}
    if "--append" in sys.argv and OUT.exists():
        with open(OUT) as f:
            rd = csv.reader(f); next(rd, None)
            for row in rd:
                LOG.append(tuple(None if c == "" else c for c in row))
        print(f"appending to existing {len(LOG)} rows", flush=True)
    print("traces:", {k: len(v) for k, v in traces.items()}, flush=True)
    slos = [200, 500, 1000] if "--multi-slo" in sys.argv else [500]
    max_w = 6 if "--quick" in sys.argv else 30
    cfg = SweepLLMConfig(decode_gate_mode="goodput", prefill_envelope=True)
    header = ["kind", "trace", "slo", "window", "gpu", "tp", "freq", "il", "ol",
              "req_slo", "sel_slo", "source", "capacity"]

    def flush_csv():
        with open(OUT, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(header); wr.writerows(LOG)

    for slo in slos:
        for name, tr in traces.items():
            _CTX["trace"], _CTX["slo"] = name, slo
            strat = SweepLLMStrategy(scheds, cfg)
            t0 = time.time()
            try:
                nw = replay(strat, tr, slo, max_windows=max_w)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  slo={slo} {name}: REPLAY FAILED {e}", flush=True)
                continue
            flush_csv()  # incremental: never lose completed traces
            print(f"  slo={slo} {name:18s}: {nw} windows, {len(LOG)} cum lookups, "
                  f"{time.time()-t0:.0f}s (csv flushed)", flush=True)
    flush_csv()
    print(f"\nWrote {len(LOG)} lookups to {OUT}", flush=True)


if __name__ == "__main__":
    main()
