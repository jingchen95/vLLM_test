"""Generate the frozen synthetic evaluation tables from the authoritative sweep.

Single source of truth: results/paper/section42_frozen_main/summary.csv (+ logs/).
Emits, to stdout and to results/paper/section42_frozen_main/derived/:
  1. raw_energy_violation.csv  - all strategies, per trace, representative SLO/tau
  2. table_a_feasible_frontier.csv - SWEEP vs best feasible (0% viol) non-SWEEP
  3. aggregate.csv             - mean energy/violation over all 36 runs per strategy
  4. sweep_route_share.csv     - request-weighted SWEEP route fractions per trace

Run: python paper/scripts/make_frozen_synthetic_tables.py
"""
import csv
import os
from collections import defaultdict

BASE = "results/paper/section42_frozen_main"
SUMMARY = os.path.join(BASE, "summary.csv")
LOGS = os.path.join(BASE, "logs")
OUT = os.path.join(BASE, "derived")
REP_SLO, REP_TAU = "500", "16"          # representative setting
TRACES = ["T1", "T2", "T3", "T4"]
ORDER = ["static_disagg", "greenllm", "dynamollm", "dualscale", "hierarchical_disagg", "sweep_llm"]
DISPLAY = {
    "static_disagg": "Static-Disagg",
    "greenllm": "GreenLLM-OracleDVFS",
    "dynamollm": "DynamoLLM-Mono",
    "dualscale": "DualScale-Ext",
    "hierarchical_disagg": "Hierarchical-Disagg",
    "sweep_llm": "SWEEP-LLM",
}

def kj(r): return float(r["total_energy_j"]) / 1000.0
def viol(r): return float(r["slo_violation_rate_pct"])
def tau_match(r): return r["tau_kv_us_per_tok"] in (REP_TAU, REP_TAU + ".0")

rows = list(csv.DictReader(open(SUMMARY)))
os.makedirs(OUT, exist_ok=True)

def write_csv(name, header, data):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(data)
    return path

# --- 1. Raw energy/violation at representative setting ---
rep = {(r["trace"], r["strategy"]): r for r in rows if r["slo_ttft_ms"] == REP_SLO and tau_match(r)}
raw = []
print(f"=== 1. Raw energy(kJ)/violation(%), SLO={REP_SLO}ms (TPOT=200), tau_kv={REP_TAU} ===")
print(f"{'strategy':<20} " + " ".join(f"{t:>14}" for t in TRACES))
for s in ORDER:
    cells, drow = [], [DISPLAY[s]]
    for t in TRACES:
        r = rep.get((t, s))
        cells.append(f"{kj(r):6.1f}/{viol(r):4.1f}%" if r else "n/a")
        drow += [round(kj(r), 1) if r else "", round(viol(r), 1) if r else ""]
    print(f"{DISPLAY[s]:<20} " + " ".join(f"{c:>14}" for c in cells))
    raw.append(drow)
hdr = ["strategy"] + [f"{t}_{k}" for t in TRACES for k in ("kJ", "viol%")]
write_csv("raw_energy_violation.csv", hdr, raw)

# --- 2. Aggregate mean over all 36 runs/strategy ---
agg = defaultdict(list)
for r in rows:
    agg[r["strategy"]].append((kj(r), viol(r)))
sweep_mean = sum(e for e, v in agg["sweep_llm"]) / len(agg["sweep_llm"])
print("\n=== 2. Aggregate mean over 36 runs/strategy ===")
print(f"{'strategy':<20} {'mean_kJ':>9} {'mean_viol%':>11} {'vs_SWEEP':>9}")
agg_rows = []
for s in ORDER:
    es = [e for e, v in agg[s]]; vs = [v for e, v in agg[s]]
    m = sum(es) / len(es); mv = sum(vs) / len(vs)
    rel = (m - sweep_mean) / m * 100 if s != "sweep_llm" else 0.0
    print(f"{DISPLAY[s]:<20} {m:>9.1f} {mv:>10.1f}% {rel:>+8.0f}%")
    agg_rows.append([DISPLAY[s], round(m, 1), round(mv, 1), round(rel, 1)])
write_csv("aggregate.csv", ["strategy", "mean_kJ", "mean_viol_pct", "pct_vs_sweep"], agg_rows)

# --- 3. Table A: feasible frontier (0% viol) at representative setting ---
print(f"\n=== 3. Table A: feasible frontier (0% modeled viol), SLO={REP_SLO}/tau={REP_TAU} ===")
print(f"{'trace':<5} {'best feasible non-SWEEP':<22} {'base_kJ':>8} {'SWEEP_kJ':>9} {'saving%':>8}")
ta_rows, savings = [], []
for t in TRACES:
    sweep_e = kj(rep[(t, "sweep_llm")])
    feas = [(s, kj(rep[(t, s)])) for s in ORDER
            if s != "sweep_llm" and (t, s) in rep and viol(rep[(t, s)]) == 0.0]
    if feas:
        bs, be = min(feas, key=lambda x: x[1])
        sav = (be - sweep_e) / be * 100
        savings.append(sav)
        print(f"{t:<5} {DISPLAY[bs]:<22} {be:>8.1f} {sweep_e:>9.1f} {sav:>7.1f}")
        ta_rows.append([t, DISPLAY[bs], round(be, 1), round(sweep_e, 1), round(sav, 1)])
write_csv("table_a_feasible_frontier.csv",
          ["trace", "best_feasible_non_sweep", "baseline_kJ", "sweep_kJ", "saving_pct"], ta_rows)
if savings:
    print(f"  mean saving = {sum(savings)/len(savings):.1f}%  range = {min(savings):.1f}-{max(savings):.1f}%")

# --- 4. SWEEP route share (request-weighted) at representative setting ---
print(f"\n=== 4. SWEEP route share (request-weighted), SLO={REP_SLO}/tau={REP_TAU} ===")
print(f"{'trace':<5} {'AA':>7} {'AB':>7} {'BA':>7} {'BB':>7}")
rs_rows = []
for t in TRACES:
    path = os.path.join(LOGS, f"{t}_sweep_llm_slo{REP_SLO}_kv{REP_TAU}.csv")
    acc = defaultdict(float); tot = 0.0
    for w in csv.DictReader(open(path)):
        n = float(w["n_requests"])
        if n <= 0:
            continue
        for k in ("route_aa_frac", "route_ab_frac", "route_ba_frac", "route_bb_frac"):
            v = w.get(k, "")
            acc[k] += (float(v) if v not in ("", None) else 0.0) * n
        tot += n
    sh = {k: (acc[k] / tot if tot else 0.0) for k in acc}
    print(f"{t:<5} {sh['route_aa_frac']:>7.3f} {sh['route_ab_frac']:>7.3f} "
          f"{sh['route_ba_frac']:>7.3f} {sh['route_bb_frac']:>7.3f}")
    rs_rows.append([t, round(sh['route_aa_frac'], 3), round(sh['route_ab_frac'], 3),
                    round(sh['route_ba_frac'], 3), round(sh['route_bb_frac'], 3)])
write_csv("sweep_route_share.csv", ["trace", "AA", "AB", "BA", "BB"], rs_rows)

# --- 5. Table B: DualScale-Ext cadence sensitivity (runner-based) ---
# Cadence variants live in a separate sweep dir; 5-min (= `dualscale`) and the
# SWEEP reference come from the frozen main sweep. All use the runner windowizer.
CAD = "results/paper/dualscale_cadence_rhogate/summary.csv"
if os.path.exists(CAD):
    cad_rows = list(csv.DictReader(open(CAD)))
    def pick(rowset, trace, strat):
        for r in rowset:
            if (r["trace"] == trace and r["strategy"] == strat
                    and r["slo_ttft_ms"] == REP_SLO and tau_match(r)):
                return r
        return None
    CAD_COLS = [("dualscale_20s", "20s", cad_rows), ("dualscale_1min", "1min", cad_rows),
                ("dualscale_2min", "2min", cad_rows), ("dualscale", "5min", rows)]
    print(f"\n=== 5. Table B: DualScale-Ext cadence sensitivity (runner), SLO={REP_SLO}/tau={REP_TAU} ===")
    print(f"{'trace':<5} " + " ".join(f"{lbl:>13}" for _, lbl, _ in CAD_COLS) + f" {'SWEEP':>13}")
    tb_rows = []
    any_feasible_beats = False
    for t in TRACES:
        cells, drow = [], [t]
        for strat, lbl, src in CAD_COLS:
            r = pick(src, t, strat)
            cells.append(f"{kj(r):6.1f}/{viol(r):4.1f}%" if r else "n/a")
            drow += [round(kj(r), 1) if r else "", round(viol(r), 1) if r else ""]
        sw = rep[(t, "sweep_llm")]
        cells.append(f"{kj(sw):6.1f}/{viol(sw):4.1f}%")
        drow += [round(kj(sw), 1), round(viol(sw), 1)]
        for strat, lbl, src in CAD_COLS:
            r = pick(src, t, strat)
            if r and viol(r) == 0.0 and kj(r) < kj(sw):
                any_feasible_beats = True
        print(f"{t:<5} " + " ".join(f"{c:>13}" for c in cells))
        tb_rows.append(drow)
    hdr = ["trace"] + [f"DS_{lbl}_{k}" for _, lbl, _ in CAD_COLS for k in ("kJ", "viol%")] + ["SWEEP_kJ", "SWEEP_viol%"]
    write_csv("table_b_cadence.csv", hdr, tb_rows)
    print(f"  => any DS cadence with 0% viol AND energy < SWEEP: {any_feasible_beats}")
    n_csv = 5
else:
    print(f"\n(Table B skipped: {CAD} not found.)")
    n_csv = 4

# --- 6. Ablation table: each SWEEP knob removed (separate frozen sweep) ---
ABL = "results/paper/section42_frozen_ablation/summary.csv"
ABL_ORDER = ["sweep_llm", "sweep_llm_no_routing", "sweep_llm_no_dvfs",
             "sweep_llm_no_tp", "sweep_llm_no_capacity"]
ABL_DISP = {
    "sweep_llm": "SWEEP-LLM (full)",
    "sweep_llm_no_routing": "-Routing",
    "sweep_llm_no_dvfs": "-DVFS",
    "sweep_llm_no_tp": "-TP",
    "sweep_llm_no_capacity": "-Capacity",
}
if os.path.exists(ABL):
    abl = list(csv.DictReader(open(ABL)))
    aagg = defaultdict(list)
    for r in abl:
        aagg[r["strategy"]].append((kj(r), viol(r)))
    full_mean = sum(e for e, v in aagg["sweep_llm"]) / len(aagg["sweep_llm"])
    print("\n=== 6a. Ablation: mean over 36 runs, normalized to full SWEEP ===")
    print(f"{'config':<18} {'mean_kJ':>9} {'norm':>7} {'mean_viol%':>11}")
    abl_rows = []
    for s in ABL_ORDER:
        es = [e for e, v in aagg[s]]; vs = [v for e, v in aagg[s]]
        m = sum(es) / len(es); mv = sum(vs) / len(vs); norm = m / full_mean
        print(f"{ABL_DISP[s]:<18} {m:>9.1f} {norm:>7.2f} {mv:>10.1f}%")
        abl_rows.append([ABL_DISP[s], round(m, 1), round(norm, 3), round(mv, 1)])
    write_csv("ablation_aggregate.csv", ["config", "mean_kJ", "norm_to_full", "mean_viol_pct"], abl_rows)

    arep = {(r["trace"], r["strategy"]): r for r in abl if r["slo_ttft_ms"] == REP_SLO and tau_match(r)}
    print(f"\n=== 6b. Ablation per-trace (SLO={REP_SLO}/tau={REP_TAU}), kJ/viol%, norm to full ===")
    print(f"{'config':<18} " + " ".join(f"{t:>16}" for t in TRACES))
    abl_pt = []
    for s in ABL_ORDER:
        cells, drow = [], [ABL_DISP[s]]
        for t in TRACES:
            r = arep.get((t, s)); full = arep.get((t, "sweep_llm"))
            if r and full:
                nrm = kj(r) / kj(full)
                cells.append(f"{kj(r):6.1f}/{viol(r):4.1f}%/{nrm:.2f}")
                drow += [round(kj(r), 1), round(viol(r), 1), round(nrm, 3)]
            else:
                cells.append("n/a"); drow += ["", "", ""]
        print(f"{ABL_DISP[s]:<18} " + " ".join(f"{c:>16}" for c in cells))
        abl_pt.append(drow)
    hdr = ["config"] + [f"{t}_{k}" for t in TRACES for k in ("kJ", "viol%", "norm")]
    write_csv("ablation_per_trace.csv", hdr, abl_pt)
    n_csv += 2
else:
    print(f"\n(Ablation table skipped: {ABL} not found.)")

# --- 7. Scheduler overhead: simulator decision time per strategy (tau=16) ---
print(f"\n=== 7. Scheduler overhead (simulator decision time, ms), tau_kv={REP_TAU} ===")
ov = defaultdict(lambda: {"mean": [], "p50": [], "p99": [], "max": []})
for r in rows:  # frozen main summary (6 strategies)
    if not tau_match(r):
        continue
    ov[r["strategy"]]["mean"].append(float(r["decision_ms_mean"]))
    ov[r["strategy"]]["p50"].append(float(r["decision_ms_p50"]))
    ov[r["strategy"]]["p99"].append(float(r["decision_ms_p99"]))
    ov[r["strategy"]]["max"].append(float(r["decision_ms_max"]))
print(f"{'strategy':<20} {'Mean':>8} {'p50':>8} {'p99':>8} {'Max':>8}")
ov_rows = []
for s in ORDER:
    a = ov[s]
    mean = sum(a["mean"]) / len(a["mean"]); p50 = sum(a["p50"]) / len(a["p50"])
    p99 = max(a["p99"]); mx = max(a["max"])  # worst-case across runs
    print(f"{DISPLAY[s]:<20} {mean:>8.1f} {p50:>8.1f} {p99:>8.0f} {mx:>8.0f}")
    ov_rows.append([DISPLAY[s], round(mean, 1), round(p50, 1), round(p99), round(mx)])
write_csv("overhead.csv", ["strategy", "mean_ms", "p50_ms", "p99_ms", "max_ms"], ov_rows)
n_csv += 1

# --- 8. Idle-power sensitivity: energy per strategy across idle settings ---
IDLE = {"zero": "results/paper/idle_frozen_zero/summary.csv",
        "low": "results/paper/idle_frozen_low/summary.csv",
        "baseline": SUMMARY,  # frozen main = default idle (90/18)
        "high": "results/paper/idle_frozen_high/summary.csv"}
if all(os.path.exists(p) for p in IDLE.values()):
    def load_rep(path):
        return {(r["trace"], r["strategy"]): kj(r) for r in csv.DictReader(open(path))
                if r["slo_ttft_ms"] == REP_SLO and tau_match(r)}
    idle = {k: load_rep(p) for k, p in IDLE.items()}
    print(f"\n=== 8. Idle-power sensitivity (kJ), SLO={REP_SLO}/tau={REP_TAU} ===")
    print(f"{'trace':<4} {'strategy':<20} {'zero':>7} {'low':>7} {'base':>7} {'high':>7} {'z->h%':>7}")
    idle_rows = []
    for t in TRACES:
        for s in ORDER:
            z = idle["zero"][(t, s)]; lo = idle["low"][(t, s)]
            b = idle["baseline"][(t, s)]; h = idle["high"][(t, s)]
            rel = (h - z) / z * 100 if z else 0.0
            print(f"{t:<4} {DISPLAY[s]:<20} {z:>7.1f} {lo:>7.1f} {b:>7.1f} {h:>7.1f} {rel:>6.1f}%")
            idle_rows.append([t, DISPLAY[s], round(z, 1), round(lo, 1), round(b, 1), round(h, 1), round(rel, 1)])
    write_csv("idle_sensitivity.csv", ["trace", "strategy", "zero_kJ", "low_kJ", "base_kJ", "high_kJ", "z_to_h_pct"], idle_rows)
    n_csv += 1
else:
    print("\n(Idle table skipped: idle_frozen_{zero,low,high} not all present.)")

print(f"\nWrote {n_csv} CSVs to {OUT}/")
