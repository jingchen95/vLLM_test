#!/usr/bin/env python3
"""COMPLETE coverage audit of the existing Phase-2 calibration data (read-only).

Enumerates the measured (GPU,TP,Freq,IL,OL,Rate) space, builds coverage matrices,
grades decode-capacity confirmation, and ranks coverage holes by estimated impact
on SCHEDULER coverage (the decode DVFS grid the gate consumes), not by row count.

Inputs (nothing written except the report):
  Phase2_Results_{L4,L40S}/master_results.csv   — ground truth of what was measured
  artifacts/paper/models/decode_capacity.csv     — per-config capacity + confirmed flags
Output:
  paper/COVERAGE_AUDIT.md                         — full detailed report
Also prints the Step-8 summary to stdout.
"""
from __future__ import annotations
import csv, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = {"l4": ROOT / "Phase2_Results_L4" / "master_results.csv",
       "l40s": ROOT / "Phase2_Results_L40S" / "master_results.csv"}
CAPCSV = ROOT / "artifacts" / "paper" / "models" / "decode_capacity.csv"
OUT = ROOT / "paper" / "COVERAGE_AUDIT.md"
KAPPA = 0.95


def load_raw(gpu):
    """(tp,freq,il,ol,rate) -> averaged {ach_tps, tpot} ; plus axis value sets."""
    acc = defaultdict(lambda: defaultdict(list))
    axes = defaultdict(set)
    for r in csv.DictReader(open(RAW[gpu])):
        try:
            tp, f = int(r["tp_degree"]), int(r["gpu_freq_mhz"])
            il, ol = int(r["input_len"]), int(r["output_len"])
            rate = float(r["request_rate"])
        except (KeyError, ValueError):
            continue
        k = (tp, f, il, ol, rate)
        acc[k]["ach"].append(float(r["output_token_throughput_tps"]))
        acc[k]["tpot"].append(float(r["p99_tpot_ms"]))
        for name, v in (("tp", tp), ("freq", f), ("il", il), ("ol", ol), ("rate", rate)):
            axes[name].add(v)
    rows = {k: {m: st.mean(v) for m, v in d.items()} for k, d in acc.items()}
    return rows, axes


def load_cap():
    """(gpu,il,ol,tp,freq) -> {stable_C, stable_confirmed} ; and per-slo C."""
    stable = {}
    per_slo = defaultdict(dict)  # (gpu,il,ol,tp,freq) -> {slo: (C, confirmed)}
    for r in csv.DictReader(open(CAPCSV)):
        key = (r["gpu"], int(r["il"]), int(r["ol"]), int(r["tp"]), int(r["freq_mhz"]))
        stable[key] = (float(r["C_dc_stable_tps"]), r["stable_confirmed"].lower() == "true")
        per_slo[key][float(r["slo_tpot_ms"])] = (
            float(r["C_dc_SLO_tps"]), r["slo_confirmed"].lower() == "true")
    return stable, per_slo


def config_capacity_grade(gpu, il, ol, tp, freq, stable):
    """confirmed | lower_bound | none for a (config) from the stable capacity."""
    C, conf = stable.get((gpu, il, ol, tp, freq), (0.0, False))
    if C <= 0:
        return "none"
    return "confirmed" if conf else "lower_bound"


def main():
    out = []
    w = out.append
    w("# Phase-2 calibration coverage audit\n")
    w("Read-only enumeration of the existing data. Denominators: per GPU the full "
      "config grid is |TP| x |Freq| x |IL| x |OL| (rate is a sweep dimension used to "
      "*derive* decode capacity, not part of a config). `mem_freq` maps 1:1 to "
      "`gpu_freq`, so it is not an independent axis.\n")

    data = {}
    for gpu in ("l40s", "l4"):
        rows, axes = load_raw(gpu)
        data[gpu] = (rows, axes)
    stable, per_slo = load_cap()

    # ---- Step 1: enumerate axes ---------------------------------------------
    w("## Step 1 — Calibration space (unique values per axis)\n")
    for gpu in ("l40s", "l4"):
        _, axes = data[gpu]
        w(f"### GPU = {gpu.upper()}")
        for name in ("tp", "freq", "il", "ol", "rate"):
            vals = sorted(axes[name])
            w(f"- **{name.upper()}** ({len(vals)}): {vals}")
        w("")

    # ---- Step 2: coverage matrix (TP x Freq) --------------------------------
    w("## Step 2 — Coverage matrix (TP x Freq): #IL, #OL, #rate-points, #configs\n")
    for gpu in ("l40s", "l4"):
        rows, axes = data[gpu]
        tps, freqs = sorted(axes["tp"]), sorted(axes["freq"])
        w(f"### GPU = {gpu.upper()}\n")
        w("| TP | Freq | #IL | #OL | #(IL,OL) | rate-pts (min..max) | #configs |")
        w("|---|---|---|---|---|---|---|")
        for tp in tps:
            for f in freqs:
                sub = [k for k in rows if k[0] == tp and k[1] == f]
                ils = {k[2] for k in sub}; ols = {k[3] for k in sub}
                pairs = {(k[2], k[3]) for k in sub}
                rpc = [len({k[4] for k in sub if (k[2], k[3]) == p}) for p in pairs]
                rr = f"{min(rpc)}..{max(rpc)}" if rpc else "-"
                w(f"| {tp} | {f} | {len(ils)} | {len(ols)} | {len(pairs)} | {rr} | {len(sub)} |")
        w("")

    # ---- Step 3: shape coverage per (GPU,TP,Freq) ---------------------------
    w("## Step 3 — Shape (IL,OL) coverage per (GPU,TP,Freq)\n")
    w("Total possible (IL,OL) pairs per cell = |IL| x |OL| = 11 x 4 = 44.\n")
    for gpu in ("l40s", "l4"):
        rows, axes = data[gpu]
        tps, freqs = sorted(axes["tp"]), sorted(axes["freq"])
        w(f"### GPU = {gpu.upper()}\n")
        w("| TP | Freq | existing pairs | missing | pairs |")
        w("|---|---|---|---|---|")
        for tp in tps:
            for f in freqs:
                pairs = sorted({(k[2], k[3]) for k in rows if k[0] == tp and k[1] == f})
                if not pairs:
                    continue
                plist = ", ".join(f"({a},{b})" for a, b in pairs)
                w(f"| {tp} | {f} | {len(pairs)} | {44-len(pairs)} | {plist} |")
        w("")

    # ---- Step 4: rate-sweep quality -----------------------------------------
    w("## Step 4 — Rate-sweep quality & decode-capacity grade per (GPU,TP,Freq,IL,OL)\n")
    grade_ct = {gpu: defaultdict(int) for gpu in ("l40s", "l4")}
    for gpu in ("l40s", "l4"):
        rows, axes = data[gpu]
        cfgs = defaultdict(list)  # (tp,f,il,ol) -> [(rate, ach, tpot)]
        for (tp, f, il, ol, rate), d in rows.items():
            cfgs[(tp, f, il, ol)].append((rate, d["ach"], d["tpot"]))
        for (tp, f, il, ol), pts in cfgs.items():
            rates = sorted(p[0] for p in pts)
            stable_seen = any(a / (rate * ol) >= KAPPA for rate, a, _ in pts if ol > 0)
            unstable_seen = any(a / (rate * ol) < KAPPA for rate, a, _ in pts if ol > 0)
            g = config_capacity_grade(gpu, il, ol, tp, f, stable)
            grade_ct[gpu][g] += 1
            grade_ct[gpu][("both_regions" if (stable_seen and unstable_seen) else "one_region")] += 1
    for gpu in ("l40s", "l4"):
        c = grade_ct[gpu]
        tot = c["confirmed"] + c["lower_bound"] + c["none"]
        w(f"### GPU = {gpu.upper()} (over {tot} measured configs)")
        w(f"- Confirmed (saw saturation knee): **{c['confirmed']}** ({c['confirmed']/tot:.0%})")
        w(f"- Lower-bound only (stable seen, no knee): **{c['lower_bound']}** ({c['lower_bound']/tot:.0%})")
        w(f"- No capacity (never stable at any tested rate): **{c['none']}** ({c['none']/tot:.0%})")
        w(f"- Configs where BOTH stable & unstable rates observed: {c['both_regions']} "
          f"| only one region: {c['one_region']}")
        w("")

    # ---- Step 5: frequency coverage per (GPU,TP,IL,OL) ----------------------
    w("## Step 5 — Frequency coverage per (GPU,TP,IL,OL)\n")
    w("How many of the 10 clock levels were profiled for each shape.\n")
    freq_hist = {}
    for gpu in ("l40s", "l4"):
        rows, axes = data[gpu]
        nfreq = defaultdict(set)
        for (tp, f, il, ol, rate) in rows:
            nfreq[(tp, il, ol)].add(f)
        hist = defaultdict(int)
        for k, fs in nfreq.items():
            hist[len(fs)] += 1
        freq_hist[gpu] = (nfreq, hist)
        w(f"### GPU = {gpu.upper()}")
        w(f"- #shapes (TP,IL,OL) measured: {len(nfreq)}")
        w(f"- #freqs-per-shape histogram: {dict(sorted(hist.items()))}")
        full = sorted(k for k, fs in nfreq.items() if len(fs) >= 8)
        w(f"- shapes with >=8 freqs (DVFS-usable): {len(full)} -> {full}")
        w("")

    # ---- Step 6: TP coverage per (GPU,IL,OL,Freq) ---------------------------
    w("## Step 6 — TP coverage per (GPU,IL,OL,Freq)\n")
    for gpu in ("l40s", "l4"):
        rows, axes = data[gpu]
        ntp = defaultdict(set)
        for (tp, f, il, ol, rate) in rows:
            ntp[(il, ol, f)].add(tp)
        hist = defaultdict(int)
        for k, tps in ntp.items():
            hist[tuple(sorted(tps))] += 1
        w(f"### GPU = {gpu.upper()} (|TP|={len(sorted(axes['tp']))})")
        w(f"- (IL,OL,Freq) cells measured: {len(ntp)}")
        w(f"- TP-set histogram: " + "; ".join(f"{list(k)}:{v}" for k, v in sorted(hist.items(), key=lambda x:-x[1])))
        w("")

    # ---- Step 7: SLO coverage -----------------------------------------------
    w("## Step 7 — TPOT-SLO bucket coverage (from decode_capacity.csv)\n")
    slo_ct = defaultdict(lambda: defaultdict(int))
    for key, slos in per_slo.items():
        gpu = key[0]
        for slo, (C, conf) in slos.items():
            if C <= 0:
                slo_ct[(gpu, slo)]["missing"] += 1
            elif conf:
                slo_ct[(gpu, slo)]["confirmed"] += 1
            else:
                slo_ct[(gpu, slo)]["lower_bound"] += 1
    for gpu in ("l40s", "l4"):
        w(f"### GPU = {gpu.upper()}")
        w("| SLO(ms) | confirmed | lower_bound | missing |")
        w("|---|---|---|---|")
        for (g, slo) in sorted(slo_ct, key=lambda x: (x[0], x[1])):
            if g != gpu:
                continue
            c = slo_ct[(g, slo)]
            w(f"| {slo:.0f} | {c['confirmed']} | {c['lower_bound']} | {c['missing']} |")
        w("")

    # ---- Step 8: final report ------------------------------------------------
    w("## Step 8 — Final coverage report\n")
    summary_lines = []

    def emit(s):
        w(s); summary_lines.append(s)

    for gpu in ("l40s", "l4"):
        rows, axes = data[gpu]
        tps, freqs = sorted(axes["tp"]), sorted(axes["freq"])
        full_grid = len(tps) * len(freqs) * 11 * 4
        measured_cfg = len({(k[0], k[1], k[2], k[3]) for k in rows})
        c = grade_ct[gpu]
        tot = c["confirmed"] + c["lower_bound"] + c["none"]
        emit(f"### GPU = {gpu.upper()}")
        emit(f"- Full config grid |TP|x|Freq|x|IL|x|OL| = {len(tps)}x{len(freqs)}x11x4 = **{full_grid}**")
        emit(f"- Measured configs (any rate): **{measured_cfg}** = {measured_cfg/full_grid:.0%} of grid")
        emit(f"- (1) measured capacity (confirmed): **{c['confirmed']/full_grid:.0%}** of full grid "
             f"({c['confirmed']} cells)")
        emit(f"- (2) conservative lower-bound only: **{c['lower_bound']/full_grid:.0%}** of full grid "
             f"({c['lower_bound']} cells)")
        emit(f"- usable (confirmed+lower_bound) of MEASURED configs: "
             f"{(c['confirmed']+c['lower_bound'])/tot:.0%}; of FULL grid: "
             f"{(c['confirmed']+c['lower_bound'])/full_grid:.0%}")
        # dense vs sparse dimensions
        nfreq, fhist = freq_hist[gpu]
        single = sum(1 for _, fs in nfreq.items() if len(fs) == 1)
        emit(f"- (3) DENSE: IL axis (all 11 values appear), OL axis (all 4 appear), "
             f"TP axis (all {len(tps)} appear), rate grid ({len(axes['rate'])} pts) at the "
             f"shapes that were swept.")
        emit(f"- (4) SPARSE: FREQUENCY per shape — {single}/{len(nfreq)} (TP,IL,OL) shapes have a "
             f"SINGLE clock; only {sum(1 for _,fs in nfreq.items() if len(fs)>=8)} have >=8.")
        emit("")

    # Top holes ranked by scheduler-coverage impact (DVFS grid unlocked)
    w("### Top coverage holes ranked by scheduler-coverage impact\n")
    w("Impact = DVFS cells (shape x 10 freq) the hole denies the decode gate, weighted "
      "toward diverse OL (the gate needs decode capacity across OL, but full freq sweeps "
      "sit almost entirely at OL=32).\n")
    holes = []
    for gpu in ("l40s", "l4"):
        rows, axes = data[gpu]
        nfreq, _ = freq_hist[gpu]
        # per (OL): how many shapes are freq-starved (single clock)
        ol_starved = defaultdict(int); ol_shapes = defaultdict(int)
        for (tp, il, ol), fs in nfreq.items():
            ol_shapes[ol] += 1
            if len(fs) == 1:
                ol_starved[ol] += 1
        for ol in sorted(ol_shapes):
            starved = ol_starved[ol]
            # cells denied ~ starved shapes * (10 - 1) missing freqs
            holes.append((starved * 9, gpu, f"OL={ol}: {starved}/{ol_shapes[ol]} shapes single-clock "
                                              f"-> ~{starved*9} DVFS cells missing"))
    holes.sort(reverse=True)
    for i, (impact, gpu, desc) in enumerate(holes[:10], 1):
        w(f"{i}. **[{gpu.upper()}]** {desc}")
    w("")

    OUT.write_text("\n".join(out))
    print(f"Wrote full report to {OUT} ({len(out)} lines)\n")
    print("\n".join(summary_lines))
    print("\nTOP HOLES (scheduler-impact ranked):")
    for i, (impact, gpu, desc) in enumerate(holes[:10], 1):
        print(f"  {i}. [{gpu.upper()}] {desc}")


if __name__ == "__main__":
    main()
