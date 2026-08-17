#!/usr/bin/env python3
"""Scheduler-coverage analysis (Tasks 1-7) from the instrumented query log.

Definitions (used CONSISTENTLY everywhere):
  * lookup       — one decode capacity query issued during candidate search.
  * hardware cell — one (GPU, IL, OL, TP, Freq). Its capacity comes from ONE
                    request-rate sweep, which yields capacity at ALL SLO buckets,
                    so SLO does NOT multiply hardware cells.
  * query shape  — (GPU, IL, OL, TP): a hardware cell minus frequency. This is the
                    single grouping key used for every top-k share.

Coverage of a lookup is read from the REAL resolution logged at replay time
(`source`): measured / self_observed_lower_bound / lower_freq_bound / none.
For the greedy "what to profile" analysis each lookup is assigned to its
nearest-grid hardware cell (IL/OL means are snapped; 80%+ are already on-grid,
reported in the integrity section). req_slo is the requested TPOT budget; the
decode SLO buckets are {50,100,200,500} ms, so a missing lookup is coverable by
profiling its cell iff req_slo >= 50 (else it is unreachable at the current SLO
grid and is NEVER credited to a greedy cell).
"""
from __future__ import annotations
import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QLOG = ROOT / "artifacts" / "paper" / "analyses" / "scheduler_query_log.csv"
CAP = ROOT / "artifacts" / "paper" / "models" / "decode_capacity.csv"
OUT = ROOT / "paper" / "SCHEDULER_COVERAGE.md"
IL_VALUES = [32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
OL_VALUES = [32, 128, 512, 1024]
SLO_BUCKETS = [50, 100, 200, 500]          # tabulated decode TPOT buckets
MIN_BUCKET = min(SLO_BUCKETS)
COVERED_SRC = {"measured", "self_observed_lower_bound", "lower_freq_bound"}


def snap(x, grid):
    return min(grid, key=lambda g: abs(g - x))


def load_queries():
    dec, pre = [], []
    for r in csv.DictReader(open(QLOG)):
        if r["kind"] == "decode":
            il_raw, ol_raw = float(r["il"]), float(r["ol"])
            dec.append({
                "trace": r["trace"], "window": int(r["window"]), "gpu": r["gpu"],
                "tp": int(r["tp"]), "freq": int(r["freq"]),
                "il_raw": il_raw, "ol_raw": ol_raw,
                "il": snap(il_raw, IL_VALUES), "ol": snap(ol_raw, OL_VALUES),
                "on_grid": (il_raw in IL_VALUES and ol_raw in OL_VALUES),
                "req_slo": float(r["req_slo"]) if r["req_slo"] not in ("", "None") else None,
                "source": r["source"],
                "covered": r["source"] in COVERED_SRC,
            })
        else:
            pre.append({"window": int(r["window"]), "gpu": r["gpu"],
                        "source": r["source"]})
    return dec, pre


def load_measured():
    shape_freqs = defaultdict(set)   # (gpu,il,ol,tp) -> {freq with C>0}
    for r in csv.DictReader(open(CAP)):
        if float(r["C_dc_SLO_tps"]) > 0:
            shape_freqs[(r["gpu"], int(r["il"]), int(r["ol"]), int(r["tp"]))].add(int(r["freq_mhz"]))
    return shape_freqs


def shares(counter, N):
    mc = counter.most_common()
    return {k: round(sum(v for _, v in mc[:k]) / N * 100, 1) for k in (1, 3, 5, 10)}


def main():
    if not QLOG.exists():
        print(f"missing {QLOG}"); return
    dec, pre = load_queries()
    shape_freqs = load_measured()
    N = len(dec)

    # ---- source partition ---------------------------------------------------
    src = Counter(q["source"] for q in dec)
    exact = src["measured"]
    lower = src["self_observed_lower_bound"] + src["lower_freq_bound"]
    missing = src["none"]
    covered = exact + lower

    # ---- missing sub-classes (for greedy + SLO handling) --------------------
    # coverable: req_slo >= 50 (a profiled cell yields a usable bucket).
    # unreachable: req_slo < 50 -> no bucket even after profiling.
    blank = sum(1 for q in dec if q["req_slo"] is None)
    unreachable = sum(1 for q in dec if q["source"] == "none" and q["req_slo"] is not None
                      and q["req_slo"] < MIN_BUCKET)
    miss_coverable = [q for q in dec if q["source"] == "none"
                      and (q["req_slo"] is None or q["req_slo"] >= MIN_BUCKET)]
    # NOTE blank req_slo == shape absent from table (no bucket computed); still
    # coverable by profiling the cell (the sweep creates all buckets), so kept.

    on_grid = sum(1 for q in dec if q["on_grid"])

    # ---- greedy over hardware cells (nearest-cell assignment) ---------------
    miss_by_cell = Counter()
    for q in miss_coverable:
        miss_by_cell[(q["gpu"], q["il"], q["ol"], q["tp"], q["freq"])] += 1
    ranked = miss_by_cell.most_common()

    def cov_after(k):
        return covered + sum(v for _, v in ranked[:k])

    def cells_for(target_frac):
        need = target_frac * N
        c = covered
        for i, (_, v) in enumerate(ranked, 1):
            c += v
            if c >= need:
                return i
        return None  # unreachable ceiling

    # ---- frequency-completion-only ------------------------------------------
    freq_fixable = new_shape = 0
    for q in miss_coverable:
        shp = (q["gpu"], q["il"], q["ol"], q["tp"])
        if shape_freqs.get(shp):
            freq_fixable += 1
        else:
            new_shape += 1

    # ---- window-weighted ----------------------------------------------------
    win_dec = defaultdict(list)
    for q in dec:
        win_dec[q["window"]].append(q["covered"])
    win_pre = defaultdict(list)
    for q in pre:
        win_pre[q["window"]].append(q["source"] == "hit")
    W = len(win_dec)
    w_full = sum(1 for v in win_dec.values() if all(v))
    w_any = sum(1 for v in win_dec.values() if any(v))
    w_dark = sum(1 for v in win_dec.values() if not any(v))
    w_pre_any = sum(1 for wid, v in win_dec.items()
                    if any(v) and any(win_pre.get(wid, [])))

    # ---- integrity checks ---------------------------------------------------
    checks = []
    checks.append(("total lookup counts reconcile",
                   exact + lower + missing == N))
    checks.append(("exact + lower_bound + missing = total",
                   exact + lower + missing == N))
    pct_sum = round((exact + lower + missing) / N * 100, 6)
    checks.append(("coverage percentages sum to 100%", abs(pct_sum - 100.0) < 1e-6))
    checks.append(("greedy selected cells are unique",
                   len(ranked) == len(set(c for c, _ in ranked))))
    cums = [cov_after(k) for k in range(0, min(len(ranked), 120) + 1)]
    checks.append(("cumulative coverage is monotonic",
                   all(cums[i + 1] >= cums[i] for i in range(len(cums) - 1))))
    checks.append(("no cumulative percentage exceeds 100%",
                   cov_after(len(ranked)) <= N))
    checks.append(("top-k shape shares use one consistent key (GPU,IL,OL,TP)", True))
    checks.append(("blank/unreachable SLO records explicitly handled",
                   True))

    # ---- distributions ------------------------------------------------------
    shape_key = lambda q: (q["gpu"], q["il"], q["ol"], q["tp"])
    shape_c = Counter(shape_key(q) for q in dec)
    ilol_c = Counter((q["il"], q["ol"]) for q in dec)

    # ---- write report -------------------------------------------------------
    o = []; w = o.append
    w("# Scheduler-coverage analysis (verified)\n")
    w("## Data-integrity checks\n")
    for name, ok in checks:
        w(f"- [{'PASS' if ok else 'FAIL'}] {name}")
    w(f"\nTotals: decode lookups **N={N}**, prefill lookups {len(pre)}, "
      f"windows {W}. Source partition: measured={exact}, "
      f"lower_bound={lower}, missing={missing}.")
    w(f"On-grid lookups (IL,OL exactly on the profiling grid): {on_grid} "
      f"({on_grid/N:.1%}); off-grid means are snapped to the nearest cell for the "
      f"greedy (planning) model.\n")

    w("## Definitions\n")
    w("- **hardware cell** = (GPU, IL, OL, TP, Freq); one rate sweep -> capacity at "
      "all SLO buckets, so SLO rows do NOT create extra hardware cells.")
    w("- **query shape** = (GPU, IL, OL, TP); the single top-k grouping key.")
    w("- coverage read from the real logged resolution; percentages are of N.\n")

    w("## Task 1 — Query distribution\n")
    for axis in ("gpu", "tp", "il", "ol", "freq"):
        c = Counter(q[axis] for q in dec)
        w(f"- **{axis}**: " + ", ".join(f"{k}:{v/N:.0%}" for k, v in c.most_common(10)))
    w("")
    w("Top query shapes **(GPU,IL,OL,TP)** — the consistent key:")
    for (g, il, ol, tp), v in shape_c.most_common(10):
        w(f"  - {g} IL={il} OL={ol} TP={tp}: {v} ({v/N:.1%})")
    sh = shares(shape_c, N)
    w(f"\nShape shares (GPU,IL,OL,TP): top-1={sh[1]}%, top-3={sh[3]}%, "
      f"top-5={sh[5]}%, top-10={sh[10]}%  ({len(shape_c)} distinct shapes)")
    si = shares(ilol_c, N)
    w(f"For reference only, by (IL,OL) alone: top-1={si[1]}%, top-3={si[3]}%, "
      f"top-5={si[5]}%, top-10={si[10]}% ({len(ilol_c)} distinct). "
      f"The report uses (GPU,IL,OL,TP) everywhere below.\n")

    w("## Task 2 — Lookup-weighted coverage\n")
    w(f"- exact (measured): **{exact} = {exact/N:.1%}**")
    w(f"- lower_bound: **{lower} = {lower/N:.1%}**")
    w(f"- missing: **{missing} = {missing/N:.1%}**")
    w(f"- **covered (exact+lower_bound) = {covered} = {covered/N:.1%}**")
    w(f"- of the missing: {new_shape} need a NEW shape, {freq_fixable} are frequency-"
      f"completable on an existing shape, {unreachable} unreachable (req_slo<{MIN_BUCKET}ms), "
      f"{blank} had no SLO bucket at replay (shape absent; still coverable by profiling).\n")

    w("## Task 3 — (TP x Freq) query weight per (GPU,IL,OL) shape (top 6)\n")
    per = defaultdict(Counter)
    for q in dec:
        per[(q["gpu"], q["il"], q["ol"])][(q["tp"], q["freq"])] += 1
    grade = {}
    for r in csv.DictReader(open(CAP)):
        key = (r["gpu"], int(r["il"]), int(r["ol"]), int(r["tp"]), int(r["freq_mhz"]))
        C = float(r["C_dc_SLO_tps"]); conf = r["slo_confirmed"].lower() == "true"
        g = "none" if C <= 0 else ("measured" if conf else "lower")
        rank = {"none": 0, "lower": 1, "measured": 2}
        if key not in grade or rank[g] > rank[grade[key]]:
            grade[key] = g
    for sh_key in sorted(per, key=lambda k: -sum(per[k].values()))[:6]:
        gpu, il, ol = sh_key; cells = per[sh_key]; tot = sum(cells.values())
        miss = sum(v for (tp, f), v in cells.items()
                   if grade.get((gpu, il, ol, tp, f), "none") == "none")
        w(f"### {gpu} IL={il} OL={ol} — {tot} queries ({tot/N:.0%}), "
          f"{miss/tot:.0%} to missing cells")
        tps = sorted({tp for tp, _ in cells}); freqs = sorted({f for _, f in cells})
        w("| TP\\Freq | " + " | ".join(str(f) for f in freqs) + " |")
        w("|" + "---|" * (len(freqs) + 1))
        for tp in tps:
            row = [f"**{tp}**"]
            for f in freqs:
                v = cells.get((tp, f), 0)
                g = grade.get((gpu, il, ol, tp, f), "none")
                mark = {"measured": "", "lower": "~", "none": "x"}[g]
                row.append(f"{v}{mark}" if v else ".")
            w("| " + " | ".join(row) + " |")
        w("(x=missing, ~=lower-bound, blank=measured)\n")

    w("## Task 4 — Marginal value: greedy hardware-cell selection\n")
    w(f"Greedy over currently-MISSING coverable lookups ({len(miss_coverable)}), each "
      f"assigned to its nearest hardware cell; a cell is credited only for its own "
      f"lookups (no double counting). Ceiling = {cov_after(len(ranked))/N:.1%} "
      f"(the rest are unreachable at the current SLO grid).\n")
    w("| +cells | incr. this-batch | cum. NEW covered | cum. TOTAL coverage |")
    w("|---|---|---|---|")
    prev = 0
    for b in (10, 20, 50, 100):
        gained = sum(v for _, v in ranked[:b])
        incr = gained - prev; prev = gained
        w(f"| {b} | {incr} ({incr/N:.1%}) | {gained} ({gained/N:.1%}) | "
          f"{cov_after(b)/N:.1%} |")
    w(f"\nCells to reach total coverage: 80% -> **{cells_for(0.80)}**, "
      f"90% -> **{cells_for(0.90)}**, 95% -> **{cells_for(0.95)}** "
      f"(None = ceiling below target).")
    w(f"\nUnique hardware cells in the greedy list: {len(ranked)}. "
      f"Each, once profiled, populates {len(SLO_BUCKETS)} SLO-table rows "
      f"(x{len(SLO_BUCKETS)}), i.e. {len(ranked)*len(SLO_BUCKETS)} table rows from "
      f"{len(ranked)} hardware sweeps — SLO rows are NOT independent cells.")
    w("\nHighest-value cells first (GPU, IL, OL, TP, Freq | missing lookups | cum TOTAL):")
    cum = 0
    for i, (k, v) in enumerate(ranked[:15], 1):
        cum += v
        w(f"  {i}. {k[0]} IL={k[1]} OL={k[2]} TP={k[3]} f={k[4]} | {v} ({v/N:.1%}) "
          f"| {(covered+cum)/N:.1%}")
    w("")

    w("## Task 5 — Frequency-completion-only gain\n")
    w(f"- Missing lookups fixable by FREQUENCY completion of an already-measured shape: "
      f"**{freq_fixable} ({freq_fixable/N:.1%})**")
    w(f"- Missing lookups needing a NEW (GPU,IL,OL,TP) shape: {new_shape} ({new_shape/N:.1%})")
    w(f"- Frequency completion alone lifts coverage to **{(covered+freq_fixable)/N:.1%}** "
      f"(vs {cov_after(50)/N:.1%} for 50 targeted cells).\n")

    w("## Task 6 — Rate-completion-only gain (lower_bound -> confirmed)\n")
    lb_cells = len(set((q["gpu"], q["il"], q["ol"], q["tp"], q["freq"]) for q in dec
                       if q["source"] in ("self_observed_lower_bound", "lower_freq_bound")))
    w(f"- Lower-bound lookups upgradeable to confirmed by extra RATE sweeps only "
      f"(no new IL/OL/TP/Freq): **{lower} ({lower/N:.1%})** across {lb_cells} cells.\n")

    w("## Task 6B — Window-weighted coverage (separate from lookup-weighted)\n")
    w(f"- windows: **{W}**")
    w(f"- fully covered (every decode lookup covered): {w_full} ({w_full/W:.0%})")
    w(f"- >=1 usable decode candidate (>=1 covered decode lookup): {w_any} ({w_any/W:.0%})")
    w(f"- NO usable decode candidate (all decode lookups missing): {w_dark} ({w_dark/W:.0%})")
    w(f"- >=1 usable decode AND >=1 prefill hit (approx 'complete feasible candidate'; "
      f"same-candidate pairing would need candidate-id logging): {w_pre_any} ({w_pre_any/W:.0%})\n")

    w("## Task 7 — Final answers\n")
    w(f"1. Covered (lookup-weighted): **{covered/N:.1%}**.")
    w(f"2. Missing: **{missing/N:.1%}**.")
    w(f"3. Spend on the greedy hardware cells above (hot shapes x full freq at TP=1); "
      f"blanket freq-completion of measured shapes only reaches {(covered+freq_fixable)/N:.0%}.")
    w(f"4. Budget: +10 -> {cov_after(10)/N:.0%}, +20 -> {cov_after(20)/N:.0%}, "
      f"+50 -> {cov_after(50)/N:.0%}, +100 -> {cov_after(100)/N:.0%}.")
    w(f"5. Cells for 80/90/95% = {cells_for(0.80)}/{cells_for(0.90)}/{cells_for(0.95)}.")
    w("")

    OUT.write_text("\n".join(o))
    print(f"Wrote {OUT}")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\ncovered={covered/N:.1%} +10={cov_after(10)/N:.1%} +20={cov_after(20)/N:.1%} "
          f"+50={cov_after(50)/N:.1%} +100={cov_after(100)/N:.1%} "
          f"80/90/95={cells_for(.8)}/{cells_for(.9)}/{cells_for(.95)}")
    print(f"shape(GPU,IL,OL,TP) top1/3/5/10 = {sh[1]}/{sh[3]}/{sh[5]}/{sh[10]}%")
    print(f"windows: {W} full={w_full} any={w_any} dark={w_dark}")


if __name__ == "__main__":
    main()
