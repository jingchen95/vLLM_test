"""Generate the production-replay evaluation tables from the frozen Azure sweeps.

Sources of truth:
  results/paper/prod_conv_frozen/summary.csv   -> tab:prod_conv
  results/paper/prod_code_frozen/summary.csv   -> tab:prod_code
  results/paper/prod_diurnal_frozen/summary.csv (optional) -> tab:prod_diurnal

For each workload it prints, per utilization level (50/70/85%):
  - energy (kJ) and modeled violation (%v) per strategy
  - SWEEP-LLM savings vs the best feasible / near-feasible baseline
and emits a paper-ready LaTeX tabular body to stdout.

Run: python paper/scripts/make_production_tables.py
"""
import csv
import os

ORDER = ["static_disagg", "greenllm", "dynamollm", "dualscale", "hierarchical_disagg", "sweep_llm"]
DISPLAY = {
    "static_disagg": "Static-Disagg",
    "greenllm": "GreenLLM-OracleDVFS",
    "dynamollm": "DynamoLLM-Mono",
    "dualscale": "DualScale-Ext",
    "hierarchical_disagg": "Hierarchical-Disagg",
    "sweep_llm": "SWEEP-LLM",
}
UTILS = ["50", "70", "85"]


def kj(r):
    return float(r["total_energy_j"]) / 1000.0


def viol(r):
    return float(r["slo_violation_rate_pct"])


def load(summary_path, prefix):
    """Return {(util, strategy): row}. prefix is 'conv'/'code'/'conv_diurnal'."""
    if not os.path.exists(summary_path):
        return None
    out = {}
    for r in csv.DictReader(open(summary_path)):
        trace = r["trace"]              # e.g. conv_50pct, code_85pct, conv_diurnal_70pct
        for u in UTILS:
            if trace == f"{prefix}_{u}pct":
                out[(u, r["strategy"])] = r
    return out


def emit_table(name, data, caption_kj_fmt="%.0f"):
    print(f"\n===== {name} =====")
    print(f"{'Strategy':<22}" + "".join(f"{u + '%':>16}" for u in UTILS))
    for s in ORDER:
        cells = []
        for u in UTILS:
            r = data.get((u, s))
            cells.append(f"{kj(r):.0f} / {viol(r):.1f}" if r else "n/a")
        print(f"{DISPLAY[s]:<22}" + "".join(f"{c:>16}" for c in cells))

    # LaTeX body
    print(f"\n--- LaTeX body for {name} ---")
    for s in ORDER:
        cells = []
        for u in UTILS:
            r = data.get((u, s))
            cells.append(f"{kj(r):.0f} / {viol(r):.1f}" if r else "--")
        disp = f"\\textbf{{{DISPLAY[s]}}}" if s == "sweep_llm" else DISPLAY[s]
        body = " & ".join((f"\\textbf{{{c}}}" if s == "sweep_llm" else c) for c in cells)
        print(f"{disp:<22} & {body} \\\\")

    # SWEEP savings, per util. "Feasible" is self-consistent: a baseline is feasible
    # iff its modeled violation <= SWEEP's own violation at that util (so on conv,
    # where SWEEP is 0%, only true-0% baselines qualify). We always also report the
    # margin over Hierarchical-Disagg, the paper's primary routing-aware comparator.
    print(f"\n--- {name}: SWEEP-LLM savings ---")
    for u in UTILS:
        sw = data.get((u, "sweep_llm"))
        if sw is None:
            continue
        sw_kj, sw_v = kj(sw), viol(sw)
        feas = [(s, kj(data[(u, s)]), viol(data[(u, s)]))
                for s in ORDER if s != "sweep_llm"
                and (u, s) in data and viol(data[(u, s)]) <= sw_v + 1e-9]
        line = f"  util={u}%  SWEEP={sw_kj:.0f}kJ/{sw_v:.2f}%v"
        if feas:
            best = min(feas, key=lambda x: x[1])
            sav = 100.0 * (best[1] - sw_kj) / best[1]
            line += f"  | best-feasible={DISPLAY[best[0]]}({best[1]:.0f}kJ) savings={sav:.1f}%"
        else:
            line += "  | no feasible (viol<=SWEEP) non-SWEEP baseline (cluster undersized)"
        hr = data.get((u, "hierarchical_disagg"))
        if hr:
            line += f"  | vs Hierarchical={100.0 * (kj(hr) - sw_kj) / kj(hr):.1f}%"
        # flag any baseline cheaper than SWEEP (it is then infeasible by construction)
        cheaper = [(s, kj(data[(u, s)]), viol(data[(u, s)]))
                   for s in ORDER if s != "sweep_llm"
                   and (u, s) in data and kj(data[(u, s)]) < sw_kj]
        if cheaper:
            tags = ", ".join(f"{DISPLAY[s]}({e:.0f}kJ@{v:.1f}%v)" for s, e, v in cheaper)
            line += f"  | cheaper-but-infeasible: {tags}"
        print(line)


def main():
    conv = load("results/paper/prod_conv_frozen/summary.csv", "conv")
    code = load("results/paper/prod_code_frozen/summary.csv", "code")
    diur = load("results/paper/prod_diurnal_frozen/summary.csv", "conv_diurnal")

    if conv:
        emit_table("tab:prod_conv (Azure conversation replay)", conv)
    if code:
        emit_table("tab:prod_code (Azure code replay; cluster undersized)", code)
    if diur and len(diur) >= len(ORDER) * len(UTILS):
        emit_table("tab:prod_diurnal (1-week-derived diurnal segment)", diur)
    elif diur:
        print("\n[diurnal] partial results present "
              f"({len(diur)} cells, need {len(ORDER) * len(UTILS)}) -- skipping table")
    else:
        print("\n[diurnal] results/paper/prod_diurnal_frozen/summary.csv not found -- skipping")


if __name__ == "__main__":
    main()
