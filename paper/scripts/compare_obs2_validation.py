#!/usr/bin/env python3
"""
Compare the Obs2 hardware fill-in measurements (validate_obs2_l4.sh output)
against the predictive model, for the cells the model EXTRAPOLATED to.

Usage:
  python compare_obs2_validation.py [--results ../../Obs2_Validation_L4/obs2_validation_results.csv]

Prints, per measured cell: measured P99 TTFT/TPOT (median over runs), the model
prediction at the same (il,ol,tp,freq,rate), and the pass/fail verdict vs the
motivation SLO (TTFT<=300, TPOT<=120). The point is mechanism confirmation, not
tight numerical agreement (the model is known to be conservative).
"""
from __future__ import annotations
import argparse, csv, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import paper_model_dir
from scheduler import EnergyScheduler

TTFT_SLO, TPOT_SLO = 300, 120

def med(vals):
    vals = [v for v in vals if v is not None and v > 0]
    return st.median(vals) if vals else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[2] / "Obs2_Validation_L4" / "obs2_validation_results.csv"))
    args = ap.parse_args()

    sch = EnergyScheduler(str(paper_model_dir("models_l4")))

    # group measured rows by (tp,il,ol,freq,rate)
    cells = {}
    for r in csv.DictReader(open(args.results)):
        try:
            k = (int(float(r["tp_degree"])), int(float(r["input_len"])), int(float(r["output_len"])),
                 int(float(r["gpu_freq_mhz"])), float(r["request_rate"]))
        except Exception:
            continue
        cells.setdefault(k, {"ttft": [], "tpot": [], "pw": [], "e": []})
        for col, key in (("p99_ttft_ms", "ttft"), ("p99_tpot_ms", "tpot"), ("avg_power_w", "pw"), ("energy_j", "e")):
            try: cells[k][key].append(float(r[col]))
            except Exception: pass

    print(f"{'tp':>3}{'il':>6}{'ol':>6}{'freq':>6}{'rate':>6} | "
          f"{'meas_TTFT':>10}{'mdl_TTFT':>9} | {'meas_TPOT':>10}{'mdl_TPOT':>9} | {'meas_P':>7} verdict")
    print("-" * 100)
    for k in sorted(cells):
        tp, il, ol, freq, rate = k
        m = cells[k]
        mt, mp = med(m["ttft"]), med(m["tpot"])
        pw = med(m["pw"])
        # model predictions (prefill -> TTFT; decode -> TPOT). ol large => decode-bound cell.
        pf = sch.predict_config(il=il, ol=ol, tp=tp, freq=freq, rate=rate, slo=TTFT_SLO, phase="prefill")
        dc = sch.predict_config(il=il, ol=ol, tp=tp, freq=freq, rate=rate, slo=TPOT_SLO, phase="decode")
        mdl_ttft = pf.get("p99_ttft_ms", pf.get("latency_ms"))
        mdl_tpot = dc.get("p99_tpot_ms", dc.get("latency_ms"))
        verdict = []
        if mt is not None: verdict.append(f"TTFT {'PASS' if mt <= TTFT_SLO else 'FAIL'}")
        if mp is not None: verdict.append(f"TPOT {'PASS' if mp <= TPOT_SLO else 'FAIL'}")
        f2 = lambda x: f"{x:>10.1f}" if x is not None else f"{'-':>10}"
        f1 = lambda x: f"{x:>9.1f}" if x is not None else f"{'-':>9}"
        print(f"{tp:>3}{il:>6}{ol:>6}{freq:>6}{rate:>6.0f} | "
              f"{f2(mt)}{f1(mdl_ttft)} | {f2(mp)}{f1(mdl_tpot)} | {(f'{pw:.0f}' if pw else '-'):>7} {', '.join(verdict)}")

if __name__ == "__main__":
    main()
