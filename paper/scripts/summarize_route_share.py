"""
summarize_route_share.py — Aggregate per-window route fractions from section42 logs.

For each (trace, strategy, slo, tau_kv) run, computes the request-weighted
fraction of traffic using each cross-pool route (AA, AB, BA, BB).

Usage:
  python summarize_route_share.py --log-dir results/paper/section42_sweep/logs
  python summarize_route_share.py --log-dir results/paper/hier_disagg_smoke/logs \
                                  --out-csv results/paper/analyses/route_share_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

# Filename pattern: {trace}_{strategy}_slo{N}_kv{M}.csv
# strategy may contain underscores (e.g. sweep_llm, hierarchical_disagg)
_PAT = re.compile(r"^(T\d+)_(.+)_slo(\d+)_kv(\d+(?:\.\d+)?)$")


def process_log(path: Path) -> dict | None:
    m = _PAT.match(path.stem)
    if not m:
        return None
    trace, strategy, slo_str, tau_str = m.groups()
    slo = int(slo_str)
    tau = float(tau_str)

    try:
        windows = list(csv.DictReader(path.open()))
    except Exception:
        return None
    if not windows:
        return None

    # Route fraction columns may be missing (non-disagg strategies)
    has_routes = "route_aa_frac" in windows[0]

    total_req = sum(int(w["n_requests"]) for w in windows)
    total_energy = sum(float(w["energy_j"]) for w in windows)
    n_viol = sum(1 for w in windows if w.get("slo_met", "true").lower() == "false")
    viol_rate = 100.0 * n_viol / len(windows) if windows else 0.0

    if has_routes and total_req > 0:
        def wt(col: str) -> float:
            return sum(float(w.get(col, 0)) * int(w["n_requests"]) for w in windows) / total_req

        aa = wt("route_aa_frac")
        ab = wt("route_ab_frac")
        ba = wt("route_ba_frac")
        bb = wt("route_bb_frac")
    else:
        aa = ab = ba = bb = float("nan")

    return {
        "trace": trace,
        "strategy": strategy,
        "slo_ms": slo,
        "tau_kv_us": tau,
        "total_energy_j": round(total_energy, 1),
        "slo_violation_rate_pct": round(viol_rate, 2),
        "n_windows": len(windows),
        "total_requests": total_req,
        "route_aa_pct": round(100 * aa, 2) if aa == aa else "",
        "route_ab_pct": round(100 * ab, 2) if ab == ab else "",
        "route_ba_pct": round(100 * ba, 2) if ba == ba else "",
        "route_bb_pct": round(100 * bb, 2) if bb == bb else "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True,
                    help="Directory containing per-window CSV logs")
    ap.add_argument("--out-csv",
                    default="results/paper/analyses/route_share_summary.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"ERROR: log-dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in sorted(log_dir.glob("*.csv")):
        r = process_log(f)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No matching log files found.", file=sys.stderr)
        sys.exit(1)

    # Sort for readability
    rows.sort(key=lambda r: (r["trace"], r["strategy"], r["slo_ms"], r["tau_kv_us"]))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trace", "strategy", "slo_ms", "tau_kv_us",
              "total_energy_j", "slo_violation_rate_pct", "n_windows", "total_requests",
              "route_aa_pct", "route_ab_pct", "route_ba_pct", "route_bb_pct"]

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    # Print table to stdout for quick inspection
    disagg = [r for r in rows if r["route_aa_pct"] != ""]
    if disagg:
        print(f"\n{'trace':>3} {'strategy':>22} {'slo':>5} {'tau':>5} | "
              f"{'E(J)':>10} {'viol%':>6} | {'AA%':>6} {'AB%':>6} {'BA%':>6} {'BB%':>6}")
        print("-" * 85)
        for r in disagg:
            print(f"{r['trace']:>3} {r['strategy']:>22} {r['slo_ms']:>5} {r['tau_kv_us']:>5.0f} | "
                  f"{r['total_energy_j']:>10.0f} {r['slo_violation_rate_pct']:>5.1f}% | "
                  f"{r['route_aa_pct']:>6} {r['route_ab_pct']:>6} {r['route_ba_pct']:>6} {r['route_bb_pct']:>6}")


if __name__ == "__main__":
    main()
