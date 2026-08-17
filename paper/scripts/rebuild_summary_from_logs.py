"""
rebuild_summary_from_logs.py — Rebuild a clean summary.csv from per-run log files.

Use this when summary.csv has been corrupted (e.g., duplicate rows from two concurrent
sweep processes writing to the same file).  Reads each log CSV, computes the same
aggregate statistics as run_section42_sweep.py, and writes a deduplicated summary.

Usage:
    python rebuild_summary_from_logs.py --log-dir results/paper/hier_disagg_full/logs \
                                        --out-csv results/paper/hier_disagg_full/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

_PAT = re.compile(r"^(T\d+)_(.+)_slo(\d+)_kv(\d+(?:\.\d+)?)$")

SUMMARY_FIELDS = [
    "trace", "strategy", "slo_ttft_ms", "tau_kv_us_per_tok",
    "num_windows", "total_requests", "total_energy_j",
    "mean_power_w", "peak_power_w", "slo_violation_rate_pct",
    "mean_arrival_rate_rps", "peak_arrival_rate_rps",
    "decision_ms_mean", "decision_ms_p50", "decision_ms_p99", "decision_ms_max",
    "wall_clock_s", "status",
]


def _pct(values: list[float], p: float) -> float:
    """p-th percentile (0–100) via linear interpolation."""
    if not values:
        return float("nan")
    s = sorted(values)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def process_log(path: Path) -> dict | None:
    m = _PAT.match(path.stem)
    if not m:
        return None
    trace, strategy, slo_str, tau_str = m.groups()
    slo = int(slo_str)
    tau = float(tau_str)

    try:
        rows = list(csv.DictReader(path.open()))
    except Exception as e:
        print(f"  WARN: could not read {path.name}: {e}", file=sys.stderr)
        return None
    if not rows:
        return None

    n_windows = len(rows)
    total_requests = sum(int(r["n_requests"]) for r in rows)
    total_energy_j = sum(float(r["energy_j"]) for r in rows)
    powers = [float(r["power_w"]) for r in rows]
    arrival_rates = [float(r["arrival_rate_rps"]) for r in rows]
    dec_times = [float(r["decision_ms"]) for r in rows]

    n_viol = sum(1 for r in rows if r.get("slo_met", "true").lower() == "false")
    viol_pct = round(100.0 * n_viol / n_windows, 2) if n_windows else 0.0

    mean_power = round(total_energy_j / (n_windows * 5.0), 2)  # 5s window
    peak_power = round(max(powers), 1)
    mean_arr = round(statistics.mean(arrival_rates), 3)
    peak_arr = round(max(arrival_rates), 1)
    dec_mean = round(statistics.mean(dec_times), 3)
    dec_p50 = round(_pct(dec_times, 50), 3)
    dec_p99 = round(_pct(dec_times, 99), 3)
    dec_max = round(max(dec_times), 3)

    status = "PASS"  # matches sweep script: FAIL only means a Python exception

    return {
        "trace": trace,
        "strategy": strategy,
        "slo_ttft_ms": slo,
        "tau_kv_us_per_tok": tau,
        "num_windows": n_windows,
        "total_requests": total_requests,
        "total_energy_j": round(total_energy_j, 1),
        "mean_power_w": mean_power,
        "peak_power_w": peak_power,
        "slo_violation_rate_pct": viol_pct,
        "mean_arrival_rate_rps": mean_arr,
        "peak_arrival_rate_rps": peak_arr,
        "decision_ms_mean": dec_mean,
        "decision_ms_p50": dec_p50,
        "decision_ms_p99": dec_p99,
        "decision_ms_max": dec_max,
        "wall_clock_s": "",
        "status": status,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"ERROR: {log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in sorted(log_dir.glob("*.csv")):
        r = process_log(f)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No matching log files found.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (r["trace"], r["strategy"], r["slo_ttft_ms"], r["tau_kv_us_per_tok"]))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    # Quick table
    print(f"\n{'trace':>3} {'strategy':>22} {'slo':>5} {'tau':>5} | "
          f"{'E(J)':>10} {'viol%':>6} | {'status':>6}")
    print("-" * 75)
    for r in rows:
        print(f"{r['trace']:>3} {r['strategy']:>22} {r['slo_ttft_ms']:>5} "
              f"{r['tau_kv_us_per_tok']:>5.0f} | "
              f"{r['total_energy_j']:>10.0f} {r['slo_violation_rate_pct']:>5.1f}% | "
              f"{r['status']:>6}")


if __name__ == "__main__":
    main()
