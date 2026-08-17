"""
parse_disagg_results.py — Parse B5 benchmark results into a lookup table CSV.

Walks a B5 results directory, extracts bench metrics from vllm bench serve
output files and power from NVML CSV files, and emits a unified
b5_lookup_table.csv.

Experiments recognised:
  A  — A_monolithic_l40s/bench_f{freq}_tp{tp}_il{il}_ol{ol}_r{rate}.txt
  B  — B_monolithic_l4/bench_f{freq}_tp{tp}_il{il}_ol{ol}_r{rate}.txt
  C  — CD_prefill_decode_only/prefill_l40s_f{freq}_il{il}_r{rate}.txt
  D  — CD_prefill_decode_only/decode_l4_f{freq}_ol{ol}_r{rate}.txt
  E  — E_disaggregated/disagg_l40sf{lf}_l4f{lf}_il{il}_ol{ol}_r{rate}.txt
       Power: monitor_l40sf{lf}_l4f{lf}_il{il}_ol{ol}_r{rate}_l40s_gpu0.csv
              monitor_l40sf{lf}_l4f{lf}_il{il}_ol{ol}_r{rate}_l4_gpu0.csv
       Reports avg_power_l40s_w, avg_power_l4_w, avg_power_total_w.
       energy_per_token_j uses total power.

Usage:
    python parse_disagg_results.py --results-dir results/disagg_20241201_120000
    python parse_disagg_results.py --results-dir results/disagg_20241201_120000 \\
        --output my_table.csv --plot
"""

import argparse
import os
import re
import sys
import csv
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# vllm bench serve output parser
# ---------------------------------------------------------------------------

# Regex patterns for each metric line in the bench output
_BENCH_PATTERNS = {
    "successful_requests":  re.compile(r"Successful requests:\s+([\d]+)"),
    "benchmark_duration_s": re.compile(r"Benchmark duration \(s\):\s+([\d.]+)"),
    "throughput_tps":       re.compile(r"Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)"),
    # TTFT — vllm bench default: Mean + Median(P50) + P99
    # with --percentile-metrics ttft also emits P25/P75/P90/P95
    "mean_ttft_ms":         re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)"),
    "p50_ttft_ms":          re.compile(r"(?:Median|P50) TTFT \(ms\):\s+([\d.]+)"),
    "p95_ttft_ms":          re.compile(r"P95 TTFT \(ms\):\s+([\d.]+)"),
    "p99_ttft_ms":          re.compile(r"P99 TTFT \(ms\):\s+([\d.]+)"),
    # TPOT — same structure
    "mean_tpot_ms":         re.compile(r"Mean TPOT \(ms\):\s+([\d.]+)"),
    "p50_tpot_ms":          re.compile(r"(?:Median|P50) TPOT \(ms\):\s+([\d.]+)"),
    "p95_tpot_ms":          re.compile(r"P95 TPOT \(ms\):\s+([\d.]+)"),
    "p99_tpot_ms":          re.compile(r"P99 TPOT \(ms\):\s+([\d.]+)"),
}


def parse_bench_file(path: str) -> Optional[Dict]:
    """
    Parse a vllm bench serve output .txt file.

    Returns a dict with keys from _BENCH_PATTERNS, or None if the file
    does not look like a completed bench run.
    """
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as exc:
        warnings.warn(f"Cannot read bench file {path}: {exc}")
        return None

    if "Serving Benchmark Result" not in text:
        # File exists but run did not complete — skip
        return None

    result = {}
    for key, pat in _BENCH_PATTERNS.items():
        m = pat.search(text)
        if m:
            result[key] = float(m.group(1))
        else:
            result[key] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Power CSV parser
# ---------------------------------------------------------------------------

def parse_power_csv(path: str, use_last_fraction: float = 0.8) -> Optional[float]:
    """
    Compute average power (W) from an NVML power CSV.

    Preferred method: derive avg power from hardware energy counter
    (energy_consumed_mj column, added in newer script version).
      avg_power_w = Δenergy_mj / (Δtime_s × 1000)  per GPU, then averaged.

    Fallback: average power_w over the last `use_last_fraction` of the trace.

    Returns average_power_w, or None if the file is missing / unreadable.
    """
    if not os.path.isfile(path):
        return None

    try:
        df = pd.read_csv(path, header=0, on_bad_lines="skip")
    except Exception as exc:
        warnings.warn(f"Cannot parse power CSV {path}: {exc}")
        return None

    if df.empty:
        return None

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # --- Preferred: hardware energy counter ---
    # Support both column naming conventions
    energy_col = ("total_energy_mj" if "total_energy_mj" in df.columns
                  else "energy_consumed_mj" if "energy_consumed_mj" in df.columns
                  else None)
    ts_col = ("timestamp" if "timestamp" in df.columns
              else "timestamp_s" if "timestamp_s" in df.columns
              else None)
    if energy_col and ts_col:
        df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
        df[ts_col]     = pd.to_numeric(df[ts_col],     errors="coerce")
        df = df.dropna(subset=[energy_col, ts_col])

        if not df.empty:
            # Compute per-GPU Δenergy / Δtime
            power_estimates = []
            # Single-GPU CSVs have no gpu_id column — treat as one group
            if "gpu_id" in df.columns:
                groups = df.groupby("gpu_id")
            else:
                groups = [("gpu0", df)]
            for gpu_id, grp in groups:
                grp = grp.sort_values(ts_col)
                delta_e = grp[energy_col].iloc[-1] - grp[energy_col].iloc[0]
                delta_t = grp[ts_col].iloc[-1]     - grp[ts_col].iloc[0]
                if delta_t > 1.0 and delta_e > 0:
                    power_estimates.append((delta_e / 1000.0) / delta_t)  # mJ→J / s = W
            if power_estimates:
                return float(np.mean(power_estimates))

    # --- Fallback: sampled power_w average ---
    if "power_w" not in df.columns:
        return None

    df["power_w"] = pd.to_numeric(df["power_w"], errors="coerce")
    df = df.dropna(subset=["power_w"])
    if df.empty:
        return None

    # Use only the last fraction of the trace to avoid startup transients
    ts_fallback = "timestamp" if "timestamp" in df.columns else "timestamp_s"
    if ts_fallback in df.columns:
        t_col = pd.to_numeric(df[ts_fallback], errors="coerce")
        t_min, t_max = t_col.min(), t_col.max()
        duration = t_max - t_min
        if duration > 0:
            cutoff = t_min + (1.0 - use_last_fraction) * duration
            df = df[t_col >= cutoff]

    if df.empty:
        return None

    return float(df["power_w"].mean())


# ---------------------------------------------------------------------------
# Filename parsers for each experiment type
# ---------------------------------------------------------------------------

def _parse_ab_bench_name(fname: str, gpu_type: str) -> Optional[Dict]:
    """
    Parse bench_f{freq}_tp{tp}_il{il}_ol{ol}_r{rate}.txt
    Returns metadata dict or None.
    """
    m = re.match(
        r"bench_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r([\d.]+)\.txt$", fname
    )
    if not m:
        return None
    return {
        "gpu_type":  gpu_type,
        "freq_mhz":  int(m.group(1)),
        "tp":        int(m.group(2)),
        "input_len": int(m.group(3)),
        "output_len": int(m.group(4)),
        "rate_rps":  float(m.group(5)),
    }


def _parse_c_bench_name(fname: str) -> Optional[Dict]:
    """
    Parse prefill_l40s_f{freq}_il{il}_r{rate}.txt
    Experiment C: prefill-only on L40S (output_len fixed at 1).
    """
    m = re.match(r"prefill_l40s_f(\d+)_il(\d+)_r([\d.]+)\.txt$", fname)
    if not m:
        return None
    return {
        "gpu_type":  "l40s",
        "freq_mhz":  int(m.group(1)),
        "tp":        1,
        "input_len": int(m.group(2)),
        "output_len": 1,
        "rate_rps":  float(m.group(3)),
    }


def _parse_d_bench_name(fname: str) -> Optional[Dict]:
    """
    Parse decode_l4_f{freq}_ol{ol}_r{rate}.txt
    Experiment D: decode-only on L4 (input_len fixed at 2).
    """
    m = re.match(r"decode_l4_f(\d+)_ol(\d+)_r([\d.]+)\.txt$", fname)
    if not m:
        return None
    return {
        "gpu_type":  "l4",
        "freq_mhz":  int(m.group(1)),
        "tp":        1,
        "input_len": 2,
        "output_len": int(m.group(2)),
        "rate_rps":  float(m.group(3)),
    }


# ---------------------------------------------------------------------------
# Power file locator for A/B experiments
# ---------------------------------------------------------------------------

def _find_power_csv_ab(exp_dir: str, gpu_type: str,
                        freq_mhz: int, tp: int,
                        input_len: int = 0, output_len: int = 0,
                        rate_rps: float = 0.0) -> Optional[str]:
    """Return path to the per-workload monitor CSV for A/B experiments.

    Tries the per-workload pattern first:
      monitor_f{freq}_tp{tp}_il{il}_ol{ol}_r{rate}_gpu0.csv
    Falls back to the legacy consolidated pattern:
      power_{gpu_type}_f{freq}_tp{tp}.csv
    """
    if input_len and output_len and rate_rps:
        rate_str = str(int(rate_rps)) if rate_rps == int(rate_rps) else str(rate_rps)
        stem = f"monitor_f{freq_mhz}_tp{tp}_il{input_len}_ol{output_len}_r{rate_str}"
        # Try gpu0 first, then search for any gpu index (tp2 may use gpu2, gpu4, etc.)
        for gpu_idx in range(8):
            fname = f"{stem}_gpu{gpu_idx}.csv"
            full = os.path.join(exp_dir, fname)
            if os.path.isfile(full):
                return full
    # Legacy fallback
    fname = f"power_{gpu_type}_f{freq_mhz}_tp{tp}.csv"
    full = os.path.join(exp_dir, fname)
    return full if os.path.isfile(full) else None


def _find_power_csv_c(exp_dir: str, freq_mhz: int,
                       input_len: int = 0, rate_rps: float = 0.0) -> Optional[str]:
    """Return path to monitor_prefill_f{freq}_il{il}_r{rate}_gpu0.csv."""
    if input_len and rate_rps:
        rate_str = str(int(rate_rps)) if rate_rps == int(rate_rps) else str(rate_rps)
        fname = f"monitor_prefill_f{freq_mhz}_il{input_len}_r{rate_str}_gpu0.csv"
        full = os.path.join(exp_dir, fname)
        if os.path.isfile(full):
            return full
    # Legacy fallback
    fname = f"power_l40s_prefill_f{freq_mhz}.csv"
    full = os.path.join(exp_dir, fname)
    return full if os.path.isfile(full) else None


def _find_power_csv_d(exp_dir: str, freq_mhz: int,
                       output_len: int = 0, rate_rps: float = 0.0) -> Optional[str]:
    """Return path to monitor_decode_f{freq}_ol{ol}_r{rate}_gpu0.csv."""
    if output_len and rate_rps:
        rate_str = str(int(rate_rps)) if rate_rps == int(rate_rps) else str(rate_rps)
        fname = f"monitor_decode_f{freq_mhz}_ol{output_len}_r{rate_str}_gpu0.csv"
        full = os.path.join(exp_dir, fname)
        if os.path.isfile(full):
            return full
    # Legacy fallback
    fname = f"power_l4_decode_f{freq_mhz}.csv"
    full = os.path.join(exp_dir, fname)
    return full if os.path.isfile(full) else None


# ---------------------------------------------------------------------------
# Experiment E helpers
# ---------------------------------------------------------------------------

def _parse_e_bench_name(fname: str) -> Optional[Dict]:
    """
    Parse disagg_l40sf{l40s_freq}_l4f{l4_freq}_il{il}_ol{ol}_r{rate}.txt
    Returns metadata dict or None.
    """
    m = re.match(
        r"disagg_l40sf(\d+)_l4f(\d+)_il(\d+)_ol(\d+)_r([\d.]+)\.txt$", fname
    )
    if not m:
        return None
    return {
        "l40s_freq_mhz": int(m.group(1)),
        "l4_freq_mhz":   int(m.group(2)),
        "input_len":     int(m.group(3)),
        "output_len":    int(m.group(4)),
        "rate_rps":      float(m.group(5)),
        "tp":            1,
    }


def _find_power_csvs_e(exp_dir: str, l40s_freq: int, l4_freq: int,
                        il: int, ol: int, rate: float) -> Dict[str, Optional[str]]:
    """
    Return paths for both L40S and L4 monitor CSVs for an E workload.
    Pattern: monitor_l40sf{lf}_l4f{lf}_il{il}_ol{ol}_r{rate}_{node}_gpu0.csv
    rate is stored as int in the filename when it has no decimal part.
    """
    rate_str = str(int(rate)) if rate == int(rate) else str(rate)
    stem = f"monitor_l40sf{l40s_freq}_l4f{l4_freq}_il{il}_ol{ol}_r{rate_str}"
    l40s_csv = os.path.join(exp_dir, f"{stem}_l40s_gpu0.csv")
    l4_csv   = os.path.join(exp_dir, f"{stem}_l4_gpu0.csv")
    return {
        "l40s": l40s_csv if os.path.isfile(l40s_csv) else None,
        "l4":   l4_csv   if os.path.isfile(l4_csv)   else None,
    }


# ---------------------------------------------------------------------------
# Walk the results directory
# ---------------------------------------------------------------------------

def collect_records(results_dir: str) -> List[Dict]:
    """
    Walk B5 results directory and collect all parsed records.

    Returns a list of dicts, one per bench file that was successfully parsed.
    """
    results_dir = os.path.abspath(results_dir)
    records: List[Dict] = []

    # ---- Experiment A ----
    exp_a_dir = os.path.join(results_dir, "A_monolithic_l40s")
    if os.path.isdir(exp_a_dir):
        for fname in os.listdir(exp_a_dir):
            meta = _parse_ab_bench_name(fname, "l40s")
            if meta is None:
                continue
            bench_metrics = parse_bench_file(os.path.join(exp_a_dir, fname))
            if bench_metrics is None:
                continue
            power_path = _find_power_csv_ab(
                exp_a_dir, "l40s", meta["freq_mhz"], meta["tp"],
                meta["input_len"], meta["output_len"], meta["rate_rps"],
            )
            avg_power = parse_power_csv(power_path) if power_path else float("nan")
            records.append({
                "exp_type": "A",
                **meta,
                **bench_metrics,
                "avg_power_w": avg_power,
                "power_csv": power_path or "",
            })

    # ---- Experiment B ----
    exp_b_dir = os.path.join(results_dir, "B_monolithic_l4")
    if os.path.isdir(exp_b_dir):
        for fname in os.listdir(exp_b_dir):
            meta = _parse_ab_bench_name(fname, "l4")
            if meta is None:
                continue
            bench_metrics = parse_bench_file(os.path.join(exp_b_dir, fname))
            if bench_metrics is None:
                continue
            power_path = _find_power_csv_ab(
                exp_b_dir, "l4", meta["freq_mhz"], meta["tp"],
                meta["input_len"], meta["output_len"], meta["rate_rps"],
            )
            avg_power = parse_power_csv(power_path) if power_path else float("nan")
            records.append({
                "exp_type": "B",
                **meta,
                **bench_metrics,
                "avg_power_w": avg_power,
                "power_csv": power_path or "",
            })

    # ---- Experiments C + D ----
    exp_cd_dir = os.path.join(results_dir, "CD_prefill_decode_only")
    if os.path.isdir(exp_cd_dir):
        for fname in os.listdir(exp_cd_dir):
            # Experiment C
            meta = _parse_c_bench_name(fname)
            if meta is not None:
                bench_metrics = parse_bench_file(os.path.join(exp_cd_dir, fname))
                if bench_metrics is not None:
                    power_path = _find_power_csv_c(exp_cd_dir, meta["freq_mhz"],
                                                   meta["input_len"], meta["rate_rps"])
                    avg_power = parse_power_csv(power_path) if power_path else float("nan")
                    records.append({
                        "exp_type": "C",
                        **meta,
                        **bench_metrics,
                        "avg_power_w": avg_power,
                        "power_csv": power_path or "",
                    })
                continue

            # Experiment D
            meta = _parse_d_bench_name(fname)
            if meta is not None:
                bench_metrics = parse_bench_file(os.path.join(exp_cd_dir, fname))
                if bench_metrics is not None:
                    power_path = _find_power_csv_d(exp_cd_dir, meta["freq_mhz"],
                                                   meta["output_len"], meta["rate_rps"])
                    avg_power = parse_power_csv(power_path) if power_path else float("nan")
                    records.append({
                        "exp_type": "D",
                        **meta,
                        **bench_metrics,
                        "avg_power_w": avg_power,
                        "power_csv": power_path or "",
                    })

    # ---- Experiment E ----
    exp_e_dir = os.path.join(results_dir, "E_disaggregated")
    if os.path.isdir(exp_e_dir):
        for fname in os.listdir(exp_e_dir):
            meta = _parse_e_bench_name(fname)
            if meta is None:
                continue
            bench_metrics = parse_bench_file(os.path.join(exp_e_dir, fname))
            if bench_metrics is None:
                continue
            power_paths = _find_power_csvs_e(
                exp_e_dir,
                meta["l40s_freq_mhz"], meta["l4_freq_mhz"],
                meta["input_len"], meta["output_len"], meta["rate_rps"],
            )
            avg_power_l40s = parse_power_csv(power_paths["l40s"]) if power_paths["l40s"] else float("nan")
            avg_power_l4   = parse_power_csv(power_paths["l4"])   if power_paths["l4"]   else float("nan")
            # Total power: sum of both nodes (NaN if either is missing)
            if not (avg_power_l40s != avg_power_l40s or avg_power_l4 != avg_power_l4):
                avg_power_total = avg_power_l40s + avg_power_l4
            else:
                avg_power_total = float("nan")
            records.append({
                "exp_type":         "E",
                "gpu_type":         "l40s+l4",
                "freq_mhz":         meta["l40s_freq_mhz"],  # L40S freq (primary axis)
                **meta,
                **bench_metrics,
                "avg_power_l40s_w": avg_power_l40s,
                "avg_power_l4_w":   avg_power_l4,
                "avg_power_w":      avg_power_total,
                "power_csv":        f"{power_paths['l40s'] or ''}|{power_paths['l4'] or ''}",
            })

    return records


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print key metrics grouped by (gpu_type, exp_type)."""
    if df.empty:
        print("No records to summarise.")
        return

    metrics = ["p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
               "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
               "throughput_tps", "avg_power_w", "energy_per_token_j"]
    # Only include columns that are present (some may be NaN-only)
    metrics = [m for m in metrics if m in df.columns]

    grouped = df.groupby(["gpu_type", "exp_type"])[metrics].agg(
        lambda x: f"{x.mean():.1f} ± {x.std():.1f}"
    )
    print("\n=== B5 Results Summary (mean ± std across configs) ===")
    print(grouped.to_string())
    print(f"\nTotal records: {len(df)}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, output_path: str) -> None:
    """
    Generate a 4-subplot figure:
      1. P99 TTFT vs freq_mhz
      2. P99 TPOT vs freq_mhz
      3. avg_power_w vs freq_mhz
      4. Energy-per-token (J/tok) vs freq_mhz
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping plots.")
        return

    if df.empty:
        warnings.warn("Empty dataframe; skipping plots.")
        return

    # Compute energy-per-token (J/tok) = avg_power_w / throughput_tps
    df = df.copy()
    df["energy_per_token_j"] = df["avg_power_w"] / df["throughput_tps"].replace(0, float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("B5 Benchmark: DVFS Sweep Results", fontsize=14)

    # Colour / marker per (gpu_type, exp_type)
    color_map = {
        "l40s_A": ("tab:blue",   "o"),
        "l40s_C": ("tab:cyan",   "^"),
        "l4_B":   ("tab:orange", "s"),
        "l4_D":   ("tab:red",    "v"),
    }

    plot_specs = [
        (axes[0, 0], "p99_ttft_ms",       "P99 TTFT (ms)"),
        (axes[0, 1], "p99_tpot_ms",       "P99 TPOT (ms)"),
        (axes[1, 0], "avg_power_w",        "Avg Power (W)"),
        (axes[1, 1], "energy_per_token_j", "Energy per Token (J/tok)"),
    ]

    for ax, metric, ylabel in plot_specs:
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        for (gpu_type, exp_type), grp in df.groupby(["gpu_type", "exp_type"]):
            key = f"{gpu_type}_{exp_type}"
            color, marker = color_map.get(key, ("gray", "x"))
            # Average metric per freq_mhz for this group
            agg = (grp.groupby("freq_mhz")[metric]
                      .mean()
                      .dropna()
                      .sort_index())
            if agg.empty:
                continue
            label = f"{gpu_type.upper()} Exp-{exp_type}"
            ax.plot(agg.index, agg.values, marker=marker, color=color,
                    label=label, linewidth=1.5, markersize=5)

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse B5 disaggregated benchmark results into a CSV lookup table."
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to the B5 results directory (e.g. results/disagg_20241201_120000)"
    )
    parser.add_argument(
        "--output", default="b5_lookup_table.csv",
        help="Output CSV path (default: b5_lookup_table.csv)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib figure saved as b5_lookup_table_plots.png"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results-dir does not exist: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {args.results_dir}")
    records = collect_records(args.results_dir)

    if not records:
        print("WARNING: No completed bench files found. Check the results directory.")
        sys.exit(0)

    # Build dataframe and enforce column order
    col_order = [
        "exp_type", "gpu_type", "freq_mhz", "tp", "input_len", "output_len",
        "rate_rps",
        # Exp E extra: per-node frequencies
        "l40s_freq_mhz", "l4_freq_mhz",
        "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
        "throughput_tps", "benchmark_duration_s", "successful_requests",
        # Power: avg_power_w = total; E also has per-node breakdown
        "avg_power_w", "avg_power_l40s_w", "avg_power_l4_w",
        "energy_per_token_j", "power_csv",
    ]
    df = pd.DataFrame(records)
    # Compute energy-per-token: avg_power_w / throughput_tps (W / tok/s = J/tok)
    df["energy_per_token_j"] = (
        df["avg_power_w"] / df["throughput_tps"].replace(0, float("nan"))
    ).round(6)
    # Add any missing columns as NaN
    for col in col_order:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[col_order]

    # Sort for readability
    df = df.sort_values(
        ["exp_type", "gpu_type", "freq_mhz", "l40s_freq_mhz", "l4_freq_mhz",
         "tp", "input_len", "output_len", "rate_rps"],
        na_position="first",
    ).reset_index(drop=True)

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} records to: {args.output}")

    print_summary(df)

    if args.plot:
        plot_path = os.path.splitext(args.output)[0] + "_plots.png"
        make_plots(df, plot_path)


if __name__ == "__main__":
    main()
