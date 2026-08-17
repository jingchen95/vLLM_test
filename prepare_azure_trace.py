"""
B6: Azure LLM Production Trace Download and Preprocessing.

Downloads the Azure LLM Inference Dataset (2023) and converts it to:
  1. vllm bench serve format (JSON dataset file)
  2. Our internal Request trace format (for jsep_simulator.py)
  3. Summary statistics (arrival rate distribution, token length distribution)

Dataset:
  Azure LLM Inference Trace (2023), released with:
  "Characterization of Large Language Model Development in the Datacenter"
  Repository: https://github.com/Azure/AzurePublicDataset

  Two workload types:
    conv  — conversational (shorter prompts, shorter outputs)
    code  — coding assistant (longer prompts, longer outputs)

Usage:
  # Download and preprocess both workload types
  python prepare_azure_trace.py --download

  # Preprocess already-downloaded files
  python prepare_azure_trace.py \
      --conv-file data/AzureLLMInferenceTrace_conv.csv \
      --code-file data/AzureLLMInferenceTrace_code.csv

  # Generate only vllm bench format for a specific duration
  python prepare_azure_trace.py --download --duration 600 --scale-rate 10
"""

import argparse
import csv
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Dataset URLs (Azure Public Dataset GitHub)
# ---------------------------------------------------------------------------
AZURE_BASE = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset"
    "/master/data"
)
DATASET_FILES = {
    "conv": f"{AZURE_BASE}/AzureLLMInferenceTrace_conv.csv",
    "code": f"{AZURE_BASE}/AzureLLMInferenceTrace_code.csv",
}

# Expected CSV columns (verify after download)
# TIMESTAMP    : Unix timestamp (seconds) of request arrival
# ContextTokens: Number of input/prompt tokens
# GeneratedTokens: Number of output tokens generated
COL_TIMESTAMP = "TIMESTAMP"
COL_CONTEXT   = "ContextTokens"
COL_GENERATED = "GeneratedTokens"


@dataclass
class Request:
    arrival_time: float   # seconds since trace start
    input_len:   int
    output_len:  int


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(output_dir: str = "data") -> dict:
    """Download Azure LLM trace files."""
    os.makedirs(output_dir, exist_ok=True)
    local_paths = {}

    for name, url in DATASET_FILES.items():
        out_path = os.path.join(output_dir, f"AzureLLMInferenceTrace_{name}.csv")
        if os.path.exists(out_path):
            print(f"  [{name}] Already exists: {out_path}")
        else:
            print(f"  [{name}] Downloading from {url} ...")
            try:
                urllib.request.urlretrieve(url, out_path)
                print(f"  [{name}] Saved to {out_path}")
            except Exception as e:
                print(f"  [{name}] Download failed: {e}")
                print(f"  [{name}] Manual download:")
                print(f"         wget '{url}' -O {out_path}")
                out_path = None
        local_paths[name] = out_path

    return local_paths


# ---------------------------------------------------------------------------
# Parse CSV
# ---------------------------------------------------------------------------

def _parse_ts(s: str) -> float:
    """Parse an Azure TIMESTAMP to epoch seconds.

    Accepts either a numeric Unix timestamp or a datetime string such as
    "2023-11-16 18:15:46.6805900" (the released dataset format, with up to 7
    fractional digits) or an ISO string with a timezone offset.
    """
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    import datetime as _dt
    iso = s.replace("Z", "+00:00")
    # Trim fractional seconds to 6 digits (microseconds) for parsers.
    if "." in iso:
        head, rest = iso.split(".", 1)
        digits = ""
        i = 0
        while i < len(rest) and rest[i].isdigit():
            digits += rest[i]; i += 1
        iso = f"{head}.{digits[:6].ljust(6, '0')}{rest[i:]}"
    try:
        return _dt.datetime.fromisoformat(iso).timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return _dt.datetime.strptime(iso, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {s!r}")


def parse_azure_csv(filepath: str) -> List[Request]:
    """Parse Azure trace CSV into Request list."""
    requests = []
    min_ts = None

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        print(f"  Columns: {header}")

        # Detect column names (dataset may use different capitalizations)
        ts_col = _find_col(header, [COL_TIMESTAMP, "Timestamp", "timestamp", "time"])
        ctx_col = _find_col(header, [COL_CONTEXT, "context_tokens", "input_tokens",
                                      "prompt_tokens", "InputTokens"])
        gen_col = _find_col(header, [COL_GENERATED, "generated_tokens", "output_tokens",
                                      "completion_tokens", "OutputTokens"])

        if not all([ts_col, ctx_col, gen_col]):
            print(f"  ERROR: Could not find required columns.")
            print(f"  Looking for: timestamp, context_tokens, generated_tokens")
            print(f"  Found: {header}")
            return []

        print(f"  Using columns: ts='{ts_col}', ctx='{ctx_col}', gen='{gen_col}'")

        for row in reader:
            try:
                ts  = _parse_ts(row[ts_col])
                ctx = int(float(row[ctx_col]))
                gen = int(float(row[gen_col]))
            except (ValueError, KeyError):
                continue

            if ctx <= 0 or gen <= 0:
                continue

            if min_ts is None:
                min_ts = ts
            requests.append(Request(
                arrival_time=ts - min_ts,
                input_len=ctx,
                output_len=gen,
            ))

    # Sort by arrival time
    requests.sort(key=lambda r: r.arrival_time)
    return requests


def _find_col(header, candidates):
    """Find first matching column name (case-insensitive)."""
    if header is None:
        return None
    header_lower = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in header_lower:
            return header_lower[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------

def analyze_trace(requests: List[Request], name: str):
    """Print summary statistics of the trace."""
    if not requests:
        print(f"  [{name}] Empty trace")
        return

    duration = requests[-1].arrival_time - requests[0].arrival_time
    n = len(requests)

    ils = [r.input_len  for r in requests]
    ols = [r.output_len for r in requests]

    # Compute per-window arrival rates (60s windows)
    window = 60.0
    rates = []
    t0 = requests[0].arrival_time
    t_end = requests[-1].arrival_time
    t = t0
    while t < t_end:
        count = sum(1 for r in requests if t <= r.arrival_time < t + window)
        rates.append(count / window)
        t += window

    print(f"\n  [{name}] Trace summary:")
    print(f"    Requests:      {n:,}")
    print(f"    Duration:      {duration/3600:.2f} hours")
    print(f"    Arrival rate:")
    print(f"      Mean:        {np.mean(rates):.2f} req/s")
    print(f"      P50:         {np.percentile(rates, 50):.2f} req/s")
    print(f"      P95:         {np.percentile(rates, 95):.2f} req/s")
    print(f"      Max:         {np.max(rates):.2f} req/s")
    print(f"    Input tokens:")
    print(f"      Mean:        {np.mean(ils):.0f}")
    print(f"      P50:         {np.percentile(ils, 50):.0f}")
    print(f"      P95:         {np.percentile(ils, 95):.0f}")
    print(f"      Max:         {np.max(ils):.0f}")
    print(f"    Output tokens:")
    print(f"      Mean:        {np.mean(ols):.0f}")
    print(f"      P50:         {np.percentile(ols, 50):.0f}")
    print(f"      P95:         {np.percentile(ols, 95):.0f}")
    print(f"      Max:         {np.max(ols):.0f}")
    print(f"    D_prefill (avg tok/s): {np.mean(rates)*np.mean(ils):.0f}")
    print(f"    D_decode  (avg tok/s): {np.mean(rates)*np.mean(ols):.0f}")
    print(f"    Prefill/Decode ratio:  {np.mean(ils)/np.mean(ols):.2f}")


# ---------------------------------------------------------------------------
# Trace slicing: extract a representative window
# ---------------------------------------------------------------------------

def slice_trace(requests: List[Request],
                duration_s: float = 3600.0,
                start_frac: float = 0.1) -> List[Request]:
    """
    Extract a contiguous window from the trace.
    start_frac: starting position as fraction of total trace (skip warmup).
    """
    if not requests:
        return []

    total_duration = requests[-1].arrival_time
    start_time = total_duration * start_frac
    end_time   = start_time + duration_s

    sliced = [r for r in requests
              if start_time <= r.arrival_time < end_time]

    # Renormalize arrival times to start at 0
    if sliced:
        t0 = sliced[0].arrival_time
        sliced = [Request(r.arrival_time - t0, r.input_len, r.output_len)
                  for r in sliced]

    return sliced


# ---------------------------------------------------------------------------
# Rate scaling: replay at a target mean rate
# ---------------------------------------------------------------------------

def scale_trace_rate(requests: List[Request],
                     target_rate_rps: float) -> List[Request]:
    """
    Scale arrival times to achieve a target mean request rate.
    Preserves burstiness pattern (shape of inter-arrival distribution).
    """
    if not requests or len(requests) < 2:
        return requests

    current_duration = requests[-1].arrival_time
    current_rate = len(requests) / current_duration
    scale_factor = current_rate / target_rate_rps

    return [Request(r.arrival_time * scale_factor, r.input_len, r.output_len)
            for r in requests]


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

def save_vllm_bench_dataset(requests: List[Request], output_path: str):
    """
    Save in vllm bench serve dataset format.
    Each entry: {"conversations": [{"role": "user", "content": "<PAD>*input_len"}],
                 "output_len": N}
    This format is compatible with --dataset-name custom in recent vLLM versions.

    Alternatively saved as ShareGPT format which vllm bench supports directly:
    [{"id": "...", "conversations": [{"from": "human", "value": "..."},
                                      {"from": "gpt",   "value": "..."}]}]
    """
    dataset = []
    for i, r in enumerate(requests):
        # Approximate prompt by repeating a short phrase to reach target token count
        # vllm bench will tokenize this — actual token count may vary slightly
        prompt = ("hello " * math.ceil(r.input_len / 2))[:r.input_len * 6]
        response = ("word " * math.ceil(r.output_len / 2))[:r.output_len * 6]

        dataset.append({
            "id": f"azure-{i:06d}",
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt",   "value": response},
            ]
        })

    with open(output_path, "w") as f:
        json.dump(dataset, f)
    print(f"  Saved vllm bench dataset: {output_path} ({len(dataset)} entries)")


def save_arrival_times(requests: List[Request], output_path: str):
    """
    Save arrival times + token counts as CSV.
    Used by our jsep_simulator.py trace loader.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arrival_time_s", "input_len", "output_len"])
        for r in requests:
            writer.writerow([f"{r.arrival_time:.4f}", r.input_len, r.output_len])
    print(f"  Saved arrival trace: {output_path} ({len(requests)} requests)")


def save_rate_profile(requests: List[Request], output_path: str,
                      window_s: float = 60.0):
    """Save per-window rate profile for plotting."""
    if not requests:
        return
    t_end = requests[-1].arrival_time
    rows = []
    t = 0.0
    while t < t_end:
        window_reqs = [r for r in requests if t <= r.arrival_time < t + window_s]
        n = len(window_reqs)
        rate = n / window_s
        avg_il = np.mean([r.input_len  for r in window_reqs]) if window_reqs else 0
        avg_ol = np.mean([r.output_len for r in window_reqs]) if window_reqs else 0
        d_pf = rate * avg_il
        d_dc = rate * avg_ol
        rows.append([t, rate, avg_il, avg_ol, d_pf, d_dc])
        t += window_s

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_start_s", "rate_rps", "avg_input_len",
                          "avg_output_len", "d_prefill_toks", "d_decode_toks"])
        writer.writerows(rows)
    print(f"  Saved rate profile: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess Azure LLM production trace")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from Azure GitHub")
    parser.add_argument("--conv-file", default=None,
                        help="Path to already-downloaded conv trace CSV")
    parser.add_argument("--code-file", default=None,
                        help="Path to already-downloaded code trace CSV")
    parser.add_argument("--data-dir", default="data",
                        help="Directory to store downloaded/processed data")
    parser.add_argument("--output-dir", default="traces",
                        help="Output directory for processed trace files")
    parser.add_argument("--duration", type=float, default=3600.0,
                        help="Duration of trace window to extract (seconds, default 1h)")
    parser.add_argument("--scale-rate", type=float, default=None,
                        help="Scale to target mean request rate (req/s). "
                             "None = use original rate")
    parser.add_argument("--workload", choices=["conv", "code", "both"],
                        default="both")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: locate CSV files
    local_paths = {}
    if args.download:
        print("Downloading Azure LLM traces...")
        local_paths = download_dataset(args.data_dir)
    if args.conv_file:
        local_paths["conv"] = args.conv_file
    if args.code_file:
        local_paths["code"] = args.code_file

    if not local_paths:
        print("ERROR: No input files. Use --download or --conv-file/--code-file.")
        sys.exit(1)

    # Step 2: process each workload type
    workloads = (["conv", "code"] if args.workload == "both"
                 else [args.workload])

    for wl_type in workloads:
        csv_path = local_paths.get(wl_type)
        if not csv_path or not os.path.exists(csv_path):
            print(f"\n[{wl_type}] File not found: {csv_path}. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {wl_type}")
        print(f"{'='*60}")

        # Parse
        print(f"  Parsing {csv_path}...")
        all_requests = parse_azure_csv(csv_path)
        if not all_requests:
            print(f"  [{wl_type}] Failed to parse. Skipping.")
            continue

        # Full trace stats
        analyze_trace(all_requests, f"{wl_type} (full)")

        # Slice to requested duration
        sliced = slice_trace(all_requests,
                             duration_s=args.duration,
                             start_frac=0.1)  # skip first 10% as warmup
        print(f"\n  Sliced to {args.duration/3600:.1f}h window: {len(sliced)} requests")

        # Optional rate scaling
        if args.scale_rate:
            sliced = scale_trace_rate(sliced, args.scale_rate)
            print(f"  Scaled to {args.scale_rate} req/s mean rate")

        analyze_trace(sliced, f"{wl_type} (sliced)")

        # Save outputs
        prefix = os.path.join(args.output_dir, f"azure_{wl_type}")

        save_arrival_times(sliced, f"{prefix}_arrival.csv")
        save_vllm_bench_dataset(sliced, f"{prefix}_vllm_dataset.json")
        save_rate_profile(sliced, f"{prefix}_rate_profile.csv")

        # Also save a shorter "quick test" subset (10 minutes)
        quick = slice_trace(all_requests, duration_s=600.0, start_frac=0.2)
        if args.scale_rate:
            quick = scale_trace_rate(quick, args.scale_rate)
        save_arrival_times(quick,
                           f"{prefix}_10min_arrival.csv")
        save_vllm_bench_dataset(quick,
                                f"{prefix}_10min_vllm_dataset.json")

    print(f"\n{'='*60}")
    print("Done. Files written to:", args.output_dir)
    print("")
    print("To use with vllm bench serve:")
    print("  vllm bench serve \\")
    print("    --backend openai \\")
    print("    --base-url http://localhost:8000 \\")
    print("    --model mistralai/Mistral-7B-v0.1 \\")
    print("    --dataset-name sharegpt \\")
    print(f"    --dataset-path traces/azure_conv_vllm_dataset.json \\")
    print("    --num-prompts 1000")
    print("")
    print("To load in jsep_simulator.py (B6 production trace eval):")
    print("  see load_azure_trace() in jsep_traces.py")


if __name__ == "__main__":
    main()
