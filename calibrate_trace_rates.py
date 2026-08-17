#!/usr/bin/env python3
"""
calibrate_trace_rates.py

Step 1 of the SWEEP-LLM evaluation calibration pipeline.

Consumes existing disagg benchmark results (A/B/CD/E experiments) and
produces four CSV artifacts plus a console summary:

  1. reference_capacity.csv   — per-GPU and pool-extrapolated C_pf^ref and
                                C_dc^ref across measured shapes.
  2. feasibility_by_config.csv — per-record SLO and saturation flags.
  3. max_sustainable_rate.csv  — per (exp, pool, tp, freq, il, ol) × SLO,
                                 the largest configured rate that was both
                                 SLO-compliant and not saturated.
  4. trace_peak_rates.csv      — per (trace, SLO), the feasible aggregate λ
                                 under the static-disagg baseline, plus
                                 D_pf, D_dc, (x, y), and classifier state.

Reference:
  - paper/evaluation_rewrite.tex (SLO levels, trace mixes T1–T4)
  - paper/design_rewrite.tex    (classifier x/y, C_pf^ref / C_dc^ref)
  - paper/models.tex            (capacity model, plateau definition)

Shape mapping (paper → v2 data):
  SS  (64/128)   ≈ 128/128   (approx; no il=64 in v2 for short ol)
  SL  (64/512)   ≈ 128/512   (approx; same)
  LS  (1024/128) = 1024/128  (exact)
  LL  (1024/512) ≈ no exact match; decode-side proxied by 128/512 (SL)
                   and prefill-side by 1024/128 (LS); flagged as approx

Usage:
  python3 calibrate_trace_rates.py \
      --results-dir results/disagg_20260321_v2 \
      --out-dir calibration_out/

Requires only Python 3.9+ stdlib.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------
# Constants from paper
# --------------------------------------------------------------------------

SLO_LEVELS = {
    "tight":    {"ttft_ms": 200.0,  "tpot_ms": 100.0},
    "moderate": {"ttft_ms": 500.0,  "tpot_ms": 200.0},
    "loose":    {"ttft_ms": 1000.0, "tpot_ms": 400.0},
}

CLUSTER = {
    "l40s": {"gpu_count": 4, "max_freq_mhz": 2520},
    "l4":   {"gpu_count": 8, "max_freq_mhz": 2040},
}

CLASSES = {
    "SS": {"il_paper": 64,   "ol_paper": 128,
           "il_proxy": 128,  "ol_proxy": 128,  "approx": True},
    "SL": {"il_paper": 64,   "ol_paper": 512,
           "il_proxy": 128,  "ol_proxy": 512,  "approx": True},
    "LS": {"il_paper": 1024, "ol_paper": 128,
           "il_proxy": 1024, "ol_proxy": 128,  "approx": False},
    "LL": {"il_paper": 1024, "ol_paper": 512,
           "il_proxy": None, "ol_proxy": None, "approx": True},
}

TRACE_MIXES = {
    "T1":       {"SS": 0.25, "SL": 0.00, "LS": 0.50, "LL": 0.25},
    "T2":       {"SS": 0.25, "SL": 0.50, "LS": 0.00, "LL": 0.25},
    "T3_start": {"SS": 0.25, "SL": 0.00, "LS": 0.50, "LL": 0.25},
    "T3_end":   {"SS": 0.25, "SL": 0.50, "LS": 0.00, "LL": 0.25},
    "T4":       {"SS": 0.25, "SL": 0.25, "LS": 0.25, "LL": 0.25},
}

# A rate is "sustained" only if measured req throughput ≥ this × configured rate.
SATURATION_SAFETY_FRAC = 0.9

# Placeholder classifier thresholds (paper flags these as TBD; see Table 1
# in evaluation_rewrite.tex). Used only for state-coverage verification.
TAU_IMB_DEFAULT  = 2.0
TAU_LOAD_DEFAULT = 0.5

# KV transfer cost per token for synthesized cross-pool (disagg) routes.
# The v2 cluster uses commodity Ethernet (measured ~14,700 µs/tok), which
# is not representative of disagg-serving deployments. The paper assumes
# one of the following reference interconnects instead:
#   5  µs/tok — Infiniband HDR (200 Gbps, as in DistServe/Splitwise)
#   16 µs/tok — PCIe 4.0 with NCCL overhead (current jsep_disagg_scheduler.py)
#   41 µs/tok — 25 GbE (older commercial DC)
# Passed to the script as --tau-kv-us-per-tok (comma-separated for sweep).
TAU_KV_DEFAULTS = [5.0, 16.0, 41.0]

# Mistral-7B KV bytes per token: 32 layers × 2 (K,V) × 8 kv_heads × 128 d_head × 2 B = 131,072 B
MISTRAL_7B_KV_BYTES_PER_TOK = 131_072


# --------------------------------------------------------------------------
# Benchmark file parser
# --------------------------------------------------------------------------

_BENCH_PATTERNS = {
    "successful_requests":  re.compile(r"Successful requests:\s+([\d]+)"),
    "benchmark_duration_s": re.compile(r"Benchmark duration \(s\):\s+([\d.]+)"),
    "throughput_tps":       re.compile(r"Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)"),
    "req_throughput_rps":   re.compile(r"Request throughput \(req/s\):\s+([\d.]+)"),
    "p99_ttft_ms":          re.compile(r"P99 TTFT \(ms\):\s+([\d.]+)"),
    "p99_tpot_ms":          re.compile(r"P99 TPOT \(ms\):\s+([\d.]+)"),
    "mean_ttft_ms":         re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)"),
    "mean_tpot_ms":         re.compile(r"Mean TPOT \(ms\):\s+([\d.]+)"),
}


def parse_bench_file(path):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    if "Serving Benchmark Result" not in text:
        return None
    out = {}
    for key, pat in _BENCH_PATTERNS.items():
        m = pat.search(text)
        out[key] = float(m.group(1)) if m else None
    return out


_RE_AB = re.compile(r"^bench_f(\d+)_tp(\d+)_il(\d+)_ol(\d+)_r([\d.]+)\.txt$")
_RE_C  = re.compile(r"^prefill_l40s_f(\d+)_il(\d+)_r([\d.]+)\.txt$")
_RE_D  = re.compile(r"^decode_l4_f(\d+)_ol(\d+)_r([\d.]+)\.txt$")
_RE_E  = re.compile(r"^disagg_l40sf(\d+)_l4f(\d+)_il(\d+)_ol(\d+)_r([\d.]+)\.txt$")


def collect_records(results_dir):
    records = []
    results_dir = os.path.abspath(results_dir)

    def ingest_ab(dir_name, pool, exp):
        d = os.path.join(results_dir, dir_name)
        if not os.path.isdir(d):
            return
        for fname in sorted(os.listdir(d)):
            m = _RE_AB.match(fname)
            if not m:
                continue
            meta = parse_bench_file(os.path.join(d, fname))
            if meta is None:
                continue
            records.append({
                "exp": exp, "pool": pool,
                "freq_mhz": int(m.group(1)),
                "tp":       int(m.group(2)),
                "il":       int(m.group(3)),
                "ol":       int(m.group(4)),
                "l4_freq_mhz": None,
                "rate_configured_rps": float(m.group(5)),
                **meta,
            })

    ingest_ab("A_monolithic_l40s", "l40s", "A")
    ingest_ab("B_monolithic_l4",   "l4",   "B")

    cd_dir = os.path.join(results_dir, "CD_prefill_decode_only")
    if os.path.isdir(cd_dir):
        for fname in sorted(os.listdir(cd_dir)):
            m = _RE_C.match(fname)
            if m:
                meta = parse_bench_file(os.path.join(cd_dir, fname))
                if meta is not None:
                    records.append({
                        "exp": "C", "pool": "l40s",
                        "freq_mhz": int(m.group(1)), "tp": 1,
                        "il": int(m.group(2)), "ol": 1,
                        "l4_freq_mhz": None,
                        "rate_configured_rps": float(m.group(3)),
                        **meta,
                    })
                continue
            m = _RE_D.match(fname)
            if m:
                meta = parse_bench_file(os.path.join(cd_dir, fname))
                if meta is not None:
                    records.append({
                        "exp": "D", "pool": "l4",
                        "freq_mhz": int(m.group(1)), "tp": 1,
                        "il": 2, "ol": int(m.group(2)),
                        "l4_freq_mhz": None,
                        "rate_configured_rps": float(m.group(3)),
                        **meta,
                    })

    e_dir = os.path.join(results_dir, "E_disaggregated")
    if os.path.isdir(e_dir):
        for fname in sorted(os.listdir(e_dir)):
            m = _RE_E.match(fname)
            if not m:
                continue
            meta = parse_bench_file(os.path.join(e_dir, fname))
            if meta is None:
                continue
            records.append({
                "exp": "E", "pool": "l40s+l4",
                "freq_mhz": int(m.group(1)), "tp": 1,
                "l4_freq_mhz": int(m.group(2)),
                "il": int(m.group(3)), "ol": int(m.group(4)),
                "rate_configured_rps": float(m.group(5)),
                **meta,
            })

    return records


# --------------------------------------------------------------------------
# Feasibility annotation
# --------------------------------------------------------------------------

def annotate_feasibility(records):
    for rec in records:
        p99_ttft = rec.get("p99_ttft_ms")
        p99_tpot = rec.get("p99_tpot_ms")
        for name, slo in SLO_LEVELS.items():
            if p99_ttft is None or p99_tpot is None:
                rec[f"slo_{name}_ok"] = None
            else:
                rec[f"slo_{name}_ok"] = (
                    p99_ttft <= slo["ttft_ms"] and p99_tpot <= slo["tpot_ms"]
                )
        thr = rec.get("req_throughput_rps")
        cfg = rec["rate_configured_rps"]
        rec["saturated"] = (
            None if thr is None else thr < SATURATION_SAFETY_FRAC * cfg
        )
    return records


# --------------------------------------------------------------------------
# Artifact 1: reference capacities
# --------------------------------------------------------------------------

def estimate_reference_capacities(records):
    """
    Per-GPU prefill capacity from Experiment C (prefill-only L40S, ol=1) at
    L40S max freq, TP=1. Per-GPU decode capacity from Experiment D
    (decode-only L4, il=2) at L4 max freq, TP=1. Pool totals are linear
    extrapolations by GPU count — flagged as approximate since the v2
    dataset is single-instance.
    """
    l40s_max = CLUSTER["l40s"]["max_freq_mhz"]
    l4_max   = CLUSTER["l4"]["max_freq_mhz"]

    l40s_pf = defaultdict(float)  # il -> max per-GPU tok/s
    l4_dc   = defaultdict(float)  # ol -> max per-GPU tok/s

    for rec in records:
        thr = rec.get("throughput_tps")
        if thr is None:
            continue
        if rec["exp"] == "C" and rec["freq_mhz"] == l40s_max and rec["tp"] == 1:
            l40s_pf[rec["il"]] = max(l40s_pf[rec["il"]], thr)
        elif rec["exp"] == "D" and rec["freq_mhz"] == l4_max and rec["tp"] == 1:
            l4_dc[rec["ol"]] = max(l4_dc[rec["ol"]], thr)

    rows = []
    for il, per_gpu in sorted(l40s_pf.items()):
        rows.append({
            "phase": "prefill", "pool": "l40s",
            "freq_mhz": l40s_max, "tp": 1,
            "shape_label": f"il={il}, ol=1",
            "il": il, "ol": 1,
            "per_gpu_max_tps": round(per_gpu, 1),
            "pool_total_tps_linear": round(per_gpu * CLUSTER["l40s"]["gpu_count"], 1),
            "n_instances_assumed": CLUSTER["l40s"]["gpu_count"],
            "note": "linear extrapolation from single-instance measurement",
        })
    for ol, per_gpu in sorted(l4_dc.items()):
        rows.append({
            "phase": "decode", "pool": "l4",
            "freq_mhz": l4_max, "tp": 1,
            "shape_label": f"il=2, ol={ol}",
            "il": 2, "ol": ol,
            "per_gpu_max_tps": round(per_gpu, 1),
            "pool_total_tps_linear": round(per_gpu * CLUSTER["l4"]["gpu_count"], 1),
            "n_instances_assumed": CLUSTER["l4"]["gpu_count"],
            "note": "linear extrapolation from single-instance measurement",
        })

    c_pf_per_gpu = max(l40s_pf.values()) if l40s_pf else 0.0
    c_dc_per_gpu = max(l4_dc.values())   if l4_dc   else 0.0
    c_pf_ref = c_pf_per_gpu * CLUSTER["l40s"]["gpu_count"]
    c_dc_ref = c_dc_per_gpu * CLUSTER["l4"]["gpu_count"]
    return rows, c_pf_ref, c_dc_ref


# --------------------------------------------------------------------------
# Artifacts 2 & 3
# --------------------------------------------------------------------------

def build_feasibility_table(records):
    rows = []
    for r in records:
        rows.append({
            "exp": r["exp"], "pool": r["pool"],
            "tp": r["tp"], "freq_mhz": r["freq_mhz"],
            "l4_freq_mhz": r.get("l4_freq_mhz"),
            "il": r["il"], "ol": r["ol"],
            "rate_configured_rps": r["rate_configured_rps"],
            "req_throughput_rps": r.get("req_throughput_rps"),
            "throughput_tps":    r.get("throughput_tps"),
            "p99_ttft_ms":       r.get("p99_ttft_ms"),
            "p99_tpot_ms":       r.get("p99_tpot_ms"),
            "saturated":         r.get("saturated"),
            "slo_tight_ok":      r.get("slo_tight_ok"),
            "slo_moderate_ok":   r.get("slo_moderate_ok"),
            "slo_loose_ok":      r.get("slo_loose_ok"),
        })
    return rows


def max_sustainable_rate(records):
    """
    Key: (exp, pool, tp, freq_mhz, l4_freq_mhz, il, ol). For E, include L4 freq
    so we can pick the max-both-freq reference. For A/B/C/D, l4_freq_mhz is None.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["exp"], r["pool"], r["tp"], r["freq_mhz"],
               r.get("l4_freq_mhz"), r["il"], r["ol"])
        groups[key].append(r)

    rows = []
    for key, recs in sorted(groups.items(), key=lambda kv: tuple(
            (-1 if x is None else x) if isinstance(x, (int, float)) else x for x in kv[0])):
        exp, pool, tp, freq, l4_freq, il, ol = key
        row = {
            "exp": exp, "pool": pool, "tp": tp,
            "freq_mhz": freq, "l4_freq_mhz": l4_freq,
            "il": il, "ol": ol,
            "rates_measured": sorted({r["rate_configured_rps"] for r in recs}),
        }
        for name in SLO_LEVELS:
            feasible = [
                r for r in recs
                if r.get(f"slo_{name}_ok") and r.get("saturated") is False
            ]
            row[f"max_rate_{name}_rps"] = (
                max(r["rate_configured_rps"] for r in feasible) if feasible else None
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Artifact 4: trace peak rates
#
# Two feasibility paths per class:
#   a) Co-located (single-pool) — the class is served end-to-end on one pool.
#      Sourced from A (L40S monolithic) or B (L4 monolithic) at max freq, TP=1.
#      Independent of interconnect (no cross-pool transfer).
#   b) Disaggregated (cross-pool) — prefill on L40S, decode on L4.
#      Synthesized from C (prefill-only L40S, ol=1) + τ_kv × il + D (decode-only
#      L4, il=2) at each pool's max freq, TP=1. Depends on τ_kv (interconnect).
#
# The SWEEP-LLM-best peak λ per class = max of (a) and (b) over whichever is
# SLO-feasible. The trace peak λ = min over classes of (per-class best / π_c).
# --------------------------------------------------------------------------

def lookup_max_rate(sust_rows, *, exp, pool, tp, freq, l4_freq, il, ol, slo_name):
    for r in sust_rows:
        if (r["exp"] == exp and r["pool"] == pool and r["tp"] == tp
                and r["freq_mhz"] == freq and r["l4_freq_mhz"] == l4_freq
                and r["il"] == il and r["ol"] == ol):
            return r[f"max_rate_{slo_name}_rps"]
    return None


def _per_instance_best_thr_under_slo(recs, slo_name, check_ttft=True, check_tpot=True,
                                      extra_ttft_ms=0.0):
    """Over a list of records, return max per-instance req_throughput_rps that
    meets the SLO. Uses actual served throughput, not configured rate.
    extra_ttft_ms adds a fixed overhead to TTFT (for KV transfer cost).
    """
    slo = SLO_LEVELS[slo_name]
    best = None
    for r in recs:
        ttft = r.get("p99_ttft_ms")
        tpot = r.get("p99_tpot_ms")
        thr  = r.get("req_throughput_rps")
        if thr is None:
            continue
        if check_ttft:
            if ttft is None: continue
            if ttft + extra_ttft_ms > slo["ttft_ms"]: continue
        if check_tpot:
            if tpot is None: continue
            if tpot > slo["tpot_ms"]: continue
        if best is None or thr > best:
            best = thr
    return best


def _colocated_rate_limit(records, il, ol, slo_name):
    """Cluster feasible rate for a class served co-located (single pool).
    Picks the better of L40S-only (×4 GPUs) and L4-only (×8 GPUs),
    each using monolithic A or B data at max freq, TP=1.
    """
    l40s_max = CLUSTER["l40s"]["max_freq_mhz"]
    l4_max   = CLUSTER["l4"]["max_freq_mhz"]
    n_a = CLUSTER["l40s"]["gpu_count"]
    n_b = CLUSTER["l4"]["gpu_count"]

    a_recs = [r for r in records
              if r["exp"] == "A" and r["freq_mhz"] == l40s_max
              and r["tp"] == 1 and r["il"] == il and r["ol"] == ol]
    b_recs = [r for r in records
              if r["exp"] == "B" and r["freq_mhz"] == l4_max
              and r["tp"] == 1 and r["il"] == il and r["ol"] == ol]

    a_pi = _per_instance_best_thr_under_slo(a_recs, slo_name)
    b_pi = _per_instance_best_thr_under_slo(b_recs, slo_name)
    a_cluster = a_pi * n_a if a_pi is not None else None
    b_cluster = b_pi * n_b if b_pi is not None else None

    if a_cluster is None and b_cluster is None:
        return None, "no_data"
    if a_cluster is None:
        return b_cluster, f"L4×{n_b}({b_pi:.2f}rps/inst)"
    if b_cluster is None:
        return a_cluster, f"L40S×{n_a}({a_pi:.2f}rps/inst)"
    if a_cluster >= b_cluster:
        return a_cluster, f"L40S×{n_a}({a_pi:.2f}rps/inst); L4×{n_b}={b_cluster:.1f}"
    return b_cluster, f"L4×{n_b}({b_pi:.2f}rps/inst); L40S×{n_a}={a_cluster:.1f}"


def build_cd_lookup(records):
    """Return (c_lookup, d_lookup) dicts keyed by (freq_mhz, shape_size, rate).
    c_lookup: (freq, il, rate) → C record (prefill-only L40S, ol=1)
    d_lookup: (freq, ol, rate) → D record (decode-only L4, il=2)
    """
    c_lookup = {}
    d_lookup = {}
    for r in records:
        if r["exp"] == "C":
            c_lookup[(r["freq_mhz"], r["il"], r["rate_configured_rps"])] = r
        elif r["exp"] == "D":
            d_lookup[(r["freq_mhz"], r["ol"], r["rate_configured_rps"])] = r
    return c_lookup, d_lookup


def _disagg_rate_limit(c_lookup, d_lookup, il, ol, slo_name, tau_kv_us):
    """Cluster feasible rate for a class under cross-pool (disagg) routing,
    synthesized from C and D phase-isolated data at each pool's max freq,
    TP=1, with KV transfer cost = il × tau_kv_us applied to the prefill side.

    Per-instance capacities:
      prefill (L40S):  max req_throughput_rps over C records at (il, f=2520)
                       such that (C.p99_ttft + il × τ_kv / 1000) ≤ SLO_ttft.
                       No TPOT check (C has no decode phase).
      decode  (L4):    max req_throughput_rps over D records at (ol, f=2040)
                       such that D.p99_tpot ≤ SLO_tpot.
                       No TTFT check (D's decode-only TTFT is not meaningful).

    Cluster capacities:
      prefill_cluster = per_inst × 4  (L40S has 4 GPUs)
      decode_cluster  = per_inst × 8  (L4  has 8 GPUs)

    The class rate is bounded by whichever side saturates first:
      limit = min(prefill_cluster, decode_cluster)

    If the exact (il or ol) is not in the C/D grid, the nearest measured
    shape is used (flagged with '*' in the source string).
    """
    l40s_max = CLUSTER["l40s"]["max_freq_mhz"]
    l4_max   = CLUSTER["l4"]["max_freq_mhz"]
    n_a = CLUSTER["l40s"]["gpu_count"]
    n_b = CLUSTER["l4"]["gpu_count"]

    c_ils = sorted({k[1] for k in c_lookup if k[0] == l40s_max})
    d_ols = sorted({k[1] for k in d_lookup if k[0] == l4_max})
    if not c_ils or not d_ols:
        return None, "no_cd_data"
    c_il = min(c_ils, key=lambda x: abs(x - il))
    d_ol = min(d_ols, key=lambda x: abs(x - ol))
    c_il_star = "*" if c_il != il else ""
    d_ol_star = "*" if d_ol != ol else ""

    kv_ms = il * tau_kv_us / 1000.0

    # Prefill-side per-instance sustained rate on L40S
    c_recs = [c_lookup[(l40s_max, c_il, rate)]
              for rate in sorted({k[2] for k in c_lookup
                                  if k[0] == l40s_max and k[1] == c_il})]
    pf_per_inst = _per_instance_best_thr_under_slo(
        c_recs, slo_name, check_ttft=True, check_tpot=False,
        extra_ttft_ms=kv_ms,
    )

    # Decode-side per-instance sustained rate on L4
    d_recs = [d_lookup[(l4_max, d_ol, rate)]
              for rate in sorted({k[2] for k in d_lookup
                                  if k[0] == l4_max and k[1] == d_ol})]
    dc_per_inst = _per_instance_best_thr_under_slo(
        d_recs, slo_name, check_ttft=False, check_tpot=True,
    )

    src_base = (f"C_il={c_il}{c_il_star}, D_ol={d_ol}{d_ol_star}, "
                f"kv+={kv_ms:.1f}ms")
    if pf_per_inst is None and dc_per_inst is None:
        return None, f"neither_side_feasible ({src_base})"
    if pf_per_inst is None:
        return None, f"prefill_infeasible ({src_base})"
    if dc_per_inst is None:
        return None, f"decode_infeasible ({src_base})"

    pf_cluster = pf_per_inst * n_a
    dc_cluster = dc_per_inst * n_b
    limit = min(pf_cluster, dc_cluster)
    binding = "prefill(L40S)" if pf_cluster <= dc_cluster else "decode(L4)"
    return limit, (f"min(pf×{n_a}={pf_cluster:.1f}, dc×{n_b}={dc_cluster:.1f}) "
                   f"bind={binding}; {src_base}")


def compute_trace_peak_rates(records, c_lookup, d_lookup, c_pf_ref, c_dc_ref,
                             tau_kv_list,
                             tau_imb=TAU_IMB_DEFAULT, tau_load=TAU_LOAD_DEFAULT):
    rows = []
    for trace, mix in TRACE_MIXES.items():
        for slo_name in SLO_LEVELS:
            # Co-located path — independent of τ_kv
            coloc = {}
            for cls, frac in mix.items():
                if frac <= 0:
                    continue
                info = CLASSES[cls]
                # Co-located lookup uses proxy shape (since A/B don't have
                # il=64 or the LL shape 1024/512 directly)
                il_used, ol_used = info.get("il_proxy"), info.get("ol_proxy")
                if il_used is None or ol_used is None:
                    # LL has no A/B proxy — no co-located limit available
                    limit, source = None, "no_AB_proxy"
                else:
                    limit, source = _colocated_rate_limit(
                        records, il_used, ol_used, slo_name
                    )
                coloc[cls] = {"frac": frac, "limit_rps": limit, "source": source,
                              "il": il_used, "ol": ol_used}

            # Disagg path — per τ_kv
            for tau_kv in tau_kv_list:
                disagg = {}
                for cls, frac in mix.items():
                    if frac <= 0:
                        continue
                    info = CLASSES[cls]
                    il_paper = info["il_paper"]
                    ol_paper = info["ol_paper"]
                    limit, source = _disagg_rate_limit(
                        c_lookup, d_lookup, il_paper, ol_paper, slo_name, tau_kv
                    )
                    disagg[cls] = {"frac": frac, "limit_rps": limit, "source": source,
                                   "il": il_paper, "ol": ol_paper}

                # Best-of: per class, the scheduler picks whichever is higher
                best = {}
                for cls in mix:
                    if mix[cls] <= 0:
                        continue
                    co_lim = coloc[cls]["limit_rps"]
                    dg_lim = disagg[cls]["limit_rps"]
                    if co_lim is None and dg_lim is None:
                        lim, src = None, "no_path"
                    elif co_lim is None:
                        lim, src = dg_lim, "disagg"
                    elif dg_lim is None:
                        lim, src = co_lim, "colocated"
                    else:
                        if dg_lim >= co_lim:
                            lim, src = dg_lim, "disagg"
                        else:
                            lim, src = co_lim, "colocated"
                    best[cls] = {"frac": mix[cls], "limit_rps": lim, "path": src}

                def peak_from(per_class):
                    bounds = [(d["limit_rps"] / d["frac"], c)
                              for c, d in per_class.items()
                              if d["limit_rps"] is not None and d["frac"] > 0]
                    if not bounds:
                        return None, None
                    return min(bounds)

                # Co-located peak is τ_kv-independent but we recompute once per tau_kv
                # row for readability (same value).
                co_peak, co_bind = peak_from(coloc)
                dg_peak, dg_bind = peak_from(disagg)
                be_peak, be_bind = peak_from(best)

                # (x, y) and state computed at the SWEEP-LLM-best peak λ using
                # paper-canonical shapes for D_pf and D_dc.
                if be_peak is not None:
                    D_pf = be_peak * sum(mix[c] * CLASSES[c]["il_paper"] for c in mix)
                    D_dc = be_peak * sum(mix[c] * CLASSES[c]["ol_paper"] for c in mix)
                    x = D_pf / c_pf_ref if c_pf_ref > 0 else None
                    y = D_dc / c_dc_ref if c_dc_ref > 0 else None
                    state = classify_state(x, y, tau_imb, tau_load)
                else:
                    D_pf = D_dc = x = y = None
                    state = None

                detail = "; ".join(
                    f"{cls}(π={d['frac']:.2f}, "
                    f"co={coloc[cls]['limit_rps']}, "
                    f"dg={disagg[cls]['limit_rps']}, "
                    f"best={best[cls]['limit_rps']} via {best[cls]['path']})"
                    for cls, d in best.items()
                )
                rows.append({
                    "trace": trace, "slo": slo_name,
                    "tau_kv_us_per_tok": tau_kv,
                    "colocated_peak_rps": round(co_peak, 3) if co_peak is not None else None,
                    "colocated_binding": co_bind,
                    "disagg_peak_rps":    round(dg_peak, 3) if dg_peak is not None else None,
                    "disagg_binding":     dg_bind,
                    "best_peak_rps":      round(be_peak, 3) if be_peak is not None else None,
                    "best_binding":       be_bind,
                    "D_pf_tok_per_s": round(D_pf, 1) if D_pf is not None else None,
                    "D_dc_tok_per_s": round(D_dc, 1) if D_dc is not None else None,
                    "x_normalized":   round(x, 3) if x is not None else None,
                    "y_normalized":   round(y, 3) if y is not None else None,
                    "classifier_state": state or "",
                    "tau_imb_used": tau_imb, "tau_load_used": tau_load,
                    "per_class_detail": detail,
                })
    return rows


def classify_state(x, y, tau_imb, tau_load):
    if x is None or y is None:
        return None
    if y > 0 and x / y >= tau_imb:
        return "PREFILL_HEAVY"
    if x > 0 and y / x >= tau_imb:
        return "DECODE_HEAVY"
    if x + y < tau_load:
        return "BOTH_LOW"
    return "BOTH_HEAVY"


# --------------------------------------------------------------------------
# CSV writer & summary printer
# --------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        print(f"  [skip] no rows for {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            out = {}
            for k, v in r.items():
                if isinstance(v, (list, tuple)):
                    out[k] = ",".join(str(x) for x in v)
                elif v is None:
                    out[k] = ""
                elif isinstance(v, bool):
                    out[k] = "true" if v else "false"
                else:
                    out[k] = v
            w.writerow(out)
    print(f"  wrote {len(rows):>4d} rows: {path}")


def fmt(v, width, fmt_spec=""):
    if v is None or v == "":
        return f"{'—':>{width}}"
    if fmt_spec:
        return f"{v:{fmt_spec}}".rjust(width)
    return str(v).rjust(width)


def print_trace_summary(peak_rows, tau_kv_list):
    print("\n=== Trace peak rates (max freq both pools) ===")
    print("Columns:")
    print("  co    = co-located peak λ (interconnect-independent; A+B best)")
    print("  dg(τ) = disaggregated peak λ at τ µs/tok KV transfer (C+D synth)")
    print("  best  = SWEEP-LLM-best per-class choice; x,y,state at best λ")
    dg_cols = "".join(f"{'dg(' + str(int(t)) + ')':>10}" for t in tau_kv_list)
    header = (f"{'trace':<10}{'slo':<10}{'co':>8}"
              f"{dg_cols}{'best':>10}{'x':>8}{'y':>8}  state")
    print(header)
    print("-" * len(header))

    # Group rows by (trace, slo) and pivot tau_kv to columns
    from collections import OrderedDict
    grouped = OrderedDict()
    for r in peak_rows:
        key = (r["trace"], r["slo"])
        grouped.setdefault(key, {})[r["tau_kv_us_per_tok"]] = r

    for (trace, slo), by_tau in grouped.items():
        first = next(iter(by_tau.values()))
        co_val = fmt(first["colocated_peak_rps"], 8, ".2f")
        dg_vals = ""
        # best_peak varies with τ_kv; pick the τ that gives the highest best
        best_across_tau = max(
            ((by_tau[t]["best_peak_rps"] or -1, t) for t in tau_kv_list),
            default=(-1, None),
        )
        best_tau = best_across_tau[1]
        for t in tau_kv_list:
            r = by_tau.get(t)
            dg_vals += fmt(r["disagg_peak_rps"] if r else None, 10, ".2f")
        # Report best and state at the τ_kv-dependent best
        r_best = by_tau.get(best_tau) if best_tau is not None else first
        print(f"{trace:<10}{slo:<10}{co_val}"
              f"{dg_vals}"
              f"{fmt(r_best['best_peak_rps'], 10, '.2f')}"
              f"{fmt(r_best['x_normalized'], 8, '.2f')}"
              f"{fmt(r_best['y_normalized'], 8, '.2f')}"
              f"  {r_best['classifier_state'] or '—'}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", required=True,
                    help="Path to the v2 results dir (contains A_/B_/CD_/E_ subdirs)")
    ap.add_argument("--out-dir", default="calibration_out",
                    help="Output directory for CSV artifacts")
    ap.add_argument("--tau-imb", type=float, default=TAU_IMB_DEFAULT,
                    help=f"Phase imbalance threshold for classifier (default {TAU_IMB_DEFAULT})")
    ap.add_argument("--tau-load", type=float, default=TAU_LOAD_DEFAULT,
                    help=f"Low-load threshold for classifier (default {TAU_LOAD_DEFAULT})")
    ap.add_argument("--tau-kv-us-per-tok", type=str,
                    default=",".join(str(t) for t in TAU_KV_DEFAULTS),
                    help=f"Comma-separated KV transfer cost(s) in µs/tok for synthesized "
                         f"disagg (default: {','.join(str(t) for t in TAU_KV_DEFAULTS)}; "
                         f"5=IB HDR, 16=PCIe 4.0, 41=25 GbE)")
    args = ap.parse_args()

    try:
        tau_kv_list = [float(x.strip()) for x in args.tau_kv_us_per_tok.split(",") if x.strip()]
    except ValueError:
        print(f"ERROR: --tau-kv-us-per-tok must be a comma-separated list of numbers", file=sys.stderr)
        sys.exit(1)
    if not tau_kv_list:
        print(f"ERROR: --tau-kv-us-per-tok produced no valid values", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results-dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing: {args.results_dir}")
    records = collect_records(args.results_dir)
    print(f"  parsed {len(records)} bench records")
    if not records:
        sys.exit(1)

    # Distribution by experiment
    by_exp = defaultdict(int)
    for r in records:
        by_exp[r["exp"]] += 1
    print("  by experiment: " + ", ".join(f"{k}={v}" for k, v in sorted(by_exp.items())))

    records = annotate_feasibility(records)

    print("\nEstimating reference capacities …")
    ref_rows, c_pf_ref, c_dc_ref = estimate_reference_capacities(records)
    print(f"  C_pf^ref (L40S pool, linear extrap from C): {c_pf_ref:,.0f} tok/s")
    print(f"  C_dc^ref (L4 pool,   linear extrap from D): {c_dc_ref:,.0f} tok/s")
    write_csv(os.path.join(args.out_dir, "reference_capacity.csv"), ref_rows)

    print("\nBuilding per-record feasibility table …")
    feas_rows = build_feasibility_table(records)
    write_csv(os.path.join(args.out_dir, "feasibility_by_config.csv"), feas_rows)

    print("\nComputing max sustainable rate per config …")
    sust_rows = max_sustainable_rate(records)
    write_csv(os.path.join(args.out_dir, "max_sustainable_rate.csv"), sust_rows)

    print("\nComputing trace peak rates …")
    c_lookup, d_lookup = build_cd_lookup(records)
    print(f"  C (prefill-only L40S) entries: {len(c_lookup)}")
    print(f"  D (decode-only L4)   entries: {len(d_lookup)}")
    print(f"  τ_kv sweep: {tau_kv_list} µs/tok")

    peak_rows = compute_trace_peak_rates(
        records, c_lookup, d_lookup, c_pf_ref, c_dc_ref,
        tau_kv_list=tau_kv_list,
        tau_imb=args.tau_imb, tau_load=args.tau_load,
    )
    write_csv(os.path.join(args.out_dir, "trace_peak_rates.csv"), peak_rows)

    print_trace_summary(peak_rows, tau_kv_list)

    print("\nNotes:")
    print(f"  - tau_imb={args.tau_imb}, tau_load={args.tau_load} are placeholders;")
    print("    paper Table 1 lists them as TBD.")
    print("  - Co-located peak (co) is τ_kv-independent and uses A/B monolithic data.")
    print("  - Disaggregated peak (dg) is synthesized from C+D phase-isolated runs")
    print("    plus KV transfer cost il × τ_kv. Does NOT use v2 E measurements,")
    print("    which were on commodity Ethernet (~14,700 µs/tok).")
    print("  - C_pf^ref / C_dc^ref are linear extrapolations from single-instance")
    print("    measurements; confirm with multi-instance runs if tightness matters.")
    print("  - SS/SL use il=64 (paper); C lookup falls back to nearest il (128).")


if __name__ == "__main__":
    main()
