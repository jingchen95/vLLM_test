"""
jsep_experiment_f.py — Experiment F: Trace-driven scheduler comparison.

Evaluates 5 scheduling strategies over realistic workload traces using the
B5 lookup table as a ground-truth performance oracle.

Strategies:
  1. StaticL40SStrategy   — L40S at max freq (2520 MHz), TP=4. No adaptation.
  2. StaticL4Strategy     — L4  at max freq (2040 MHz), TP=4. No adaptation.
  3. StaticDisaggStrategy — L40S prefill + L4 decode, both at max freq, TP=1.
  4. DynamoLLMStrategy    — DVFS on L40S only; min power that meets SLO.
  5. SWEEPLLMStrategy     — Full dynamic: routing + DVFS + TP + disagg.

Usage:
    python jsep_experiment_f.py --lookup-table ../b5_lookup_table.csv
    python jsep_experiment_f.py --slo 300 --trace bursty --plot
    python jsep_experiment_f.py --trace azure --azure-csv my_azure.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import warnings
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from jsep_traces import (
    Request, generate_steady, generate_bursty, generate_diurnal,
    load_azure_trace,
    generate_T1_prefill_heavy, generate_T2_decode_heavy,
    generate_T3_phase_shift, generate_T4_overload_burst,
)
from jsep_cluster import GPU_SPECS
from jsep_lookup_table import LookupTable
from jsep_disagg_scheduler import DisaggStateClassifier, KV_TRANSFER_MS_PER_TOKEN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_S = 30.0       # evaluation window duration (seconds)
MIN_RATE  = 0.01      # req/s floor to avoid division by zero

# Per-GPU idle power (W) when not serving
IDLE_W_L40S = GPU_SPECS["l40s"]["idle_power_w"]   # 90 W per GPU
IDLE_W_L4   = GPU_SPECS["l4"]["idle_power_w"]      # 18 W per GPU
N_L40S_TOTAL = GPU_SPECS["l40s"]["total_gpus"]     # 4
N_L4_TOTAL   = GPU_SPECS["l4"]["total_gpus"]        # 8
IDLE_POWER_L40S = N_L40S_TOTAL * IDLE_W_L40S
IDLE_POWER_L4   = N_L4_TOTAL * IDLE_W_L4
IDLE_POWER_TOTAL = IDLE_POWER_L40S + IDLE_POWER_L4


# ---------------------------------------------------------------------------
# SchedulingDecision dataclass
# ---------------------------------------------------------------------------

@dataclass
class SchedulingDecision:
    """Result from a single strategy.decide() call."""
    gpu_type: str              # "l40s", "l4", or "disagg"
    l40s_freq: Optional[int]  # None when L40S not used
    l40s_tp:   Optional[int]
    l4_freq:   Optional[int]  # None when L4 not used
    l4_tp:     Optional[int]
    alpha: float               # fraction of prefill routed to L40S
    beta:  float               # fraction of decode routed to L4
    # Pool sizing
    N_l40s_pf: int = 0         # L40S GPUs assigned to prefill
    N_l40s_dc: int = 0         # L40S GPUs assigned to decode
    N_l4_pf:   int = 0         # L4 GPUs assigned to prefill
    N_l4_dc:   int = 0         # L4 GPUs assigned to decode
    state: str = "N/A"         # classifier state (SWEEP-LLM only)
    # Ideal config (what joint search recommends if reconfiguration were free)
    ideal_tp_l40s: Optional[int] = None
    ideal_tp_l4:   Optional[int] = None
    ideal_N_l40s:  int = 0
    ideal_N_l4:    int = 0
    ideal_power_w: float = 0.0
    # Predictions (for the actually applied config)
    predicted_ttft_ms: float = 0.0
    predicted_tpot_ms: float = 0.0
    predicted_power_w: float = 0.0
    slo_met: bool = False

    @property
    def active_gpus(self) -> int:
        return self.N_l40s_pf + self.N_l40s_dc + self.N_l4_pf + self.N_l4_dc


# ---------------------------------------------------------------------------
# Capacity model helper
# ---------------------------------------------------------------------------

def _pool_query(lut: LookupTable, gpu_type: str, N: int, tp: int, freq: int,
                il: int, ol: int, demand_rate: float,
                metric: str, exp_type: str) -> float:
    """Query LUT at per-instance rate for a pool with N GPUs at given TP."""
    if N == 0 or tp == 0:
        return float("inf") if "ttft" in metric or "tpot" in metric else 0.0
    k = N // tp  # number of independent instances
    if k == 0:
        return float("inf") if "ttft" in metric or "tpot" in metric else 0.0
    per_inst_rate = max(demand_rate / k, 0.01)
    return lut.query(gpu_type, freq, tp, il, ol, per_inst_rate, metric,
                     exp_type=exp_type)


def _pool_power(lut: LookupTable, gpu_type: str, N: int, tp: int, freq: int,
                il: int = 512, ol: int = 128, demand_rate: float = 10.0,
                exp_type: str = "A") -> float:
    """Total active power for a pool of N GPUs at (freq, tp).
    Queries LUT at the per-instance rate for workload-specific per-GPU power.
    Returns N × per_gpu_power (since LUT avg_power_w is per-GPU from monitor CSV)."""
    if N == 0 or tp == 0:
        return 0.0
    k = N // tp
    if k == 0:
        return 0.0
    per_inst_rate = max(demand_rate / k, 0.01)
    # LUT avg_power_w is per-GPU power (from single-GPU monitor CSV)
    power_per_gpu = lut.query(gpu_type, freq, tp, il, ol, per_inst_rate,
                              "avg_power_w", exp_type=exp_type)
    if math.isnan(power_per_gpu):
        power_per_gpu = lut.get_power(gpu_type, freq, tp)
    # All N GPUs in the pool draw similar power
    return N * power_per_gpu


def _cluster_power_detailed(lut, dec, il: int = 512, ol: int = 128,
                             rate: float = 10.0) -> float:
    """Total cluster power: active pools at workload-specific power + idle GPUs."""
    p = 0.0
    # Active L40S pools
    if dec.l40s_freq is not None and dec.l40s_tp is not None:
        if dec.N_l40s_pf > 0:
            pf_rate = dec.alpha * rate
            p += _pool_power(lut, "l40s", dec.N_l40s_pf, dec.l40s_tp, dec.l40s_freq,
                             il, 1, pf_rate, "C")
        if dec.N_l40s_dc > 0:
            dc_rate = (1 - dec.beta) * rate
            p += _pool_power(lut, "l40s", dec.N_l40s_dc, dec.l40s_tp, dec.l40s_freq,
                             2, ol, dc_rate, "A")
    # Active L4 pools
    if dec.l4_freq is not None and dec.l4_tp is not None:
        if dec.N_l4_dc > 0:
            dc_rate = dec.beta * rate
            p += _pool_power(lut, "l4", dec.N_l4_dc, dec.l4_tp, dec.l4_freq,
                             2, ol, dc_rate, "D")
        if dec.N_l4_pf > 0:
            pf_rate = (1 - dec.alpha) * rate
            p += _pool_power(lut, "l4", dec.N_l4_pf, dec.l4_tp, dec.l4_freq,
                             il, 1, pf_rate, "B")
    # Idle GPUs
    n_l40s_active = dec.N_l40s_pf + dec.N_l40s_dc
    n_l4_active   = dec.N_l4_pf + dec.N_l4_dc
    p += (N_L40S_TOTAL - n_l40s_active) * IDLE_W_L40S
    p += (N_L4_TOTAL - n_l4_active) * IDLE_W_L4
    return p


def _cluster_power(lut, dec: SchedulingDecision) -> float:
    """Backward-compatible: total cluster power using avg workload power.
    get_power returns per-GPU power, so multiply by N (not k)."""
    p = 0.0
    if dec.l40s_freq is not None and dec.l40s_tp is not None:
        N = dec.N_l40s_pf + dec.N_l40s_dc
        if N > 0:
            p += N * lut.get_power("l40s", dec.l40s_freq, dec.l40s_tp)
    if dec.l4_freq is not None and dec.l4_tp is not None:
        N = dec.N_l4_pf + dec.N_l4_dc
        if N > 0:
            p += N * lut.get_power("l4", dec.l4_freq, dec.l4_tp)
    n_l40s_active = dec.N_l40s_pf + dec.N_l40s_dc
    n_l4_active   = dec.N_l4_pf + dec.N_l4_dc
    p += (N_L40S_TOTAL - n_l40s_active) * IDLE_W_L40S
    p += (N_L4_TOTAL - n_l4_active) * IDLE_W_L4
    return p


# ---------------------------------------------------------------------------
# Strategy 1: Static L40S
# ---------------------------------------------------------------------------

# =========================================================================
# STRATEGIES
# =========================================================================

# ---------------------------------------------------------------------------
# Strategy 1: Static L40S — all traffic on L40S, N=4, TP=4, max freq
# ---------------------------------------------------------------------------

class StaticL40SStrategy:
    """All traffic on L40S, N=4, TP=4, max freq. All L4 idle."""

    name = "StaticL40S"

    def __init__(self, lut: LookupTable, slo_ms: float):
        self.lut = lut
        self.slo_ms = slo_ms
        self.freq = GPU_SPECS["l40s"]["max_freq_mhz"]
        self.tp = 4
        self.N = N_L40S_TOTAL  # 4

    def decide(self, rate: float, avg_il: float, avg_ol: float) -> SchedulingDecision:
        il, ol = max(1, int(round(avg_il))), max(1, int(round(avg_ol)))
        ttft = _pool_query(self.lut, "l40s", self.N, self.tp, self.freq,
                           il, ol, rate, "p99_ttft_ms", "A")
        tpot = _pool_query(self.lut, "l40s", self.N, self.tp, self.freq,
                           il, ol, rate, "p99_tpot_ms", "A")
        dec = SchedulingDecision(
            gpu_type="l40s", l40s_freq=self.freq, l40s_tp=self.tp,
            l4_freq=None, l4_tp=None, alpha=1.0, beta=0.0,
            N_l40s_pf=self.N, N_l40s_dc=0, N_l4_pf=0, N_l4_dc=0,
            predicted_ttft_ms=ttft, predicted_tpot_ms=tpot,
            slo_met=(ttft <= self.slo_ms),
        )
        dec.predicted_power_w = _cluster_power(self.lut, dec)
        return dec


# ---------------------------------------------------------------------------
# Strategy 2: Static L4
# ---------------------------------------------------------------------------

class StaticL4Strategy:
    """All traffic on L4, N=8, TP=4, max freq. All L40S idle."""

    name = "StaticL4"

    def __init__(self, lut: LookupTable, slo_ms: float):
        self.lut = lut
        self.slo_ms = slo_ms
        self.freq = GPU_SPECS["l4"]["max_freq_mhz"]
        self.tp = 4
        self.N = N_L4_TOTAL  # 8

    def decide(self, rate: float, avg_il: float, avg_ol: float) -> SchedulingDecision:
        il, ol = max(1, int(round(avg_il))), max(1, int(round(avg_ol)))
        ttft = _pool_query(self.lut, "l4", self.N, self.tp, self.freq,
                           il, ol, rate, "p99_ttft_ms", "B")
        tpot = _pool_query(self.lut, "l4", self.N, self.tp, self.freq,
                           il, ol, rate, "p99_tpot_ms", "B")
        dec = SchedulingDecision(
            gpu_type="l4", l40s_freq=None, l40s_tp=None,
            l4_freq=self.freq, l4_tp=self.tp, alpha=0.0, beta=1.0,
            N_l40s_pf=0, N_l40s_dc=0, N_l4_pf=0, N_l4_dc=self.N,
            predicted_ttft_ms=ttft, predicted_tpot_ms=tpot,
            slo_met=(ttft <= self.slo_ms),
        )
        dec.predicted_power_w = _cluster_power(self.lut, dec)
        return dec


# ---------------------------------------------------------------------------
# Strategy 3: Static Disaggregated — N=4 L40S prefill, N=8 L4 decode, TP=1
# ---------------------------------------------------------------------------

class StaticDisaggStrategy:
    """Fixed disagg: L40S prefill (N=4, TP=1) + L4 decode (N=8, TP=1), max freq."""

    name = "StaticDisagg"

    def __init__(self, lut: LookupTable, slo_ms: float):
        self.lut = lut
        self.slo_ms = slo_ms
        self.l40s_freq = GPU_SPECS["l40s"]["max_freq_mhz"]
        self.l4_freq = GPU_SPECS["l4"]["max_freq_mhz"]

    def decide(self, rate: float, avg_il: float, avg_ol: float) -> SchedulingDecision:
        il, ol = max(1, int(round(avg_il))), max(1, int(round(avg_ol)))
        N_pf, N_dc, tp = N_L40S_TOTAL, N_L4_TOTAL, 1

        ttft_pf = _pool_query(self.lut, "l40s", N_pf, tp, self.l40s_freq,
                              il, 1, rate, "p99_ttft_ms", "C")
        kv_cost = KV_TRANSFER_MS_PER_TOKEN * il
        ttft = ttft_pf + kv_cost

        tpot = _pool_query(self.lut, "l4", N_dc, tp, self.l4_freq,
                           2, ol, rate, "p99_tpot_ms", "D")

        dec = SchedulingDecision(
            gpu_type="disagg",
            l40s_freq=self.l40s_freq, l40s_tp=tp,
            l4_freq=self.l4_freq, l4_tp=tp,
            alpha=1.0, beta=1.0,
            N_l40s_pf=N_pf, N_l40s_dc=0, N_l4_pf=0, N_l4_dc=N_dc,
            predicted_ttft_ms=ttft, predicted_tpot_ms=tpot,
            slo_met=(ttft <= self.slo_ms),
        )
        dec.predicted_power_w = _cluster_power(self.lut, dec)
        return dec


# ---------------------------------------------------------------------------
# Strategy 4: DynamoLLM — L40S only, adapts TP+freq, N=4
# ---------------------------------------------------------------------------

class DynamoLLMStrategy:
    """DVFS + TP on L40S only, N=4. No routing or disaggregation."""

    name = "DynamoLLM"

    def __init__(self, lut: LookupTable, slo_ms: float):
        self.lut = lut
        self.slo_ms = slo_ms
        self.N = N_L40S_TOTAL
        self.freqs = lut.get_valid_freqs("l40s", exp_type="A") or sorted(GPU_SPECS["l40s"]["frequencies"])
        self.tps = lut.get_valid_tps("l40s", exp_type="A") or GPU_SPECS["l40s"]["tp_degrees"]

    def decide(self, rate: float, avg_il: float, avg_ol: float) -> SchedulingDecision:
        il, ol = max(1, int(round(avg_il))), max(1, int(round(avg_ol)))
        best_dec = None
        best_power = float("inf")

        for tp in self.tps:
            if self.N % tp != 0:
                continue
            for freq in self.freqs:
                ttft = _pool_query(self.lut, "l40s", self.N, tp, freq,
                                   il, ol, rate, "p99_ttft_ms", "A")
                tpot = _pool_query(self.lut, "l40s", self.N, tp, freq,
                                   il, ol, rate, "p99_tpot_ms", "A")
                dec = SchedulingDecision(
                    gpu_type="l40s",
                    l40s_freq=freq, l40s_tp=tp,
                    l4_freq=None, l4_tp=None,
                    alpha=1.0, beta=0.0,
                    N_l40s_pf=self.N, N_l40s_dc=0, N_l4_pf=0, N_l4_dc=0,
                    predicted_ttft_ms=ttft, predicted_tpot_ms=tpot,
                    slo_met=(ttft <= self.slo_ms),
                )
                dec.predicted_power_w = _cluster_power(self.lut, dec)
                if dec.slo_met and dec.predicted_power_w < best_power:
                    best_power = dec.predicted_power_w
                    best_dec = dec

        if best_dec is not None:
            return best_dec
        # Fallback: max freq, max TP
        return SchedulingDecision(
            gpu_type="l40s",
            l40s_freq=self.freqs[-1], l40s_tp=self.tps[-1],
            l4_freq=None, l4_tp=None, alpha=1.0, beta=0.0,
            N_l40s_pf=self.N, N_l40s_dc=0, N_l4_pf=0, N_l4_dc=0,
            predicted_ttft_ms=float("nan"), predicted_tpot_ms=float("nan"),
            predicted_power_w=IDLE_POWER_TOTAL + 300, slo_met=False,
        )


# ---------------------------------------------------------------------------
# Strategy 5: SWEEP-LLM — Full joint optimization (brute-force)
# ---------------------------------------------------------------------------

class SWEEPLLMStrategy:
    """
    Full SWEEP-LLM with four-step scheduling pipeline:
      Step 1: Window classification (four-state classifier)
      Step 2: Joint target search (ideal + constrained)
      Step 3: Knob classification (fast / medium / slow)
      Step 4: Stability gating (apply slow knobs only after K consistent windows)
    """

    name = "SWEEP-LLM"

    ALPHA_CANDIDATES = [1.0, 0.7, 0.5, 0.3, 0.0]
    BETA_CANDIDATES  = [1.0, 0.7, 0.5, 0.3, 0.0]

    # Stability gating thresholds (number of consecutive windows)
    K_TP = 4     # TP change requires 4 consistent windows (~2 min at 30s windows)
    K_N  = 10    # Instance scaling requires 10 consistent windows (~5 min)

    def __init__(self, lut: LookupTable, slo_ms: float, slo_tpot: float = 50.0):
        self.lut = lut
        self.slo_ms = slo_ms
        self.slo_tpot = slo_tpot
        self.classifier = DisaggStateClassifier()
        # LUT-backed search dimensions
        self.l40s_freqs = lut.get_valid_freqs("l40s", exp_type="A") or sorted(GPU_SPECS["l40s"]["frequencies"])
        self.l4_freqs = lut.get_valid_freqs("l4", exp_type="B") or sorted(GPU_SPECS["l4"]["frequencies"])
        self.l40s_tps = lut.get_valid_tps("l40s", exp_type="A") or GPU_SPECS["l40s"]["tp_degrees"]
        self.l4_tps = lut.get_valid_tps("l4", exp_type="B") or GPU_SPECS["l4"]["tp_degrees"]
        self.c_freqs = lut.get_valid_freqs("l40s", exp_type="C") or self.l40s_freqs
        self.d_freqs = lut.get_valid_freqs("l4", exp_type="D") or self.l4_freqs

        # Current physical state (slow knobs)
        self.cur_tp_l40s = self.l40s_tps[0]   # start at lowest TP
        self.cur_tp_l4   = self.l4_tps[0]
        self.cur_N_l40s  = 1                    # start with 1 active GPU per type
        self.cur_N_l4    = 1

        # Stability counters for medium/slow knobs
        self.tp_l40s_target = self.cur_tp_l40s
        self.tp_l4_target   = self.cur_tp_l4
        self.N_l40s_target  = self.cur_N_l40s
        self.N_l4_target    = self.cur_N_l4
        self.ctr_tp_l40s = 0
        self.ctr_tp_l4   = 0
        self.ctr_N_l40s  = 0
        self.ctr_N_l4    = 0

    def decide(self, rate: float, avg_il: float, avg_ol: float) -> SchedulingDecision:
        il = max(1, int(round(avg_il)))
        ol = max(1, int(round(avg_ol)))

        # === Step 1: Window Classification ===
        state = self.classifier.classify(rate, il, ol)

        # === Step 2: Joint Target Search ===
        # 2a: Ideal config (all knobs free)
        ideal = self._brute_force_search(rate, il, ol, state)
        # 2b: Constrained config (TP and N locked to current physical state)
        constrained = self._brute_force_search(
            rate, il, ol, state,
            lock_tp_l40s=self.cur_tp_l40s, lock_tp_l4=self.cur_tp_l4,
            lock_N_l40s=self.cur_N_l40s + self.cur_N_l4,  # not used directly
        )

        # === Step 3: Knob Classification ===
        # Fast knobs (freq, α, β): take from constrained search
        # Medium knobs (TP): target from ideal search
        # Slow knobs (N): target from ideal search

        # === Step 4: Stability Gating ===
        # TP gating
        ideal_tp_l40s = ideal.l40s_tp or self.cur_tp_l40s
        ideal_tp_l4 = ideal.l4_tp or self.cur_tp_l4
        self.cur_tp_l40s = self._gate_knob(
            ideal_tp_l40s, "tp_l40s", self.K_TP)
        self.cur_tp_l4 = self._gate_knob(
            ideal_tp_l4, "tp_l4", self.K_TP)

        # N gating (total active GPUs per type)
        ideal_N_l40s = (ideal.N_l40s_pf + ideal.N_l40s_dc)
        ideal_N_l4 = (ideal.N_l4_pf + ideal.N_l4_dc)
        self.cur_N_l40s = self._gate_knob(
            ideal_N_l40s, "N_l40s", self.K_N)
        self.cur_N_l4 = self._gate_knob(
            ideal_N_l4, "N_l4", self.K_N)

        # Build final applied config: fast knobs from constrained, slow from gated
        applied = constrained
        applied.state = state
        # Record ideal config for analysis (what would be optimal without reconfig cost)
        applied.ideal_tp_l40s = ideal.l40s_tp
        applied.ideal_tp_l4 = ideal.l4_tp
        applied.ideal_N_l40s = ideal.N_l40s_pf + ideal.N_l40s_dc
        applied.ideal_N_l4 = ideal.N_l4_pf + ideal.N_l4_dc
        applied.ideal_power_w = ideal.predicted_power_w
        return applied

    def _gate_knob(self, target_val, knob_name: str, K: int) -> int:
        """Stability gating: only change if target is consistent for K windows."""
        ctr_attr = f"ctr_{knob_name}"
        tgt_attr = f"{knob_name}_target"
        cur_attr = f"cur_{knob_name}"

        prev_target = getattr(self, tgt_attr)
        cur_val = getattr(self, cur_attr)

        if target_val == prev_target:
            setattr(self, ctr_attr, getattr(self, ctr_attr) + 1)
        else:
            setattr(self, ctr_attr, 1)
            setattr(self, tgt_attr, target_val)

        if getattr(self, ctr_attr) >= K:
            setattr(self, ctr_attr, 0)
            return target_val  # apply the change
        return cur_val  # keep current

    def _brute_force_search(self, rate: float, il: int, ol: int,
                             state: str,
                             lock_tp_l40s: Optional[int] = None,
                             lock_tp_l4: Optional[int] = None,
                             lock_N_l40s: Optional[int] = None) -> SchedulingDecision:
        """
        Enumerate all feasible (α, β, N_splits, TP, freq) configurations.
        TP and freq are shared per GPU type (physical constraint).

        If lock_tp_*/lock_N_* are set, those knobs are fixed (constrained search).
        """
        best_dec: Optional[SchedulingDecision] = None
        best_power = float("inf")

        kv_per_req = KV_TRANSFER_MS_PER_TOKEN * il

        # Determine search ranges (locked or full)
        tps_l40s = [lock_tp_l40s] if lock_tp_l40s is not None else self.l40s_tps
        tps_l4 = [lock_tp_l4] if lock_tp_l4 is not None else self.l4_tps

        for alpha in self.ALPHA_CANDIDATES:
            for beta in self.BETA_CANDIDATES:
                # Skip disagg search during BURST (use monolithic max)
                if state == "BURST" and 0 < alpha < 1:
                    continue
                if state == "BURST" and 0 < beta < 1:
                    continue

                # Demand routing
                rate_pf_l40s = alpha * rate
                rate_pf_l4 = (1 - alpha) * rate
                rate_dc_l4 = beta * rate
                rate_dc_l40s = (1 - beta) * rate

                # KV transfer fraction
                kv_frac = alpha * beta + (1 - alpha) * (1 - beta)

                # Enumerate GPU type configs
                for tp_l40s in tps_l40s:
                    for tp_l4 in tps_l4:
                        # Enumerate N splits for L40S (N_pf + N_dc ≤ 4)
                        for N_l40s_pf in range(0, N_L40S_TOTAL + 1, tp_l40s):
                            for N_l40s_dc in range(0, N_L40S_TOTAL - N_l40s_pf + 1, tp_l40s):
                                # Enumerate N splits for L4 (N_pf + N_dc ≤ 8)
                                for N_l4_pf in range(0, N_L4_TOTAL + 1, tp_l4):
                                    for N_l4_dc in range(0, N_L4_TOTAL - N_l4_pf + 1, tp_l4):
                                        dec = self._evaluate_config(
                                            rate, il, ol, alpha, beta,
                                            N_l40s_pf, N_l40s_dc, N_l4_pf, N_l4_dc,
                                            tp_l40s, tp_l4, kv_frac, kv_per_req,
                                            rate_pf_l40s, rate_pf_l4,
                                            rate_dc_l4, rate_dc_l40s,
                                        )
                                        if dec is not None and dec.slo_met and dec.predicted_power_w < best_power:
                                            best_power = dec.predicted_power_w
                                            best_dec = dec

        if best_dec is not None:
            return best_dec
        return self._fallback(rate, il, ol)

    def _evaluate_config(self, rate, il, ol, alpha, beta,  # noqa: C901
                          N_l40s_pf, N_l40s_dc, N_l4_pf, N_l4_dc,
                          tp_l40s, tp_l4, kv_frac, kv_per_req,
                          rate_pf_l40s, rate_pf_l4,
                          rate_dc_l4, rate_dc_l40s) -> Optional[SchedulingDecision]:
        """Evaluate one candidate config. Returns None if infeasible."""

        # Feasibility: pools with demand must have GPUs
        if rate_pf_l40s > 0.01 and N_l40s_pf < tp_l40s:
            return None
        if rate_dc_l40s > 0.01 and N_l40s_dc < tp_l40s:
            return None
        if rate_pf_l4 > 0.01 and N_l4_pf < tp_l4:
            return None
        if rate_dc_l4 > 0.01 and N_l4_dc < tp_l4:
            return None
        # Skip configs with GPUs but no demand (waste)
        if rate_pf_l40s <= 0.01 and N_l40s_pf > 0:
            return None
        if rate_dc_l40s <= 0.01 and N_l40s_dc > 0:
            return None
        if rate_pf_l4 <= 0.01 and N_l4_pf > 0:
            return None
        if rate_dc_l4 <= 0.01 and N_l4_dc > 0:
            return None

        # Search best freq combo (ascending = lowest power first)
        best_freq_dec = None
        best_freq_power = float("inf")

        l40s_freqs = self.c_freqs if N_l40s_pf > 0 else self.l40s_freqs
        l4_freqs = self.d_freqs if N_l4_dc > 0 else self.l4_freqs

        for freq_l40s in l40s_freqs:
            for freq_l4 in l4_freqs:
                # TTFT: from whichever pool(s) handle prefill
                ttft_vals = []
                if rate_pf_l40s > 0.01:
                    t = _pool_query(self.lut, "l40s", N_l40s_pf, tp_l40s, freq_l40s,
                                    il, 1, rate_pf_l40s, "p99_ttft_ms", "C")
                    ttft_vals.append(t)
                if rate_pf_l4 > 0.01:
                    t = _pool_query(self.lut, "l4", N_l4_pf, tp_l4, freq_l4,
                                    il, 1, rate_pf_l4, "p99_ttft_ms", "B")
                    ttft_vals.append(t)

                ttft_eff = max(ttft_vals) if ttft_vals else 0.0
                ttft_eff += kv_frac * kv_per_req  # KV transfer overhead

                if ttft_eff > self.slo_ms:
                    continue  # SLO violated

                # TPOT: from whichever pool(s) handle decode
                tpot_vals = []
                if rate_dc_l4 > 0.01:
                    t = _pool_query(self.lut, "l4", N_l4_dc, tp_l4, freq_l4,
                                    2, ol, rate_dc_l4, "p99_tpot_ms", "D")
                    tpot_vals.append(t)
                if rate_dc_l40s > 0.01:
                    t = _pool_query(self.lut, "l40s", N_l40s_dc, tp_l40s, freq_l40s,
                                    2, ol, rate_dc_l40s, "p99_tpot_ms", "A")
                    tpot_vals.append(t)

                tpot_eff = max(tpot_vals) if tpot_vals else 0.0

                if tpot_eff > self.slo_tpot:
                    continue

                # Build candidate
                dec = SchedulingDecision(
                    gpu_type="disagg" if (alpha < 1.0 or beta < 1.0 or
                             (N_l40s_pf > 0 and N_l4_dc > 0)) else
                            ("l40s" if N_l40s_pf + N_l40s_dc > 0 and N_l4_pf + N_l4_dc == 0 else "l4"),
                    l40s_freq=freq_l40s if N_l40s_pf + N_l40s_dc > 0 else None,
                    l40s_tp=tp_l40s if N_l40s_pf + N_l40s_dc > 0 else None,
                    l4_freq=freq_l4 if N_l4_pf + N_l4_dc > 0 else None,
                    l4_tp=tp_l4 if N_l4_pf + N_l4_dc > 0 else None,
                    alpha=alpha, beta=beta,
                    N_l40s_pf=N_l40s_pf, N_l40s_dc=N_l40s_dc,
                    N_l4_pf=N_l4_pf, N_l4_dc=N_l4_dc,
                    predicted_ttft_ms=ttft_eff,
                    predicted_tpot_ms=tpot_eff,
                    slo_met=True,
                )
                dec.predicted_power_w = _cluster_power_detailed(
                    self.lut, dec, il, ol, rate)

                if dec.predicted_power_w < best_freq_power:
                    best_freq_power = dec.predicted_power_w
                    best_freq_dec = dec

                break  # lowest freq_l4 that works — move to next freq_l40s
            # If we found a valid config at this freq_l40s, keep going
            # (lower l40s freq might yield lower total power)

        return best_freq_dec

    def _fallback(self, rate: float, il: int, ol: int) -> SchedulingDecision:
        """Max-power fallback: all GPUs active at max freq."""
        freq_l40s = self.l40s_freqs[-1]
        freq_l4 = self.l4_freqs[-1]
        tp_l40s = self.l40s_tps[-1]
        tp_l4 = self.l4_tps[-1]
        ttft = _pool_query(self.lut, "l40s", N_L40S_TOTAL, tp_l40s, freq_l40s,
                           il, ol, rate, "p99_ttft_ms", "A")
        tpot = _pool_query(self.lut, "l40s", N_L40S_TOTAL, tp_l40s, freq_l40s,
                           il, ol, rate, "p99_tpot_ms", "A")
        dec = SchedulingDecision(
            gpu_type="l40s",
            l40s_freq=freq_l40s, l40s_tp=tp_l40s,
            l4_freq=None, l4_tp=None, alpha=1.0, beta=0.0,
            N_l40s_pf=N_L40S_TOTAL, N_l40s_dc=0, N_l4_pf=0, N_l4_dc=0,
            predicted_ttft_ms=ttft, predicted_tpot_ms=tpot,
            slo_met=(ttft <= self.slo_ms),
        )
        dec.predicted_power_w = _cluster_power(self.lut, dec)
        return dec


# ---------------------------------------------------------------------------
# Trace generator
# ---------------------------------------------------------------------------

_TRACE_BUILDERS = {
    "steady": lambda d, s, **kw: generate_steady(d, rate_rps=10.0, seed=s),
    "bursty": lambda d, s, **kw: generate_bursty(d, seed=s),
    "diurnal": lambda d, s, **kw: generate_diurnal(d, seed=s),
    "T1": lambda d, s, **kw: generate_T1_prefill_heavy(d, seed=s),
    "T2": lambda d, s, **kw: generate_T2_decode_heavy(d, seed=s),
    "T3": lambda d, s, **kw: generate_T3_phase_shift(d, seed=s),
    "T4": lambda d, s, **kw: generate_T4_overload_burst(d, seed=s),
}

def build_trace(trace_name: str, duration_s: float, seed: int,
                azure_csv: Optional[str] = None) -> List[Request]:
    """Generate a request trace by name."""
    if trace_name in _TRACE_BUILDERS:
        return _TRACE_BUILDERS[trace_name](duration_s, seed)
    if trace_name == "azure":
        if azure_csv is None:
            raise ValueError("--azure-csv must be set when --trace azure is used")
        return load_azure_trace(azure_csv, duration_s=duration_s)
    raise ValueError(f"Unknown trace: {trace_name}. "
                     f"Choose from {list(_TRACE_BUILDERS.keys()) + ['azure']}")


# ---------------------------------------------------------------------------
# Window statistics
# ---------------------------------------------------------------------------

def window_stats(requests: List[Request],
                  t_start: float, t_end: float) -> Tuple[float, float, float]:
    """
    Compute (rate_rps, avg_il, avg_ol) for requests in [t_start, t_end).
    Returns (0, default_il, default_ol) if no requests in window.
    """
    win = [r for r in requests if t_start <= r.arrival_time < t_end]
    if not win:
        return 0.0, 512.0, 128.0   # defaults for idle window
    duration = t_end - t_start
    rate     = len(win) / duration
    avg_il   = float(np.mean([r.input_len  for r in win]))
    avg_ol   = float(np.mean([r.output_len for r in win]))
    return rate, avg_il, avg_ol


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def simulate(trace: List[Request], strategies: Dict[str, object],
             slo_ms: float, window_s: float = WINDOW_S) -> pd.DataFrame:
    """
    Slide a window over the trace and query each strategy each window.

    Returns a DataFrame with columns:
      window_start, window_end, rate_rps, avg_il, avg_ol,
      strategy, gpu_type, l40s_freq, l40s_tp, l4_freq, l4_tp,
      alpha, beta, predicted_ttft_ms, predicted_tpot_ms,
      predicted_power_w, slo_met, window_energy_j, n_requests
    """
    if not trace:
        return pd.DataFrame()

    t_end_trace = trace[-1].arrival_time + window_s
    windows     = np.arange(0.0, t_end_trace, window_s)

    records = []
    for t_start in windows:
        t_end = t_start + window_s
        rate, avg_il, avg_ol = window_stats(trace, t_start, t_end)
        n_req = sum(1 for r in trace if t_start <= r.arrival_time < t_end)

        # Use a small floor rate to keep strategies from seeing zero
        eff_rate = max(rate, MIN_RATE)

        for strat_name, strategy in strategies.items():
            dec = strategy.decide(eff_rate, avg_il, avg_ol)
            records.append({
                "window_start":        t_start,
                "window_end":          t_end,
                "rate_rps":            round(rate, 3),
                "avg_il":              round(avg_il, 1),
                "avg_ol":              round(avg_ol, 1),
                "n_requests":          n_req,
                "strategy":            strat_name,
                "gpu_type":            dec.gpu_type,
                "l40s_freq":           dec.l40s_freq,
                "l40s_tp":             dec.l40s_tp,
                "l4_freq":             dec.l4_freq,
                "l4_tp":               dec.l4_tp,
                "alpha":               dec.alpha,
                "beta":                dec.beta,
                "N_l40s_pf":           dec.N_l40s_pf,
                "N_l40s_dc":           dec.N_l40s_dc,
                "N_l4_pf":             dec.N_l4_pf,
                "N_l4_dc":             dec.N_l4_dc,
                "active_gpus":         dec.active_gpus,
                "state":               dec.state,
                "ideal_tp_l40s":       dec.ideal_tp_l40s,
                "ideal_tp_l4":         dec.ideal_tp_l4,
                "ideal_N_l40s":        dec.ideal_N_l40s,
                "ideal_N_l4":          dec.ideal_N_l4,
                "ideal_power_w":       round(dec.ideal_power_w, 2) if dec.ideal_power_w else None,
                "predicted_ttft_ms":   round(dec.predicted_ttft_ms, 2),
                "predicted_tpot_ms":   round(dec.predicted_tpot_ms, 2),
                "predicted_power_w":   round(dec.predicted_power_w, 2),
                "slo_met":             dec.slo_met,
                "window_energy_j":     round(dec.predicted_power_w * window_s, 1),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, slo_ms: float) -> None:
    """Print per-strategy summary table to stdout."""
    if df.empty:
        print("No simulation results.")
        return

    rows = []
    for strat_name, grp in df.groupby("strategy"):
        total_energy   = grp["window_energy_j"].sum()
        slo_viol_pct   = 100.0 * (1 - grp["slo_met"].mean())
        met_windows    = grp[grp["slo_met"]]
        n_met_req      = met_windows["n_requests"].sum()
        met_energy     = met_windows["window_energy_j"].sum()
        # Energy per SLO-met request (J / request)
        energy_per_req = (met_energy / n_met_req) if n_met_req > 0 else float("nan")
        mean_gpus = grp["active_gpus"].mean() if "active_gpus" in grp.columns else float("nan")
        rows.append({
            "strategy":               strat_name,
            "total_energy_j":         round(total_energy, 0),
            "slo_violation_pct":      round(slo_viol_pct, 1),
            "energy_per_slo_req_j":   round(energy_per_req, 3),
            "mean_power_w":           round(grp["predicted_power_w"].mean(), 1),
            "mean_ttft_ms":           round(grp["predicted_ttft_ms"].mean(), 1),
            "mean_active_gpus":       round(mean_gpus, 1),
            "n_windows":              len(grp),
        })

    summary = pd.DataFrame(rows).sort_values("total_energy_j")

    print(f"\n=== Experiment F — SLO: {slo_ms:.0f} ms ===")
    print(summary.to_string(index=False))
    print()

    # Highlight energy savings vs worst baseline
    if len(summary) >= 2:
        max_energy = summary["total_energy_j"].max()
        best_row   = summary.iloc[0]  # already sorted by energy
        saving_pct = 100.0 * (max_energy - best_row["total_energy_j"]) / max_energy
        print(f"Best strategy '{best_row['strategy']}' saves "
              f"{saving_pct:.1f}% energy vs worst baseline, "
              f"SLO violation rate: {best_row['slo_violation_pct']:.1f}%")
        print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, output_path: str, slo_ms: float) -> None:
    """
    Generate a comparison figure with 4 subplots:
      1. Power over time per strategy
      2. Cumulative energy per strategy
      3. SLO violation windows (boolean heatmap)
      4. Bar chart: total energy + SLO violation %
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

    strategies = df["strategy"].unique()
    colors     = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    color_map  = dict(zip(strategies, colors))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Experiment F — Strategy Comparison (SLO: {slo_ms:.0f} ms)", fontsize=13)

    # --- Subplot 1: Power over time ---
    ax = axes[0, 0]
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("window_start")
        ax.plot(sub["window_start"], sub["predicted_power_w"],
                label=strat, color=color_map[strat], linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Cluster Power vs Time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Cumulative energy ---
    ax = axes[0, 1]
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("window_start")
        cum_energy = sub["window_energy_j"].cumsum() / 1000.0  # → kJ
        ax.plot(sub["window_start"].values, cum_energy.values,
                label=strat, color=color_map[strat], linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Energy (kJ)")
    ax.set_title("Cumulative Energy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: SLO violations ---
    ax = axes[1, 0]
    windows = sorted(df["window_start"].unique())
    matrix  = np.zeros((len(strategies), len(windows)))
    win_idx = {w: i for i, w in enumerate(windows)}
    for i, strat in enumerate(strategies):
        sub = df[df["strategy"] == strat]
        for _, row in sub.iterrows():
            j = win_idx[row["window_start"]]
            matrix[i, j] = 0 if row["slo_met"] else 1
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=8)
    ax.set_xlabel("Window index")
    ax.set_title("SLO Violations (red = violated)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # --- Subplot 4: Total energy + SLO violation bar ---
    ax4a = axes[1, 1]
    summary_rows = []
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        summary_rows.append({
            "strategy": strat,
            "total_kj":      sub["window_energy_j"].sum() / 1000.0,
            "viol_pct":      100.0 * (1 - sub["slo_met"].mean()),
        })
    sdf = pd.DataFrame(summary_rows)

    x = np.arange(len(strategies))
    bars = ax4a.bar(x, sdf["total_kj"], color=[color_map[s] for s in strategies],
                     alpha=0.8, width=0.5)
    ax4a.set_xticks(x)
    ax4a.set_xticklabels(sdf["strategy"], rotation=20, ha="right", fontsize=8)
    ax4a.set_ylabel("Total Energy (kJ)")
    ax4a.set_title("Total Energy + SLO Violation %")
    ax4a.grid(True, axis="y", alpha=0.3)

    ax4b = ax4a.twinx()
    ax4b.plot(x, sdf["viol_pct"], "D--k", markersize=6, linewidth=1.5,
               label="SLO viol %")
    ax4b.set_ylabel("SLO Violation (%)")
    ax4b.set_ylim(0, max(sdf["viol_pct"].max() * 1.3, 5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment F: Trace-driven scheduling strategy comparison."
    )
    parser.add_argument(
        "--lookup-table",
        default=os.path.join(os.path.dirname(__file__), "..", "b5_lookup_table.csv"),
        help="Path to b5_lookup_table.csv (default: ../b5_lookup_table.csv)"
    )
    parser.add_argument(
        "--slo", type=float, default=500.0,
        help="P99 TTFT SLO in ms (default: 500)"
    )
    parser.add_argument(
        "--slo-tpot", type=float, default=None,
        help="P99 TPOT SLO in ms (default: 50). Set separately from TTFT SLO."
    )
    parser.add_argument(
        "--trace", default="steady",
        choices=["steady", "bursty", "diurnal", "T1", "T2", "T3", "T4", "azure"],
        help="Workload trace to use (default: steady)"
    )
    parser.add_argument(
        "--azure-csv", default=None,
        help="Path to Azure trace CSV (required when --trace azure)"
    )
    parser.add_argument(
        "--trace-duration", type=float, default=600.0,
        help="Trace duration in seconds (default: 600)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic traces (default: 42)"
    )
    parser.add_argument(
        "--window", type=float, default=WINDOW_S,
        help=f"Evaluation window size in seconds (default: {WINDOW_S})"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: experiment_f_{trace}_{slo}ms.csv)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib comparison figure"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load lookup table
    # ------------------------------------------------------------------
    lut_path = os.path.abspath(args.lookup_table)
    if not os.path.isfile(lut_path):
        warnings.warn(
            f"Lookup table not found: {lut_path}. "
            "Proceeding with synthetic fallback values — results will be approximate."
        )
    lut = LookupTable(lut_path)
    print(f"Lookup table: {lut}")

    # ------------------------------------------------------------------
    # Generate trace
    # ------------------------------------------------------------------
    print(f"Generating trace: {args.trace} ({args.trace_duration:.0f} s, seed={args.seed})")
    trace = build_trace(args.trace, args.trace_duration, args.seed, args.azure_csv)
    print(f"  {len(trace)} requests generated")

    # ------------------------------------------------------------------
    # Build strategies
    # ------------------------------------------------------------------
    slo_ms = args.slo
    slo_tpot = args.slo_tpot if args.slo_tpot is not None else 50.0
    print(f"SLO: TTFT={slo_ms:.0f}ms, TPOT={slo_tpot:.0f}ms")
    strategies = {
        StaticL40SStrategy.name:   StaticL40SStrategy(lut, slo_ms),
        StaticL4Strategy.name:     StaticL4Strategy(lut, slo_ms),
        StaticDisaggStrategy.name: StaticDisaggStrategy(lut, slo_ms),
        DynamoLLMStrategy.name:    DynamoLLMStrategy(lut, slo_ms),
        SWEEPLLMStrategy.name:     SWEEPLLMStrategy(lut, slo_ms, slo_tpot=slo_tpot),
    }

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    print(f"Running simulation: {len(strategies)} strategies × "
          f"{int(args.trace_duration / args.window)} windows ...")
    df = simulate(trace, strategies, slo_ms, window_s=args.window)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_csv = args.output or f"experiment_f_{args.trace}_ttft{int(slo_ms)}_tpot{int(slo_tpot)}ms.csv"
    df.to_csv(out_csv, index=False)
    print(f"Results written to: {out_csv}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_summary(df, slo_ms)

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if args.plot:
        plot_path = os.path.splitext(out_csv)[0] + "_plot.png"
        make_plots(df, plot_path, slo_ms)


if __name__ == "__main__":
    main()
