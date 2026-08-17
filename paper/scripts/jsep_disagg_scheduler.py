"""
JSEP Phase B: Prefill-Decode Disaggregated Scheduling Strategy.

Key innovations over Phase A (JSEPStrategy):
  1. Token-driven load classification:
       D_prefill = rate × avg_il,  D_decode = rate × avg_ol
     instead of request-rate-only HP/LP detection.

  2. Hardware-matched pool assignment:
       L40S = prefill pool  (compute-bound, low TTFT)
       L4   = decode pool   (memory-bound, energy-efficient)

  3. (α, β) spill parameters:
       α = fraction of prefill on L40S  (1-α spills to L4)
       β = fraction of decode on L4     (1-β spills to L40S)
     Default: α=1, β=1 (fully disaggregated).
     When a pool saturates, the parameter decreases to spill load.

  4. Four-state classifier:
       BURST           — rapid rate increase (dλ/dt >> 0)
       PREFILL_HEAVY   — D_prefill >> D_decode (prefill bottleneck)
       DECODE_HEAVY    — D_decode >> D_prefill (decode bottleneck)
       BALANCED_LP     — overall demand low (power-gate idle GPUs)

  5. KV transfer cost:
       T_kv = KV_TRANSFER_MS_PER_TOKEN × input_len
     Overhead is added to effective TTFT SLO budget when prefill
     and decode pools are on different hardware types.

Pool modeling approximation (using existing EnergyScheduler):
  • Prefill pool call:  predict_config(il=actual_il, ol=PREFILL_OL, ...)
      PREFILL_OL = 32 → model treats workload as compute-bound.
  • Decode pool call:   predict_config(il=DECODE_IL, ol=actual_ol, ...)
      DECODE_IL = 32  → minimal prefill; KV cache already loaded.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
from typing import Dict, Optional, Tuple, List

from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import SchedulingResult, _find_min_safe_freq
from paths import paper_model_dir
from scheduler import EnergyScheduler


# ---------------------------------------------------------------------------
# KV transfer cost model
# ---------------------------------------------------------------------------

# Mistral-7B: 32 layers × 8 heads × 128 head_dim × 2(K+V) × 2 bytes = 128 KB/token
# Measured B4 (kv_transfer_benchmark.py): T_kv ≈ linear in KV size (throughput-dominated for il≥128)
#   Slope: ~0.57 ms/MB ≈ 0.073 ms/token  →  effective BW ~14 Gbps (~14% of 100 GbE line rate)
#   bs=1: il=512 → 37.4 ms, il=1024 → 75.5 ms, il=2048 → 127.7 ms
#   Note: il≤64 has ~5 ms fixed TCP overhead not captured by this linear model
KV_TRANSFER_MS_PER_TOKEN = 0.075  # ms per input token (measured B4, TCP 100 GbE, il≥128)


def kv_transfer_cost_ms(input_len: int) -> float:
    """KV cache transfer latency (ms) for a single request with given prompt length."""
    return KV_TRANSFER_MS_PER_TOKEN * input_len


# ---------------------------------------------------------------------------
# Four-state classifier
# ---------------------------------------------------------------------------

class DisaggStateClassifier:
    """
    Classifies current system load using token-demand ratios.

    Token demands:
      D_prefill = rate × avg_input_len   [prefill tokens/s]
      D_decode  = rate × avg_output_len  [decode tokens/s]

    States (with hysteresis — only transition on clear signal):
      BURST:          rate jumped > BURST_RATE_FACTOR × prev_rate
      PREFILL_HEAVY:  D_prefill / D_decode >= PD_RATIO_THRESHOLD
      DECODE_HEAVY:   D_decode  / D_prefill >= DP_RATIO_THRESHOLD
      BALANCED_LP:    D_total < LP_TOK_S_THRESHOLD
    """

    BURST_RATE_FACTOR    = 2.0   # rate must double to trigger BURST
    PD_RATIO_THRESHOLD   = 3.0   # D_prefill/D_decode ≥ this → PREFILL_HEAVY
    DP_RATIO_THRESHOLD   = 3.0   # D_decode/D_prefill  ≥ this → DECODE_HEAVY
    LP_TOK_S_THRESHOLD   = 150   # D_total (tok/s) below this → BALANCED_LP

    def __init__(self):
        self.prev_rate     = 0.0
        self.current_state = "BALANCED_LP"

    def classify(self, rate: float, il: int, ol: int) -> str:
        """Return state string; updates internal state with hysteresis."""
        D_prefill = rate * il
        D_decode  = rate * ol
        D_total   = D_prefill + D_decode

        # Priority order: BURST > LP > imbalance (with hysteresis)
        if rate > self.BURST_RATE_FACTOR * max(self.prev_rate, 0.1):
            self.current_state = "BURST"
        elif D_total < self.LP_TOK_S_THRESHOLD:
            self.current_state = "BALANCED_LP"
        elif D_prefill >= self.PD_RATIO_THRESHOLD * max(D_decode, 1):
            self.current_state = "PREFILL_HEAVY"
        elif D_decode >= self.DP_RATIO_THRESHOLD * max(D_prefill, 1):
            self.current_state = "DECODE_HEAVY"
        else:
            # Balanced moderate load: neither LP nor imbalanced.
            # Explicitly transition out of BURST when rate stabilises
            # (prevents hysteresis from keeping BURST forever).
            self.current_state = "BALANCED_LP"

        self.prev_rate = rate
        return self.current_state


# ---------------------------------------------------------------------------
# Helper: optimize a single-GPU-type pool
# ---------------------------------------------------------------------------

def _optimize_pool(
    sched: EnergyScheduler,
    spec: dict,
    il: int,
    ol: int,
    rate: float,
    slo_budget_ms: float,
    max_active_instances: Optional[int] = None,
    allow_powergating: bool = False,
) -> Optional[dict]:
    """
    Find the minimum-power (TP, freq, n_active) config for a pool.

    Args:
        sched:                 EnergyScheduler for this GPU type
        spec:                  GPU_SPECS[gpu_type]
        il, ol:                representative workload
        rate:                  request rate assigned to this pool (req/s)
        slo_budget_ms:         SLO budget for this pool (ms)
        max_active_instances:  cap on active TP groups (None = unconstrained)
        allow_powergating:     if True, can reduce active_instances below max

    Returns:
        dict with keys: tp, freq_mhz, n_active, power_per_instance_w,
                        pool_power_w, rho, is_safe
        None if no feasible config found.
    """
    # Snap slo_budget_ms to the nearest supported SLO threshold
    supported_slos = sorted(sched.SLO_THRESHOLDS)
    snapped_slo = min(supported_slos, key=lambda s: abs(s - slo_budget_ms))

    best_power = float("inf")
    best_cfg   = None

    tp_degrees = spec["tp_degrees"]
    total_gpus  = spec["total_gpus"]

    for tp in tp_degrees:
        max_inst = total_gpus // tp
        if max_active_instances is not None:
            max_inst = min(max_inst, max_active_instances)

        inst_range = range(1, max_inst + 1) if allow_powergating else [max_inst]

        for n_active in inst_range:
            if n_active == 0 or rate == 0:
                continue

            rate_per_inst = rate / n_active

            # Find minimum safe freq
            best_freq = None
            best_pred = None
            for freq in sched.FREQUENCIES:
                pred = sched.predict_config(il, ol, tp, freq, rate_per_inst,
                                            snapped_slo)
                if pred["is_safe"]:
                    best_freq = freq
                    best_pred = pred
                    break  # FREQUENCIES sorted ascending → first safe is lowest

            if best_freq is None:
                continue  # no safe freq at this (tp, n_active)

            active_power  = n_active * best_pred["total_power_w"]
            idle_gpus     = total_gpus - n_active * tp
            idle_power    = idle_gpus * spec["idle_power_w"]
            total_pool_pw = active_power + idle_power

            if total_pool_pw < best_power:
                best_power = total_pool_pw
                best_cfg = {
                    "tp":                  tp,
                    "freq_mhz":            best_freq,
                    "n_active":            n_active,
                    "power_per_instance_w": round(best_pred["total_power_w"], 1),
                    "pool_power_w":        round(total_pool_pw, 1),
                    "rho":                 best_pred["rho"],
                    "is_safe":             True,
                }

    return best_cfg


def _idle_pool_entry(spec: dict) -> dict:
    """Power cost when a pool is completely idle."""
    idle_total = spec["total_gpus"] * spec["idle_power_w"]
    return {
        "tp": 1, "freq_mhz": 0, "n_active": 0,
        "power_per_instance_w": 0,
        "pool_power_w": round(idle_total, 1),
        "rho": 0.0, "is_safe": True,
    }


# ---------------------------------------------------------------------------
# Main disaggregated JSEP strategy
# ---------------------------------------------------------------------------

class JSEPDisaggStrategy:
    """
    Disaggregated JSEP: joint prefill-pool + decode-pool optimization
    with (α, β) spill parameters and four-state load classification.

    Default assignment:
      L40S → prefill pool  (compute-bound; fast TTFT)
      L4   → decode pool   (memory-bound; efficient TPOT)

    Spill:
      If L40S prefill is saturated (ρ > threshold), α decreases:
        → (1-α) fraction of prefill spills to L4.
      If L4 decode is saturated (ρ > threshold), β decreases:
        → (1-β) fraction of decode spills to L40S.

    KV transfer overhead:
      Requests where prefill_pool ≠ decode_pool incur T_kv latency.
      In the default (α=1, β=1) configuration, ALL traffic incurs T_kv
      (L40S prefill → L4 decode).  When α<1 or β<1, some traffic is
      co-located and avoids the transfer.
    """

    # SLO budget fractions
    PREFILL_SLO_FRAC = 0.40   # TTFT budget = slo * this
    DECODE_SLO_FRAC  = 0.50   # decode latency budget = slo * this
    # (remaining 10% = KV transfer overhead headroom)

    # Proxy workload for pool-specific capacity estimation
    PREFILL_OL = 32    # ol used when modeling the prefill-only pool
    DECODE_IL  = 32    # il used when modeling the decode-only pool

    # Spill search: alpha values to try (fraction of prefill on L40S)
    ALPHA_CANDIDATES = [1.0, 0.7, 0.5, 0.3, 0.0]
    # Beta values to try (fraction of decode on L4)
    BETA_CANDIDATES  = [1.0, 0.7, 0.5, 0.3, 0.0]

    # TP reconfiguration cooldown
    RECONFIG_COOLDOWN_S = 30.0

    def __init__(self, schedulers: Dict[str, EnergyScheduler]):
        self.schedulers        = schedulers
        self.state_classifier  = DisaggStateClassifier()
        self.prev_tp           = {"l40s": 1, "l4": 1}
        self.last_reconfig_time = -999.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decide(self, il: int, ol: int, rate: float, slo: int,
               cluster: ClusterState,
               current_time: float = 0.0) -> SchedulingResult:
        state = self.state_classifier.classify(rate, il, ol)

        if state == "BURST":
            result = self._burst_decide(il, ol, rate, slo, cluster,
                                        current_time)
        elif state == "PREFILL_HEAVY":
            result = self._directed_decide(il, ol, rate, slo, cluster,
                                           current_time,
                                           prefer_alpha=1.0, prefer_beta=1.0)
        elif state == "DECODE_HEAVY":
            result = self._directed_decide(il, ol, rate, slo, cluster,
                                           current_time,
                                           prefer_alpha=1.0, prefer_beta=1.0)
        else:  # BALANCED_LP
            result = self._balanced_lp_decide(il, ol, rate, slo, cluster,
                                               current_time)

        # Map internal state → phase label for simulator bookkeeping
        result.phase = state
        return result

    # ------------------------------------------------------------------
    # State-specific decisions
    # ------------------------------------------------------------------

    def _burst_decide(self, il, ol, rate, slo, cluster,
                       current_time) -> SchedulingResult:
        """BURST: All GPUs at max freq; use default disagg (α=1, β=1)."""
        slo_pf = max(1, int(slo * self.PREFILL_SLO_FRAC))
        slo_dc = max(1, int(slo * self.DECODE_SLO_FRAC))

        spec_l40s = GPU_SPECS["l40s"]
        spec_l4   = GPU_SPECS["l4"]

        # L40S: prefill, all instances, max freq
        l40s_sched = self.schedulers["l40s"]
        n_l40s     = spec_l40s["total_gpus"]  # TP=1 → n_inst = n_gpus
        rate_per_l40s = rate / n_l40s
        pred_l40s = l40s_sched.predict_config(
            il, self.PREFILL_OL, 1, spec_l40s["max_freq_mhz"],
            rate_per_l40s, slo_pf)
        l40s_power = n_l40s * pred_l40s["total_power_w"]

        # L4: decode, all instances, max freq
        l4_sched  = self.schedulers["l4"]
        n_l4      = spec_l4["total_gpus"]
        rate_per_l4 = rate / n_l4
        pred_l4   = l4_sched.predict_config(
            self.DECODE_IL, ol, 1, spec_l4["max_freq_mhz"],
            rate_per_l4, slo_dc)
        l4_power  = n_l4 * pred_l4["total_power_w"]

        total_power = l40s_power + l4_power
        slo_met = pred_l40s["is_safe"] and pred_l4["is_safe"]

        config = {
            "l40s": {
                "tp": 1, "freq_mhz": spec_l40s["max_freq_mhz"],
                "active_instances": n_l40s,
                "role": "prefill", "alpha": 1.0,
                "rate_per_instance": round(rate_per_l40s, 2),
                "power_per_instance_w": round(pred_l40s["total_power_w"], 1),
                "pool_power_w": round(l40s_power, 1),
            },
            "l4": {
                "tp": 1, "freq_mhz": spec_l4["max_freq_mhz"],
                "active_instances": n_l4,
                "role": "decode", "beta": 1.0,
                "rate_per_instance": round(rate_per_l4, 2),
                "power_per_instance_w": round(pred_l4["total_power_w"], 1),
                "pool_power_w": round(l4_power, 1),
            },
        }
        return SchedulingResult(total_power, config, slo_met, phase="BURST")

    def _directed_decide(self, il, ol, rate, slo, cluster, current_time,
                          prefer_alpha: float,
                          prefer_beta: float) -> SchedulingResult:
        """
        Try the preferred (α, β) first; fall back to full joint search
        if the preferred configuration is infeasible.
        """
        result = self._try_alpha_beta(il, ol, rate, slo, cluster,
                                       current_time,
                                       prefer_alpha, prefer_beta)
        if result is not None:
            return result

        # Preferred config infeasible — run joint search
        return self._joint_search(il, ol, rate, slo, cluster, current_time,
                                   allow_powergating=False)

    def _balanced_lp_decide(self, il, ol, rate, slo, cluster,
                              current_time) -> SchedulingResult:
        """BALANCED_LP: allow power-gating; minimize total energy."""
        result = self._joint_search(il, ol, rate, slo, cluster,
                                     current_time, allow_powergating=True)
        if result is not None:
            return result
        # Fallback: burst mode (max power, always safe)
        return self._burst_decide(il, ol, rate, slo, cluster, current_time)

    # ------------------------------------------------------------------
    # Core optimizers
    # ------------------------------------------------------------------

    def _try_alpha_beta(self, il, ol, rate, slo, cluster, current_time,
                         alpha: float, beta: float) -> Optional[SchedulingResult]:
        """
        Try a specific (α, β) split.
        α = fraction of prefill on L40S; β = fraction of decode on L4.

        Returns SchedulingResult if feasible, None otherwise.
        """
        slo_pf = max(1, int(slo * self.PREFILL_SLO_FRAC))
        slo_dc = max(1, int(slo * self.DECODE_SLO_FRAC))

        kv_cost   = kv_transfer_cost_ms(il)
        kv_frac   = alpha * beta + (1 - alpha) * (1 - beta)  # fraction needing transfer
        kv_impact = kv_frac * kv_cost

        # Adjust SLO budgets for KV transfer overhead
        slo_pf_eff = max(50, slo_pf - kv_impact / 2)
        slo_dc_eff = max(50, slo_dc - kv_impact / 2)

        spec_l40s = GPU_SPECS["l40s"]
        spec_l4   = GPU_SPECS["l4"]

        # --- L40S: handles α fraction of prefill + (1-β) fraction of decode ---
        l40s_pf_rate = alpha * rate      # prefill on L40S
        l40s_dc_rate = (1 - beta) * rate  # decode spilled to L40S

        l40s_cfg = None
        l40s_power = spec_l40s["total_gpus"] * spec_l40s["idle_power_w"]  # idle default

        if l40s_pf_rate > 0 or l40s_dc_rate > 0:
            # Combined L40S workload: blended il/ol
            l40s_il = il if l40s_pf_rate > 0 else self.DECODE_IL
            l40s_ol = (self.PREFILL_OL if l40s_dc_rate == 0
                       else ol if l40s_pf_rate == 0
                       else self.PREFILL_OL)  # prefill-dominant workload model
            l40s_rate_total = l40s_pf_rate + l40s_dc_rate
            l40s_slo_use = slo_pf_eff if l40s_pf_rate >= l40s_dc_rate else slo_dc_eff

            l40s_cfg = _optimize_pool(
                self.schedulers["l40s"], spec_l40s,
                l40s_il, l40s_ol, l40s_rate_total, l40s_slo_use,
                allow_powergating=False,
            )
            if l40s_cfg is None:
                return None  # infeasible
            l40s_power = l40s_cfg["pool_power_w"]

        # --- L4: handles β fraction of decode + (1-α) fraction of prefill ---
        l4_dc_rate = beta * rate          # decode on L4
        l4_pf_rate = (1 - alpha) * rate   # prefill spilled to L4

        l4_cfg   = None
        l4_power = spec_l4["total_gpus"] * spec_l4["idle_power_w"]  # idle default

        if l4_dc_rate > 0 or l4_pf_rate > 0:
            l4_il = self.DECODE_IL if l4_pf_rate == 0 else il
            l4_ol = ol if l4_dc_rate > 0 else self.PREFILL_OL
            l4_rate_total = l4_dc_rate + l4_pf_rate
            l4_slo_use = slo_dc_eff if l4_dc_rate >= l4_pf_rate else slo_pf_eff

            l4_cfg = _optimize_pool(
                self.schedulers["l4"], spec_l4,
                l4_il, l4_ol, l4_rate_total, l4_slo_use,
                allow_powergating=False,
            )
            if l4_cfg is None:
                return None  # infeasible
            l4_power = l4_cfg["pool_power_w"]

        total_power = l40s_power + l4_power
        slo_met = (l40s_cfg is None or l40s_cfg["is_safe"]) and \
                  (l4_cfg is None or l4_cfg["is_safe"])

        config = self._make_config(l40s_cfg, l4_cfg, spec_l40s, spec_l4,
                                    alpha, beta,
                                    l40s_pf_rate, l40s_dc_rate,
                                    l4_dc_rate, l4_pf_rate)
        return SchedulingResult(total_power, config, slo_met)

    def _joint_search(self, il, ol, rate, slo, cluster, current_time,
                       allow_powergating: bool) -> Optional[SchedulingResult]:
        """
        Joint search over (α, β) × (TP_L40S, TP_L4, freq) to minimize power.
        """
        best_power  = float("inf")
        best_result = None

        for alpha in self.ALPHA_CANDIDATES:
            for beta in self.BETA_CANDIDATES:
                # Skip trivial/redundant combos where both pools are idle
                if alpha == 0 and beta == 0:
                    continue  # all decode on L40S, all prefill on L4
                # (still valid; just uncommon — keep it in search for completeness)

                # TP reconfig check
                result = self._try_alpha_beta_with_tp_search(
                    il, ol, rate, slo, cluster, current_time,
                    alpha, beta, allow_powergating)

                if result is not None and result.total_power_w < best_power:
                    best_power  = result.total_power_w
                    best_result = result

        return best_result

    def _try_alpha_beta_with_tp_search(self, il, ol, rate, slo, cluster,
                                        current_time, alpha, beta,
                                        allow_powergating) -> Optional[SchedulingResult]:
        """
        For a fixed (α, β), search over TP degrees for L40S and L4.
        Returns best-power feasible result or None.
        """
        slo_pf = max(1, int(slo * self.PREFILL_SLO_FRAC))
        slo_dc = max(1, int(slo * self.DECODE_SLO_FRAC))

        kv_cost   = kv_transfer_cost_ms(il)
        kv_frac   = alpha * beta + (1 - alpha) * (1 - beta)
        kv_impact = kv_frac * kv_cost
        slo_pf_eff = max(50, slo_pf - kv_impact / 2)
        slo_dc_eff = max(50, slo_dc - kv_impact / 2)

        spec_l40s = GPU_SPECS["l40s"]
        spec_l4   = GPU_SPECS["l4"]

        best_power  = float("inf")
        best_result = None

        for tp_l40s in spec_l40s["tp_degrees"]:
            for tp_l4 in spec_l4["tp_degrees"]:
                # TP reconfig cooldown
                tp_changed = (tp_l40s != self.prev_tp.get("l40s", tp_l40s) or
                              tp_l4   != self.prev_tp.get("l4",   tp_l4))
                if (tp_changed and
                        current_time - self.last_reconfig_time < self.RECONFIG_COOLDOWN_S):
                    continue

                total_power = 0.0
                l40s_cfg = l4_cfg = None

                # --- L40S pool ---
                l40s_pf_rate    = alpha * rate
                l40s_dc_rate    = (1 - beta) * rate
                l40s_total_rate = l40s_pf_rate + l40s_dc_rate

                if l40s_total_rate > 0:
                    l40s_il = il if l40s_pf_rate >= l40s_dc_rate else self.DECODE_IL
                    l40s_ol = self.PREFILL_OL if l40s_pf_rate >= l40s_dc_rate else ol
                    l40s_slo = slo_pf_eff if l40s_pf_rate >= l40s_dc_rate else slo_dc_eff

                    max_l40s_inst = spec_l40s["total_gpus"] // tp_l40s
                    l40s_cfg = _optimize_pool(
                        self.schedulers["l40s"], spec_l40s,
                        l40s_il, l40s_ol, l40s_total_rate, l40s_slo,
                        max_active_instances=max_l40s_inst,
                        allow_powergating=allow_powergating,
                    )
                    if l40s_cfg is None:
                        continue  # infeasible for this (α, β, TP_l40s)
                    total_power += l40s_cfg["pool_power_w"]
                else:
                    # L40S idle
                    idle_pw = spec_l40s["total_gpus"] * spec_l40s["idle_power_w"]
                    total_power += idle_pw
                    l40s_cfg = _idle_pool_entry(spec_l40s)

                # --- L4 pool ---
                l4_dc_rate    = beta * rate
                l4_pf_rate    = (1 - alpha) * rate
                l4_total_rate = l4_dc_rate + l4_pf_rate

                if l4_total_rate > 0:
                    l4_il  = self.DECODE_IL if l4_pf_rate == 0 else il
                    l4_ol  = ol if l4_dc_rate > 0 else self.PREFILL_OL
                    l4_slo = slo_dc_eff if l4_dc_rate >= l4_pf_rate else slo_pf_eff

                    max_l4_inst = spec_l4["total_gpus"] // tp_l4
                    l4_cfg = _optimize_pool(
                        self.schedulers["l4"], spec_l4,
                        l4_il, l4_ol, l4_total_rate, l4_slo,
                        max_active_instances=max_l4_inst,
                        allow_powergating=allow_powergating,
                    )
                    if l4_cfg is None:
                        continue  # infeasible for this (α, β, TP_l4)
                    total_power += l4_cfg["pool_power_w"]
                else:
                    idle_pw = spec_l4["total_gpus"] * spec_l4["idle_power_w"]
                    total_power += idle_pw
                    l4_cfg = _idle_pool_entry(spec_l4)

                if total_power < best_power:
                    best_power = total_power
                    slo_met = l40s_cfg["is_safe"] and l4_cfg["is_safe"]

                    config = self._make_config(
                        l40s_cfg, l4_cfg, spec_l40s, spec_l4,
                        alpha, beta,
                        l40s_pf_rate, l40s_dc_rate,
                        l4_dc_rate, l4_pf_rate,
                    )
                    best_result = SchedulingResult(total_power, config, slo_met)

                    if tp_changed:
                        best_result._tp_changed = True
                        best_result._new_tp = {"l40s": tp_l40s, "l4": tp_l4}
                    else:
                        best_result._tp_changed = False

        # Update TP state if best result found
        if best_result is not None and getattr(best_result, "_tp_changed", False):
            self.last_reconfig_time = current_time
            self.prev_tp = best_result._new_tp

        return best_result

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    @staticmethod
    def _make_config(l40s_cfg, l4_cfg, spec_l40s, spec_l4,
                     alpha, beta,
                     l40s_pf_rate, l40s_dc_rate,
                     l4_dc_rate, l4_pf_rate) -> dict:
        """Assemble the per-pool config dict for SchedulingResult."""
        config = {}

        if l40s_cfg and l40s_cfg["n_active"] > 0:
            config["l40s"] = {
                "tp":                   l40s_cfg.get("tp", 1),
                "freq_mhz":             l40s_cfg["freq_mhz"],
                "active_instances":     l40s_cfg["n_active"],
                "role":                 "prefill" if l40s_pf_rate >= l40s_dc_rate else "decode_spill",
                "alpha":                alpha,
                "rate_per_instance":    round((l40s_pf_rate + l40s_dc_rate) /
                                              max(l40s_cfg["n_active"], 1), 2),
                "power_per_instance_w": l40s_cfg["power_per_instance_w"],
                "pool_power_w":         l40s_cfg["pool_power_w"],
            }
        else:
            config["l40s"] = {
                "tp": 1, "freq_mhz": 0, "active_instances": 0,
                "role": "idle", "alpha": alpha,
                "rate_per_instance": 0, "power_per_instance_w": 0,
                "pool_power_w": spec_l40s["total_gpus"] * spec_l40s["idle_power_w"],
            }

        if l4_cfg and l4_cfg["n_active"] > 0:
            config["l4"] = {
                "tp":                   l4_cfg.get("tp", 1),
                "freq_mhz":             l4_cfg["freq_mhz"],
                "active_instances":     l4_cfg["n_active"],
                "role":                 "decode" if l4_dc_rate >= l4_pf_rate else "prefill_spill",
                "beta":                 beta,
                "rate_per_instance":    round((l4_dc_rate + l4_pf_rate) /
                                              max(l4_cfg["n_active"], 1), 2),
                "power_per_instance_w": l4_cfg["power_per_instance_w"],
                "pool_power_w":         l4_cfg["pool_power_w"],
            }
        else:
            config["l4"] = {
                "tp": 1, "freq_mhz": 0, "active_instances": 0,
                "role": "idle", "beta": beta,
                "rate_per_instance": 0, "power_per_instance_w": 0,
                "pool_power_w": spec_l4["total_gpus"] * spec_l4["idle_power_w"],
            }

        return config


# ---------------------------------------------------------------------------
# Ablation variant: fixed α=1, β=1 (no spill, no LP)
# ---------------------------------------------------------------------------

class JSEPDisaggNoSpillStrategy(JSEPDisaggStrategy):
    """
    Ablation: fully disaggregated (α=1, β=1) with no spill.
    Always routes all prefill to L40S and all decode to L4,
    regardless of saturation. LP mode disabled.
    """

    def decide(self, il, ol, rate, slo, cluster, current_time=0.0):
        result = self._try_alpha_beta(il, ol, rate, slo, cluster,
                                       current_time, alpha=1.0, beta=1.0)
        if result is None:
            result = self._burst_decide(il, ol, rate, slo, cluster,
                                        current_time)
        result.phase = "PREFILL+DECODE_DISAGG"
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_disagg_strategies(model_dir_l40s: str | None = None,
                              model_dir_l4:   str | None = None) -> dict:
    """Create disaggregated JSEP strategies with shared Phase A models."""
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4":   EnergyScheduler(model_dir=model_dir_l4),
    }
    return {
        "jsep_disagg":         JSEPDisaggStrategy(schedulers),
        "jsep_disagg_nospill": JSEPDisaggNoSpillStrategy(schedulers),
    }
