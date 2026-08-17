"""SWEEP-LLM ablation variants (extracted verbatim)."""
from __future__ import annotations

from typing import List, Tuple

from jsep_cluster import GPU_SPECS

from .common import (
    POOL_TO_GPU,
    ROUTE_AB,
    RequestClassSummary,
    PoolBundle,
)
from .sweep import SweepLLMStrategy


class SweepLLMNoRoutingStrategy(SweepLLMStrategy):
    """Ablation: SWEEP-LLM with class-aware routing disabled. Every class is
    pinned to ROUTE_AB; DVFS, TP, and active capacity remain free."""

    def _admissible_routes(self, cls: RequestClassSummary, state: str) -> Tuple[Tuple[str, str], ...]:
        return (ROUTE_AB,)


class SweepLLMNoDVFSStrategy(SweepLLMStrategy):
    """Ablation: SWEEP-LLM with DVFS disabled. Frequency is pinned to the
    max for each pool; routing, TP, and active capacity remain free."""

    def _select_frequency_pairs(self, state: str, burst: bool,
                                expanded: bool = False) -> List[Tuple[int, int]]:
        return [(GPU_SPECS["l40s"]["max_freq_mhz"], GPU_SPECS["l4"]["max_freq_mhz"])]


class SweepLLMNoTPStrategy(SweepLLMStrategy):
    """Ablation: SWEEP-LLM with TP fixed at 1 on both pools. The set of
    deployment bundles is restricted to TP=1 candidates; routing, DVFS,
    and active capacity remain free."""

    def _enumerate_pool_bundles(self, pool_label: str) -> List[PoolBundle]:
        return [b for b in super()._enumerate_pool_bundles(pool_label) if b.tp == 1]

    def _prefill_heavy_power_gate_anchors(self, state: str) -> List[Tuple[PoolBundle, PoolBundle]]:
        # TP=1 only — restrict to TP=1 anchors (symmetric and asymmetric)
        if state not in ("PREFILL_HEAVY", "BOTH_HEAVY"):
            return []
        power_gated_a = PoolBundle(tp=1, n_pf=0, n_dc=0)
        n_l4_total = GPU_SPECS["l4"]["total_gpus"]
        anchors = []
        for n_pf in range(1, n_l4_total + 1):
            for n_dc in range(1, n_pf + 1):  # n_dc <= n_pf, both TP=1
                if n_pf + n_dc <= n_l4_total:
                    anchors.append((power_gated_a, PoolBundle(tp=1, n_pf=n_pf, n_dc=n_dc)))
        return anchors


class SweepLLMNoCapacityStrategy(SweepLLMStrategy):
    """Ablation: SWEEP-LLM with active capacity locked at the pool maximum.
    Bundles are restricted to those that activate every GPU on the pool;
    routing, DVFS, and TP remain free."""

    def _enumerate_pool_bundles(self, pool_label: str) -> List[PoolBundle]:
        gpu_type = POOL_TO_GPU[pool_label]
        total = GPU_SPECS[gpu_type]["total_gpus"]
        return [b for b in super()._enumerate_pool_bundles(pool_label)
                if b.total_active_gpus == total]

    def _prefill_heavy_power_gate_anchors(self, state: str) -> List[Tuple[PoolBundle, PoolBundle]]:
        # no_capacity requires all GPUs active — power-gating L40S violates this constraint
        return []


