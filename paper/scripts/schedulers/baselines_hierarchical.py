"""Hierarchical-disaggregation baseline (extracted verbatim)."""
from __future__ import annotations

import math
from itertools import product
from typing import Dict, List, Optional, Tuple

from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import SchedulingResult
from jsep_traces import Request

from .common import (
    POOL_TO_GPU,
    ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB, ALL_ROUTES,
    PoolBundle, SearchCandidate, SearchResult,
)
from .sweep import SweepLLMStrategy


class HierarchicalDisaggStrategy(SweepLLMStrategy):
    """Sequential greedy baseline with full per-class cross-pool routing access.

    Answers: "Is SWEEP-LLM's advantage due to joint search, or just due to
    routing?"  Receives the same per-class routing options as SWEEP-LLM
    (ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB per class), but makes decisions
    sequentially rather than jointly:

        route (per class) → TP → DVFS → capacity

    each stage decided greedily given prior choices.

    Algorithm (per window):
      1. Route (per class): enumerate all 4^n_classes per-class route
         assignments; for each, evaluate at TP=1, max freq, max capacity.
         Pick the assignment with minimum energy (feasibility required).
      2. TP: given selected routes, enumerate all (tp_A, tp_B) pairs;
         pick the energy-minimum feasible pair at max freq + max capacity.
      3. DVFS: given routes + TP, enumerate all (freq_A, freq_B) pairs;
         pick the energy-minimum feasible pair at max capacity.
      4. Capacity: given routes + TP + freq, enumerate all valid bundle
         pairs; pick the energy-minimum feasible configuration.

    Key design properties:
    - Per-class routing: matches SWEEP-LLM's routing granularity; avoids
      the strawman of forcing a single route for all classes.
    - Energy-minimum TP (stage 2): not "first feasible" — avoids
      artificially handicapping the baseline with an asymmetric heuristic.
    - Stages 1–4 are independent and greedy; no bundle stability or
      state-guided candidate pruning.
    """

    @staticmethod
    def _pool_roles(routes: Dict[str, Tuple[str, str]]) -> Dict[str, Tuple[bool, bool]]:
        """Return {pool: (needs_pf, needs_dc)} for the given per-class route map."""
        roles: Dict[str, List[bool]] = {"A": [False, False], "B": [False, False]}
        for pf_pool, dc_pool in routes.values():
            roles[pf_pool][0] = True
            roles[dc_pool][1] = True
        return {k: (v[0], v[1]) for k, v in roles.items()}

    @staticmethod
    def _bundle_for_roles(pool_label: str, tp: int,
                          needs_pf: bool, needs_dc: bool) -> PoolBundle:
        """Max-capacity bundle for pool_label given which phases it handles."""
        total = GPU_SPECS[POOL_TO_GPU[pool_label]]["total_gpus"]
        if not needs_pf and not needs_dc:
            return PoolBundle(tp=1, n_pf=0, n_dc=0)
        if needs_pf and not needs_dc:
            return PoolBundle(tp=tp, n_pf=(total // tp) * tp, n_dc=0)
        if needs_dc and not needs_pf:
            return PoolBundle(tp=tp, n_pf=0, n_dc=(total // tp) * tp)
        # Both phases on same pool — split evenly
        per_phase = max(((total // 2) // tp) * tp, tp)
        while per_phase * 2 > total and per_phase > tp:
            per_phase -= tp
        return PoolBundle(tp=tp, n_pf=per_phase, n_dc=per_phase)

    def _eval_max_capacity(self, summary: WindowSummary, slo: int,
                           routes: Dict[str, Tuple[str, str]],
                           tp_a: int, tp_b: int,
                           freq_a: Optional[int] = None,
                           freq_b: Optional[int] = None,
                           require_safe: bool = True) -> Optional[SearchResult]:
        """Evaluate at max capacity for given routes, TP, and freq."""
        if freq_a is None:
            freq_a = GPU_SPECS["l40s"]["max_freq_mhz"]
        if freq_b is None:
            freq_b = GPU_SPECS["l4"]["max_freq_mhz"]
        pool_roles = self._pool_roles(routes)
        bun_a = self._bundle_for_roles("A", tp_a, *pool_roles["A"])
        bun_b = self._bundle_for_roles("B", tp_b, *pool_roles["B"])
        return self._evaluate_complete_candidate(
            summary, slo, SearchCandidate(bun_a, bun_b, freq_a, freq_b),
            routes, require_safe)

    def _hier_select_routes(self, summary: WindowSummary,
                            slo: int) -> Dict[str, Tuple[str, str]]:
        """Stage 1: per-class route selection at TP=1, max freq, max capacity.

        Enumerates all 4^n_classes per-class route assignments (≤256 for
        n_classes≤4); picks the assignment with minimum energy.
        """
        class_ids = [c.class_id for c in summary.classes]
        best_result: Optional[SearchResult] = None
        best_routes: Optional[Dict[str, Tuple[str, str]]] = None

        for route_combo in product(ALL_ROUTES, repeat=len(class_ids)):
            routes = dict(zip(class_ids, route_combo))
            r = self._eval_max_capacity(summary, slo, routes, tp_a=1, tp_b=1,
                                         require_safe=True)
            if r is not None:
                if best_result is None or r.total_energy_j < best_result.total_energy_j:
                    best_result = r
                    best_routes = routes

        if best_routes is None:
            # No route combo feasible at TP=1/max-freq/max-capacity.
            # Fall back to standard disagg; TP stage will handle the rest.
            best_routes = {cid: ROUTE_AB for cid in class_ids}
        return best_routes

    def _hier_select_tp(self, summary: WindowSummary, slo: int,
                        routes: Dict[str, Tuple[str, str]]) -> Tuple[int, int]:
        """Stage 2: energy-minimum feasible (tp_A, tp_B) at max freq + max capacity."""
        spec_a = GPU_SPECS["l40s"]
        spec_b = GPU_SPECS["l4"]
        best_result: Optional[SearchResult] = None
        best_tp: Optional[Tuple[int, int]] = None

        for tp_a in sorted(spec_a["tp_degrees"]):
            for tp_b in sorted(spec_b["tp_degrees"]):
                r = self._eval_max_capacity(summary, slo, routes,
                                             tp_a=tp_a, tp_b=tp_b, require_safe=True)
                if r is not None:
                    if best_result is None or r.total_energy_j < best_result.total_energy_j:
                        best_result = r
                        best_tp = (tp_a, tp_b)

        # Fallback: smallest TP (Stage 3+ will handle feasibility if infeasible here)
        return best_tp if best_tp is not None else (
            sorted(spec_a["tp_degrees"])[0], sorted(spec_b["tp_degrees"])[0])

    def _hier_select_freq(self, summary: WindowSummary, slo: int,
                          routes: Dict[str, Tuple[str, str]],
                          tp_a: int, tp_b: int) -> Tuple[int, int]:
        """Stage 3: energy-minimum feasible (freq_A, freq_B) at max capacity + selected TP."""
        spec_a = GPU_SPECS["l40s"]
        spec_b = GPU_SPECS["l4"]
        best_pair: Optional[Tuple[int, int]] = None
        best_energy = float("inf")

        for freq_a in spec_a["frequencies"]:
            for freq_b in spec_b["frequencies"]:
                r = self._eval_max_capacity(summary, slo, routes,
                                             tp_a=tp_a, tp_b=tp_b,
                                             freq_a=freq_a, freq_b=freq_b,
                                             require_safe=True)
                if r is not None and r.total_energy_j < best_energy:
                    best_energy = r.total_energy_j
                    best_pair = (freq_a, freq_b)

        return best_pair if best_pair is not None else (
            spec_a["max_freq_mhz"], spec_b["max_freq_mhz"])

    def _hier_select_capacity(self, summary: WindowSummary, slo: int,
                              routes: Dict[str, Tuple[str, str]],
                              tp_a: int, tp_b: int,
                              freq_a: int, freq_b: int) -> Optional[SearchResult]:
        """Stage 4: energy-minimum bundle pair at selected routes + TP + freq."""
        pool_roles = self._pool_roles(routes)
        # Keep only bundles consistent with pool roles and selected TP.
        def _admissible(bun: PoolBundle, needs_pf: bool, needs_dc: bool) -> bool:
            if bun.total_active_gpus == 0:
                return True  # power-gated is always admissible
            if needs_pf and bun.n_pf == 0:
                return False
            if needs_dc and bun.n_dc == 0:
                return False
            return True

        bundles_a = [b for b in self._enumerate_pool_bundles("A")
                     if (b.tp == tp_a or b.total_active_gpus == 0)
                     and _admissible(b, *pool_roles["A"])]
        bundles_b = [b for b in self._enumerate_pool_bundles("B")
                     if (b.tp == tp_b or b.total_active_gpus == 0)
                     and _admissible(b, *pool_roles["B"])]

        best: Optional[SearchResult] = None
        for bun_a in bundles_a:
            for bun_b in bundles_b:
                r = self._evaluate_complete_candidate(
                    summary, slo,
                    SearchCandidate(bun_a, bun_b, freq_a, freq_b),
                    routes, require_safe=True)
                if r is not None:
                    if best is None or r.total_energy_j < best.total_energy_j:
                        best = r
        return best

    def decide_window(self, requests: List[Request], slo,
                      cluster: ClusterState,
                      current_time: float = 0.0) -> SchedulingResult:
        del cluster, current_time

        if not requests:
            idle_power = sum(
                spec["total_gpus"] * spec["idle_power_w"] for spec in GPU_SPECS.values()
            )
            return SchedulingResult(
                idle_power,
                {
                    "l40s": {"tp": 1, "freq_mhz": 0, "active_instances": 0,
                             "pool_power_w": GPU_SPECS["l40s"]["total_gpus"] * GPU_SPECS["l40s"]["idle_power_w"]},
                    "l4":   {"tp": 1, "freq_mhz": 0, "active_instances": 0,
                             "pool_power_w": GPU_SPECS["l4"]["total_gpus"] * GPU_SPECS["l4"]["idle_power_w"]},
                    "routes": {},
                },
                slo_met=True,
                phase="IDLE",
            )

        summary = self._summarize_requests(requests)
        state, burst = self._classify_window(summary)
        self._predict_cache = {}

        routes = self._hier_select_routes(summary, slo)
        tp_a, tp_b = self._hier_select_tp(summary, slo, routes)
        freq_a, freq_b = self._hier_select_freq(summary, slo, routes, tp_a, tp_b)
        best = self._hier_select_capacity(summary, slo, routes, tp_a, tp_b, freq_a, freq_b)

        if best is None:
            best = self._fallback(summary, slo)

        empty_search = self._empty_search_stats(skipped=True, reason="hierarchical_disagg")
        search_stats = {
            "state": state,
            "burst": burst,
            "num_classes": len(summary.classes),
            "arrival_rate_rps": round(summary.arrival_rate, 2),
            "decision_time_s": 0.0,
            "emergency_override": False,
            "ideal": empty_search,
            "fast": empty_search,
        }
        config = self._build_runtime_config(best, state=state, burst=burst,
                                             search_stats=search_stats)
        return SchedulingResult(
            best.total_power_w,
            config,
            slo_met=best.slo_met,
            phase=state + "+HIER",
        )


