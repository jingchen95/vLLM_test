"""Static-disaggregation and GreenLLM baselines (extracted verbatim)."""
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import List, Optional

from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import SchedulingResult
from jsep_traces import Request

from .common import (
    ROUTE_AB,
    PoolBundle, SearchCandidate, SearchResult,
)
from .sweep import SweepLLMStrategy


class StaticDisaggStrategy(SweepLLMStrategy):
    """Static disaggregated baseline.

    Every window uses the same fixed configuration:
      * route A->B (L40S prefill, L4 decode) for every request class
      * L40S pool: TP=1, all 4 GPUs assigned to prefill, max frequency
      * L4 pool:   TP=1, all 8 GPUs assigned to decode,  max frequency
      * no DVFS, no TP search, no GPU allocation changes, no class-aware
        routing, no bundle stability gating, no emergency override

    Energy and feasibility predictions reuse the same EnergyScheduler models
    and the same KV-transfer cost (cfg.kv_transfer_ms_per_token x input_len)
    that SweepLLMStrategy uses, so the comparison is apples-to-apples under
    one declared interconnect assumption.
    """

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
        self._predict_cache = {}

        candidate = SearchCandidate(
            bundle_a=PoolBundle(1, GPU_SPECS["l40s"]["total_gpus"], 0),
            bundle_b=PoolBundle(1, 0, GPU_SPECS["l4"]["total_gpus"]),
            freq_a=GPU_SPECS["l40s"]["max_freq_mhz"],
            freq_b=GPU_SPECS["l4"]["max_freq_mhz"],
        )
        routes = {cls.class_id: ROUTE_AB for cls in summary.classes}

        # slo_met uses the SAME common rho-gate as every other strategy: the
        # config is feasible iff the rho-gated evaluation accepts it.  Energy
        # and per-class metrics come from the require_safe=False evaluation so
        # they are recorded whether or not the static config meets SLO.
        slo_met = self._evaluate_complete_candidate(
            summary, slo, candidate, routes, require_safe=True,
        ) is not None
        result = self._evaluate_complete_candidate(
            summary, slo, candidate, routes, require_safe=False,
        )
        if result is None:
            result = self._fallback(summary, slo)
        result = dc_replace(result, slo_met=slo_met)

        empty_search = self._empty_search_stats(skipped=True, reason="static_disagg")
        search_stats = {
            "state": "STATIC",
            "burst": False,
            "num_classes": len(summary.classes),
            "arrival_rate_rps": round(summary.arrival_rate, 2),
            "decision_time_s": 0.0,
            "emergency_override": False,
            "ideal": empty_search,
            "fast": empty_search,
        }
        config = self._build_runtime_config(result, state="STATIC", burst=False,
                                            search_stats=search_stats)
        return SchedulingResult(
            result.total_power_w,
            config,
            slo_met=slo_met,
            phase="STATIC",
        )


class GreenLLMStrategy(SweepLLMStrategy):
    """GreenLLM-inspired DVFS-only baseline (a.k.a. GreenLLM-OracleDVFS).

    Captures GreenLLM's (Ye et al., 2025) premise — fixed routing + fixed pool
    layout, only frequency adapts — but is **not** a faithful reproduction of
    its TPS-based frequency heuristic.  Instead it gives the baseline an
    *oracle* exhaustive sweep over all (freq_A, freq_B) pairs, picking the
    lowest-power pair that passes the common feasibility gate.  That makes it a
    *generous* DVFS baseline (conservative for SWEEP-LLM), not a literal
    GreenLLM.  In the paper it should be labelled "GreenLLM-OracleDVFS".

      * Routing fixed: A->B (L40S prefill, L4 decode) for every class.
      * Pool layout fixed: TP=1, all GPUs assigned to their phase. No TP
        search and no GPU allocation changes.
      * Frequency: oracle scan of all (freq_A, freq_B) pairs, lowest power
        among those the common rho-gate accepts; fall back to max freq if none.

    Uses the same EnergyScheduler backend and the same common rho-gate as every
    other strategy, so the comparison is apples-to-apples.
    """

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
        self._predict_cache = {}

        bundle_a = PoolBundle(1, GPU_SPECS["l40s"]["total_gpus"], 0)
        bundle_b = PoolBundle(1, 0, GPU_SPECS["l4"]["total_gpus"])
        routes = {cls.class_id: ROUTE_AB for cls in summary.classes}

        freqs_a = sorted(GPU_SPECS["l40s"]["frequencies"])
        freqs_b = sorted(GPU_SPECS["l4"]["frequencies"])

        best: Optional[SearchResult] = None
        for freq_a in freqs_a:
            for freq_b in freqs_b:
                candidate = SearchCandidate(bundle_a, bundle_b, freq_a, freq_b)
                result = self._evaluate_complete_candidate(
                    summary, slo, candidate, routes, require_safe=True,
                )
                if result is None:
                    continue
                if best is None or result.total_power_w < best.total_power_w:
                    best = result

        if best is None:
            # No rho-feasible (freq_a, freq_b) pair; report the max-freq config
            # so the simulator still records metrics.  slo_met uses the SAME
            # common rho-gate as every other strategy (feasible iff the
            # rho-gated evaluation accepts the max-freq config), not the
            # latency regression.
            candidate = SearchCandidate(
                bundle_a, bundle_b,
                GPU_SPECS["l40s"]["max_freq_mhz"],
                GPU_SPECS["l4"]["max_freq_mhz"],
            )
            slo_met = self._evaluate_complete_candidate(
                summary, slo, candidate, routes, require_safe=True,
            ) is not None
            best = self._evaluate_complete_candidate(
                summary, slo, candidate, routes, require_safe=False,
            )
            if best is None:
                best = self._fallback(summary, slo)
            best = dc_replace(best, slo_met=slo_met)

        empty_search = self._empty_search_stats(skipped=True, reason="greenllm_dvfs_only")
        search_stats = {
            "state": "GREENLLM",
            "burst": False,
            "num_classes": len(summary.classes),
            "arrival_rate_rps": round(summary.arrival_rate, 2),
            "decision_time_s": 0.0,
            "emergency_override": False,
            "ideal": empty_search,
            "fast": empty_search,
        }
        config = self._build_runtime_config(best, state="GREENLLM", burst=False,
                                            search_stats=search_stats)
        return SchedulingResult(
            best.total_power_w,
            config,
            slo_met=best.slo_met,
            phase="GREENLLM",
        )


