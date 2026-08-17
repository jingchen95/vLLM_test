"""DualScale-Ext two-tier baseline (extracted verbatim)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import replace as dc_replace
from itertools import product
from typing import Dict, List, Optional, Tuple

from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import SchedulingResult
from jsep_traces import Request
from scheduler import EnergyScheduler

from .common import (
    ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB,
    SweepLLMConfig, WindowSummary,
    PoolBundle, SearchCandidate, SearchResult,
)
from .sweep import SweepLLMStrategy


class DualScaleStrategy(SweepLLMStrategy):
    """DualScale-style two-tier baseline.

    Mirrors Basit et al. (2025) DualScale in its key architectural choices:

    Coarse tier (Tier 1) — runs every _DS_COARSE_EVERY windows (5 min, matching
    Basit et al.'s provisioning cadence):
      * Provisions for the PEAK load over the trailing lookback window
        (_DS_LOOKBACK_WINDOWS), scaled by the safety margin _DS_PEAK_MARGIN
        (α=5% in the paper).  This mirrors the paper's "past 5 min predicts
        next 5 min" prediction and its (1+α)R capacity constraint.  The chosen
        placement is then HELD for the whole 5-min block — reconfiguration is
        expensive (model weight loading, request draining), which is the paper's
        core motivation for the two-tier split (§3 C1).
      * Enumerates ALL four disaggregated placements (ROUTE_AA, ROUTE_AB,
        ROUTE_BA, ROUTE_BB) and all valid (TP, n_pf, n_dc) GPU allocation
        combinations, using each pool's max frequency as a design-point probe.
      * For monolithic routes (AA/BB): both phases on one GPU type, inactive
        pool is power-gated.  Mirrors DualScale's within-type disaggregation
        on a homogeneous cluster.
      * For cross-pool routes (AB/BA): prefill on one GPU type, decode on the
        other, with KV-transfer cost — the natural extension of DualScale's
        ILP to our heterogeneous cluster (the ILP would optimise over all GPU
        type assignments).
      * Selects the placement that minimises predicted energy at peak load.

    Fine tier (Tier 2) — runs every window:
      * Accepts the route and GPU allocation fixed by the coarse tier; placement
        does NOT change between coarse triggers.  During low-load windows the
        peak-sized placement is over-provisioned, and the surplus GPUs incur
        idle power (they cannot be power-gated without re-provisioning) — the
        very inefficiency the joint design is meant to avoid.
      * For monolithic routes: sweeps all frequencies of the active pool.
      * For cross-pool routes: sweeps all (freq_a, freq_b) pairs across both
        pools to find the lowest-energy SLO-feasible setting.
        This mirrors DualScale's per-phase DVFS adaptation (MPC for prefill,
        per-batch for decode), approximated here as an oracle frequency sweep.
      * If no frequency meets the SLO for the locked placement, it falls back to
        max frequency (the paper's Tier-2 fallback), accepting a violation
        rather than re-provisioning.

    Faithfulness note: The real DualScale system uses MPC-based prefill
    frequency prediction and slack-tracking decode frequency control. We
    implement the fine tier as a full-frequency sweep over the same
    EnergyScheduler predictive model used by all other strategies, which is
    an oracle upper-bound on DualScale's per-phase frequency selection quality.
    This makes the comparison conservative from SWEEP-LLM's perspective.
    """

    _DS_COARSE_EVERY: int = 60          # coarse tier interval (windows ≈ 5 min)
    _DS_LOOKBACK_WINDOWS: int = 60      # peak-load prediction window (≈ 5 min)
    _DS_PEAK_MARGIN: float = 1.05       # (1+α) provisioning margin, α=5%

    def __init__(self, schedulers: Dict[str, EnergyScheduler],
                 config: Optional[SweepLLMConfig] = None,
                 window_s: float = 5.0,
                 coarse_every: Optional[int] = None,
                 lookback_windows: Optional[int] = None,
                 strict_classifier_gate: bool = False) -> None:
        super().__init__(schedulers, config=config, window_s=window_s)
        # SLO-feasibility gate.  Default False -> use the SAME common rho-gate
        # as every other strategy (apples-to-apples).  True -> additionally
        # enable the pool-level is_safe classifier (a stricter sensitivity mode,
        # reported only in the appendix, never the main result).
        self._ds_strict_gate = strict_classifier_gate
        # Coarse-tier cadence and peak-prediction window are configurable for
        # cadence-sensitivity studies; defaults preserve the nominal 5-min
        # behaviour (60 windows at window_s=5s).  Lookback defaults to the
        # cadence so each block predicts its peak from the matching trailing
        # window ("past N min predicts next N min").
        if coarse_every is not None:
            self._DS_COARSE_EVERY = coarse_every
        if lookback_windows is not None:
            self._DS_LOOKBACK_WINDOWS = lookback_windows
        elif coarse_every is not None:
            self._DS_LOOKBACK_WINDOWS = coarse_every
        self._ds_route_type: Optional[Tuple[str, str]] = None
        self._ds_bundle_a: Optional[PoolBundle] = None
        self._ds_bundle_b: Optional[PoolBundle] = None
        self._ds_windows_since_coarse: int = 999  # force coarse on first window
        self._ds_history: deque = deque(maxlen=self._DS_LOOKBACK_WINDOWS)

    # _admissible_routes not used by DualScaleStrategy (coarse tier handles all routes directly)

    def decide_window(self, requests: List[Request], slo,
                      cluster: ClusterState,
                      current_time: float = 0.0) -> SchedulingResult:
        del cluster, current_time

        if not requests:
            idle_power = sum(
                spec["total_gpus"] * spec["idle_power_w"]
                for spec in GPU_SPECS.values()
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
        decision_start = time.perf_counter()

        # --- Coarse-tier trigger: purely periodic (≈5 min), matching the
        # paper's fixed provisioning cadence.  No load-change retrigger. ---
        run_coarse = (
            self._ds_route_type is None or
            self._ds_windows_since_coarse >= self._DS_COARSE_EVERY
        )
        if run_coarse:
            # Tier 1 provisions from PAST history only — no lookahead.  The
            # paper predicts the next 5-min window from the trailing 5 min.
            # The very first block has no history, so we bootstrap from the
            # current window; that block is warmup and is excluded from
            # reported metrics (the paper likewise reports from minute 5 on).
            peak_summary = (
                self._ds_peak_summary() if self._ds_history
                else self._scale_summary(summary, self._DS_PEAK_MARGIN)
            )
            placement = self._coarse_tier_search(peak_summary, slo)
            if placement is not None:
                self._ds_route_type, self._ds_bundle_a, self._ds_bundle_b = placement
            self._ds_windows_since_coarse = 0

        # --- Fine-tier: sweep frequencies given fixed coarse placement.
        # Placement is locked between coarse triggers; only frequency moves. ---
        result = None
        if self._ds_route_type is not None:
            result = self._fine_tier_search(summary, slo)

        # Last resort only if no placement has ever been set (first window with
        # an infeasible peak).  The fine tier already falls back to max freq for
        # a locked placement, so this should rarely trigger.
        if result is None:
            result = self._fallback(summary, slo)

        # Advance the coarse-tier clock once per window (including the window a
        # coarse trigger just fired on), so the next trigger lands exactly
        # _DS_COARSE_EVERY windows later rather than one window late.
        self._ds_windows_since_coarse += 1

        # Record this window for FUTURE coarse triggers — appended only after
        # serving it, so Tier 1 never sees the window it is provisioning for.
        self._ds_history.append(summary)

        decision_time_s = time.perf_counter() - decision_start
        search_stats = {
            "state": state,
            "burst": burst,
            "num_classes": len(summary.classes),
            "arrival_rate_rps": round(summary.arrival_rate, 2),
            "decision_time_s": round(decision_time_s, 4),
            "emergency_override": False,
            "ideal": self._empty_search_stats(skipped=(not run_coarse),
                                              reason="dualscale_coarse"),
            "fast": self._empty_search_stats(skipped=False,
                                             reason="dualscale_fine"),
        }
        config = self._build_runtime_config(result, state=state, burst=burst,
                                            search_stats=search_stats)
        return SchedulingResult(
            result.total_power_w, config,
            slo_met=result.slo_met,
            phase=state + ("+BURST" if burst else ""),
        )

    def _ds_peak_summary(self) -> WindowSummary:
        """Peak-load window over the trailing lookback, scaled by the margin.

        Tier 1 provisions for predicted peak load: the paper uses the past
        5 min as the prediction for the next 5 min and sizes capacity to
        (1+α)R.  We take the highest-arrival-rate window in the trailing
        history (its actual class mix) and scale all per-class rates by
        _DS_PEAK_MARGIN.
        """
        peak = max(self._ds_history, key=lambda s: s.arrival_rate)
        return self._scale_summary(peak, self._DS_PEAK_MARGIN)

    @staticmethod
    def _scale_summary(summary: WindowSummary, factor: float) -> WindowSummary:
        scaled = tuple(
            dc_replace(cls, request_rate=cls.request_rate * factor)
            for cls in summary.classes
        )
        return WindowSummary(arrival_rate=summary.arrival_rate * factor,
                             classes=scaled)

    def _evaluate_dualscale_candidate(
        self, summary: WindowSummary, slo: int,
        candidate: SearchCandidate, routes: Dict[str, Tuple[str, str]],
        require_safe: bool = True,
    ) -> Optional[SearchResult]:
        """SLO-feasibility evaluation for DualScale's two tiers.

        Default (strict_classifier_gate=False): delegate to the SAME common
        rho-gate used by every other strategy (_evaluate_complete_candidate),
        so the cross-strategy comparison is apples-to-apples — no strategy is
        held to a stricter or looser feasibility standard than the others.

        Strict mode (strict_classifier_gate=True, sensitivity only): also enable
        the pool-level is_safe classifier so DualScale cannot bank energy on
        placements the looser rho gate would accept.  Reported in the appendix,
        never the main result.
        """
        if not self._ds_strict_gate:
            return self._evaluate_complete_candidate(
                summary, slo, candidate, routes, require_safe=require_safe
            )
        if len(routes) != len(summary.classes):
            return None
        loads = self._build_phase_loads(summary, routes)
        pool_details, total_power = self._predict_pool_details(
            candidate, loads, slo, require_safe=require_safe
        )
        if pool_details is None:
            return None
        per_class = self._predict_class_metrics(
            summary, candidate, loads, routes, slo, require_safe=require_safe
        )
        if per_class is None:
            return None
        return SearchResult(
            candidate=candidate,
            routes=dict(routes),
            total_power_w=round(total_power, 1),
            total_energy_j=round(total_power * self.window_s, 1),
            slo_met=require_safe,
            per_class_metrics=per_class,
            pool_details=pool_details,
        )

    def _coarse_tier_search(
        self, summary: WindowSummary, slo: int
    ) -> Optional[Tuple[Tuple[str, str], PoolBundle, PoolBundle]]:
        """Pick disaggregated placement and bundle at max-freq design point.

        Enumerates all four routes (AA, AB, BA, BB).  Monolithic routes
        (AA/BB) put both phases on one pool; cross-pool routes (AB/BA) split
        prefill and decode across pool types, mirroring DualScale's ILP
        extended to our heterogeneous cluster.
        """
        spec_a = GPU_SPECS["l40s"]
        spec_b = GPU_SPECS["l4"]
        max_freq_a = spec_a["max_freq_mhz"]
        max_freq_b = spec_b["max_freq_mhz"]
        min_freq_a = list(self.schedulers["l40s"].FREQUENCIES)[0]
        min_freq_b = list(self.schedulers["l4"].FREQUENCIES)[0]
        inactive = PoolBundle(tp=1, n_pf=0, n_dc=0)
        best: Optional[Tuple[Tuple[str, str], PoolBundle, PoolBundle, float]] = None

        # --- Monolithic routes: both phases on one pool, other pool idle ---
        for route_type, active_label, spec, design_freq, inactive_freq in (
            (ROUTE_AA, "A", spec_a, max_freq_a, min_freq_b),
            (ROUTE_BB, "B", spec_b, max_freq_b, min_freq_a),
        ):
            routes = {cls.class_id: route_type for cls in summary.classes}
            for tp in spec["tp_degrees"]:
                for n_pf in range(tp, spec["total_gpus"] + 1, tp):
                    for n_dc in range(tp, spec["total_gpus"] + 1, tp):
                        if n_pf + n_dc > spec["total_gpus"]:
                            continue
                        active_bundle = PoolBundle(tp=tp, n_pf=n_pf, n_dc=n_dc)
                        if active_label == "A":
                            ba, bb = active_bundle, inactive
                            cand = SearchCandidate(ba, bb, design_freq, inactive_freq)
                        else:
                            ba, bb = inactive, active_bundle
                            cand = SearchCandidate(ba, bb, inactive_freq, design_freq)
                        result = self._evaluate_dualscale_candidate(
                            summary, slo, cand, routes, require_safe=True
                        )
                        if result is None:
                            continue
                        if best is None or result.total_energy_j < best[3]:
                            best = (route_type, ba, bb, result.total_energy_j)

        # --- Cross-pool routes: prefill on one pool, decode on the other ---
        # ROUTE_AB: L40S prefill (bundle_a has n_pf>0, n_dc=0)
        #           L4 decode   (bundle_b has n_pf=0, n_dc>0)
        # ROUTE_BA: L4 prefill  (bundle_b has n_pf>0, n_dc=0)
        #           L40S decode (bundle_a has n_pf=0, n_dc>0)
        for route_type, pf_spec, dc_spec in (
            (ROUTE_AB, spec_a, spec_b),
            (ROUTE_BA, spec_b, spec_a),
        ):
            routes = {cls.class_id: route_type for cls in summary.classes}
            is_ab = (route_type == ROUTE_AB)
            for tp_pf in pf_spec["tp_degrees"]:
                for n_pf in range(tp_pf, pf_spec["total_gpus"] + 1, tp_pf):
                    for tp_dc in dc_spec["tp_degrees"]:
                        for n_dc in range(tp_dc, dc_spec["total_gpus"] + 1, tp_dc):
                            if is_ab:
                                # L40S = prefill only, L4 = decode only
                                ba = PoolBundle(tp=tp_pf, n_pf=n_pf, n_dc=0)
                                bb = PoolBundle(tp=tp_dc, n_pf=0, n_dc=n_dc)
                            else:
                                # L4 = prefill only, L40S = decode only
                                ba = PoolBundle(tp=tp_dc, n_pf=0, n_dc=n_dc)
                                bb = PoolBundle(tp=tp_pf, n_pf=n_pf, n_dc=0)
                            cand = SearchCandidate(ba, bb, max_freq_a, max_freq_b)
                            result = self._evaluate_dualscale_candidate(
                                summary, slo, cand, routes, require_safe=True
                            )
                            if result is None:
                                continue
                            if best is None or result.total_energy_j < best[3]:
                                best = (route_type, ba, bb, result.total_energy_j)

        if best is None:
            return None
        return best[0], best[1], best[2]

    def _fine_tier_search(
        self, summary: WindowSummary, slo: int
    ) -> Optional[SearchResult]:
        """Sweep frequencies for locked placement; return min-energy SLO-feasible result.

        Monolithic routes (AA/BB): sweep the active pool's frequencies only.
        Cross-pool routes (AB/BA): sweep all (freq_a, freq_b) pairs across
        both pools, mirroring DualScale's independent per-phase DVFS control.
        """
        routes = {cls.class_id: self._ds_route_type for cls in summary.classes}
        freqs_a = list(self.schedulers["l40s"].FREQUENCIES)
        freqs_b = list(self.schedulers["l4"].FREQUENCIES)
        min_freq_a = freqs_a[0]
        min_freq_b = freqs_b[0]

        if self._ds_route_type in (ROUTE_AA, ROUTE_BB):
            # Only one pool is active; pin the idle pool to its minimum freq.
            is_aa = (self._ds_route_type == ROUTE_AA)
            sweep_freqs = freqs_a if is_aa else freqs_b
            freq_pairs = [
                (f, min_freq_b) if is_aa else (min_freq_a, f)
                for f in sweep_freqs
            ]
        else:
            # Both pools active — sweep the full Cartesian product.
            from itertools import product as iproduct
            freq_pairs = list(iproduct(freqs_a, freqs_b))

        best: Optional[SearchResult] = None
        for freq_a, freq_b in freq_pairs:
            cand = SearchCandidate(self._ds_bundle_a, self._ds_bundle_b, freq_a, freq_b)
            result = self._evaluate_dualscale_candidate(
                summary, slo, cand, routes, require_safe=True
            )
            if result is None:
                continue
            if best is None or result.total_energy_j < best.total_energy_j:
                best = result
        if best is not None:
            return best

        # Tier-2 fallback: no frequency meets the SLO for the locked placement.
        # The paper falls back to max frequency to preserve SLO compliance; if
        # the placement is fundamentally under-provisioned, this still records a
        # violation (slo_met=False) rather than re-provisioning the cluster.
        max_freq_a = list(self.schedulers["l40s"].FREQUENCIES)[-1]
        max_freq_b = list(self.schedulers["l4"].FREQUENCIES)[-1]
        max_cand = SearchCandidate(self._ds_bundle_a, self._ds_bundle_b,
                                   max_freq_a, max_freq_b)
        return self._evaluate_dualscale_candidate(
            summary, slo, max_cand, routes, require_safe=False
        )




# ---------------------------------------------------------------------------
# Cadence variants for the provisioning-cadence sensitivity study.
# 5-min (60 windows) is the nominal DualScale-Ext (the base class default);
# these subclasses only change the coarse-provisioning interval and the
# matching peak-prediction lookback.  20s is an aggressive cadence (not
# faithful to the paper) reported as an upper bound.
# ---------------------------------------------------------------------------
class DualScaleExt20s(DualScaleStrategy):
    """DualScale-Ext, 20s coarse cadence (aggressive upper bound; not faithful)."""
    _DS_COARSE_EVERY = 4
    _DS_LOOKBACK_WINDOWS = 4


class DualScaleExt1Min(DualScaleStrategy):
    """DualScale-Ext, 1-min coarse cadence."""
    _DS_COARSE_EVERY = 12
    _DS_LOOKBACK_WINDOWS = 12


class DualScaleExt2Min(DualScaleStrategy):
    """DualScale-Ext, 2-min coarse cadence."""
    _DS_COARSE_EVERY = 24
    _DS_LOOKBACK_WINDOWS = 24
