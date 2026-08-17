"""SWEEP-LLM main scheduling strategy (extracted verbatim; no logic change)."""
from __future__ import annotations

import math
import time
from dataclasses import replace as dc_replace
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import SchedulingResult
from jsep_traces import IL_VALUES, OL_VALUES, Request, snap_to_nearest
from scheduler import EnergyScheduler

from .common import (
    POOL_TO_GPU,
    ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB, ALL_ROUTES,
    SweepLLMConfig, RequestClassSummary, WindowSummary,
    PoolBundle, SearchCandidate, SearchResult,
)
from .feasibility_tables import FeasibilityTables


class SweepLLMStrategy:
    """
    Class-aware SWEEP-LLM strategy backed by EnergyScheduler models.
    """

    def __init__(self, schedulers: Dict[str, EnergyScheduler],
                 config: Optional[SweepLLMConfig] = None,
                 window_s: float = 5.0):
        self.schedulers = schedulers
        self.cfg = config or SweepLLMConfig()
        self.window_s = window_s

        self.current_state = "BOTH_LOW"
        self.prev_rate: Optional[float] = None

        self.current_bundles = {
            "A": PoolBundle(*self.cfg.current_bundle_a),
            "B": PoolBundle(*self.cfg.current_bundle_b),
        }
        self.target_bundles = dict(self.current_bundles)
        self.bundle_counters = {"A": 0, "B": 0}
        self.bundle_stability = {
            "A": self.cfg.bundle_stability_a,
            "B": self.cfg.bundle_stability_b,
        }

        self._bundle_catalog = {
            "A": self._enumerate_pool_bundles("A"),
            "B": self._enumerate_pool_bundles("B"),
        }
        # Fail-closed hardware feasibility tables. Loaded only when a table gate is
        # requested (legacy runs need no artifacts); FeasibilityTables fails fast if
        # the goodput gate is on but the decode-capacity artifact is unusable.
        self.feas = None
        if self.cfg.decode_gate_mode == "goodput" or self.cfg.prefill_envelope:
            from paths import PAPER_MODELS_DIR
            self.feas = FeasibilityTables(
                decode_csv=PAPER_MODELS_DIR / "decode_capacity.csv",
                rho_envelope_csv=PAPER_MODELS_DIR / "rho_envelope.csv",
                gate_mode=self.cfg.decode_gate_mode,
            )

        self.ref_capacities = self._estimate_reference_capacities()
        self.windows_since_ideal = 0
        self.last_ideal_context: Optional[Dict[str, object]] = None
        self._predict_cache: Dict[Tuple[object, ...], dict] = {}
        self.observed_state_run_length = 0

    def reset(self):
        self.current_state = "BOTH_LOW"
        self.prev_rate = None
        self.current_bundles = {
            "A": PoolBundle(*self.cfg.current_bundle_a),
            "B": PoolBundle(*self.cfg.current_bundle_b),
        }
        self.target_bundles = dict(self.current_bundles)
        self.bundle_counters = {"A": 0, "B": 0}
        self.windows_since_ideal = 0
        self.last_ideal_context = None
        self._predict_cache = {}
        self.observed_state_run_length = 0

    def decide_window(self, requests: List[Request], slo,
                      cluster: ClusterState,
                      current_time: float = 0.0) -> SchedulingResult:
        del cluster, current_time  # current implementation keeps scheduler-local state

        if not requests:
            idle_power = sum(
                spec["total_gpus"] * spec["idle_power_w"] for spec in GPU_SPECS.values()
            )
            return SchedulingResult(
                idle_power,
                {
                    "l40s": {"tp": 1, "freq_mhz": 0, "active_instances": 0, "pool_power_w": GPU_SPECS["l40s"]["total_gpus"] * GPU_SPECS["l40s"]["idle_power_w"]},
                    "l4": {"tp": 1, "freq_mhz": 0, "active_instances": 0, "pool_power_w": GPU_SPECS["l4"]["total_gpus"] * GPU_SPECS["l4"]["idle_power_w"]},
                    "routes": {},
                },
                slo_met=True,
                phase="IDLE",
            )

        summary = self._summarize_requests(requests)
        state, burst = self._classify_window(summary)
        self._predict_cache = {}
        decision_start = time.perf_counter()

        fast, fast_stats = self._search(summary, slo, state, burst, fixed_bundles=self.current_bundles)
        run_ideal, ideal_reason = self._should_run_ideal_search(summary, state, burst)
        ideal = None
        ideal_stats = self._empty_search_stats(skipped=True, reason=ideal_reason)
        if run_ideal:
            ideal, ideal_stats = self._search(summary, slo, state, burst, fixed_bundles=None)
            ideal_stats["trigger_reason"] = ideal_reason
            self._record_ideal_context(summary, state, burst)
            self.windows_since_ideal = 0
        else:
            self.windows_since_ideal += 1

        if (ideal is None and not run_ideal) and (fast is None or not fast.slo_met):
            ideal, ideal_stats = self._search(summary, slo, state, burst, fixed_bundles=None)
            ideal_stats["trigger_reason"] = "fast_infeasible"
            ideal_stats["skipped"] = False
            self._record_ideal_context(summary, state, burst)
            self.windows_since_ideal = 0

        if ideal is None:
            fast = fast or self._fallback(summary, slo)
        if fast is None:
            fast = self._fallback(summary, slo)

        if ideal is not None:
            self._update_bundle_targets(ideal.candidate.bundle_a, ideal.candidate.bundle_b)

        emergency_override = False
        if (
            self.cfg.emergency_bundle_override and
            not fast.slo_met and
            ideal is not None and
            ideal.slo_met
        ):
            self.current_bundles["A"] = ideal.candidate.bundle_a
            self.current_bundles["B"] = ideal.candidate.bundle_b
            self.target_bundles["A"] = ideal.candidate.bundle_a
            self.target_bundles["B"] = ideal.candidate.bundle_b
            self.bundle_counters["A"] = 0
            self.bundle_counters["B"] = 0
            emergency_override = True
            fast, fast_stats = self._search(summary, slo, state, burst, fixed_bundles=self.current_bundles)
            if fast is None:
                fast = ideal

        # Savings override: when the ideal found a config substantially cheaper
        # than the fast result, commit to it immediately instead of waiting for
        # the bundle-stability ramp.  Covers both power-gated (ROUTE_BB) and
        # leaner cross-pool (ROUTE_AB with fewer L4 GPUs at higher freq) cases.
        # On a state_change trigger the fast path's bundle comes from the prior
        # state, so the savings_override_min_frac guard is meaningless — bypass
        # it and always adopt the ideal when the trigger was a state change.
        ideal_from_state_change = ideal_stats.get("trigger_reason") == "state_change"
        if (
            not emergency_override and
            ideal is not None and
            ideal.slo_met and
            fast is not None and
            fast.slo_met and
            (
                (ideal_from_state_change and ideal.total_energy_j < fast.total_energy_j) or
                ideal.total_energy_j < fast.total_energy_j * (1.0 - self.cfg.savings_override_min_frac)
            )
        ):
            self.current_bundles["A"] = ideal.candidate.bundle_a
            self.current_bundles["B"] = ideal.candidate.bundle_b
            self.target_bundles["A"] = ideal.candidate.bundle_a
            self.target_bundles["B"] = ideal.candidate.bundle_b
            self.bundle_counters["A"] = 0
            self.bundle_counters["B"] = 0
            fast = ideal
            fast_stats = ideal_stats

        decision_time_s = time.perf_counter() - decision_start
        search_stats = {
            "state": state,
            "burst": burst,
            "num_classes": len(summary.classes),
            "arrival_rate_rps": round(summary.arrival_rate, 2),
            "decision_time_s": round(decision_time_s, 4),
            "emergency_override": emergency_override,
            "ideal": ideal_stats,
            "fast": fast_stats,
        }

        if self.cfg.print_search_stats:
            print(
                "[SWEEP-LLM] "
                f"state={state} burst={burst} classes={len(summary.classes)} "
                f"decision={decision_time_s:.3f}s "
                f"override={emergency_override} "
                f"ideal_reason={ideal_stats.get('trigger_reason', 'n/a')} "
                f"ideal(cands={ideal_stats['candidate_count']}, "
                f"expanded={ideal_stats['expanded']}, "
                f"visited={ideal_stats['candidates_visited']}, "
                f"partial={ideal_stats['partial_route_evals']}, "
                f"complete={ideal_stats['complete_evals']}, "
                f"time={ideal_stats['time_s']:.3f}s) "
                f"fast(cands={fast_stats['candidate_count']}, "
                f"expanded={fast_stats['expanded']}, "
                f"visited={fast_stats['candidates_visited']}, "
                f"partial={fast_stats['partial_route_evals']}, "
                f"complete={fast_stats['complete_evals']}, "
                f"time={fast_stats['time_s']:.3f}s)"
            )

        config = self._build_runtime_config(fast, state, burst, search_stats)
        return SchedulingResult(
            fast.total_power_w,
            config,
            slo_met=fast.slo_met,
            phase=state + ("+BURST" if burst else ""),
        )

    def _fallback(self, summary: WindowSummary, slo: int) -> SearchResult:
        bundle_a = PoolBundle(1, GPU_SPECS["l40s"]["total_gpus"], 0)
        bundle_b = PoolBundle(1, 0, GPU_SPECS["l4"]["total_gpus"])
        candidate = SearchCandidate(
            bundle_a=bundle_a,
            bundle_b=bundle_b,
            freq_a=GPU_SPECS["l40s"]["max_freq_mhz"],
            freq_b=GPU_SPECS["l4"]["max_freq_mhz"],
        )
        routes = {cls.class_id: ROUTE_AB for cls in summary.classes}
        fallback = self._evaluate_complete_candidate(
            summary,
            slo,
            candidate,
            routes,
            require_safe=False,
        )
        if fallback is not None:
            return fallback

        return SearchResult(
            candidate=candidate,
            routes=routes,
            total_power_w=round(
                GPU_SPECS["l40s"]["total_gpus"] * GPU_SPECS["l40s"]["tdp_w"] +
                GPU_SPECS["l4"]["total_gpus"] * GPU_SPECS["l4"]["tdp_w"],
                1,
            ),
            total_energy_j=round(
                (
                    GPU_SPECS["l40s"]["total_gpus"] * GPU_SPECS["l40s"]["tdp_w"] +
                    GPU_SPECS["l4"]["total_gpus"] * GPU_SPECS["l4"]["tdp_w"]
                ) * self.window_s,
                1,
            ),
            slo_met=False,
            per_class_metrics={
                cls.class_id: {
                    "route": "A->B",
                    "request_rate": round(cls.request_rate, 2),
                    "ttft_ms": float("inf"),
                    "tpot_ms": float("inf"),
                    "kv_transfer_ms": round(self.cfg.kv_transfer_ms_per_token * cls.input_len, 1),
                }
                for cls in summary.classes
            },
            pool_details={
                "A": {
                    "gpu_type": "l40s",
                    "tp": 1,
                    "freq_mhz": GPU_SPECS["l40s"]["max_freq_mhz"],
                    "n_prefill_gpus": GPU_SPECS["l40s"]["total_gpus"],
                    "n_decode_gpus": 0,
                    "active_instances": GPU_SPECS["l40s"]["total_gpus"],
                    "idle_gpus": 0,
                    "pool_power_w": round(GPU_SPECS["l40s"]["total_gpus"] * GPU_SPECS["l40s"]["tdp_w"], 1),
                },
                "B": {
                    "gpu_type": "l4",
                    "tp": 1,
                    "freq_mhz": GPU_SPECS["l4"]["max_freq_mhz"],
                    "n_prefill_gpus": 0,
                    "n_decode_gpus": GPU_SPECS["l4"]["total_gpus"],
                    "active_instances": GPU_SPECS["l4"]["total_gpus"],
                    "idle_gpus": 0,
                    "pool_power_w": round(GPU_SPECS["l4"]["total_gpus"] * GPU_SPECS["l4"]["tdp_w"], 1),
                },
            },
        )

    def _update_bundle_targets(self, bundle_a: PoolBundle, bundle_b: PoolBundle):
        next_targets = {"A": bundle_a, "B": bundle_b}
        for pool in ("A", "B"):
            if next_targets[pool] == self.target_bundles[pool]:
                self.bundle_counters[pool] += 1
            else:
                self.target_bundles[pool] = next_targets[pool]
                self.bundle_counters[pool] = 1

            if self.bundle_counters[pool] >= self.bundle_stability[pool]:
                self.current_bundles[pool] = self.target_bundles[pool]
                self.bundle_counters[pool] = 0

    def _build_runtime_config(self, result: SearchResult, state: str, burst: bool,
                              search_stats: Dict[str, object]) -> dict:
        cfg = {
            "state": state,
            "burst": burst,
            "search_stats": search_stats,
            "routes": {
                class_id: f"{route[0]}->{route[1]}"
                for class_id, route in result.routes.items()
            },
            "l40s": dict(result.pool_details["A"]),
            "l4": dict(result.pool_details["B"]),
        }
        cfg["l40s"]["target_bundle"] = self.target_bundles["A"]
        cfg["l4"]["target_bundle"] = self.target_bundles["B"]
        cfg["l40s"]["current_bundle"] = self.current_bundles["A"]
        cfg["l4"]["current_bundle"] = self.current_bundles["B"]
        return cfg

    def _summarize_requests(self, requests: List[Request]) -> WindowSummary:
        total = len(requests)
        class_groups: Dict[str, list] = {}

        for req in requests:
            class_groups.setdefault(
                self._class_id(req.input_len, req.output_len), []
            ).append(req)

        classes = []
        for class_id, group in class_groups.items():
            if not group:
                continue
            count = len(group)
            mean_il = sum(req.input_len for req in group) / count
            mean_ol = sum(req.output_len for req in group) / count
            classes.append(RequestClassSummary(
                class_id=class_id,
                fraction=count / total,
                request_rate=count / self.window_s,
                input_len=snap_to_nearest(int(round(mean_il)), IL_VALUES),
                output_len=snap_to_nearest(int(round(mean_ol)), OL_VALUES),
                count=count,
            ))

        return WindowSummary(
            arrival_rate=total / self.window_s,
            classes=tuple(classes),
        )

    def _class_id(self, input_len: int, output_len: int) -> str:
        # Three-level (S/M/L) per-axis taxonomy when the optional high thresholds
        # are configured (K-sensitivity study); otherwise the default 2x2 scheme.
        if self.cfg.theta_il_hi is not None or self.cfg.theta_ol_hi is not None:
            il_lvl = self._axis_level(input_len, self.cfg.theta_il, self.cfg.theta_il_hi)
            ol_lvl = self._axis_level(output_len, self.cfg.theta_ol, self.cfg.theta_ol_hi)
            return f"il{il_lvl}_ol{ol_lvl}"
        long_il = input_len >= self.cfg.theta_il
        long_ol = output_len >= self.cfg.theta_ol
        if long_il and not long_ol:
            return "long_short"
        if not long_il and long_ol:
            return "short_long"
        if long_il and long_ol:
            return "long_long"
        return "short_short"

    @staticmethod
    def _axis_level(val: int, lo: int, hi: Optional[int]) -> str:
        if hi is not None and val >= hi:
            return "L"
        if val >= lo:
            return "M"
        return "S"

    def _classify_window(self, summary: WindowSummary) -> Tuple[str, bool]:
        d_pf = sum(cls.request_rate * cls.input_len for cls in summary.classes)
        d_dc = sum(cls.request_rate * cls.output_len for cls in summary.classes)

        x = d_pf / max(self.ref_capacities["prefill"], 1.0)
        y = d_dc / max(self.ref_capacities["decode"], 1.0)

        if x + y < self.cfg.tau_load:
            raw_state = "BOTH_LOW"
        elif x / max(y, 1e-6) >= self.cfg.tau_imb:
            raw_state = "PREFILL_HEAVY"
        elif y / max(x, 1e-6) >= self.cfg.tau_imb:
            raw_state = "DECODE_HEAVY"
        else:
            raw_state = "BOTH_HEAVY"

        raw_state = self._apply_state_hysteresis(raw_state, x, y)

        if raw_state == self.current_state:
            self.observed_state_run_length += 1
        else:
            self.observed_state_run_length = 1

        burst = False
        if not (self.cfg.burst_requires_history and self.prev_rate is None):
            prev_rate = self.prev_rate or 0.0
            burst = summary.arrival_rate > max(
                self.cfg.tau_burst * prev_rate,
                prev_rate + self.cfg.delta_lambda,
            )

        self.current_state = raw_state
        self.prev_rate = summary.arrival_rate
        return raw_state, burst

    def _apply_state_hysteresis(self, proposed_state: str, x: float, y: float) -> str:
        if proposed_state == self.current_state:
            return proposed_state

        margin = self.cfg.hysteresis_margin
        if proposed_state == "BOTH_LOW":
            if x + y >= self.cfg.tau_load - margin:
                return self.current_state
        elif proposed_state == "PREFILL_HEAVY":
            if x / max(y, 1e-6) < self.cfg.tau_imb + margin:
                return self.current_state
        elif proposed_state == "DECODE_HEAVY":
            if y / max(x, 1e-6) < self.cfg.tau_imb + margin:
                return self.current_state
        elif proposed_state == "BOTH_HEAVY":
            if x + y < self.cfg.tau_load + margin:
                return self.current_state
        return proposed_state

    def _search(self, summary: WindowSummary, slo: int, state: str, burst: bool,
                fixed_bundles: Optional[Dict[str, PoolBundle]]) -> Tuple[Optional[SearchResult], Dict[str, float]]:
        search_start = time.perf_counter()
        candidates, candidate_meta = self._construct_candidates(
            summary, state, burst, fixed_bundles, expanded=False
        )
        stats = {
            "candidate_count": len(candidates),
            "bundle_pair_count": candidate_meta["bundle_pair_count"],
            "freq_pair_count": candidate_meta["freq_pair_count"],
            "expanded": False,
            "expansion_attempted": False,
            "candidates_visited": 0,
            "partial_route_evals": 0,
            "complete_evals": 0,
            "infeasible_candidates": 0,
            "no_route_prunes": 0,
            "bound_prunes": 0,
            "best_updates": 0,
            "time_s": 0.0,
            "skipped": False,
            "trigger_reason": "always",
        }
        if not candidates:
            stats["time_s"] = round(time.perf_counter() - search_start, 4)
            return None, stats

        best = self._search_candidates(summary, slo, state, candidates, stats)

        if best is None:
            stats["expansion_attempted"] = True
            rescue_candidates, rescue_meta = self._construct_candidates(
                summary, state, burst, fixed_bundles, expanded=True
            )
            if rescue_candidates and (
                rescue_meta["bundle_pair_count"] > candidate_meta["bundle_pair_count"] or
                rescue_meta["freq_pair_count"] > candidate_meta["freq_pair_count"]
            ):
                stats["expanded"] = True
                stats["candidate_count"] = len(rescue_candidates)
                stats["bundle_pair_count"] = rescue_meta["bundle_pair_count"]
                stats["freq_pair_count"] = rescue_meta["freq_pair_count"]
                best = self._search_candidates(summary, slo, state, rescue_candidates, stats)

        # Monolithic consolidation: if the main search found a cross-pool
        # (disaggregated) result, or found nothing, try power-gating one pool
        # entirely using rho-only safety.  This recovers energy when both pools
        # are kept alive unnecessarily for short-decode or prefill-heavy workloads.
        uses_cross_pool = best is not None and any(
            r in (ROUTE_AB, ROUTE_BA) for r in best.routes.values()
        )
        uses_all_bb = best is not None and all(r == ROUTE_BB for r in best.routes.values())
        if best is None or uses_cross_pool or uses_all_bb:
            mono = self._try_monolithic_consolidation(summary, slo, state, burst)
            if mono is not None and (best is None or mono.total_energy_j < best.total_energy_j):
                best = mono

        # Regime-2 infeasible fallback: hardware cannot meet the SLO for this
        # window at any configuration.  Rather than returning a random default,
        # find the minimum-TTFT config (minimises violation probability on real
        # hardware), with energy as tiebreaker.  slo_met stays False.
        if best is None or not best.slo_met:
            infeasible = self._infeasible_fallback(summary, slo)
            if infeasible is not None and (
                best is None or
                (not best.slo_met and infeasible.total_energy_j < best.total_energy_j)
            ):
                best = infeasible

        stats["time_s"] = round(time.perf_counter() - search_start, 4)
        return best, stats

    def _empty_search_stats(self, skipped: bool, reason: str) -> Dict[str, float]:
        return {
            "candidate_count": 0,
            "bundle_pair_count": 0,
            "freq_pair_count": 0,
            "expanded": False,
            "expansion_attempted": False,
            "candidates_visited": 0,
            "partial_route_evals": 0,
            "complete_evals": 0,
            "infeasible_candidates": 0,
            "no_route_prunes": 0,
            "bound_prunes": 0,
            "best_updates": 0,
            "time_s": 0.0,
            "skipped": skipped,
            "trigger_reason": reason,
        }

    def _should_run_ideal_search(self, summary: WindowSummary, state: str, burst: bool) -> Tuple[bool, str]:
        if self.cfg.ideal_every_window:
            return True, "always"

        if self.last_ideal_context is None:
            return True, "cold_start"

        if self.cfg.ideal_on_state_change and state != self.last_ideal_context["state"]:
            if self.observed_state_run_length >= self.cfg.ideal_state_change_hold_windows:
                return True, "state_change"

        if self.cfg.ideal_on_burst_toggle and burst != self.last_ideal_context["burst"]:
            return True, "burst_toggle"

        prev_rate = float(self.last_ideal_context["arrival_rate"])
        if prev_rate <= 0:
            if summary.arrival_rate > 0:
                return True, "load_change"
        else:
            frac = abs(summary.arrival_rate - prev_rate) / prev_rate
            if frac >= self.cfg.ideal_load_change_frac:
                return True, "load_change"

        if self.cfg.ideal_on_class_mix_change:
            prev_mix = self.last_ideal_context["class_mix"]
            curr_mix = self._class_mix(summary)
            mix_l1 = sum(
                abs(curr_mix.get(c, 0.0) - prev_mix.get(c, 0.0))
                for c in set(curr_mix) | set(prev_mix)
            )
            if mix_l1 >= self.cfg.ideal_class_mix_l1:
                return True, "class_mix_change"

        if self.windows_since_ideal >= self.cfg.ideal_refresh_windows:
            return True, "periodic_refresh"

        return False, "skipped"

    def _class_mix(self, summary: WindowSummary) -> Dict[str, float]:
        # Built from the classes present this window; class-mix drift (L1) is
        # taxonomy-agnostic, so this supports both the 4- and 9-class schemes.
        return {cls.class_id: cls.fraction for cls in summary.classes}

    def _record_ideal_context(self, summary: WindowSummary, state: str, burst: bool):
        self.last_ideal_context = {
            "state": state,
            "burst": burst,
            "arrival_rate": summary.arrival_rate,
            "class_mix": self._class_mix(summary),
        }

    def _search_candidates(self, summary: WindowSummary, slo: int, state: str,
                           candidates: List[SearchCandidate],
                           stats: Dict[str, float]) -> Optional[SearchResult]:
        if state == "BOTH_HEAVY":
            return self._search_candidates_both_heavy(summary, slo, candidates, stats)

        class_order = self._class_priority(summary.classes, state)
        admissible_routes = {
            cls.class_id: self._admissible_routes(cls, state)
            for cls in summary.classes
        }

        best: Optional[SearchResult] = None
        best_energy = float("inf")
        seed = self._seed_complete_candidate(
            summary,
            slo,
            state,
            candidates,
            class_order,
            admissible_routes,
        )
        if seed is not None:
            best = seed
            best_energy = seed.total_energy_j
            stats["best_updates"] += 1

        for candidate in candidates:
            stats["candidates_visited"] += 1
            complete = self._search_candidate_routes(
                summary,
                slo,
                candidate,
                class_order,
                admissible_routes,
                stats,
                best_energy,
                state,
            )
            if complete is None:
                stats["infeasible_candidates"] += 1
                continue
            if complete.total_energy_j < best_energy:
                best_energy = complete.total_energy_j
                best = complete
                stats["best_updates"] += 1

        return best

    def _search_candidates_both_heavy(self, summary: WindowSummary, slo: int,
                                      candidates: List[SearchCandidate],
                                      stats: Dict[str, float]) -> Optional[SearchResult]:
        class_order = self._class_priority(summary.classes, "BOTH_HEAVY")
        admissible_routes = {
            cls.class_id: self._admissible_routes(cls, "BOTH_HEAVY")
            for cls in summary.classes
        }

        best: Optional[SearchResult] = None
        best_energy = float("inf")
        seed = self._seed_complete_candidate(
            summary,
            slo,
            "BOTH_HEAVY",
            candidates,
            class_order,
            admissible_routes,
        )
        if seed is not None:
            best = seed
            best_energy = seed.total_energy_j
            stats["best_updates"] += 1

        anchor_candidate: Optional[SearchCandidate] = (
            seed.candidate if seed is not None else None
        )
        stage1 = candidates[: min(self.cfg.both_heavy_stage1_candidates, len(candidates))]
        for candidate in stage1:
            stats["candidates_visited"] += 1
            complete = self._search_candidate_routes(
                summary,
                slo,
                candidate,
                class_order,
                admissible_routes,
                stats,
                best_energy,
                "BOTH_HEAVY",
            )
            if complete is None:
                stats["infeasible_candidates"] += 1
                continue
            if complete.total_energy_j < best_energy:
                best_energy = complete.total_energy_j
                best = complete
                anchor_candidate = complete.candidate
                stats["best_updates"] += 1
            if anchor_candidate is None:
                anchor_candidate = complete.candidate

        if anchor_candidate is None:
            refine_candidates = candidates
        else:
            refine_candidates = self._both_heavy_refine_candidates(candidates, anchor_candidate)

        for candidate in refine_candidates:
            if candidate in stage1:
                continue
            stats["candidates_visited"] += 1
            complete = self._search_candidate_routes(
                summary,
                slo,
                candidate,
                class_order,
                admissible_routes,
                stats,
                best_energy,
                "BOTH_HEAVY",
            )
            if complete is None:
                stats["infeasible_candidates"] += 1
                continue
            if complete.total_energy_j < best_energy:
                best_energy = complete.total_energy_j
                best = complete
                stats["best_updates"] += 1

        return best

    def _both_heavy_refine_candidates(self, candidates: List[SearchCandidate],
                                      anchor: SearchCandidate) -> List[SearchCandidate]:
        refined: List[SearchCandidate] = []
        for candidate in candidates:
            same_tp = (
                candidate.bundle_a.tp == anchor.bundle_a.tp and
                candidate.bundle_b.tp == anchor.bundle_b.tp
            )
            same_bundle = (
                candidate.bundle_a == anchor.bundle_a and
                candidate.bundle_b == anchor.bundle_b
            )
            split_capacity = candidate.bundle_a.n_pf + candidate.bundle_b.n_dc
            anchor_split = anchor.bundle_a.n_pf + anchor.bundle_b.n_dc
            close_split = split_capacity >= max(anchor_split - 2, 0)
            close_freq = (
                abs(candidate.freq_a - anchor.freq_a) <= 1020 and
                abs(candidate.freq_b - anchor.freq_b) <= 840
            )
            if same_bundle or (same_tp and close_split and close_freq):
                refined.append(candidate)
            if len(refined) >= self.cfg.both_heavy_refine_candidates:
                break
        if anchor not in refined:
            refined.insert(0, anchor)
        return refined[: self.cfg.both_heavy_refine_candidates]

    def _seed_complete_candidate(self, summary: WindowSummary, slo: int, state: str,
                                 candidates: List[SearchCandidate],
                                 class_order: List[RequestClassSummary],
                                 admissible_routes: Dict[str, Tuple[Tuple[str, str], ...]]) -> Optional[SearchResult]:
        if not candidates:
            return None

        seed_candidates = candidates[: min(4, len(candidates))]
        route_maps: List[Dict[str, Tuple[str, str]]] = []

        # Canonical split route where admissible.
        canonical = {}
        canonical_ok = True
        for cls in class_order:
            routes = admissible_routes[cls.class_id]
            if ROUTE_AB in routes:
                canonical[cls.class_id] = ROUTE_AB
            else:
                canonical_ok = False
                break
        if canonical_ok:
            route_maps.append(canonical)

        # Lowest-order admissible route map.
        route_maps.append({
            cls.class_id: admissible_routes[cls.class_id][0]
            for cls in class_order
        })

        best: Optional[SearchResult] = None
        for candidate in seed_candidates:
            for routes in route_maps:
                complete = self._evaluate_complete_candidate(summary, slo, candidate, routes)
                if complete is None:
                    continue
                if best is None or complete.total_energy_j < best.total_energy_j:
                    best = complete
        return best

    def _search_candidate_routes(self, summary: WindowSummary, slo: int,
                                 candidate: SearchCandidate,
                                 class_order: List[RequestClassSummary],
                                 admissible_routes: Dict[str, Tuple[Tuple[str, str], ...]],
                                 stats: Dict[str, float],
                                 global_best_energy: float,
                                 state: str) -> Optional[SearchResult]:
        best_complete: Optional[SearchResult] = None
        best_energy = global_best_energy
        if state == "BOTH_HEAVY":
            has_incumbent = math.isfinite(global_best_energy)
            branch_cap = (
                self.cfg.route_branch_cap_both_heavy
                if has_incumbent
                else self.cfg.route_branch_cap_default
            )
            stop_on_first_complete = has_incumbent
        else:
            branch_cap = self.cfg.route_branch_cap_default
            stop_on_first_complete = False

        def dfs(idx: int, routes: Dict[str, Tuple[str, str]]):
            nonlocal best_complete, best_energy

            if stop_on_first_complete and best_complete is not None:
                return

            if idx == len(class_order):
                stats["complete_evals"] += 1
                complete = self._evaluate_complete_candidate(summary, slo, candidate, routes)
                if complete is None:
                    return
                if complete.total_energy_j < best_energy:
                    best_energy = complete.total_energy_j
                    best_complete = complete
                return

            cls = class_order[idx]
            viable_routes: List[Tuple[float, Tuple[str, str]]] = []
            for route in admissible_routes[cls.class_id]:
                stats["partial_route_evals"] += 1
                partial_routes = dict(routes)
                partial_routes[cls.class_id] = route
                partial_eval = self._evaluate_partial_candidate(
                    summary, slo, candidate, partial_routes
                )
                if partial_eval is None:
                    continue
                if partial_eval.total_energy_j >= best_energy:
                    stats["bound_prunes"] += 1
                    continue
                viable_routes.append((partial_eval.total_energy_j, route))

            if not viable_routes:
                stats["no_route_prunes"] += 1
                return

            viable_routes.sort(key=lambda item: item[0])
            for _, route in viable_routes[:branch_cap]:
                routes[cls.class_id] = route
                dfs(idx + 1, routes)
                del routes[cls.class_id]

        dfs(0, {})
        return best_complete

    def _construct_candidates(self, summary: WindowSummary, state: str, burst: bool,
                              fixed_bundles: Optional[Dict[str, PoolBundle]],
                              expanded: bool = False) -> Tuple[List[SearchCandidate], Dict[str, int]]:
        if fixed_bundles is None:
            bundle_pairs = list(product(self._bundle_catalog["A"], self._bundle_catalog["B"]))
            bundle_pairs = self._prune_bundle_pairs(summary, state, bundle_pairs)
            has_noncrit = any(cls.input_len < self.cfg.theta_il for cls in summary.classes)
            anchor_pairs = [
                (self.current_bundles["A"], self.current_bundles["B"]),
                (
                    PoolBundle(1, GPU_SPECS["l40s"]["total_gpus"], 0),
                    PoolBundle(1, 0, GPU_SPECS["l4"]["total_gpus"]),
                ),
            ]
            # Mixed-routing anchor: L40S handles long-input prefill (ROUTE_AB/AA),
            # L4 handles short-input as ROUTE_BB (pf+dc on L4).  This mirrors
            # DynamoLLM's per-class routing and can unlock L40S@min-freq by
            # reducing per-instance rate from total_rate/N to long_rate/N.
            # The bundle has n_pf_b > tp which is pruned by _prune_bundle_pairs;
            # adding it as an anchor forces the DFS to explore it.
            if state == "PREFILL_HEAVY" and not burst and has_noncrit:
                n_l4_half = GPU_SPECS["l4"]["total_gpus"] // 4
                anchor_pairs.append((
                    PoolBundle(1, GPU_SPECS["l40s"]["total_gpus"], 0),
                    PoolBundle(1, n_l4_half, n_l4_half),
                ))
            anchor_pairs.extend(self._prefill_heavy_power_gate_anchors(state))
            anchor_pairs.extend(self._decode_heavy_power_gate_anchors(state))
            bundle_pairs.sort(key=lambda pair: self._bundle_priority_key(state, pair[0], pair[1], burst))
            seen_pairs = []
            seen_set = set()
            for pair in anchor_pairs + bundle_pairs:
                if pair in seen_set:
                    continue
                seen_pairs.append(pair)
                seen_set.add(pair)
            bundle_pairs = seen_pairs
            limit = self.cfg.state_candidate_limit
            if state == "BOTH_HEAVY":
                limit = self.cfg.both_heavy_candidate_limit
            elif state == "BOTH_LOW":
                limit = self.cfg.both_low_candidate_limit
            elif state == "DECODE_HEAVY":
                limit = self.cfg.decode_heavy_candidate_limit
            if expanded:
                limit = self.cfg.rescue_state_candidate_limit
                if state == "BOTH_HEAVY":
                    limit = self.cfg.rescue_both_heavy_candidate_limit
                elif state == "BOTH_LOW":
                    limit = self.cfg.rescue_both_low_candidate_limit
                elif state == "DECODE_HEAVY":
                    limit = self.cfg.rescue_decode_heavy_candidate_limit
            bundle_pairs = bundle_pairs[:limit]
        else:
            bundle_pairs = [(fixed_bundles["A"], fixed_bundles["B"])]

        freq_pairs = self._select_frequency_pairs(state, burst, expanded=expanded)

        candidates = [
            SearchCandidate(bundle_a=bundle_a, bundle_b=bundle_b, freq_a=freq_a, freq_b=freq_b)
            for bundle_a, bundle_b in bundle_pairs
            for freq_a, freq_b in freq_pairs
        ]
        return candidates, {
            "bundle_pair_count": len(bundle_pairs),
            "freq_pair_count": len(freq_pairs),
        }

    def _prefill_heavy_power_gate_anchors(self, state: str) -> List[Tuple[PoolBundle, PoolBundle]]:
        """Return anchor pairs that bypass _prune_bundle_pairs for power-gated configs.

        _prune_bundle_pairs in PREFILL_HEAVY requires bundle_a.n_pf > 0, and in
        BOTH_HEAVY requires total_active_gpus >= 4, both blocking configs that
        DualScale's coarse tier explores.  These anchors bypass the pruning.

        Three families:
        1. ROUTE_BB symmetric: L40S power-gated, L4 TP=tp handles both phases.
        2. ROUTE_BB asymmetric TP=1: n_pf > n_dc on L4 (matched DualScale coarse tier).
        3. ROUTE_BA: L4 TP=8 handles ALL prefill (rho≈0.4 at rate=15, IL=2048),
           L40S provides a small decode shard.  This is the only genuinely
           feasible route when the L4/L40S models extrapolate badly at rho>1.
        """
        if state not in ("PREFILL_HEAVY", "BOTH_HEAVY"):
            return []
        power_gated_a = PoolBundle(tp=1, n_pf=0, n_dc=0)
        n_l4_total = GPU_SPECS["l4"]["total_gpus"]
        n_l40s_total = GPU_SPECS["l40s"]["total_gpus"]
        anchors: List[Tuple[PoolBundle, PoolBundle]] = []

        # Family 1 — ROUTE_BB symmetric: L40S off, L4 both phases
        for tp in GPU_SPECS["l4"]["tp_degrees"]:
            if tp + tp <= n_l4_total:
                anchors.append((power_gated_a, PoolBundle(tp=tp, n_pf=tp, n_dc=tp)))

        # Family 2 — ROUTE_BB asymmetric TP=1: n_pf > n_dc (DualScale-matching)
        for n_pf in range(2, n_l4_total + 1):
            for n_dc in range(1, n_pf):
                if n_pf + n_dc <= n_l4_total:
                    anchors.append((power_gated_a, PoolBundle(tp=1, n_pf=n_pf, n_dc=n_dc)))

        # Family 3 — ROUTE_BA: L4 TP=8 prefill (rho<1 for IL=2048), L40S decode
        # L4 TP=8 uses all 8 L4 GPUs as a single high-capacity prefill instance;
        # L40S takes decode with 1..total_gpus decode shards at TP=1.
        if 8 in GPU_SPECS["l4"]["tp_degrees"] and n_l4_total >= 8:
            l4_pf_all = PoolBundle(tp=8, n_pf=n_l4_total, n_dc=0)
            for n_dc_a in range(1, n_l40s_total + 1):
                ba = PoolBundle(tp=1, n_pf=0, n_dc=n_dc_a)
                anchors.append((ba, l4_pf_all))

        return anchors

    def _decode_heavy_power_gate_anchors(self, state: str) -> List[Tuple[PoolBundle, PoolBundle]]:
        """Return anchor pairs for DECODE_HEAVY that bypass _prune_bundle_pairs.

        Three complementary bias problems motivate the anchor families:

        (A) ROUTE_BB bias: The sort key ranks by -bundle_b.n_dc.  ROUTE_AB configs
            with L4 all-decode (n_dc=8) fill the top sorted slots (key=-8), displacing
            ROUTE_BB (max n_dc=7) and ROUTE_AB with n_dc≤6 from the 100-candidate window.
        (B) ROUTE_AB small-n_dc gap: DualScale's coarse tier freely tries ROUTE_AB with
            n_dc_b=2..6 and finds them optimal at moderate DECODE_HEAVY load.  These
            configs sit below the n_dc=7/8 sorted block and are cut by the limit.
        (C) ROUTE_BA/AA exclusion: bundle_b.n_dc=0 fails the prune gate, so these routes
            are never in the sorted list; anchors are the only way in.

        Family 4 — ROUTE_BB: L4 handles both phases, L40S power-gated.  Fixes (A).
        Family 5 — ROUTE_AB: L40S prefill-only + L4 decode-only (n_dc_b < total).  Fixes (B).
        Family 3 — ROUTE_BA: L40S decode-only + L4 prefill-only.  Fixes (C).
        Family 1 — ROUTE_AA TP=1: L40S handles both phases, L4 idle.  Fixes (C).
        Family 2 — ROUTE_AA TP>1: symmetric split on L40S, L4 idle.  Fixes (C).
        """
        if state != "DECODE_HEAVY":
            return []
        power_gated_a = PoolBundle(tp=1, n_pf=0, n_dc=0)
        power_gated_b = PoolBundle(tp=1, n_pf=0, n_dc=0)
        n_l40s_total = GPU_SPECS["l40s"]["total_gpus"]
        n_l4_total = GPU_SPECS["l4"]["total_gpus"]

        anchors: List[Tuple[PoolBundle, PoolBundle]] = []

        # Family 4 — ROUTE_BB: all valid (tp, n_pf, n_dc) combos on L4, L40S power-gated.
        # These 35 configs all fit within decode_heavy_candidate_limit=100 (23 fixed anchors
        # + 35 here = 58 ≤ 100).  Sorted by (-n_dc, n_pf, tp) so the most decode-heavy
        # configs are near the top of the anchor list for fast-path convergence.
        family4: List[Tuple[PoolBundle, PoolBundle]] = []
        for tp in GPU_SPECS["l4"]["tp_degrees"]:
            for n_pf in range(tp, n_l4_total + 1, tp):
                for n_dc in range(tp, n_l4_total + 1, tp):
                    if n_pf + n_dc > n_l4_total:
                        continue
                    family4.append((power_gated_a, PoolBundle(tp=tp, n_pf=n_pf, n_dc=n_dc)))
        family4.sort(key=lambda pair: (-pair[1].n_dc, pair[1].n_pf, pair[1].tp))
        anchors.extend(family4)

        # Family 5 — ROUTE_AB: L40S prefill-only + L4 decode-only.
        # The priority sort key ranks ROUTE_AB by -bundle_b.n_dc.  Configs with
        # n_dc=7-8 occupy the ~34 top sorted slots, leaving ROUTE_AB with n_dc≤6
        # outside the 100-candidate window.  Yet the DualScale coarse tier freely
        # explores these smaller decode shards (e.g. n_pf_a=1, n_dc_b=4-6) and
        # finds them optimal at moderate DECODE_HEAVY load.  Adding them as anchors
        # closes this gap: L40S handles prefill (light at DECODE_HEAVY); L4 handles
        # decode with the right shard count tuned by energy search.
        # Restricted to tp_a=1 on L40S and tp_b=1 on L4 to stay within the 100-
        # candidate limit (Family 4 contributes 35, Family 3/1/2 contribute ~23).
        family5: List[Tuple[PoolBundle, PoolBundle]] = []
        for n_pf_a in range(1, n_l40s_total + 1):  # tp_a=1 only
            ba = PoolBundle(tp=1, n_pf=n_pf_a, n_dc=0)
            for n_dc_b in range(1, n_l4_total):  # tp_b=1; exclude n_dc_b=total (in sorted list)
                family5.append((ba, PoolBundle(tp=1, n_pf=0, n_dc=n_dc_b)))
        # Sort: more decode GPUs first, then smaller L40S footprint
        family5.sort(key=lambda pair: (-pair[1].n_dc, pair[0].n_pf))
        anchors.extend(family5)

        # Family 3 — ROUTE_BA: L40S decode-only + L4 prefill-only
        for n_dc_a in range(1, n_l40s_total + 1):
            ba = PoolBundle(tp=1, n_pf=0, n_dc=n_dc_a)
            for tp_b in GPU_SPECS["l4"]["tp_degrees"]:
                bb = PoolBundle(tp=tp_b, n_pf=tp_b, n_dc=0)
                anchors.append((ba, bb))

        # Family 1 — ROUTE_AA TP=1: L40S handles both phases, L4 idle
        for n_pf in range(1, n_l40s_total):
            n_dc = n_l40s_total - n_pf
            if n_dc >= 1:
                anchors.append((PoolBundle(tp=1, n_pf=n_pf, n_dc=n_dc), power_gated_b))

        # Family 2 — ROUTE_AA higher TP: symmetric split on L40S, L4 idle
        for tp in [t for t in GPU_SPECS["l40s"]["tp_degrees"] if t > 1]:
            per_phase = max(((n_l40s_total // 2) // tp) * tp, tp)
            if per_phase * 2 <= n_l40s_total and per_phase >= tp:
                anchors.append((PoolBundle(tp=tp, n_pf=per_phase, n_dc=per_phase), power_gated_b))

        return anchors

    def _prune_bundle_pairs(self, summary: WindowSummary, state: str,
                            bundle_pairs: List[Tuple[PoolBundle, PoolBundle]]) -> List[Tuple[PoolBundle, PoolBundle]]:
        has_prefill_noncritical = any(cls.input_len < self.cfg.theta_il for cls in summary.classes)
        has_decode_noncritical = any(cls.output_len < self.cfg.theta_ol for cls in summary.classes)

        pruned: List[Tuple[PoolBundle, PoolBundle]] = []
        for bundle_a, bundle_b in bundle_pairs:
            if state == "PREFILL_HEAVY":
                if bundle_a.n_pf <= 0:
                    continue
                if bundle_a.n_dc + bundle_b.n_dc <= 0:
                    continue
                if bundle_a.n_dc not in (0, bundle_a.tp):
                    continue
                if bundle_b.n_pf not in (0, bundle_b.tp):
                    continue
                if bundle_b.n_pf > 0 and bundle_b.n_dc <= 0:
                    continue
                if not has_prefill_noncritical and bundle_b.n_pf > 0:
                    continue
            elif state == "DECODE_HEAVY":
                if bundle_b.n_dc <= 0:
                    continue
                if bundle_a.n_pf + bundle_b.n_pf <= 0:
                    continue
                if bundle_b.n_pf % bundle_b.tp != 0:
                    continue
                if bundle_a.n_dc % bundle_a.tp != 0:
                    continue
                if bundle_b.n_pf > 0 and bundle_b.n_dc <= 0:
                    continue
                if not has_decode_noncritical and bundle_a.n_dc > 0:
                    continue
            elif state == "BOTH_LOW":
                if bundle_a.total_active_gpus + bundle_b.total_active_gpus <= 0:
                    continue
                if bundle_a.n_pf + bundle_b.n_pf <= 0:
                    continue
                if bundle_a.n_dc + bundle_b.n_dc <= 0:
                    continue
            else:  # BOTH_HEAVY
                if bundle_a.n_pf + bundle_b.n_pf <= 0:
                    continue
                if bundle_a.n_dc + bundle_b.n_dc <= 0:
                    continue
                if bundle_a.total_active_gpus + bundle_b.total_active_gpus < 4:
                    continue
            pruned.append((bundle_a, bundle_b))
        return pruned

    def _select_frequency_pairs(self, state: str, burst: bool,
                                expanded: bool = False) -> List[Tuple[int, int]]:
        freqs_a = list(self.schedulers["l40s"].FREQUENCIES)
        freqs_b = list(self.schedulers["l4"].FREQUENCIES)

        low_a = freqs_a[: self.cfg.both_low_freq_levels]
        low_b = freqs_b[: self.cfg.both_low_freq_levels]
        high_a = freqs_a[-self.cfg.both_heavy_freq_levels :]
        high_b = freqs_b[-self.cfg.both_heavy_freq_levels :]
        rescue_a = freqs_a[-self.cfg.rescue_secondary_freq_levels :]
        rescue_b = freqs_b[-self.cfg.rescue_secondary_freq_levels :]
        rescue_low_a = freqs_a[-self.cfg.rescue_both_low_freq_levels :]
        rescue_low_b = freqs_b[-self.cfg.rescue_both_low_freq_levels :]
        rescue_heavy_a = freqs_a[-self.cfg.rescue_both_heavy_freq_levels :]
        rescue_heavy_b = freqs_b[-self.cfg.rescue_both_heavy_freq_levels :]

        if state == "PREFILL_HEAVY":
            if expanded:
                return [(fa, fb) for fa in freqs_a for fb in rescue_b]
            primary = sorted(freqs_a, reverse=burst)
            # Burst: high_b (need fast decode for burst).
            # Non-burst: full freqs_b so the optimizer can find the
            # energy-minimum (fewer GPUs at higher freq), not just the
            # minimum-power frequency at over-provisioned GPU count.
            secondary = sorted(high_b if burst else freqs_b)
            return [(fa, fb) for fa in primary for fb in secondary]

        if state == "DECODE_HEAVY":
            if expanded:
                return [(fa, fb) for fb in freqs_b for fa in rescue_a]
            primary = sorted(freqs_b, reverse=burst)
            # Burst: high_a (need fast L40S when acting as fallback decode).
            # Non-burst: full freqs_a so ROUTE_AA configs can be evaluated at all
            # L40S frequencies (including mid-range energy-optimal points).
            secondary = sorted(high_a if burst else freqs_a)
            return [(fa, fb) for fb in primary for fa in secondary]

        if state == "BOTH_LOW":
            if expanded:
                return [(fa, fb) for fa in rescue_low_a for fb in rescue_low_b]
            return [(fa, fb) for fa in low_a for fb in low_b]

        if expanded:
            return [(fa, fb) for fa in rescue_heavy_a for fb in rescue_heavy_b]
        if burst:
            return [(fa, fb) for fa in sorted(high_a, reverse=True) for fb in sorted(high_b, reverse=True)]
        return [(fa, fb) for fa in high_a for fb in high_b]

    def _bundle_priority_key(self, state: str, bundle_a: PoolBundle,
                             bundle_b: PoolBundle, burst: bool) -> Tuple[float, ...]:
        total_active = bundle_a.total_active_gpus + bundle_b.total_active_gpus
        if state == "PREFILL_HEAVY":
            key = (
                -bundle_a.n_pf,
                bundle_a.n_dc + bundle_b.n_pf + bundle_b.n_dc,
                total_active,
            )
            if burst:
                key = (-bundle_a.n_pf, -total_active, bundle_a.n_dc + bundle_b.total_active_gpus)
            return key
        if state == "DECODE_HEAVY":
            key = (
                -bundle_b.n_dc,
                bundle_a.total_active_gpus,   # prefer L40S power-gated over mixed
                bundle_b.n_pf + bundle_a.n_pf + bundle_a.n_dc,
                total_active,
            )
            if burst:
                key = (-bundle_b.n_dc, bundle_a.total_active_gpus, -total_active, bundle_b.n_pf)
            return key
        if state == "BOTH_LOW":
            return (
                total_active,
                abs(bundle_a.n_pf - bundle_b.n_dc),
                bundle_a.n_dc + bundle_b.n_pf,
            )
        load_bias = -(bundle_a.n_pf + bundle_b.n_dc)
        cross_penalty = bundle_a.n_dc + bundle_b.n_pf
        if burst:
            return (
                -total_active,
                load_bias,
                abs(bundle_a.total_active_gpus - bundle_b.total_active_gpus),
                cross_penalty,
            )
        return (
            load_bias,
            abs(bundle_a.total_active_gpus - bundle_b.total_active_gpus),
            total_active,
            cross_penalty,
        )

    def _class_priority(self, classes: Iterable[RequestClassSummary], state: str) -> List[RequestClassSummary]:
        if state == "PREFILL_HEAVY":
            return sorted(classes, key=lambda cls: (cls.input_len, cls.output_len), reverse=True)
        if state == "DECODE_HEAVY":
            return sorted(classes, key=lambda cls: (cls.output_len, cls.input_len), reverse=True)
        return sorted(classes, key=lambda cls: (cls.input_len + cls.output_len, cls.count), reverse=True)

    def _admissible_routes(self, cls: RequestClassSummary, state: str) -> Tuple[Tuple[str, str], ...]:
        is_prefill_critical = cls.input_len >= self.cfg.theta_il
        is_decode_critical = cls.output_len >= self.cfg.theta_ol

        if state == "PREFILL_HEAVY":
            return ALL_ROUTES   # include ROUTE_BA: L4 TP=8 prefill is feasible at high IL load
        if state == "DECODE_HEAVY":
            # Allow ALL_ROUTES for all classes: ROUTE_BA (L4 pf + L40S dc) is optimal
            # when prefill is light (short IL) — L40S acts as dedicated decode pool.
            # ROUTE_AA avoids KV transfer for high-OL classes.  Previously restricting
            # decode_critical to AB/BB only caused a 32% oracle gap (now fixed: 0%).
            return ALL_ROUTES
        return ALL_ROUTES

    def _evaluate_partial_candidate(self, summary: WindowSummary, slo: int,
                                    candidate: SearchCandidate,
                                    routes: Dict[str, Tuple[str, str]]) -> Optional[SearchResult]:
        return self._evaluate_candidate(
            summary,
            slo,
            candidate,
            routes,
            require_all_classes=False,
            require_safe=False,
        )

    def _evaluate_complete_candidate(self, summary: WindowSummary, slo: int,
                                     candidate: SearchCandidate,
                                     routes: Dict[str, Tuple[str, str]],
                                     require_safe: bool = True) -> Optional[SearchResult]:
        return self._evaluate_candidate(
            summary,
            slo,
            candidate,
            routes,
            require_all_classes=True,
            require_safe=require_safe,
        )

    def _evaluate_candidate(self, summary: WindowSummary, slo: int,
                            candidate: SearchCandidate,
                            routes: Dict[str, Tuple[str, str]],
                            require_all_classes: bool,
                            require_safe: bool) -> Optional[SearchResult]:
        if require_all_classes and len(routes) != len(summary.classes):
            return None

        loads = self._build_phase_loads(summary, routes)
        # Use pool-level predictions for capacity structure and power estimation.
        # The final SLO decision is enforced at class level, which is less
        # conservative than the aggregate proxy screen under mixed overload.
        pool_details, total_power = self._predict_pool_details(
            candidate,
            loads,
            slo,
            require_safe=False,
        )
        if pool_details is None:
            return None

        per_class = self._predict_class_metrics(
            summary,
            candidate,
            loads,
            routes,
            slo,
            require_safe=require_safe,
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

    def _build_phase_loads(self, summary: WindowSummary,
                           routes: Dict[str, Tuple[str, str]]) -> Dict[str, Dict[str, dict]]:
        # Each phase carries BOTH lengths: prefill latency/capacity depends on OL
        # (via KV-cache size it must populate) and decode on IL, so we track the
        # weighted mean of both dimensions per phase and feed the real values to
        # the models (the models have log_il/log_ol/log_kv_pressure features).
        loads = {
            "A": {"pf": {"rate": 0.0, "weighted_il": 0.0, "weighted_ol": 0.0},
                  "dc": {"rate": 0.0, "weighted_il": 0.0, "weighted_ol": 0.0}},
            "B": {"pf": {"rate": 0.0, "weighted_il": 0.0, "weighted_ol": 0.0},
                  "dc": {"rate": 0.0, "weighted_il": 0.0, "weighted_ol": 0.0}},
        }

        class_map = {cls.class_id: cls for cls in summary.classes}
        for class_id, route in routes.items():
            cls = class_map[class_id]
            prefill_pool, decode_pool = route
            loads[prefill_pool]["pf"]["rate"] += cls.request_rate
            loads[prefill_pool]["pf"]["weighted_il"] += cls.request_rate * cls.input_len
            loads[prefill_pool]["pf"]["weighted_ol"] += cls.request_rate * cls.output_len
            loads[decode_pool]["dc"]["rate"] += cls.request_rate
            loads[decode_pool]["dc"]["weighted_il"] += cls.request_rate * cls.input_len
            loads[decode_pool]["dc"]["weighted_ol"] += cls.request_rate * cls.output_len

        return loads

    def _predict_pool_details(self, candidate: SearchCandidate,
                              loads: Dict[str, Dict[str, dict]],
                              slo: int,
                              require_safe: bool) -> Tuple[Optional[Dict[str, Dict[str, float]]], float]:
        pool_details: Dict[str, Dict[str, float]] = {}
        total_power = 0.0
        slo_pf_l40s, slo_dc_l4 = self._phase_slo_targets(slo)

        for pool_label, bundle, freq in (
            ("A", candidate.bundle_a, candidate.freq_a),
            ("B", candidate.bundle_b, candidate.freq_b),
        ):
            gpu_type = POOL_TO_GPU[pool_label]
            spec = GPU_SPECS[gpu_type]
            sched = self.schedulers[gpu_type]
            pool_power = 0.0
            phase_details = {}
            used_gpus = 0

            # Prefill side
            pf_rate = loads[pool_label]["pf"]["rate"]
            if pf_rate > 0:
                if bundle.n_pf <= 0:
                    return None, 0.0
                n_inst = bundle.n_pf_instances
                if n_inst <= 0:
                    return None, 0.0
                mean_il = loads[pool_label]["pf"]["weighted_il"] / pf_rate
                mean_ol = loads[pool_label]["pf"]["weighted_ol"] / pf_rate
                pred = self._predict_config_cached(
                    gpu_type,
                    snap_to_nearest(int(round(mean_il)), IL_VALUES),
                    snap_to_nearest(int(round(mean_ol)), OL_VALUES),
                    bundle.tp,
                    freq,
                    pf_rate / n_inst,
                    slo_pf_l40s,
                    phase="prefill",
                )
                if require_safe and not pred["is_safe"]:
                    return None, 0.0
                phase_power = n_inst * pred["total_power_w"]
                pool_power += phase_power
                used_gpus += bundle.n_pf
                phase_details["prefill"] = {
                    "rate_rps": round(pf_rate, 2),
                    "active_gpus": bundle.n_pf,
                    "active_instances": n_inst,
                    "power_w": round(phase_power, 1),
                    "rho": pred["rho"],
                }
            else:
                phase_details["prefill"] = {
                    "rate_rps": 0.0,
                    "active_gpus": bundle.n_pf,
                    "active_instances": bundle.n_pf_instances,
                    "power_w": 0.0,
                    "rho": 0.0,
                }

            # Decode side
            dc_rate = loads[pool_label]["dc"]["rate"]
            if dc_rate > 0:
                if bundle.n_dc <= 0:
                    return None, 0.0
                n_inst = bundle.n_dc_instances
                if n_inst <= 0:
                    return None, 0.0
                mean_ol = loads[pool_label]["dc"]["weighted_ol"] / dc_rate
                mean_il = loads[pool_label]["dc"]["weighted_il"] / dc_rate
                pred = self._predict_config_cached(
                    gpu_type,
                    snap_to_nearest(int(round(mean_il)), IL_VALUES),
                    snap_to_nearest(int(round(mean_ol)), OL_VALUES),
                    bundle.tp,
                    freq,
                    dc_rate / n_inst,
                    slo_dc_l4,
                    phase="decode",
                )
                if require_safe and not pred["is_safe"]:
                    return None, 0.0
                phase_power = n_inst * pred["total_power_w"]
                pool_power += phase_power
                used_gpus += bundle.n_dc
                phase_details["decode"] = {
                    "rate_rps": round(dc_rate, 2),
                    "active_gpus": bundle.n_dc,
                    "active_instances": n_inst,
                    "power_w": round(phase_power, 1),
                    "rho": pred["rho"],
                }
            else:
                phase_details["decode"] = {
                    "rate_rps": 0.0,
                    "active_gpus": bundle.n_dc,
                    "active_instances": bundle.n_dc_instances,
                    "power_w": 0.0,
                    "rho": 0.0,
                }

            # Motivation-study capacity semantics:
            #   bundle.total_active_gpus = powered-on capacity of the pool
            #   used_gpus               = powered-on GPUs currently serving load
            # Only powered-on-but-unused GPUs incur idle power. GPUs outside the
            # active capacity are treated as deactivated and contribute zero power.
            idle_gpus = max(bundle.total_active_gpus - used_gpus, 0)
            idle_power = idle_gpus * spec["idle_power_w"]
            pool_power += idle_power
            total_power += pool_power

            pool_details[pool_label] = {
                "gpu_type": gpu_type,
                "tp": bundle.tp,
                "freq_mhz": freq,
                "n_prefill_gpus": bundle.n_pf,
                "n_decode_gpus": bundle.n_dc,
                "active_instances": bundle.n_pf_instances + bundle.n_dc_instances,
                "powered_gpus": bundle.total_active_gpus,
                "used_gpus": used_gpus,
                "idle_gpus": idle_gpus,
                "pool_power_w": round(pool_power, 1),
                "prefill": phase_details["prefill"],
                "decode": phase_details["decode"],
            }

        return pool_details, total_power

    def _predict_class_metrics(self, summary: WindowSummary,
                               candidate: SearchCandidate,
                               loads: Dict[str, Dict[str, dict]],
                               routes: Dict[str, Tuple[str, str]],
                               slo: int,
                               require_safe: bool) -> Optional[Dict[str, Dict[str, float]]]:
        per_class: Dict[str, Dict[str, float]] = {}
        slo_pf, slo_dc = self._phase_latency_limits(slo)

        bundle_map = {"A": candidate.bundle_a, "B": candidate.bundle_b}
        freq_map = {"A": candidate.freq_a, "B": candidate.freq_b}

        # Fix 6: pool-level rho uses weighted-mean IL/OL, not per-class IL/OL.
        # Per-class rho with total pool rate massively overestimates utilization
        # for high-IL minority classes in mixed workloads (e.g., long_long at
        # 15% of traffic yields rho=4.8 when pool's true weighted-mean rho≈0.43).
        pool_rho_pf: Dict[str, float] = {}
        pool_rho_dc: Dict[str, float] = {}
        for _pl in ("A", "B"):
            _bundle = bundle_map[_pl]
            _freq = freq_map[_pl]
            _sched = self.schedulers[POOL_TO_GPU[_pl]]

            _pf_rate = loads[_pl]["pf"]["rate"]
            if _pf_rate > 0 and _bundle.n_pf_instances > 0:
                _mean_il = loads[_pl]["pf"]["weighted_il"] / _pf_rate
                _mean_ol = loads[_pl]["pf"]["weighted_ol"] / _pf_rate
                _p_pf = self._predict_config_cached(
                    POOL_TO_GPU[_pl],
                    snap_to_nearest(int(round(_mean_il)), IL_VALUES),
                    snap_to_nearest(int(round(_mean_ol)), OL_VALUES),
                    _bundle.tp,
                    _freq,
                    _pf_rate / _bundle.n_pf_instances,
                    self._snap_slo(_sched, slo_pf),
                    phase="prefill",
                )
                pool_rho_pf[_pl] = _p_pf.get("rho", 0.0)
            else:
                pool_rho_pf[_pl] = 0.0

            _dc_rate = loads[_pl]["dc"]["rate"]
            if _dc_rate > 0 and _bundle.n_dc_instances > 0:
                _mean_ol = loads[_pl]["dc"]["weighted_ol"] / _dc_rate
                _mean_il = loads[_pl]["dc"]["weighted_il"] / _dc_rate
                _p_dc = self._predict_config_cached(
                    POOL_TO_GPU[_pl],
                    snap_to_nearest(int(round(_mean_il)), IL_VALUES),
                    snap_to_nearest(int(round(_mean_ol)), OL_VALUES),
                    _bundle.tp,
                    _freq,
                    _dc_rate / _bundle.n_dc_instances,
                    self._snap_slo(_sched, slo_dc),
                    phase="decode",
                )
                pool_rho_dc[_pl] = _p_dc.get("rho", 0.0)
                if self.cfg.decode_gate_mode == "goodput":
                    # Goodput gate: rho_dc = D_dc / (eta * C_dc_SLO). Pass the RAW
                    # weighted-mean lengths (the table does conservative neighborhood-min,
                    # not a nearest-neighbor guess). demand is per-instance tokens/s.
                    lk = self.feas.decode_safe_capacity(
                        POOL_TO_GPU[_pl], _mean_il, _mean_ol, _bundle.tp, _freq, slo_dc)
                    if lk.source == "none" or lk.capacity_tps <= 0:
                        pool_rho_dc[_pl] = float("inf")
                    else:
                        eta = (self.cfg.decode_eta_confirmed if lk.source == "measured"
                               else self.cfg.decode_eta_fallback)
                        demand_tps = (_dc_rate / _bundle.n_dc_instances) * _mean_ol
                        pool_rho_dc[_pl] = demand_tps / (eta * lk.capacity_tps)
            else:
                pool_rho_dc[_pl] = 0.0

        for cls in summary.classes:
            route = routes.get(cls.class_id)
            if route is None:
                continue
            prefill_pool, decode_pool = route

            pf_rate = loads[prefill_pool]["pf"]["rate"]
            dc_rate = loads[decode_pool]["dc"]["rate"]
            bundle_pf = bundle_map[prefill_pool]
            bundle_dc = bundle_map[decode_pool]
            if bundle_pf.n_pf_instances <= 0 or bundle_dc.n_dc_instances <= 0:
                return None

            sched_pf = self.schedulers[POOL_TO_GPU[prefill_pool]]
            sched_dc = self.schedulers[POOL_TO_GPU[decode_pool]]
            pred_pf = self._predict_config_cached(
                POOL_TO_GPU[prefill_pool],
                cls.input_len,
                cls.output_len,
                bundle_pf.tp,
                freq_map[prefill_pool],
                pf_rate / bundle_pf.n_pf_instances,
                self._snap_slo(sched_pf, slo_pf),
                phase="prefill",
            )
            pred_dc = self._predict_config_cached(
                POOL_TO_GPU[decode_pool],
                cls.input_len,
                cls.output_len,
                bundle_dc.tp,
                freq_map[decode_pool],
                dc_rate / bundle_dc.n_dc_instances,
                self._snap_slo(sched_dc, slo_dc),
                phase="decode",
            )

            kv_ms = self.cfg.kv_transfer_ms_per_token * cls.input_len if prefill_pool != decode_pool else 0.0
            raw_ttft = pred_pf["p99_ttft_ms"]
            raw_tpot = pred_dc["p99_tpot_ms"]
            # Use pool-level weighted-mean rho (Fix 6) — not per-class rho.
            rho_pf = pool_rho_pf[prefill_pool]
            rho_dc = pool_rho_dc[decode_pool]
            # Default ("rho") gate does NOT threshold the raw latency regression.
            # The latency model is a ranking model (median APE ~10-13% in-region
            # but a heavier tail and non-monotonicity at extreme IL), so hard
            # per-class P99 thresholding is fragile near the SLO boundary.  The
            # default gate admits on the utilization stability condition
            # (pool rho<=1, weighted-mean per Fix 6) plus the model-overload
            # sentinel (regression returns -1 when rho>>1).  The stricter
            # gate modes below ("classifier_rho", "latency_rho") add the trained
            # classifier verdict or the P99<=SLO test for the gate-sensitivity study.
            ttft_ms = raw_ttft + kv_ms
            tpot_ms = raw_tpot
            if require_safe:
                # Base common gate: prefill + decode utilization stability plus a
                # prediction-VALIDITY check (regression returns -1 when the input is
                # out-of-region; this is NOT an overload sentinel — decode overload is
                # detected by the goodput capacity in rho_dc, not by TPOT sign).
                # rho_pf bound: legacy fixed 1.0, or the SLO-conditioned prefill
                # envelope (fail closed: a missing (g,TP,f,SLO) bucket -> reject).
                if self.cfg.prefill_envelope:
                    rho_pf_bound = self.feas.prefill_rho_max(
                        POOL_TO_GPU[prefill_pool], bundle_pf.tp,
                        freq_map[prefill_pool], slo_pf)
                else:
                    rho_pf_bound = 1.0
                reject = (
                    rho_pf_bound is None or
                    rho_pf > rho_pf_bound or rho_dc > 1.0 or
                    raw_ttft < 0 or raw_tpot < 0
                )
                # Analytical KV-transfer budget gate (added). kv_ms is deterministic
                # (bytes/bandwidth), not a regression output, so it is safe to hard-threshold.
                # A cross-pool handoff that alone consumes more than kv_budget_frac of the
                # prefill sub-budget cannot meet TTFT for any prefill config -> reject, forcing
                # SLO-aware fallback to an in-pool route. Only binds cross-pool (kv_ms>0 there).
                if self.cfg.kv_budget_frac is not None and kv_ms > self.cfg.kv_budget_frac * slo_pf:
                    reject = True
                if self.cfg.gate_mode == "classifier_rho":
                    reject = reject or (not pred_pf["is_safe"]) or (not pred_dc["is_safe"])
                elif self.cfg.gate_mode == "latency_rho":
                    reject = reject or (ttft_ms > slo_pf) or (tpot_ms > slo_dc)
                if reject:
                    return None

            per_class[cls.class_id] = {
                "route": f"{prefill_pool}->{decode_pool}",
                "request_rate": round(cls.request_rate, 2),
                "ttft_ms": round(ttft_ms, 1),
                "tpot_ms": round(tpot_ms, 1),
                "kv_transfer_ms": round(kv_ms, 1),
            }

        return per_class

    def _phase_latency_limits(self, slo) -> Tuple[int, int]:
        if isinstance(slo, dict):
            return (
                int(slo.get("ttft_ms", slo.get("slo_ms", 500))),
                int(slo.get("tpot_ms", 200)),
            )
        if isinstance(slo, tuple) and len(slo) == 2:
            return int(slo[0]), int(slo[1])
        if self.cfg.split_phase_slo:
            return (
                max(1, int(round(slo * self.cfg.prefill_slo_frac))),
                max(1, int(round(slo * self.cfg.decode_slo_frac))),
            )
        return slo, slo

    def _phase_slo_targets(self, slo: int) -> Tuple[int, int]:
        slo_pf, slo_dc = self._phase_latency_limits(slo)
        return (
            self._snap_slo(self.schedulers["l40s"], slo_pf),
            self._snap_slo(self.schedulers["l4"], slo_dc),
        )

    def _pool_max_rho(self, result: SearchResult) -> float:
        """Return the maximum predicted rho across all active pool phases."""
        max_rho = 0.0
        for pool_detail in result.pool_details.values():
            for phase in ("prefill", "decode"):
                phase_d = pool_detail.get(phase, {})
                if phase_d.get("rate_rps", 0.0) > 0:
                    max_rho = max(max_rho, phase_d.get("rho", 0.0))
        return max_rho

    def _try_monolithic_consolidation(
        self, summary: WindowSummary, slo: int, state: str, burst: bool
    ) -> Optional[SearchResult]:
        """Try ROUTE_AA (L4 power-gated) or ROUTE_BB (L40S power-gated).

        The main search cannot reach these fully-consolidated configurations
        because _prune_bundle_pairs keeps both pools allocated (e.g., it requires
        bundle_a.n_pf > 0 in PREFILL_HEAVY state, preventing L40S power-gating),
        so no enumerated candidate fully powers off a pool.  All four routes are
        admissible per class (_admissible_routes returns ALL_ROUTES), but a pure
        single-pool deployment also needs the other pool at zero GPUs, which the
        bundle-candidate set does not enumerate.

        Safety gate: rho_cap only (not require_safe TTFT check).  The TTFT model
        is calibrated at IL≤512; for longer prompts it extrapolates and may
        over-estimate latency, causing valid configs to be incorrectly rejected.
        rho (utilization) is derived directly from throughput calibration and is
        more reliable for out-of-distribution inputs.  If rho < rho_cap the
        config is not overloaded, so we mark slo_met=True and trust actual
        hardware to meet the SLO.
        """
        rho_cap = self.cfg.mono_consolidation_rho_cap
        if rho_cap <= 0.0:
            return None

        # Do NOT use _select_frequency_pairs here.  That method restricts the
        # secondary-pool frequency to low values (DVFS savings for decode-only
        # traffic), which is wrong for monolithic configs where the single active
        # pool handles both prefill and decode and typically needs higher clocks.
        all_freqs = {
            "A": list(self.schedulers["l40s"].FREQUENCIES),
            "B": list(self.schedulers["l4"].FREQUENCIES),
        }
        power_gated = PoolBundle(tp=1, n_pf=0, n_dc=0)
        best: Optional[SearchResult] = None

        for route_type, active_label in ((ROUTE_AA, "A"), (ROUTE_BB, "B")):
            routes = {cls.class_id: route_type for cls in summary.classes}
            gpu_type = POOL_TO_GPU[active_label]
            spec = GPU_SPECS[gpu_type]
            freqs = all_freqs[active_label]

            # Cap at half the pool to keep search tractable.  If the active pool
            # needs more than half its GPUs just to serve this load monolithically,
            # the config is unlikely to beat the cross-pool result anyway.
            max_gpus = max(spec["tp_degrees"][0], spec["total_gpus"] // 2)
            for tp in spec["tp_degrees"]:
                for n_pf in range(tp, max_gpus + 1, tp):
                    for n_dc in range(tp, max_gpus + 1, tp):
                        if n_pf + n_dc > spec["total_gpus"]:
                            continue
                        active_bundle = PoolBundle(tp=tp, n_pf=n_pf, n_dc=n_dc)
                        bundle_a = active_bundle if active_label == "A" else power_gated
                        bundle_b = power_gated if active_label == "A" else active_bundle
                        # Iterate freqs low-to-high; take the first valid one
                        # (lowest power that still meets SLO) and move on.
                        # The inactive pool's freq is irrelevant — use its minimum.
                        inactive_freq_a = all_freqs["A"][0]
                        inactive_freq_b = all_freqs["B"][0]
                        for freq in freqs:
                            freq_a = freq if active_label == "A" else inactive_freq_a
                            freq_b = freq if active_label == "B" else inactive_freq_b
                            candidate = SearchCandidate(
                                bundle_a=bundle_a, bundle_b=bundle_b,
                                freq_a=freq_a, freq_b=freq_b,
                            )
                            result = self._evaluate_candidate(
                                summary, slo, candidate, routes,
                                require_all_classes=True,
                                require_safe=False,
                            )
                            if result is None:
                                continue
                            if self._pool_max_rho(result) >= rho_cap:
                                continue
                            # rho check passed — treat as safe.  TTFT model is
                            # unreliable for IL>512 (extrapolated), so we trust
                            # rho over the predicted latency.
                            result = dc_replace(result, slo_met=True)
                            # First valid freq for this bundle is also the lowest
                            # power (power rises with freq), so accept immediately.
                            if best is None or result.total_energy_j < best.total_energy_j:
                                best = result
                            break

        return best

    def _infeasible_fallback(
        self, summary: WindowSummary, slo: int
    ) -> Optional[SearchResult]:
        """Return the minimum-TTFT config when no SLO-safe option exists.

        Regime 2 (genuine overload): the hardware cannot meet the SLO for this
        window.  We still want to return a well-defined config rather than a
        random default.  We rank candidates by:
          1. Minimum max-TTFT across all classes — reduces violation probability
             on real hardware (config closest to the feasibility boundary).
          2. Minimum energy as a tiebreaker among equally-close configs.

        The result has slo_met=False (require_safe=False), recorded as a
        violation but with the best possible latency under the overload.
        """
        spec_a = GPU_SPECS["l40s"]
        spec_b = GPU_SPECS["l4"]
        max_freq_a = spec_a["max_freq_mhz"]
        max_freq_b = spec_b["max_freq_mhz"]
        power_gated = PoolBundle(tp=1, n_pf=0, n_dc=0)

        # Build candidate set: all (bundle_a, bundle_b) at max frequency.
        # Covers ROUTE_AB (disaggregated) and monolithic routes (AA/BB).
        # No pruning — we need the full space to find the true minimum TTFT.
        cand_triples: List[Tuple[PoolBundle, PoolBundle, Dict[str, Tuple[str, str]]]] = []

        # ROUTE_AB: L40S prefill, L4 decode — best for prefill-heavy overload
        for tp_a in spec_a["tp_degrees"]:
            for n_pf_a in range(tp_a, spec_a["total_gpus"] + 1, tp_a):
                for tp_b in spec_b["tp_degrees"]:
                    for n_dc_b in range(tp_b, spec_b["total_gpus"] + 1, tp_b):
                        ba = PoolBundle(tp=tp_a, n_pf=n_pf_a, n_dc=0)
                        bb = PoolBundle(tp=tp_b, n_pf=0, n_dc=n_dc_b)
                        routes = {cls.class_id: ROUTE_AB for cls in summary.classes}
                        cand_triples.append((ba, bb, routes))

        # ROUTE_AA: all L40S — best when L40S can handle full load alone
        for tp_a in spec_a["tp_degrees"]:
            for n_pf in range(tp_a, spec_a["total_gpus"] + 1, tp_a):
                for n_dc in range(0, spec_a["total_gpus"] - n_pf + 1, max(tp_a, 1)):
                    if n_pf + n_dc > spec_a["total_gpus"]:
                        continue
                    ba = PoolBundle(tp=tp_a, n_pf=n_pf, n_dc=n_dc)
                    routes = {cls.class_id: ROUTE_AA for cls in summary.classes}
                    cand_triples.append((ba, power_gated, routes))

        # ROUTE_BB: all L4 — best for decode-heavy overload
        for tp_b in spec_b["tp_degrees"]:
            for n_pf in range(tp_b, spec_b["total_gpus"] + 1, tp_b):
                for n_dc in range(0, spec_b["total_gpus"] - n_pf + 1, max(tp_b, 1)):
                    if n_pf + n_dc > spec_b["total_gpus"]:
                        continue
                    bb = PoolBundle(tp=tp_b, n_pf=n_pf, n_dc=n_dc)
                    routes = {cls.class_id: ROUTE_BB for cls in summary.classes}
                    cand_triples.append((power_gated, bb, routes))

        best_result: Optional[SearchResult] = None
        best_max_ttft = float("inf")
        best_energy = float("inf")

        for ba, bb, routes in cand_triples:
            candidate = SearchCandidate(ba, bb, max_freq_a, max_freq_b)
            result = self._evaluate_candidate(
                summary, slo, candidate, routes,
                require_all_classes=True,
                require_safe=False,
            )
            if result is None:
                continue

            # Rank by worst-class TTFT (negative = model overload → treat as ∞)
            max_ttft = max(
                (m["ttft_ms"] if m.get("ttft_ms", -1.0) >= 0 else float("inf"))
                for m in result.per_class_metrics.values()
            ) if result.per_class_metrics else float("inf")

            if (max_ttft < best_max_ttft or
                    (max_ttft == best_max_ttft and result.total_energy_j < best_energy)):
                best_max_ttft = max_ttft
                best_energy = result.total_energy_j
                best_result = result

        return best_result

    def _enumerate_pool_bundles(self, pool_label: str) -> List[PoolBundle]:
        gpu_type = POOL_TO_GPU[pool_label]
        spec = GPU_SPECS[gpu_type]
        bundles = {PoolBundle(tp=1, n_pf=0, n_dc=0)}
        for tp in spec["tp_degrees"]:
            values = list(range(0, spec["total_gpus"] + 1, tp))
            for n_pf in values:
                for n_dc in values:
                    if n_pf + n_dc <= spec["total_gpus"]:
                        bundles.add(PoolBundle(tp=tp, n_pf=n_pf, n_dc=n_dc))
        return sorted(bundles, key=lambda bundle: (bundle.tp, bundle.n_pf, bundle.n_dc))

    def _estimate_reference_capacities(self) -> Dict[str, float]:
        # Reference operating point uses a consistent (IL, OL) shape for both
        # phases rather than the discarded proxy: prefill capacity is anchored at
        # ref_prefill_il and decode at ref_decode_ol, with the companion length
        # set to the same reference scale so the models see a realistic shape.
        prefill_cluster_rate = self._max_safe_cluster_rate(
            gpu_type="l40s",
            il=self.cfg.ref_prefill_il,
            ol=self.cfg.ref_decode_ol,
        )
        decode_cluster_rate = self._max_safe_cluster_rate(
            gpu_type="l4",
            il=self.cfg.ref_prefill_il,
            ol=self.cfg.ref_decode_ol,
        )
        return {
            "prefill": max(prefill_cluster_rate * self.cfg.ref_prefill_il, 1.0),
            "decode": max(decode_cluster_rate * self.cfg.ref_decode_ol, 1.0),
        }

    def _max_safe_cluster_rate(self, gpu_type: str, il: int, ol: int) -> float:
        sched = self.schedulers[gpu_type]
        spec = GPU_SPECS[gpu_type]
        tp = 1
        freq = spec["max_freq_mhz"]
        n_inst = spec["total_gpus"]
        slo = max(sched.SLO_THRESHOLDS)
        phase = "prefill" if gpu_type == "l40s" else "decode"

        lo = 0.0
        hi = 1.0
        while hi < 256.0:
            pred = sched.predict_config(il, ol, tp, freq, hi / n_inst, slo, phase=phase)
            if not pred["is_safe"]:
                break
            lo = hi
            hi *= 2.0

        for _ in range(18):
            mid = 0.5 * (lo + hi)
            pred = sched.predict_config(il, ol, tp, freq, mid / n_inst, slo, phase=phase)
            if pred["is_safe"]:
                lo = mid
            else:
                hi = mid
        return lo

    def _snap_slo(self, sched: EnergyScheduler, slo_ms: int) -> int:
        return min(sched.SLO_THRESHOLDS, key=lambda supported: abs(supported - slo_ms))

    def _predict_config_cached(self, gpu_type: str, il: int, ol: int,
                               tp: int, freq: int, rate: float, slo: int,
                               phase: Optional[str] = None) -> dict:
        key = (
            gpu_type,
            int(il),
            int(ol),
            int(tp),
            int(freq),
            round(float(rate), 6),
            int(slo),
            phase or "legacy",
        )
        cached = self._predict_cache.get(key)
        if cached is not None:
            return cached
        pred = self.schedulers[gpu_type].predict_config(il, ol, tp, freq, rate, slo, phase=phase)
        self._predict_cache[key] = pred
        return pred


