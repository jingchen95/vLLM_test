#!/usr/bin/env python3
"""
Core utilities for the Observation 2 joint-knob motivation study.

This module builds controlled workload states, evaluates restricted-policy
search spaces with the existing SWEEP-LLM predictive backend, and writes
replayable selected configurations for later spot validation on hardware.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter, time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from jsep_cluster import GPU_SPECS
from scheduler import EnergyScheduler
from sweep_llm_scheduler import (
    ALL_ROUTES,
    ROUTE_AB,
    PoolBundle,
    RequestClassSummary,
    SearchCandidate,
    SweepLLMConfig,
    SweepLLMStrategy,
    WindowSummary,
)


#
# In the SWEEP-LLM route encoding, a route is (prefill_pool, decode_pool),
# with pool A = L40S and pool B = L4. So ROUTE_AB means:
#   prefill -> L40S, decode -> L4
#
STATIC_DISAGG_ROUTE = ROUTE_AB

# Physical-capacity cap for the oracle search: any active prefill/decode phase
# driven to rho >= RHO_CAP is rejected as infeasible. rho=1 marks the point
# where offered load equals modeled pool capacity; beyond it the latency model
# extrapolates unreliably (see _compute_margins).
RHO_CAP = 1.0


DEFAULT_STATE_MIXES = {
    "prefill-heavy": {
        "long_short": 0.50,
        "long_long": 0.25,
        "short_short": 0.25,
    },
    "decode-heavy": {
        "short_long": 0.50,
        "long_long": 0.25,
        "short_short": 0.25,
    },
    "both-low": {
        "short_short": 0.50,
        "short_long": 0.25,
        "long_short": 0.25,
    },
}


@dataclass(frozen=True)
class WorkloadClassSpec:
    class_id: str
    input_len: int
    output_len: int
    fraction: float


@dataclass(frozen=True)
class WorkloadStateSpec:
    name: str
    rate_rps: float
    classes: Tuple[WorkloadClassSpec, ...]
    window_s: float
    trace_duration_s: float
    trace_seed: int

    @property
    def offered_requests(self) -> float:
        return self.rate_rps * self.window_s

    @property
    def total_prefill_tokens_per_s(self) -> float:
        return sum(cls.fraction * cls.input_len for cls in self.classes) * self.rate_rps

    @property
    def total_decode_tokens_per_s(self) -> float:
        return sum(cls.fraction * cls.output_len for cls in self.classes) * self.rate_rps


@dataclass(frozen=True)
class PolicySpec:
    name: str
    allow_routing: bool
    allow_dvfs: bool
    allow_tp: bool
    allow_active_gpus: bool
    sequential: bool = False


@dataclass(frozen=True)
class SearchSpace:
    freq_l40s: Tuple[int, ...]
    freq_l4: Tuple[int, ...]
    tp_l40s: Tuple[int, ...]
    tp_l4: Tuple[int, ...]
    active_l40s: Tuple[int, ...]
    active_l4: Tuple[int, ...]


@dataclass(frozen=True)
class ReferencePoolConfig:
    freq_mhz: int
    tp: int
    active_gpus: int


@dataclass(frozen=True)
class StaticDisaggReference:
    route: Tuple[str, str]
    l40s: ReferencePoolConfig
    l4: ReferencePoolConfig


DEFAULT_POLICIES = (
    PolicySpec("static", False, False, False, False, False),
    PolicySpec("route_only", True, False, False, False, False),
    PolicySpec("route_dvfs", True, True, False, False, False),
    PolicySpec("route_dvfs_tp", True, True, True, False, False),
    PolicySpec("full_joint", True, True, True, True, False),
    PolicySpec("sequential", True, True, True, True, True),
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_reference_config() -> StaticDisaggReference:
    return StaticDisaggReference(
        route=STATIC_DISAGG_ROUTE,
        l40s=ReferencePoolConfig(
            freq_mhz=GPU_SPECS["l40s"]["max_freq_mhz"],
            tp=1,
            active_gpus=GPU_SPECS["l40s"]["total_gpus"],
        ),
        l4=ReferencePoolConfig(
            freq_mhz=GPU_SPECS["l4"]["max_freq_mhz"],
            tp=1,
            active_gpus=GPU_SPECS["l4"]["total_gpus"],
        ),
    )


def reference_config_metadata(reference: StaticDisaggReference) -> Dict[str, object]:
    return {
        "name": "static-disagg",
        "route_semantics": "prefill->L40S, decode->L4",
        "route_tuple": list(reference.route),
        "l40s": asdict(reference.l40s),
        "l4": asdict(reference.l4),
    }


def static_disagg_route_map(summary: WindowSummary) -> Dict[str, Tuple[str, str]]:
    return {cls.class_id: STATIC_DISAGG_ROUTE for cls in summary.classes}


def _normalize_mix(raw_mix: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw_mix.values())
    if total <= 0:
        raise ValueError("State mix must sum to a positive value.")
    return {k: v / total for k, v in raw_mix.items() if v > 0}


def make_state_specs(
    rate_rps: float,
    low_rate_rps: float,
    window_s: float,
    trace_duration_s: float,
    trace_seed: int,
    short_il: int,
    long_il: int,
    short_ol: int,
    long_ol: int,
    states: Sequence[str],
    state_mixes: Dict[str, Dict[str, float]] = None,
) -> List[WorkloadStateSpec]:
    state_mixes = state_mixes or DEFAULT_STATE_MIXES
    class_lengths = {
        "short_short": (short_il, short_ol),
        "short_long": (short_il, long_ol),
        "long_short": (long_il, short_ol),
        "long_long": (long_il, long_ol),
    }
    specs: List[WorkloadStateSpec] = []
    for state_name in states:
        if state_name not in state_mixes:
            raise ValueError(f"Unsupported state: {state_name}")
        mix = _normalize_mix(state_mixes[state_name])
        classes = []
        for class_id, fraction in mix.items():
            il, ol = class_lengths[class_id]
            classes.append(
                WorkloadClassSpec(
                    class_id=class_id,
                    input_len=il,
                    output_len=ol,
                    fraction=fraction,
                )
            )
        state_rate = low_rate_rps if state_name == "both-low" else rate_rps
        specs.append(
            WorkloadStateSpec(
                name=state_name,
                rate_rps=state_rate,
                classes=tuple(classes),
                window_s=window_s,
                trace_duration_s=trace_duration_s,
                trace_seed=trace_seed,
            )
        )
    return specs


def build_summary(state: WorkloadStateSpec) -> WindowSummary:
    classes = []
    for cls in state.classes:
        count = int(round(state.offered_requests * cls.fraction))
        if cls.fraction > 0 and count == 0:
            count = 1
        classes.append(
            RequestClassSummary(
                class_id=cls.class_id,
                fraction=cls.fraction,
                request_rate=state.rate_rps * cls.fraction,
                input_len=cls.input_len,
                output_len=cls.output_len,
                count=count,
            )
        )
    return WindowSummary(arrival_rate=state.rate_rps, classes=tuple(classes))


def generate_replay_trace(state: WorkloadStateSpec) -> List[Dict[str, object]]:
    rng = np.random.RandomState(state.trace_seed)
    n_requests = int(round(state.rate_rps * state.trace_duration_s))
    if n_requests <= 0:
        return []

    inter_arrivals = rng.exponential(1.0 / state.rate_rps, n_requests)
    times = np.cumsum(inter_arrivals)
    times = times[times < state.trace_duration_s]
    choices = list(state.classes)
    probs = [cls.fraction for cls in choices]
    rows = []
    for arrival in times:
        cls = rng.choice(choices, p=probs)
        rows.append(
            {
                "arrival_time_s": round(float(arrival), 6),
                "class_id": cls.class_id,
                "input_len": cls.input_len,
                "output_len": cls.output_len,
            }
        )
    return rows


def create_backend(
    model_dir_l40s: str,
    model_dir_l4: str,
    window_s: float,
) -> SweepLLMStrategy:
    schedulers = {
        "l40s": EnergyScheduler(model_dir_l40s),
        "l4": EnergyScheduler(model_dir_l4),
    }
    cfg = SweepLLMConfig(print_search_stats=False)
    return SweepLLMStrategy(schedulers=schedulers, config=cfg, window_s=window_s)


def default_search_space(strategy: SweepLLMStrategy) -> SearchSpace:
    return SearchSpace(
        freq_l40s=tuple(strategy.schedulers["l40s"].FREQUENCIES),
        freq_l4=tuple(strategy.schedulers["l4"].FREQUENCIES),
        tp_l40s=tuple(GPU_SPECS["l40s"]["tp_degrees"]),
        tp_l4=tuple(GPU_SPECS["l4"]["tp_degrees"]),
        active_l40s=tuple(range(0, GPU_SPECS["l40s"]["total_gpus"] + 1)),
        active_l4=tuple(range(0, GPU_SPECS["l4"]["total_gpus"] + 1)),
    )


def _filter_bundles(
    bundles: Iterable[PoolBundle],
    allowed_tp: Sequence[int],
    allowed_active: Sequence[int],
) -> List[PoolBundle]:
    tp_set = set(allowed_tp)
    active_set = set(allowed_active)
    filtered = [
        bundle
        for bundle in bundles
        if bundle.tp in tp_set and bundle.total_active_gpus in active_set
    ]
    return sorted(
        filtered,
        key=lambda bundle: (bundle.total_active_gpus, bundle.tp, bundle.n_pf, bundle.n_dc),
    )


def _candidate_bundle_pairs(
    strategy: SweepLLMStrategy,
    summary: WindowSummary,
    state_name: str,
    tp_l40s: Sequence[int],
    tp_l4: Sequence[int],
    active_l40s: Sequence[int],
    active_l4: Sequence[int],
) -> List[Tuple[PoolBundle, PoolBundle]]:
    bundles_a = _filter_bundles(strategy._bundle_catalog["A"], tp_l40s, active_l40s)
    bundles_b = _filter_bundles(strategy._bundle_catalog["B"], tp_l4, active_l4)
    state_map = {
        "prefill-heavy": "PREFILL_HEAVY",
        "decode-heavy": "DECODE_HEAVY",
        "both-low": "BOTH_LOW",
    }
    pairs = list(product(bundles_a, bundles_b))
    sweep_state = state_map.get(state_name, "BOTH_HEAVY")
    pairs = strategy._prune_bundle_pairs(summary, sweep_state, pairs)
    pairs.sort(
        key=lambda pair: strategy._bundle_priority_key(
            sweep_state,
            pair[0],
            pair[1],
            burst=False,
        )
    )
    return pairs


def static_reference_candidate(reference: StaticDisaggReference) -> SearchCandidate:
    return SearchCandidate(
        bundle_a=PoolBundle(
            tp=reference.l40s.tp,
            n_pf=reference.l40s.active_gpus,
            n_dc=0,
        ),
        bundle_b=PoolBundle(
            tp=reference.l4.tp,
            n_pf=0,
            n_dc=reference.l4.active_gpus,
        ),
        freq_a=reference.l40s.freq_mhz,
        freq_b=reference.l4.freq_mhz,
    )


def _freq_pairs(
    freq_l40s: Sequence[int],
    freq_l4: Sequence[int],
) -> List[Tuple[int, int]]:
    return list(product(sorted(set(freq_l40s)), sorted(set(freq_l4))))


def _route_maps(
    summary: WindowSummary,
    allow_routing: bool,
) -> List[Dict[str, Tuple[str, str]]]:
    if not allow_routing:
        return [static_disagg_route_map(summary)]
    classes = [cls.class_id for cls in summary.classes]
    maps = []
    for routes in product(ALL_ROUTES, repeat=len(classes)):
        maps.append({class_id: route for class_id, route in zip(classes, routes)})
    return maps


def _as_jsonable_route_map(route_map: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    return {class_id: f"{route[0]}->{route[1]}" for class_id, route in route_map.items()}


def _route_map_from_json(route_map_json: str) -> Dict[str, Tuple[str, str]]:
    payload = json.loads(route_map_json)
    return {
        class_id: tuple(route.split("->"))  # type: ignore[arg-type]
        for class_id, route in payload.items()
    }


def _compute_margins(
    result,
    ttft_slo_ms: int,
    tpot_slo_ms: int,
) -> Dict[str, float]:
    ttft_margins = []
    tpot_margins = []
    per_class = {}
    for class_id, metrics in result.per_class_metrics.items():
        ttft_margin = round(ttft_slo_ms - metrics["ttft_ms"], 1)
        tpot_margin = round(tpot_slo_ms - metrics["tpot_ms"], 1)
        ttft_margins.append(ttft_margin)
        tpot_margins.append(tpot_margin)
        per_class[class_id] = {
            **metrics,
            "ttft_margin_ms": ttft_margin,
            "tpot_margin_ms": tpot_margin,
        }
    min_ttft = min(ttft_margins) if ttft_margins else float("-inf")
    min_tpot = min(tpot_margins) if tpot_margins else float("-inf")

    # Physical-capacity gate. The per-pool latency model is only trustworthy
    # while a phase is under-subscribed (rho < 1): above rho=1 the decode TPOT
    # predictor degrades (TPOT decreases with load and emits negative sentinels),
    # so a candidate that drives any active phase to rho >= RHO_CAP is treated
    # as infeasible regardless of its (extrapolated) latency margin.
    max_rho = 0.0
    rho_ok = True
    for pool in result.pool_details.values():
        for phase in ("prefill", "decode"):
            ph = pool.get(phase)
            if ph and ph.get("rate_rps", 0.0) > 0:
                rho = ph.get("rho", 0.0)
                max_rho = max(max_rho, rho)
                if rho >= RHO_CAP:
                    rho_ok = False

    return {
        "feasible": (min_ttft >= 0.0 and min_tpot >= 0.0 and rho_ok),
        "min_ttft_margin_ms": round(min_ttft, 1),
        "min_tpot_margin_ms": round(min_tpot, 1),
        "worst_margin_ms": round(min(min_ttft, min_tpot), 1),
        "max_rho": round(max_rho, 4),
        "rho_feasible": rho_ok,
        "per_class": per_class,
    }


def _config_row(
    state: WorkloadStateSpec,
    policy_name: str,
    reference: StaticDisaggReference,
    result,
    margins: Dict[str, float],
    offered_rate_rps: float,
    ttft_slo_ms: int,
    tpot_slo_ms: int,
    evaluation_index: int,
    search_started_at: str,
    search_finished_at: str,
    search_started_unix_s: float,
    search_finished_unix_s: float,
    search_elapsed_s: float,
) -> Dict[str, object]:
    candidate = result.candidate
    routes = _as_jsonable_route_map(result.routes)
    return {
        "state": state.name,
        "policy": policy_name,
        "selection_status": "candidate",
        "evaluation_index": evaluation_index,
        "search_started_at": search_started_at,
        "search_finished_at": search_finished_at,
        "search_started_unix_s": round(search_started_unix_s, 6),
        "search_finished_unix_s": round(search_finished_unix_s, 6),
        "search_elapsed_s": round(search_elapsed_s, 6),
        "offered_rate_rps": offered_rate_rps,
        "served_throughput_rps": offered_rate_rps if margins["feasible"] else 0.0,
        "ttft_slo_ms": ttft_slo_ms,
        "tpot_slo_ms": tpot_slo_ms,
        "feasible": margins["feasible"],
        "min_ttft_margin_ms": margins["min_ttft_margin_ms"],
        "min_tpot_margin_ms": margins["min_tpot_margin_ms"],
        "worst_margin_ms": margins["worst_margin_ms"],
        "energy_j": result.total_energy_j,
        "power_w": result.total_power_w,
        "route_map_json": json.dumps(routes, sort_keys=True),
        "l40s_freq_mhz": candidate.freq_a,
        "l4_freq_mhz": candidate.freq_b,
        "l40s_tp": candidate.bundle_a.tp,
        "l4_tp": candidate.bundle_b.tp,
        "l40s_active_gpus": candidate.bundle_a.total_active_gpus,
        "l4_active_gpus": candidate.bundle_b.total_active_gpus,
        "l40s_prefill_gpus": candidate.bundle_a.n_pf,
        "l40s_decode_gpus": candidate.bundle_a.n_dc,
        "l4_prefill_gpus": candidate.bundle_b.n_pf,
        "l4_decode_gpus": candidate.bundle_b.n_dc,
        "pool_details_json": json.dumps(result.pool_details, sort_keys=True),
        "per_class_metrics_json": json.dumps(margins["per_class"], sort_keys=True),
        "state_prefill_tokens_per_s": round(state.total_prefill_tokens_per_s, 2),
        "state_decode_tokens_per_s": round(state.total_decode_tokens_per_s, 2),
        "trace_duration_s": state.trace_duration_s,
        "trace_seed": state.trace_seed,
        "reference_route_semantics": "prefill->L40S,decode->L4",
        "reference_l40s_freq_mhz": reference.l40s.freq_mhz,
        "reference_l4_freq_mhz": reference.l4.freq_mhz,
        "reference_l40s_tp": reference.l40s.tp,
        "reference_l4_tp": reference.l4.tp,
        "reference_l40s_active_gpus": reference.l40s.active_gpus,
        "reference_l4_active_gpus": reference.l4.active_gpus,
    }


def _pick_best(rows: List[Dict[str, object]]) -> Dict[str, object]:
    feasible_rows = [row for row in rows if row["feasible"]]
    if feasible_rows:
        selected = dict(min(feasible_rows, key=lambda row: (row["energy_j"], -row["worst_margin_ms"])))
        selected["selection_status"] = "feasible_best"
        return selected
    selected = dict(min(rows, key=lambda row: (-row["worst_margin_ms"], row["energy_j"])))
    selected["selection_status"] = "best_infeasible"
    return selected


def evaluate_policy(
    strategy: SweepLLMStrategy,
    state: WorkloadStateSpec,
    policy: PolicySpec,
    search_space: SearchSpace,
    ttft_slo_ms: int,
    tpot_slo_ms: int,
    reference: Optional[StaticDisaggReference] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Evaluate the best configuration within a restricted search space.

    This function implements the restricted-search ablations:
    each policy gets a reduced search space, but optimization is still global
    within that reduced space.
    """
    reference = reference or default_reference_config()
    summary = build_summary(state)
    offered_rate_rps = summary.arrival_rate
    slo = {"ttft_ms": ttft_slo_ms, "tpot_ms": tpot_slo_ms}
    search_started_at = iso_now()
    search_started_unix_s = time()
    search_start = perf_counter()

    if policy.sequential:
        rows, selected = evaluate_sequential_policy(
            strategy, state, search_space, ttft_slo_ms, tpot_slo_ms, reference
        )
        return rows, selected

    tp_l40s = search_space.tp_l40s if policy.allow_tp else (reference.l40s.tp,)
    tp_l4 = search_space.tp_l4 if policy.allow_tp else (reference.l4.tp,)
    active_l40s = (
        search_space.active_l40s
        if policy.allow_active_gpus
        else (reference.l40s.active_gpus,)
    )
    active_l4 = (
        search_space.active_l4
        if policy.allow_active_gpus
        else (reference.l4.active_gpus,)
    )
    freq_l40s = search_space.freq_l40s if policy.allow_dvfs else (reference.l40s.freq_mhz,)
    freq_l4 = search_space.freq_l4 if policy.allow_dvfs else (reference.l4.freq_mhz,)

    if policy.name == "static":
        route_maps = [static_disagg_route_map(summary)]
        bundle_pairs = [(static_reference_candidate(reference).bundle_a, static_reference_candidate(reference).bundle_b)]
        freq_pairs = [(reference.l40s.freq_mhz, reference.l4.freq_mhz)]
    else:
        route_maps = _route_maps(summary, allow_routing=policy.allow_routing)
        bundle_pairs = _candidate_bundle_pairs(
            strategy, summary, state.name, tp_l40s, tp_l4, active_l40s, active_l4
        )
        freq_pairs = _freq_pairs(freq_l40s, freq_l4)

    rows: List[Dict[str, object]] = []
    evaluation_index = 0
    for bundle_a, bundle_b in bundle_pairs:
        for freq_a, freq_b in freq_pairs:
            candidate = SearchCandidate(bundle_a=bundle_a, bundle_b=bundle_b, freq_a=freq_a, freq_b=freq_b)
            for route_map in route_maps:
                result = strategy._evaluate_complete_candidate(
                    summary,
                    slo,
                    candidate,
                    route_map,
                    require_safe=False,
                )
                if result is None:
                    continue
                margins = _compute_margins(result, ttft_slo_ms, tpot_slo_ms)
                evaluation_index += 1
                rows.append(
                    _config_row(
                        state=state,
                        policy_name=policy.name,
                        reference=reference,
                        result=result,
                        margins=margins,
                        offered_rate_rps=offered_rate_rps,
                        ttft_slo_ms=ttft_slo_ms,
                        tpot_slo_ms=tpot_slo_ms,
                        evaluation_index=evaluation_index,
                        search_started_at=search_started_at,
                        search_finished_at="",
                        search_started_unix_s=search_started_unix_s,
                        search_finished_unix_s=0.0,
                        search_elapsed_s=0.0,
                    )
                )

    search_elapsed_s = perf_counter() - search_start
    search_finished_at = iso_now()
    search_finished_unix_s = time()
    for row in rows:
        row["search_finished_at"] = search_finished_at
        row["search_finished_unix_s"] = round(search_finished_unix_s, 6)
        row["search_elapsed_s"] = round(search_elapsed_s, 6)

    if not rows:
        raise RuntimeError(f"No valid candidates evaluated for policy {policy.name} on state {state.name}.")
    selected = _pick_best(rows)
    return rows, selected


def evaluate_sequential_policy(
    strategy: SweepLLMStrategy,
    state: WorkloadStateSpec,
    search_space: SearchSpace,
    ttft_slo_ms: int,
    tpot_slo_ms: int,
    reference: Optional[StaticDisaggReference] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Evaluate a staged controller with commitment.

    Unlike restricted-search ablations, sequential exposes the same knob set
    as full joint optimization, but commits after each stage in the order:
    routing -> DVFS -> TP -> active GPUs.
    """
    reference = reference or default_reference_config()
    summary = build_summary(state)
    offered_rate_rps = summary.arrival_rate
    slo = {"ttft_ms": ttft_slo_ms, "tpot_ms": tpot_slo_ms}
    search_started_at = iso_now()
    search_started_unix_s = time()
    search_start = perf_counter()

    rows: List[Dict[str, object]] = []
    evaluation_index = 0

    def stage_search(
        stage_name: str,
        route_maps: Sequence[Dict[str, Tuple[str, str]]],
        freq_pairs: Sequence[Tuple[int, int]],
        tp_l40s: Sequence[int],
        tp_l4: Sequence[int],
        active_l40s: Sequence[int],
        active_l4: Sequence[int],
    ) -> Dict[str, object]:
        nonlocal evaluation_index
        stage_rows: List[Dict[str, object]] = []
        bundle_pairs = _candidate_bundle_pairs(
            strategy, summary, state.name, tp_l40s, tp_l4, active_l40s, active_l4
        )
        for bundle_a, bundle_b in bundle_pairs:
            for freq_a, freq_b in freq_pairs:
                candidate = SearchCandidate(bundle_a=bundle_a, bundle_b=bundle_b, freq_a=freq_a, freq_b=freq_b)
                for route_map in route_maps:
                    result = strategy._evaluate_complete_candidate(
                        summary,
                        slo,
                        candidate,
                        route_map,
                        require_safe=False,
                    )
                    if result is None:
                        continue
                    margins = _compute_margins(result, ttft_slo_ms, tpot_slo_ms)
                    evaluation_index += 1
                    stage_rows.append(
                        {
                            **_config_row(
                                state=state,
                                policy_name="sequential",
                                reference=reference,
                                result=result,
                                margins=margins,
                                offered_rate_rps=offered_rate_rps,
                                ttft_slo_ms=ttft_slo_ms,
                                tpot_slo_ms=tpot_slo_ms,
                                evaluation_index=evaluation_index,
                                search_started_at=search_started_at,
                                search_finished_at="",
                                search_started_unix_s=search_started_unix_s,
                                search_finished_unix_s=0.0,
                                search_elapsed_s=0.0,
                            ),
                            "sequential_stage": stage_name,
                        }
                    )
        if not stage_rows:
            raise RuntimeError(f"Sequential stage {stage_name} found no valid candidates for {state.name}.")
        rows.extend(stage_rows)
        return _pick_best(stage_rows)

    full_active_l40s = (reference.l40s.active_gpus,)
    full_active_l4 = (reference.l4.active_gpus,)
    max_freq_pair = [(reference.l40s.freq_mhz, reference.l4.freq_mhz)]

    stage1 = stage_search(
        stage_name="routing",
        route_maps=_route_maps(summary, allow_routing=True),
        freq_pairs=max_freq_pair,
        tp_l40s=(reference.l40s.tp,),
        tp_l4=(reference.l4.tp,),
        active_l40s=full_active_l40s,
        active_l4=full_active_l4,
    )
    route_map_stage1 = _route_map_from_json(stage1["route_map_json"])

    stage2 = stage_search(
        stage_name="dvfs",
        route_maps=[route_map_stage1],
        freq_pairs=_freq_pairs(search_space.freq_l40s, search_space.freq_l4),
        tp_l40s=(reference.l40s.tp,),
        tp_l4=(reference.l4.tp,),
        active_l40s=full_active_l40s,
        active_l4=full_active_l4,
    )
    fixed_freq_pair = [(int(stage2["l40s_freq_mhz"]), int(stage2["l4_freq_mhz"]))]

    stage3 = stage_search(
        stage_name="tp",
        route_maps=[route_map_stage1],
        freq_pairs=fixed_freq_pair,
        tp_l40s=search_space.tp_l40s,
        tp_l4=search_space.tp_l4,
        active_l40s=full_active_l40s,
        active_l4=full_active_l4,
    )
    fixed_tp_l40s = (int(stage3["l40s_tp"]),)
    fixed_tp_l4 = (int(stage3["l4_tp"]),)

    stage4 = stage_search(
        stage_name="active_gpus",
        route_maps=[route_map_stage1],
        freq_pairs=fixed_freq_pair,
        tp_l40s=fixed_tp_l40s,
        tp_l4=fixed_tp_l4,
        active_l40s=search_space.active_l40s,
        active_l4=search_space.active_l4,
    )

    search_elapsed_s = perf_counter() - search_start
    search_finished_at = iso_now()
    search_finished_unix_s = time()
    for row in rows:
        row["search_finished_at"] = search_finished_at
        row["search_finished_unix_s"] = round(search_finished_unix_s, 6)
        row["search_elapsed_s"] = round(search_elapsed_s, 6)

    return rows, stage4


def write_replay_trace(output_dir: Path, state: WorkloadStateSpec) -> Path:
    trace_rows = generate_replay_trace(state)
    trace_path = output_dir / f"trace_{state.name}.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arrival_time_s", "class_id", "input_len", "output_len"])
        writer.writeheader()
        writer.writerows(trace_rows)
    return trace_path


def _selected_config_record(
    state: WorkloadStateSpec,
    selected_row: Dict[str, object],
    trace_path: Path,
    ttft_slo_ms: int,
    tpot_slo_ms: int,
    reference: Optional[StaticDisaggReference] = None,
) -> Dict[str, object]:
    reference = reference or default_reference_config()
    routes = json.loads(selected_row["route_map_json"])
    pool_details = json.loads(selected_row["pool_details_json"])
    per_class = json.loads(selected_row["per_class_metrics_json"])
    return {
        "state": {
            "name": state.name,
            "rate_rps": state.rate_rps,
            "window_s": state.window_s,
            "trace_duration_s": state.trace_duration_s,
            "trace_seed": state.trace_seed,
            "prefill_tokens_per_s": round(state.total_prefill_tokens_per_s, 2),
            "decode_tokens_per_s": round(state.total_decode_tokens_per_s, 2),
            "classes": [asdict(cls) for cls in state.classes],
            "trace_csv": str(trace_path),
        },
        "policy": selected_row["policy"],
        "selection_status": selected_row["selection_status"],
        "reference_config": reference_config_metadata(reference),
        "slo": {
            "ttft_ms": ttft_slo_ms,
            "tpot_ms": tpot_slo_ms,
        },
        "selected_config": {
            "route_map": routes,
            "l40s": {
                "freq_mhz": int(selected_row["l40s_freq_mhz"]),
                "tp": int(selected_row["l40s_tp"]),
                "active_gpus": int(selected_row["l40s_active_gpus"]),
                "prefill_gpus": int(selected_row["l40s_prefill_gpus"]),
                "decode_gpus": int(selected_row["l40s_decode_gpus"]),
            },
            "l4": {
                "freq_mhz": int(selected_row["l4_freq_mhz"]),
                "tp": int(selected_row["l4_tp"]),
                "active_gpus": int(selected_row["l4_active_gpus"]),
                "prefill_gpus": int(selected_row["l4_prefill_gpus"]),
                "decode_gpus": int(selected_row["l4_decode_gpus"]),
            },
        },
        "predicted_metrics": {
            "energy_j": selected_row["energy_j"],
            "power_w": selected_row["power_w"],
            "feasible": bool(selected_row["feasible"]),
            "served_throughput_rps": selected_row["served_throughput_rps"],
            "offered_rate_rps": selected_row["offered_rate_rps"],
            "min_ttft_margin_ms": selected_row["min_ttft_margin_ms"],
            "min_tpot_margin_ms": selected_row["min_tpot_margin_ms"],
            "worst_margin_ms": selected_row["worst_margin_ms"],
            "per_class": per_class,
            "pool_details": pool_details,
        },
        "search_metadata": {
            "search_started_at": selected_row["search_started_at"],
            "search_finished_at": selected_row["search_finished_at"],
            "search_started_unix_s": selected_row["search_started_unix_s"],
            "search_finished_unix_s": selected_row["search_finished_unix_s"],
            "search_elapsed_s": selected_row["search_elapsed_s"],
        },
    }


def write_outputs(
    output_prefix: Path,
    experiment_meta: Dict[str, object],
    evaluation_rows: List[Dict[str, object]],
    selected_rows: List[Dict[str, object]],
    replay_records: List[Dict[str, object]],
) -> Dict[str, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    eval_csv = output_prefix.with_name(output_prefix.name + "_evaluations.csv")
    selected_csv = output_prefix.with_name(output_prefix.name + "_selected.csv")
    replay_jsonl = output_prefix.with_name(output_prefix.name + "_replay.jsonl")
    summary_json = output_prefix.with_name(output_prefix.name + "_meta.json")

    eval_df = pd.DataFrame(evaluation_rows)
    selected_df = pd.DataFrame(selected_rows)
    eval_df.to_csv(eval_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    with replay_jsonl.open("w") as f:
        for record in replay_records:
            f.write(json.dumps(record) + "\n")
    summary_json.write_text(json.dumps(experiment_meta, indent=2))

    paths = {
        "evaluations_csv": eval_csv,
        "selected_csv": selected_csv,
        "replay_jsonl": replay_jsonl,
        "meta_json": summary_json,
    }

    try:
        eval_parquet = output_prefix.with_name(output_prefix.name + "_evaluations.parquet")
        selected_parquet = output_prefix.with_name(output_prefix.name + "_selected.parquet")
        eval_df.to_parquet(eval_parquet, index=False)
        selected_df.to_parquet(selected_parquet, index=False)
        paths["evaluations_parquet"] = eval_parquet
        paths["selected_parquet"] = selected_parquet
    except Exception:
        pass

    return paths
