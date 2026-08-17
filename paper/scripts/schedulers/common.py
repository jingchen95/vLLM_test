"""Shared types, constants, and configuration for the SWEEP-LLM scheduler family.

Step-1 refactor: extracted verbatim from the former monolithic
sweep_llm_scheduler.py (file separation only; no logic change).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


POOL_TO_GPU = {"A": "l40s", "B": "l4"}
GPU_TO_POOL = {"l40s": "A", "l4": "B"}

ROUTE_AA = ("A", "A")
ROUTE_AB = ("A", "B")
ROUTE_BA = ("B", "A")
ROUTE_BB = ("B", "B")
ALL_ROUTES = (ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB)


@dataclass(frozen=True)
class SweepLLMConfig:
    theta_il: int = 512
    theta_ol: int = 512
    # Optional second per-axis thresholds. When set, request classification uses
    # three length levels per axis (S/M/L) instead of two (short/long), giving up
    # to 9 classes. Left None for the default 4-class (2x2) taxonomy. Used by the
    # request-class granularity (K) sensitivity study.
    theta_il_hi: Optional[int] = None
    theta_ol_hi: Optional[int] = None
    # Common feasibility gate used to admit configurations (applied identically to
    # all strategies). Options:
    #   "rho"            — utilization stability gate (rho<=1); the deployed default.
    #   "classifier_rho" — trained SLO guard-band classifier (is_safe) AND rho<=1.
    #   "latency_rho"    — predicted P99 TTFT/TPOT <= SLO AND rho<=1.
    # Used by the gate-sensitivity study (see gate_sensitivity_pilot.py).
    gate_mode: str = "rho"
    tau_imb: float = 1.5
    tau_load: float = 0.7
    tau_burst: float = 1.5
    delta_lambda: float = 5.0
    hysteresis_margin: float = 0.10
    prefill_slo_frac: float = 0.40
    decode_slo_frac: float = 0.50
    split_phase_slo: bool = False
    burst_requires_history: bool = True
    prefill_proxy_ol: int = 32
    decode_proxy_il: int = 32
    # Hardware feasibility gate (fail-closed table lookups; see feasibility_tables.py
    # and paper/DECODE_GATE_WIRING_DESIGN.md). Both DEFAULT to legacy so behavior is
    # byte-identical until the hardware freq x shape x SLO decode sweep exists (the
    # existing Phase-2 data covers only ~14-23% of the decode DVFS grid).
    #   decode_gate_mode:
    #     "model_rho" (default) — legacy capacity-model rho<=1 for decode.
    #     "goodput"             — rho_dc = D_dc/(eta*C_dc_SLO)<=1 from decode_capacity.csv;
    #                             REQUIRES the artifact (fails fast at init if missing).
    #   prefill_envelope:
    #     False (default) — legacy fixed rho_pf<=1.
    #     True            — rho_pf<=rho_pf_max(g,TP,f,SLO_TTFT) from rho_envelope.csv;
    #                       a missing (g,TP,f,SLO) bucket fails closed (candidate rejected).
    decode_gate_mode: str = "model_rho"
    prefill_envelope: bool = False
    decode_eta_confirmed: float = 0.85  # margin on measured (SLO-confirmed) decode capacity
    decode_eta_fallback: float = 0.80   # margin on borrowed / self-observed lower-bound capacity
    kv_transfer_ms_per_token: float = 0.059  # measured L40S<->L4 cross-node KV cost (us/token)
    # Analytical KV-transfer admission gate (added). When set, a cross-pool route is
    # rejected if the deterministic handoff cost kv_ms = kv_transfer_ms_per_token * il
    # alone exceeds this fraction of the prefill sub-budget slo_pf. Unlike the latency
    # regression, kv_ms is analytical (bytes/bandwidth), so it can be hard-thresholded
    # without the regression's tail fragility. None => disabled (legacy rho-gate behavior,
    # preserves frozen baselines byte-identically).
    kv_budget_frac: Optional[float] = None
    state_candidate_limit: int = 24
    both_heavy_candidate_limit: int = 36
    both_low_candidate_limit: int = 12
    decode_heavy_candidate_limit: int = 100
    directional_secondary_freq_levels: int = 3
    both_low_freq_levels: int = 3
    both_heavy_freq_levels: int = 4
    rescue_state_candidate_limit: int = 72
    rescue_both_heavy_candidate_limit: int = 48
    rescue_both_low_candidate_limit: int = 36
    rescue_decode_heavy_candidate_limit: int = 200
    rescue_secondary_freq_levels: int = 5
    rescue_both_low_freq_levels: int = 5
    rescue_both_heavy_freq_levels: int = 4
    both_heavy_stage1_candidates: int = 12
    both_heavy_refine_candidates: int = 24
    route_branch_cap_default: int = 3
    route_branch_cap_both_heavy: int = 2
    ideal_every_window: bool = True
    emergency_bundle_override: bool = False
    ideal_refresh_windows: int = 6
    ideal_load_change_frac: float = 0.25
    ideal_class_mix_l1: float = 0.20
    ideal_on_class_mix_change: bool = True
    ideal_on_state_change: bool = True
    ideal_on_burst_toggle: bool = True
    ideal_state_change_hold_windows: int = 1
    print_search_stats: bool = True
    bundle_stability_a: int = 2
    bundle_stability_b: int = 2
    ref_prefill_il: int = 1024
    ref_decode_ol: int = 1024
    current_bundle_a: Tuple[int, int, int] = (1, 4, 0)
    current_bundle_b: Tuple[int, int, int] = (1, 0, 8)
    # Monolithic consolidation: after finding the best disaggregated result,
    # try power-gating one pool entirely (ROUTE_AA or ROUTE_BB) using rho-only
    # safety instead of the conservative classifier guard band.  A candidate is
    # accepted when max predicted rho < this cap.  Set to 0.0 to disable.
    mono_consolidation_rho_cap: float = 0.85
    # Savings override: when the ideal search finds a monolithic (one pool fully
    # power-gated) result that is cheaper than the fast result by at least this
    # fraction, immediately commit to the ideal bundles rather than waiting for
    # the bundle-stability ramp.  This prevents the scheduler from staying stuck
    # on an expensive cross-pool config (ROUTE_AB) for many windows while the
    # ramp slowly transitions to the cheaper all-on-one-pool optimum.
    # Set to 1.0 to disable.
    savings_override_min_frac: float = 0.20


@dataclass(frozen=True)
class RequestClassSummary:
    class_id: str
    fraction: float
    request_rate: float
    input_len: int
    output_len: int
    count: int


@dataclass(frozen=True)
class WindowSummary:
    arrival_rate: float
    classes: Tuple[RequestClassSummary, ...]


@dataclass(frozen=True)
class PoolBundle:
    tp: int
    n_pf: int
    n_dc: int

    @property
    def total_active_gpus(self) -> int:
        return self.n_pf + self.n_dc

    @property
    def n_pf_instances(self) -> int:
        return self.n_pf // self.tp if self.n_pf > 0 else 0

    @property
    def n_dc_instances(self) -> int:
        return self.n_dc // self.tp if self.n_dc > 0 else 0


@dataclass(frozen=True)
class SearchCandidate:
    bundle_a: PoolBundle
    bundle_b: PoolBundle
    freq_a: int
    freq_b: int


@dataclass(frozen=True)
class SearchResult:
    candidate: SearchCandidate
    routes: Dict[str, Tuple[str, str]]
    total_power_w: float
    total_energy_j: float
    slo_met: bool
    per_class_metrics: Dict[str, Dict[str, float]]
    pool_details: Dict[str, Dict[str, float]]

