"""Model-free, pool-local safe DVFS exploration for stationary C1.

The controller in this module deliberately knows nothing about the measured
C1b headroom results.  It only consumes the selected SLO, discrete hardware
frequencies, and the most recent phase-specific evidence. The C1c one-step
and adaptive policies both use this state machine; only the permitted coarse
jump differs. The legacy ``FeedbackScheduler`` is not a C1c comparator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Any, Mapping, Optional, Sequence


class SearchPhase(str, Enum):
    COLD = "COLD"
    EXPAND = "EXPAND"
    REFINE = "REFINE"
    HOLD = "HOLD"
    RECOVER = "RECOVER"


class EvidenceClass(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    STALE_TELEMETRY = "STALE_TELEMETRY"
    PRESSURE_RISK = "PRESSURE_RISK"
    READBACK_FAILURE = "READBACK_FAILURE"
    INVALID_MEASUREMENT = "INVALID_MEASUREMENT"
    CONTEXT_CHANGE = "CONTEXT_CHANGE"


@dataclass(frozen=True)
class PoolEvidence:
    """One observation for one P or D pool.

    Token coverage is phase-specific: P uses input tokens and TTFT, while D
    uses completed output tokens and TPOT.  Counts are request-level counts;
    no mean-only or silently filtered evidence can be classified SAFE.
    """

    pool: str
    phase: str
    observed_frequency_mhz: int
    observation_duration_s: float
    valid_request_count: int
    input_token_coverage: int = 0
    output_token_coverage: int = 0
    ttft_statistic_ms: Optional[float] = None
    tpot_statistic_ms: Optional[float] = None
    ttft_pass_count: int = 0
    ttft_violation_count: int = 0
    tpot_pass_count: int = 0
    tpot_violation_count: int = 0
    backlog_stable: Optional[bool] = None
    telemetry_fresh: bool = False
    health_valid: bool = False
    readback_valid: bool = False
    request_accounting_valid: bool = False
    pressure_signal: str = "unknown"
    gross_energy_j: Optional[float] = None
    context_signature: str = "stationary"
    captured_at_s: float = 0.0
    readback_timestamp_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.pool not in {"P", "D"}:
            raise ValueError("pool must be P or D")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if self.valid_request_count < 0:
            raise ValueError("valid_request_count must be non-negative")
        if self.observation_duration_s < 0:
            raise ValueError("observation_duration_s must be non-negative")

    @property
    def phase_latency_ms(self) -> Optional[float]:
        return self.ttft_statistic_ms if self.phase == "prefill" else self.tpot_statistic_ms

    @property
    def phase_pass_count(self) -> int:
        return self.ttft_pass_count if self.phase == "prefill" else self.tpot_pass_count

    @property
    def phase_violation_count(self) -> int:
        return (
            self.ttft_violation_count
            if self.phase == "prefill"
            else self.tpot_violation_count
        )

    @property
    def phase_token_coverage(self) -> int:
        return self.input_token_coverage if self.phase == "prefill" else self.output_token_coverage

    def attainment(self) -> Optional[float]:
        if self.valid_request_count <= 0:
            return None
        return self.phase_pass_count / float(self.valid_request_count)

    def headroom_fraction(self, slo_ms: float) -> Optional[float]:
        value = self.phase_latency_ms
        if value is None or not math.isfinite(float(value)) or slo_ms <= 0:
            return None
        return (float(slo_ms) - float(value)) / float(slo_ms)


@dataclass(frozen=True)
class AdaptiveExplorationConfig:
    """Safe defaults for CPU/dry-run use; values are not C1b calibration."""

    ttft_slo_ms: float = 600.0
    tpot_slo_ms: float = 65.0
    attainment_gate: float = 0.95
    min_valid_requests: int = 8
    min_input_tokens: int = 1024
    min_output_tokens: int = 1024
    min_observation_duration_s: float = 1.0
    max_evidence_timeout_s: float = 120.0
    telemetry_max_age_s: float = 300.0
    # Implementation defaults only.  Freeze/calibrate before a physical C1
    # run; they are intentionally not derived from any C1b outcome.
    headroom_for_one_step: float = 0.01
    headroom_for_large_jump: float = 0.10
    large_jump_levels: int = 2
    context_invalidation_on_change: bool = True

    def __post_init__(self) -> None:
        if self.ttft_slo_ms <= 0 or self.tpot_slo_ms <= 0:
            raise ValueError("SLOs must be positive")
        if not 0 < self.attainment_gate <= 1:
            raise ValueError("attainment_gate must be in (0, 1]")
        if self.min_valid_requests <= 0:
            raise ValueError("min_valid_requests must be positive")
        if self.min_input_tokens < 0 or self.min_output_tokens < 0:
            raise ValueError("token evidence requirements must be non-negative")
        if self.min_observation_duration_s < 0 or self.max_evidence_timeout_s <= 0:
            raise ValueError("invalid evidence timing")
        if self.telemetry_max_age_s <= 0:
            raise ValueError("telemetry_max_age_s must be positive")
        if self.headroom_for_one_step < 0 or self.headroom_for_large_jump < 0:
            raise ValueError("headroom thresholds must be non-negative")
        if self.headroom_for_large_jump < self.headroom_for_one_step:
            raise ValueError("large jump threshold must be >= one-step threshold")
        if self.large_jump_levels < 1:
            raise ValueError("large_jump_levels must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, ttft_slo_ms: float, tpot_slo_ms: float) -> "AdaptiveExplorationConfig":
        return cls(
            ttft_slo_ms=float(ttft_slo_ms),
            tpot_slo_ms=float(tpot_slo_ms),
            attainment_gate=float(values.get("attainment_gate", 0.95)),
            min_valid_requests=int(values.get("min_valid_requests", 8)),
            min_input_tokens=int(values.get("min_input_tokens", 1024)),
            min_output_tokens=int(values.get("min_output_tokens", 1024)),
            min_observation_duration_s=float(values.get("min_observation_duration_s", 1.0)),
            max_evidence_timeout_s=float(values.get("max_evidence_timeout_s", 120.0)),
            telemetry_max_age_s=float(values.get("telemetry_max_age_s", 300.0)),
            headroom_for_one_step=float(values.get("headroom_for_one_step", 0.01)),
            headroom_for_large_jump=float(values.get("headroom_for_large_jump", 0.10)),
            large_jump_levels=int(values.get("large_jump_levels", 2)),
            context_invalidation_on_change=bool(values.get("context_invalidation_on_change", True)),
        )


@dataclass
class PoolSearchState:
    """Explicit independent search state for one P or D pool."""

    pool: str
    current_frequency: int
    known_safe_frequency: int
    unsafe_bound: Optional[int] = None
    candidate_frequency: Optional[int] = None
    search_status: str = SearchPhase.COLD.value
    evidence_since_action: int = 0
    last_action_reason: str = "INITIAL"
    last_action_timestamp_s: float = 0.0
    current_index: int = 0
    last_validated_safe_index: int = 0
    unsafe_index: Optional[int] = None
    safe_energy_by_frequency_j: dict[int, float] = field(default_factory=dict)
    safe_energy_observations_by_frequency: dict[int, int] = field(default_factory=dict)
    validated_safe_frequencies: set[int] = field(default_factory=set)
    revoked_frequencies: set[int] = field(default_factory=set)
    revoked_safe_energy_history: list[dict[str, Any]] = field(default_factory=list)
    recent_evidence_frequency: Optional[int] = None
    recent_valid_request_count: int = 0
    recent_phase_pass_count: int = 0
    recent_phase_token_coverage: int = 0
    recent_evidence_context_signature: str = ""
    recent_evidence_captured_at_s: float = 0.0
    best_measured_safe_frequency: Optional[int] = None
    best_measured_safe_energy_j: Optional[float] = None
    search_phase: str = SearchPhase.COLD.value
    context_signature: str = "stationary"
    confidence: float = 0.0
    transition_reason: str = "INITIAL"
    windows_observed: int = 0
    requests_observed: int = 0
    converged: bool = False
    convergence_reason: Optional[str] = None
    convergence_window_count: Optional[int] = None
    convergence_request_count: Optional[int] = None


@dataclass(frozen=True)
class AdaptiveDecision:
    pool: str
    phase: str
    slo_regime: str
    current_frequency_mhz: int
    candidate_frequency_mhz: Optional[int]
    known_safe_frequency_mhz: int
    unsafe_bound_mhz: Optional[int]
    decision: str
    action: str
    trigger: str
    reason: str
    requested_frequency_mhz: int
    readback_frequency_mhz: Optional[int]
    readback_valid: bool
    evidence_classification: str
    evidence_duration_s: float
    valid_request_count: int
    input_token_coverage: int
    output_token_coverage: int
    latency_statistic_ms: Optional[float]
    attainment: Optional[float]
    backlog_stable: Optional[bool]
    pressure_signal: str
    telemetry_fresh: bool
    headroom_fraction: Optional[float]
    jump_size_levels: int
    decision_timestamp_s: float
    action_timestamp_s: Optional[float]
    readback_timestamp_s: Optional[float]
    search_phase: str

    def as_log(self) -> dict[str, Any]:
        """Stable CSV/JSON-friendly schema for controller_decisions.csv."""
        return {
            "pool": self.pool,
            "phase": self.phase,
            "slo_regime": self.slo_regime,
            "current_freq_mhz": self.current_frequency_mhz,
            "candidate_freq_mhz": self.candidate_frequency_mhz,
            "known_safe_freq_mhz": self.known_safe_frequency_mhz,
            "unsafe_bound_freq_mhz": self.unsafe_bound_mhz,
            "decision": self.decision,
            "action": self.action,
            "trigger": self.trigger,
            "reason": self.reason,
            "requested_freq_mhz": self.requested_frequency_mhz,
            "readback_freq_mhz": self.readback_frequency_mhz,
            "readback_valid": self.readback_valid,
            "evidence_classification": self.evidence_classification,
            "evidence_duration_s": self.evidence_duration_s,
            "valid_request_count": self.valid_request_count,
            "input_token_coverage": self.input_token_coverage,
            "output_token_coverage": self.output_token_coverage,
            "latency_statistic_ms": self.latency_statistic_ms,
            "attainment": self.attainment,
            "backlog_stable": self.backlog_stable,
            "pressure_signal": self.pressure_signal,
            "telemetry_fresh": self.telemetry_fresh,
            "headroom_fraction": self.headroom_fraction,
            "jump_size_levels": self.jump_size_levels,
            "decision_timestamp_s": self.decision_timestamp_s,
            "action_timestamp_s": self.action_timestamp_s,
            "readback_timestamp_s": self.readback_timestamp_s,
            "search_phase": self.search_phase,
        }


class AdaptivePoolController:
    """A deterministic coarse-to-fine safe search for one P or D pool."""

    def __init__(
        self,
        pool: str,
        phase: str,
        frequencies_mhz: Sequence[int],
        config: AdaptiveExplorationConfig,
        *,
        slo_regime: str = "TIGHT",
        clock: Any = None,
    ) -> None:
        if pool not in {"P", "D"}:
            raise ValueError("pool must be P or D")
        if phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        ordered = sorted({int(value) for value in frequencies_mhz})
        if not ordered:
            raise ValueError("at least one supported frequency is required")
        self.pool = pool
        self.phase = phase
        self.frequencies_mhz = tuple(ordered)
        self.config = config
        self.slo_regime = str(slo_regime)
        self.clock = clock or __import__("time").time
        high_index = len(self.frequencies_mhz) - 1
        high = self.frequencies_mhz[high_index]
        self.state = PoolSearchState(
            pool=pool,
            current_frequency=high,
            known_safe_frequency=high,
            current_index=high_index,
            last_validated_safe_index=high_index,
            validated_safe_frequencies={high},
        )

    @property
    def current_frequency_mhz(self) -> int:
        return self.state.current_frequency

    def set_current_frequency(self, frequency_mhz: int, *, readback_valid: bool = True) -> None:
        """Synchronize state after the physical actuator readback."""
        value = int(frequency_mhz)
        if value not in self.frequencies_mhz:
            raise ValueError("frequency is not in the supported discrete action space")
        self.state.current_frequency = value
        self.state.current_index = self.frequencies_mhz.index(value)
        if not readback_valid:
            self.state.transition_reason = "READBACK_FAILURE"

    def record_actuation(self, requested_frequency_mhz: int, readback_frequency_mhz: Optional[int], *, readback_valid: bool, timestamp_s: float) -> None:
        """Record command/readback without treating a mismatch as safe."""
        requested = int(requested_frequency_mhz)
        valid = bool(readback_valid and readback_frequency_mhz == requested)
        if requested not in self.frequencies_mhz:
            raise ValueError("requested frequency is not supported")
        if valid:
            self.state.last_action_timestamp_s = float(timestamp_s)
            self.state.evidence_since_action = 0
            self.set_current_frequency(requested, readback_valid=True)
        else:
            self.state.transition_reason = "READBACK_FAILURE"

    def _slo_ms(self) -> float:
        return self.config.ttft_slo_ms if self.phase == "prefill" else self.config.tpot_slo_ms

    def _enough_evidence(self, evidence: PoolEvidence) -> bool:
        if evidence.valid_request_count < self.config.min_valid_requests:
            return False
        if evidence.phase_token_coverage < (
            self.config.min_input_tokens if self.phase == "prefill" else self.config.min_output_tokens
        ):
            return False
        return evidence.observation_duration_s >= self.config.min_observation_duration_s

    def _timed_out(self, evidence: PoolEvidence, now_s: float) -> bool:
        if self.state.last_action_timestamp_s <= 0:
            return evidence.observation_duration_s >= self.config.max_evidence_timeout_s
        return now_s - self.state.last_action_timestamp_s >= self.config.max_evidence_timeout_s

    def _classify(self, evidence: PoolEvidence, now_s: float) -> tuple[EvidenceClass, str]:
        if not evidence.readback_valid or evidence.observed_frequency_mhz != self.state.current_frequency:
            return EvidenceClass.READBACK_FAILURE, "requested/readback frequency mismatch"
        if not evidence.health_valid or not evidence.request_accounting_valid:
            return EvidenceClass.INVALID_MEASUREMENT, "measurement or request accounting audit failed"
        if not evidence.telemetry_fresh:
            return EvidenceClass.STALE_TELEMETRY, "telemetry is stale or missing"
        if evidence.backlog_stable is None:
            return EvidenceClass.PRESSURE_RISK, "required endpoint pressure evidence is unknown"
        if not evidence.backlog_stable:
            return EvidenceClass.PRESSURE_RISK, "backlog/capacity pressure is not stable"
        if not self._enough_evidence(evidence):
            return EvidenceClass.INSUFFICIENT_EVIDENCE, (
                "minimum request/token/duration evidence not accumulated"
            )
        attainment = evidence.attainment()
        headroom = evidence.headroom_fraction(self._slo_ms())
        if attainment is None or attainment < self.config.attainment_gate:
            return EvidenceClass.UNSAFE, "phase SLO attainment is below configured gate"
        if headroom is None or headroom < 0:
            return EvidenceClass.UNSAFE, "phase latency statistic is at or above SLO"
        return EvidenceClass.SAFE, "fresh sufficient phase evidence satisfies SLO and pressure gates"

    def _next_down_index(self, headroom: Optional[float]) -> tuple[Optional[int], int, str]:
        state = self.state
        if headroom is None or headroom < self.config.headroom_for_one_step:
            return None, 0, "headroom is below the configured one-step probe threshold"
        # A known unsafe bound constrains every later candidate.  If the bound
        # is adjacent to the safe anchor, the boundary is resolved; never
        # fall through and re-probe that unsafe level.
        if state.unsafe_index is not None:
            if state.unsafe_index + 1 < state.last_validated_safe_index:
                index = (state.unsafe_index + state.last_validated_safe_index) // 2
                return index, state.last_validated_safe_index - index, "refine safe/unsafe frequency bracket"
            return None, 0, "safe/unsafe boundary is resolved at adjacent frequency levels"
        if state.last_validated_safe_index <= 0:
            return None, 0, "no lower supported frequency remains"
        jump = (
            self.config.large_jump_levels
            if headroom is not None and headroom >= self.config.headroom_for_large_jump
            else 1
        )
        index = max(0, state.last_validated_safe_index - jump)
        if index == state.last_validated_safe_index:
            return None, 0, "no lower supported frequency remains"
        return index, min(jump, state.last_validated_safe_index - index), "headroom-based coarse-to-fine downshift"

    def _record_safe_energy(self, frequency_mhz: int, energy_j: Optional[float]) -> None:
        if energy_j is None or not math.isfinite(float(energy_j)):
            return
        frequency = int(frequency_mhz)
        count = self.state.safe_energy_observations_by_frequency.get(frequency, 0)
        previous = self.state.safe_energy_by_frequency_j.get(frequency, 0.0)
        updated = (previous * count + float(energy_j)) / float(count + 1)
        self.state.safe_energy_by_frequency_j[frequency] = updated
        self.state.safe_energy_observations_by_frequency[frequency] = count + 1
        best_frequency, best_energy = min(
            self.state.safe_energy_by_frequency_j.items(), key=lambda item: (item[1], item[0])
        )
        self.state.best_measured_safe_frequency = best_frequency
        self.state.best_measured_safe_energy_j = best_energy

    def _recompute_best_safe_energy(self) -> None:
        eligible = {
            int(frequency): float(energy)
            for frequency, energy in self.state.safe_energy_by_frequency_j.items()
            if int(frequency) in self.state.validated_safe_frequencies
            and int(frequency) not in self.state.revoked_frequencies
        }
        self.state.safe_energy_by_frequency_j = eligible
        self.state.safe_energy_observations_by_frequency = {
            int(frequency): int(count)
            for frequency, count in (
                self.state.safe_energy_observations_by_frequency.items()
            )
            if int(frequency) in eligible
        }
        if not eligible:
            self.state.best_measured_safe_frequency = None
            self.state.best_measured_safe_energy_j = None
            return
        best_frequency, best_energy = min(
            eligible.items(), key=lambda item: (item[1], item[0])
        )
        self.state.best_measured_safe_frequency = best_frequency
        self.state.best_measured_safe_energy_j = best_energy

    def _remember_safe_evidence(self, evidence: PoolEvidence) -> None:
        self.state.recent_evidence_frequency = int(
            evidence.observed_frequency_mhz
        )
        self.state.recent_valid_request_count = int(evidence.valid_request_count)
        self.state.recent_phase_pass_count = int(evidence.phase_pass_count)
        self.state.recent_phase_token_coverage = int(
            evidence.phase_token_coverage
        )
        self.state.recent_evidence_context_signature = str(
            evidence.context_signature
        )
        self.state.recent_evidence_captured_at_s = float(evidence.captured_at_s)

    def _clear_recent_evidence(self) -> None:
        self.state.recent_evidence_frequency = None
        self.state.recent_valid_request_count = 0
        self.state.recent_phase_pass_count = 0
        self.state.recent_phase_token_coverage = 0
        self.state.recent_evidence_context_signature = ""
        self.state.recent_evidence_captured_at_s = 0.0

    def is_frequency_validated_safe(self, frequency_mhz: int) -> bool:
        frequency = int(frequency_mhz)
        return (
            frequency in self.state.validated_safe_frequencies
            and frequency not in self.state.revoked_frequencies
        )

    def assert_safety_invariants(self) -> None:
        overlap = (
            self.state.validated_safe_frequencies
            & self.state.revoked_frequencies
        )
        if overlap:
            raise RuntimeError(
                "frequency cannot be both validated-safe and revoked: %s"
                % sorted(overlap)
            )
        if (
            self.state.unsafe_bound is not None
            and int(self.state.unsafe_bound)
            in self.state.validated_safe_frequencies
        ):
            raise RuntimeError(
                "unsafe bound cannot remain in the validated-safe set"
            )
        if (
            self.state.unsafe_bound is not None
            and int(self.state.known_safe_frequency)
            == int(self.state.unsafe_bound)
        ):
            raise RuntimeError(
                "validated-safe frequency cannot equal unsafe bound"
            )
        if int(self.state.known_safe_frequency) not in (
            self.state.validated_safe_frequencies
        ):
            raise RuntimeError(
                "known-safe frequency is absent from validated-safe set"
            )

    def revoke_current_frequency(
        self, reason: str, event_timestamp_s: float,
    ) -> int:
        """Revoke current lower point and select a strictly higher safe anchor."""
        state = self.state
        current = int(state.current_frequency)
        high = int(self.frequencies_mhz[-1])
        if current >= high:
            raise RuntimeError(
                "no strictly higher recovery frequency exists above HIGH"
            )
        prior_energy = state.safe_energy_by_frequency_j.pop(current, None)
        prior_observations = state.safe_energy_observations_by_frequency.pop(
            current, None
        )
        if prior_energy is not None or prior_observations is not None:
            state.revoked_safe_energy_history.append({
                "frequency_mhz": current,
                "energy_j": prior_energy,
                "observations": prior_observations,
                "context_signature": state.context_signature,
                "revocation_reason": str(reason),
                "revoked_at_s": float(event_timestamp_s),
            })
        state.validated_safe_frequencies.discard(current)
        state.revoked_frequencies.add(current)
        # Backward-compatible state may predate the explicit set; import only
        # strictly higher, non-revoked anchors already represented as safe.
        for frequency in state.safe_energy_by_frequency_j:
            if int(frequency) > current:
                state.validated_safe_frequencies.add(int(frequency))
        if (
            int(state.known_safe_frequency) > current
            and int(state.known_safe_frequency) not in state.revoked_frequencies
        ):
            state.validated_safe_frequencies.add(int(state.known_safe_frequency))
        higher = sorted(
            frequency for frequency in state.validated_safe_frequencies
            if int(frequency) > current
        )
        target = int(higher[0]) if higher else high
        # HIGH is the conservative cold-start/fail-closed anchor when no
        # measured higher lower-frequency point remains.
        state.validated_safe_frequencies.add(target)
        state.unsafe_index = self.frequencies_mhz.index(current)
        state.unsafe_bound = current
        state.known_safe_frequency = target
        state.last_validated_safe_index = self.frequencies_mhz.index(target)
        state.candidate_frequency = None
        state.search_phase = SearchPhase.RECOVER.value
        state.search_status = SearchPhase.RECOVER.value
        state.transition_reason = str(reason)
        state.last_action_reason = (
            "revoke unsafe point and recover to strictly higher safe anchor"
        )
        self._clear_recent_evidence()
        state.converged = state.last_validated_safe_index == state.unsafe_index + 1
        if state.converged:
            state.convergence_reason = (
                "safe/unsafe boundary resolved after safe-point revocation"
            )
        self._recompute_best_safe_energy()
        self.assert_safety_invariants()
        return target

    def _mark_converged(self, reason: str) -> None:
        if self.state.converged:
            return
        self.state.converged = True
        self.state.convergence_reason = reason
        self.state.convergence_window_count = self.state.windows_observed
        self.state.convergence_request_count = self.state.requests_observed

    def _best_safe_target(self) -> int:
        if (
            self.state.best_measured_safe_frequency is not None
            and self.is_frequency_validated_safe(
                self.state.best_measured_safe_frequency
            )
        ):
            return int(self.state.best_measured_safe_frequency)
        return int(self.state.known_safe_frequency)

    def _make_decision(
        self, evidence: PoolEvidence, now_s: float, *, decision: str, action: str,
        trigger: str, reason: str, target: int, classification: EvidenceClass,
        jump: int = 0,
    ) -> AdaptiveDecision:
        self.state.last_action_reason = reason
        return AdaptiveDecision(
            pool=self.pool,
            phase=self.phase,
            slo_regime=self.slo_regime,
            current_frequency_mhz=self.state.current_frequency,
            candidate_frequency_mhz=self.state.candidate_frequency,
            known_safe_frequency_mhz=self.state.known_safe_frequency,
            unsafe_bound_mhz=(
                self.frequencies_mhz[self.state.unsafe_index]
                if self.state.unsafe_index is not None else None
            ),
            decision=decision,
            action=action,
            trigger=trigger,
            reason=reason,
            requested_frequency_mhz=target,
            readback_frequency_mhz=(
                evidence.observed_frequency_mhz if evidence.readback_valid else None
            ),
            readback_valid=bool(
                evidence.readback_valid
                and evidence.observed_frequency_mhz == self.state.current_frequency
            ),
            evidence_classification=classification.value,
            evidence_duration_s=evidence.observation_duration_s,
            valid_request_count=evidence.valid_request_count,
            input_token_coverage=evidence.input_token_coverage,
            output_token_coverage=evidence.output_token_coverage,
            latency_statistic_ms=evidence.phase_latency_ms,
            attainment=evidence.attainment(),
            backlog_stable=evidence.backlog_stable,
            pressure_signal=evidence.pressure_signal,
            telemetry_fresh=evidence.telemetry_fresh,
            headroom_fraction=evidence.headroom_fraction(self._slo_ms()),
            jump_size_levels=jump,
            decision_timestamp_s=float(now_s),
            action_timestamp_s=None,
            readback_timestamp_s=evidence.readback_timestamp_s,
            search_phase=self.state.search_phase,
        )

    def decide(
        self, evidence: PoolEvidence, *, now_s: Optional[float] = None,
        allow_exploration: bool = True,
    ) -> AdaptiveDecision:
        """Consume one observation and return a safe target/action.

        The returned target is merely a recommendation until the caller has
        performed hardware actuation and readback.  A target is never counted
        as safe before a subsequent valid observation.
        """
        now = float(self.clock() if now_s is None else now_s)
        self.state.evidence_since_action += 1
        self.state.windows_observed += 1
        self.state.requests_observed += evidence.valid_request_count
        if (
            self.config.context_invalidation_on_change
            and self.state.context_signature not in {"", "stationary"}
            and evidence.context_signature != self.state.context_signature
        ):
            self.state.search_phase = SearchPhase.RECOVER.value
            self.state.search_status = SearchPhase.RECOVER.value
            self.state.transition_reason = EvidenceClass.CONTEXT_CHANGE.value
            high = self.frequencies_mhz[-1]
            self.state.known_safe_frequency = high
            self.state.last_validated_safe_index = len(self.frequencies_mhz) - 1
            self.state.unsafe_index = None
            self.state.unsafe_bound = None
            self.state.context_signature = evidence.context_signature
            self.state.confidence = 0.0
            self.state.safe_energy_by_frequency_j.clear()
            self.state.safe_energy_observations_by_frequency.clear()
            self.state.validated_safe_frequencies = {high}
            self.state.revoked_frequencies.clear()
            self.state.best_measured_safe_frequency = None
            self.state.best_measured_safe_energy_j = None
            self._clear_recent_evidence()
            self.state.converged = False
            self.state.convergence_reason = None
            self.state.convergence_window_count = None
            self.state.convergence_request_count = None
            action = "RECOVER_HIGH" if high != self.state.current_frequency else "HOLD"
            decision = self._make_decision(
                evidence, now, decision="RECOVER" if action != "HOLD" else "HOLD",
                action=action, trigger=EvidenceClass.CONTEXT_CHANGE.value,
                reason="context signature changed; invalidate prior search evidence",
                target=high, classification=EvidenceClass.CONTEXT_CHANGE,
            )
            self.state.candidate_frequency = None
            return decision
        classification, reason = self._classify(evidence, now)
        headroom = evidence.headroom_fraction(self._slo_ms())

        if classification in {
            EvidenceClass.READBACK_FAILURE,
            EvidenceClass.INVALID_MEASUREMENT,
            EvidenceClass.STALE_TELEMETRY,
            EvidenceClass.PRESSURE_RISK,
        }:
            self.state.search_phase = SearchPhase.RECOVER.value
            self.state.search_status = SearchPhase.RECOVER.value
            self.state.transition_reason = classification.value
            high = self.frequencies_mhz[-1]
            target = high
            action = "RECOVER_HIGH" if target != self.state.current_frequency else "HOLD"
            decision = self._make_decision(
                evidence, now, decision="RECOVER" if action != "HOLD" else "HOLD",
                action=action, trigger=classification.value, reason=reason,
                target=target, classification=classification,
            )
            self.state.candidate_frequency = None
            return decision

        if classification == EvidenceClass.INSUFFICIENT_EVIDENCE:
            self.state.search_phase = SearchPhase.HOLD.value
            self.state.search_status = SearchPhase.HOLD.value
            self.state.transition_reason = "INSUFFICIENT_EVIDENCE"
            if self._timed_out(evidence, now) and self.state.current_frequency != self.frequencies_mhz[-1]:
                self.state.search_phase = SearchPhase.RECOVER.value
                self.state.search_status = SearchPhase.RECOVER.value
                decision = self._make_decision(
                    evidence, now, decision="RECOVER", action="RECOVER_HIGH",
                    trigger="EVIDENCE_TIMEOUT", reason="maximum evidence timeout reached without SAFE proof",
                    target=self.frequencies_mhz[-1], classification=classification,
                )
                self.state.candidate_frequency = None
                return decision
            return self._make_decision(
                evidence, now, decision="HOLD", action="HOLD", trigger="INSUFFICIENT_EVIDENCE",
                reason=reason, target=self.state.current_frequency, classification=classification,
            )

        # A measured lower candidate becomes a known-safe anchor only here.
        if classification == EvidenceClass.UNSAFE:
            current = int(self.state.current_frequency)
            target = self.revoke_current_frequency("SLO_RISK", now)
            action = (
                "RECOVER_HIGH"
                if target == self.frequencies_mhz[-1]
                else "STEP_UP"
            )
            decision = self._make_decision(
                evidence, now, decision="RECOVER",
                action=action, trigger="SLO_RISK", reason=reason, target=target,
                classification=classification,
            )
            if target <= current:
                raise RuntimeError("unsafe lower frequency did not recover upward")
            return decision

        # SAFE evidence: retain the current point and update energy evidence.
        if classification == EvidenceClass.SAFE:
            if self.state.current_frequency in self.state.revoked_frequencies:
                raise RuntimeError(
                    "revoked frequency cannot be revalidated without context reset"
                )
            self.state.known_safe_frequency = self.state.current_frequency
            self.state.last_validated_safe_index = self.state.current_index
            self.state.validated_safe_frequencies.add(
                int(self.state.current_frequency)
            )
            self.state.candidate_frequency = None
            self.state.search_phase = SearchPhase.EXPAND.value
            self.state.search_status = SearchPhase.EXPAND.value
            self.state.transition_reason = "SAFE_EVIDENCE"
            self.state.context_signature = evidence.context_signature
            self.state.confidence = min(1.0, self.state.confidence + 0.25)
            self._record_safe_energy(self.state.current_frequency, evidence.gross_energy_j)
            self._remember_safe_evidence(evidence)
            self.assert_safety_invariants()
            if self.state.converged:
                target = self._best_safe_target()
                action = "HOLD"
                decision_name = "ACCEPT_SAFE"
                if target != self.state.current_frequency:
                    action = "STEP_UP" if target > self.state.current_frequency else "STEP_DOWN"
                    decision_name = "SELECT_BEST_SAFE"
                self.state.search_phase = SearchPhase.HOLD.value
                self.state.search_status = SearchPhase.HOLD.value
                self.state.transition_reason = "SEARCH_CONVERGED"
                return self._make_decision(
                    evidence, now, decision=decision_name, action=action,
                    trigger="FINAL_EVIDENCE" if not allow_exploration else "SEARCH_CONVERGED",
                    reason=self.state.convergence_reason or "search already converged",
                    target=target, classification=classification,
                )
            if not allow_exploration:
                index, _, boundary_reason = self._next_down_index(headroom)
                target = self.state.current_frequency
                action = "HOLD"
                decision_name = "ACCEPT_SAFE"
                if index is None:
                    self._mark_converged(boundary_reason)
                    target = self._best_safe_target()
                    if target != self.state.current_frequency:
                        action = "STEP_UP" if target > self.state.current_frequency else "STEP_DOWN"
                        decision_name = "SELECT_BEST_SAFE"
                self.state.search_phase = SearchPhase.HOLD.value
                self.state.search_status = SearchPhase.HOLD.value
                self.state.transition_reason = "FINAL_EVIDENCE"
                return self._make_decision(
                    evidence, now, decision=decision_name, action=action,
                    trigger="FINAL_EVIDENCE", reason="final window classified; no unmeasured probe allowed",
                    target=target, classification=classification,
                )
            index, jump, next_reason = self._next_down_index(headroom)
            if index is None:
                self.state.search_phase = SearchPhase.HOLD.value
                self.state.search_status = SearchPhase.HOLD.value
                self._mark_converged(next_reason)
                target = self._best_safe_target()
                action = "HOLD"
                decision_name = "ACCEPT_SAFE"
                if target != self.state.current_frequency:
                    action = "STEP_UP" if target > self.state.current_frequency else "STEP_DOWN"
                    decision_name = "SELECT_BEST_SAFE"
                return self._make_decision(
                    evidence, now, decision=decision_name, action=action,
                    trigger="ENOUGH_EVIDENCE", reason=next_reason,
                    target=target, classification=classification,
                )
            target = self.frequencies_mhz[index]
            self.state.candidate_frequency = target
            self.state.search_phase = SearchPhase.REFINE.value if self.state.unsafe_index is not None else SearchPhase.EXPAND.value
            self.state.search_status = self.state.search_phase
            trigger = "REFINE" if self.state.unsafe_index is not None else "SEARCH_DOWN"
            return self._make_decision(
                evidence, now, decision="EXPLORE_DOWN", action="STEP_DOWN",
                trigger=trigger, reason=next_reason, target=target,
                classification=classification, jump=jump,
            )

        raise AssertionError("unhandled evidence classification")


def evidence_from_window(
    pool: str,
    metrics: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    *,
    observed_frequency_mhz: int,
    slo_regime: str,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    context_signature: str = "stationary",
    captured_at_s: float = 0.0,
    endpoint_ids: Optional[Sequence[str]] = None,
) -> PoolEvidence:
    """Convert an existing C1 window into phase-specific pool evidence."""
    phase = "prefill" if pool == "P" else "decode"
    valid = []
    for row in requests:
        try:
            real_sse = str(row.get("real_sse", "True")).lower() == "true"
            valid_id = str(row.get("logical_request_id_propagated", "True")).lower() == "true"
            if real_sse and valid_id:
                valid.append(row)
        except (TypeError, ValueError):
            continue
    ttft = [float(row["ttft_ms"]) for row in valid]
    tpot = [float(row["tpot_ms"]) for row in valid]
    input_tokens = sum(int(row.get("input_len", 0)) for row in valid)
    output_tokens = sum(int(row.get("observed_output_tokens", row.get("output_len", 0))) for row in valid)
    ttft_pass = sum(value <= ttft_slo_ms for value in ttft)
    tpot_pass = sum(value <= tpot_slo_ms for value in tpot)
    close_wait_raw = metrics.get("client_waiting_at_offered_close")
    close_wait = float(close_wait_raw) if close_wait_raw is not None else None
    endpoint_load = metrics.get("endpoint_load")
    if endpoint_load is None and metrics.get("endpoint_load_json"):
        try:
            endpoint_load = json.loads(str(metrics["endpoint_load_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            endpoint_load = None
    relevant_endpoints = tuple(endpoint_ids or (
        ("P0", "P1") if pool == "P" else ("D0", "D1")
    ))
    server_queue_pressure = False
    endpoint_pressure_observed = isinstance(endpoint_load, Mapping)
    if isinstance(endpoint_load, Mapping):
        for endpoint_id in relevant_endpoints:
            value = endpoint_load.get(endpoint_id)
            if not isinstance(value, Mapping):
                endpoint_pressure_observed = False
                continue
            for key in ("waiting_mean", "waiting_max"):
                raw = value.get(key)
                if raw is None or raw == "":
                    endpoint_pressure_observed = False
                    continue
                try:
                    if float(raw) > 0.0:
                        server_queue_pressure = True
                except (TypeError, ValueError):
                    endpoint_pressure_observed = False
    backlog_growing = metrics.get("backlog_growing")
    client_pressure_observed = backlog_growing is not None and close_wait is not None
    if not endpoint_pressure_observed or not client_pressure_observed:
        backlog_stable = None
        pressure_signal = "unknown_required_pressure"
    else:
        backlog_stable = (
            not bool(backlog_growing)
            and close_wait <= 0.0
            and not server_queue_pressure
        )
        pressure_signal = (
            "stable" if backlog_stable
            else ("server_queue" if server_queue_pressure else "backlog_risk")
        )
    energy_values = [metrics.get("%s_energy_j" % item) for item in relevant_endpoints]
    gross = None
    if energy_values and all(item is not None for item in energy_values):
        gross = sum(float(item) for item in energy_values)
    return PoolEvidence(
        pool=pool,
        phase=phase,
        observed_frequency_mhz=int(observed_frequency_mhz),
        observation_duration_s=float(metrics.get("observation_duration_s") or 0.0),
        valid_request_count=len(valid),
        input_token_coverage=input_tokens,
        output_token_coverage=output_tokens,
        ttft_statistic_ms=(float(metrics["ttft_p95_ms"]) if metrics.get("ttft_p95_ms") is not None else None),
        tpot_statistic_ms=(float(metrics["tpot_p95_ms"]) if metrics.get("tpot_p95_ms") is not None else None),
        ttft_pass_count=ttft_pass,
        ttft_violation_count=len(valid) - ttft_pass,
        tpot_pass_count=tpot_pass,
        tpot_violation_count=len(valid) - tpot_pass,
        backlog_stable=backlog_stable,
        telemetry_fresh=bool(metrics.get("telemetry_fresh", metrics.get("measurement_valid", False))),
        health_valid=bool(metrics.get("measurement_valid", False)),
        readback_valid=bool(metrics.get("readback_valid", metrics.get("measurement_valid", False))),
        request_accounting_valid=bool(metrics.get("measurement_valid", False)) and len(valid) == len(requests),
        pressure_signal=pressure_signal,
        gross_energy_j=gross,
        context_signature=context_signature,
        captured_at_s=float(captured_at_s),
        readback_timestamp_s=None,
    )


__all__ = [
    "AdaptiveDecision",
    "AdaptiveExplorationConfig",
    "AdaptivePoolController",
    "EvidenceClass",
    "PoolEvidence",
    "PoolSearchState",
    "SearchPhase",
    "evidence_from_window",
]
