"""Predictor-free XpYd routing and conservative DVFS feedback."""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Callable, Dict, Optional, Tuple

from xpyd.compatibility import CompatibilityTable
from xpyd.hardware import HardwareProfile
from xpyd.registry import EndpointRegistry
from xpyd.telemetry import EndpointTelemetrySnapshot, TelemetryAggregator
from xpyd.types import EndpointSpec, LifecycleState, RoutingDecision


class NoEligibleRouteError(RuntimeError):
    """Raised when neither a measured-safe route nor a valid fallback exists."""


class DVFSAction(str, Enum):
    HOLD = "HOLD"
    STEP_UP = "STEP_UP"
    STEP_DOWN = "STEP_DOWN"
    FALLBACK_MAX = "FALLBACK_MAX"


class LatencySafetyMetric(str, Enum):
    EWMA = "ewma"
    WINDOW_P95 = "window_p95"
    WINDOW_P99 = "window_p99"


@dataclass(frozen=True)
class DVFSRecommendation:
    endpoint_id: str
    current_freq_mhz: Optional[int]
    target_freq_mhz: int
    action: DVFSAction
    reason: str
    last_recommendation_time_s: Optional[float] = None
    last_actuation_time_s: Optional[float] = None


@dataclass(frozen=True)
class RouteCandidate:
    prefill_endpoint_id: str
    decode_endpoint_id: str
    energy_per_request_j: Optional[float]
    ranking_basis: str
    workload_context_key: Optional[str] = None


@dataclass(frozen=True)
class RouteEvaluation:
    prefill_endpoint_id: str
    decode_endpoint_id: str
    eligible: bool
    reason: str
    energy_per_request_j: Optional[float]
    ranking_basis: str
    workload_context_key: Optional[str] = None


@dataclass(frozen=True)
class FeedbackSchedulerConfig:
    ttft_slo_ms: float
    tbt_slo_ms: Optional[float] = None
    tpot_slo_ms: Optional[float] = None
    safety_fraction: float = 0.8
    min_samples: int = 1
    prefill_safety_metric: LatencySafetyMetric = LatencySafetyMetric.EWMA
    decode_safety_metric: LatencySafetyMetric = LatencySafetyMetric.EWMA
    min_tail_samples: int = 20
    telemetry_max_age_s: float = 30.0
    max_queue_depth: int = 8
    max_kv_cache_usage_frac: float = 0.90
    fallback_prefill_endpoint_id: Optional[str] = None
    fallback_decode_endpoint_id: Optional[str] = None
    fallback_prefill_freq_mhz: Optional[int] = None
    fallback_decode_freq_mhz: Optional[int] = None
    dvfs_step_down_fraction: float = 0.50
    dvfs_step_up_fraction: float = 0.80
    dvfs_low_queue_depth: int = 0
    dvfs_low_kv_cache_usage_frac: float = 0.50
    severe_queue_depth: int = 16
    severe_kv_cache_usage_frac: float = 0.97
    severe_pressure_to_max: bool = True
    dvfs_min_dwell_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.ttft_slo_ms) or self.ttft_slo_ms <= 0:
            raise ValueError("ttft_slo_ms must be positive")
        if self.tbt_slo_ms is None and self.tpot_slo_ms is None:
            raise ValueError("at least one decode TBT/TPOT SLO must be configured")
        for name, value in (
            ("tbt_slo_ms", self.tbt_slo_ms),
            ("tpot_slo_ms", self.tpot_slo_ms),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("%s must be positive when configured" % name)
        if not 0 < self.safety_fraction <= 1:
            raise ValueError("safety_fraction must be in (0, 1]")
        if not isinstance(self.min_samples, int) or self.min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")
        if not isinstance(self.min_tail_samples, int) or self.min_tail_samples <= 0:
            raise ValueError("min_tail_samples must be a positive integer")
        for field_name in ("prefill_safety_metric", "decode_safety_metric"):
            raw_metric = getattr(self, field_name)
            try:
                metric = (
                    raw_metric
                    if isinstance(raw_metric, LatencySafetyMetric)
                    else LatencySafetyMetric(raw_metric)
                )
            except ValueError as exc:
                raise ValueError("invalid %s" % field_name) from exc
            object.__setattr__(self, field_name, metric)
        if not math.isfinite(self.telemetry_max_age_s) or self.telemetry_max_age_s <= 0:
            raise ValueError("telemetry_max_age_s must be positive")
        if not math.isfinite(self.dvfs_min_dwell_s) or self.dvfs_min_dwell_s < 0:
            raise ValueError("dvfs_min_dwell_s cannot be negative")
        if self.max_queue_depth < 0 or self.dvfs_low_queue_depth < 0:
            raise ValueError("queue thresholds cannot be negative")
        if self.severe_queue_depth < self.max_queue_depth:
            raise ValueError("severe_queue_depth cannot be below max_queue_depth")
        for name, value in (
            ("max_kv_cache_usage_frac", self.max_kv_cache_usage_frac),
            ("dvfs_low_kv_cache_usage_frac", self.dvfs_low_kv_cache_usage_frac),
            ("severe_kv_cache_usage_frac", self.severe_kv_cache_usage_frac),
        ):
            if not 0 <= value <= 1:
                raise ValueError("%s must be in [0, 1]" % name)
        if self.severe_kv_cache_usage_frac < self.max_kv_cache_usage_frac:
            raise ValueError("severe KV threshold cannot be below the routing threshold")
        if not 0 < self.dvfs_step_down_fraction < self.dvfs_step_up_fraction <= 1:
            raise ValueError("DVFS fractions must satisfy 0 < down < up <= 1")

        fallback_values = (
            self.fallback_prefill_endpoint_id,
            self.fallback_decode_endpoint_id,
            self.fallback_prefill_freq_mhz,
            self.fallback_decode_freq_mhz,
        )
        if any(value is not None for value in fallback_values) and not all(
            value is not None for value in fallback_values
        ):
            raise ValueError("fallback endpoint IDs and frequencies must be configured together")
        for endpoint_id in (
            self.fallback_prefill_endpoint_id,
            self.fallback_decode_endpoint_id,
        ):
            if endpoint_id is not None and not endpoint_id.strip():
                raise ValueError("fallback endpoint IDs must be non-empty")
        for freq_mhz in (
            self.fallback_prefill_freq_mhz,
            self.fallback_decode_freq_mhz,
        ):
            if freq_mhz is not None and (
                not isinstance(freq_mhz, int)
                or isinstance(freq_mhz, bool)
                or freq_mhz <= 0
            ):
                raise ValueError("fallback frequencies must be positive integers")

    @property
    def has_fallback(self) -> bool:
        return self.fallback_prefill_endpoint_id is not None


class FeedbackScheduler:
    """Interpretable routing/DVFS feedback using observations only."""

    def __init__(
        self,
        registry: EndpointRegistry,
        compatibility: CompatibilityTable,
        telemetry: TelemetryAggregator,
        hardware: HardwareProfile,
        config: FeedbackSchedulerConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.registry = registry
        self.compatibility = compatibility
        self.telemetry = telemetry
        self.hardware = hardware
        self.config = config
        self.clock = clock
        # Recommendation history is diagnostic only. Dwell is based solely on
        # separately recorded successful actuation/readback events.
        self._last_dvfs_recommendation_s: Dict[str, float] = {}
        self._last_dvfs_actuation_s: Dict[str, float] = {}

    def evaluate_routes(
        self,
        now_s: Optional[float] = None,
        workload_context_key: Optional[str] = None,
    ) -> Tuple[RouteEvaluation, ...]:
        """Explain every registered P/D pair without changing runtime state."""

        now_s = self.clock() if now_s is None else now_s
        prefill_endpoints = sorted(
            self.registry.list_by_role("prefill"),
            key=lambda endpoint: endpoint.endpoint_id,
        )
        decode_endpoints = sorted(
            self.registry.list_by_role("decode"),
            key=lambda endpoint: endpoint.endpoint_id,
        )
        evaluations = []
        for prefill in prefill_endpoints:
            for decode in decode_endpoints:
                rejection = self._hard_safety_reason(prefill, require_current_frequency=True)
                if rejection is None:
                    rejection = self._hard_safety_reason(decode, require_current_frequency=True)
                if rejection is None and not self.compatibility.is_compatible(prefill, decode):
                    rejection = "missing_or_unsupported_connector_compatibility"
                if rejection is None:
                    rejection = self._latency_safety_reason(prefill, now_s)
                if rejection is None:
                    rejection = self._latency_safety_reason(decode, now_s)

                pair_energy = None
                ranking_basis = "ineligible"
                if rejection is None:
                    pair_energy = self._pair_energy_per_request(
                        self.telemetry.snapshot(prefill.endpoint_id, now_s=now_s),
                        self.telemetry.snapshot(decode.endpoint_id, now_s=now_s),
                    )
                    ranking_basis = (
                        "measured_energy_per_request"
                        if pair_energy is not None
                        else "deterministic_endpoint_ids"
                    )
                evaluations.append(
                    RouteEvaluation(
                        prefill_endpoint_id=prefill.endpoint_id,
                        decode_endpoint_id=decode.endpoint_id,
                        eligible=rejection is None,
                        reason="eligible" if rejection is None else rejection,
                        energy_per_request_j=pair_energy,
                        ranking_basis=ranking_basis,
                        workload_context_key=workload_context_key,
                    )
                )
        return tuple(evaluations)

    def eligible_routes(
        self,
        now_s: Optional[float] = None,
        workload_context_key: Optional[str] = None,
    ) -> Tuple[RouteCandidate, ...]:
        """Return measured-safe compatible pairs in deterministic rank order."""

        candidates = [
            RouteCandidate(
                prefill_endpoint_id=evaluation.prefill_endpoint_id,
                decode_endpoint_id=evaluation.decode_endpoint_id,
                energy_per_request_j=evaluation.energy_per_request_j,
                ranking_basis=evaluation.ranking_basis,
                workload_context_key=evaluation.workload_context_key,
            )
            for evaluation in self.evaluate_routes(now_s, workload_context_key)
            if evaluation.eligible
        ]

        candidates.sort(
            key=lambda candidate: (
                candidate.energy_per_request_j is None,
                candidate.energy_per_request_j
                if candidate.energy_per_request_j is not None
                else 0.0,
                candidate.prefill_endpoint_id,
                candidate.decode_endpoint_id,
            )
        )
        return tuple(candidates)

    def choose_route(
        self,
        now_s: Optional[float] = None,
        workload_context_key: Optional[str] = None,
    ) -> RoutingDecision:
        candidates = self.eligible_routes(now_s, workload_context_key)
        if not candidates:
            return self._fallback_decision()
        selected = candidates[0]
        prefill_state = self.registry.get_state(selected.prefill_endpoint_id)
        decode_state = self.registry.get_state(selected.decode_endpoint_id)
        # eligible_routes() requires both observed frequencies to be supported.
        assert prefill_state.freq_mhz is not None
        assert decode_state.freq_mhz is not None
        return RoutingDecision(
            prefill_endpoint_id=selected.prefill_endpoint_id,
            decode_endpoint_id=selected.decode_endpoint_id,
            prefill_freq_mhz=prefill_state.freq_mhz,
            decode_freq_mhz=decode_state.freq_mhz,
        )

    def choose_frequency_adjustment(
        self,
        endpoint_id: str,
        now_s: Optional[float] = None,
    ) -> DVFSRecommendation:
        """Recommend, but never execute, one supported frequency action."""

        now_s = self.clock() if now_s is None else now_s
        endpoint = self.registry.get_spec(endpoint_id)
        state = self.registry.get_state(endpoint_id)
        self.hardware.validate_endpoint(endpoint)
        allowed = tuple(sorted(self.hardware.gpu_type(endpoint.gpu_type).allowed_frequencies_mhz))
        current = state.freq_mhz

        if current not in allowed:
            return self._recommend(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[-1],
                action=DVFSAction.FALLBACK_MAX,
                reason="observed frequency is absent or unsupported; request a known safe maximum",
                now_s=now_s,
            )
        if not state.healthy or state.lifecycle != LifecycleState.ACTIVE:
            return self._hold(endpoint_id, current, "endpoint is not healthy and ACTIVE")
        if not state.queue_depth_observed or not state.kv_cache_usage_observed:
            return self._hold(
                endpoint_id,
                current,
                "queue/KV pressure telemetry has never been observed",
            )

        snapshot = self.telemetry.snapshot(endpoint_id, now_s=now_s)
        metric, slo_ms, metric_name, tail_count, observed_at_s = self._latency_metric(
            endpoint, snapshot
        )
        if (
            observed_at_s is None
            or now_s - observed_at_s > self.config.telemetry_max_age_s
        ):
            return self._hold(endpoint_id, current, "latency telemetry is missing or stale")
        selected_policy = self._safety_metric_for(endpoint)
        if (
            snapshot.sample_count < self.config.min_samples
            or metric is None
            or slo_ms is None
            or (
                selected_policy != LatencySafetyMetric.EWMA
                and tail_count < self.config.min_tail_samples
            )
        ):
            return self._hold(endpoint_id, current, "insufficient measured latency telemetry")

        severe_pressure = (
            state.queue_depth >= self.config.severe_queue_depth
            or state.kv_cache_usage_frac >= self.config.severe_kv_cache_usage_frac
        )
        if severe_pressure and self.config.severe_pressure_to_max:
            return self._recommend(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[-1],
                action=DVFSAction.FALLBACK_MAX,
                reason="severe queue/KV pressure; request maximum supported frequency",
                now_s=now_s,
            )

        pressure = (
            metric > self.config.dvfs_step_up_fraction * slo_ms
            or state.queue_depth > self.config.max_queue_depth
            or state.kv_cache_usage_frac > self.config.max_kv_cache_usage_frac
        )
        current_index = allowed.index(current)
        if pressure:
            if current_index == len(allowed) - 1:
                return self._hold(
                    endpoint_id,
                    current,
                    "%s/queue/KV pressure, already at maximum frequency" % metric_name,
                )
            emergency = (
                metric > slo_ms
                or state.queue_depth > self.config.max_queue_depth
                or state.kv_cache_usage_frac > self.config.max_kv_cache_usage_frac
            )
            if not emergency and self._cooldown_active(endpoint_id, now_s):
                return self._hold(endpoint_id, current, "DVFS minimum dwell time is active")
            return self._recommend(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[current_index + 1],
                action=DVFSAction.STEP_UP,
                reason="%s or runtime pressure crossed the step-up threshold" % metric_name,
                now_s=now_s,
            )

        large_slack = (
            metric <= self.config.dvfs_step_down_fraction * slo_ms
            and state.queue_depth <= self.config.dvfs_low_queue_depth
            and state.kv_cache_usage_frac <= self.config.dvfs_low_kv_cache_usage_frac
        )
        if large_slack:
            if current_index == 0:
                return self._hold(
                    endpoint_id,
                    current,
                    "large SLO slack, already at minimum supported frequency",
                )
            if self._cooldown_active(endpoint_id, now_s):
                return self._hold(endpoint_id, current, "DVFS minimum dwell time is active")
            return self._recommend(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[current_index - 1],
                action=DVFSAction.STEP_DOWN,
                reason="large measured SLO headroom and low queue/KV pressure",
                now_s=now_s,
            )

        return self._hold(endpoint_id, current, "moderate measured operating region")

    def _hard_safe(self, endpoint: EndpointSpec, require_current_frequency: bool) -> bool:
        return self._hard_safety_reason(endpoint, require_current_frequency) is None

    def _hard_safety_reason(
        self,
        endpoint: EndpointSpec,
        require_current_frequency: bool,
    ) -> Optional[str]:
        state = self.registry.get_state(endpoint.endpoint_id)
        if not state.healthy:
            return "%s_unhealthy" % endpoint.endpoint_id
        if state.lifecycle != LifecycleState.ACTIVE:
            return "%s_not_active" % endpoint.endpoint_id
        if not state.queue_depth_observed:
            return "%s_queue_unknown" % endpoint.endpoint_id
        if not state.kv_cache_usage_observed:
            return "%s_kv_unknown" % endpoint.endpoint_id
        if state.queue_depth > self.config.max_queue_depth:
            return "%s_queue_pressure" % endpoint.endpoint_id
        if state.kv_cache_usage_frac > self.config.max_kv_cache_usage_frac:
            return "%s_kv_pressure" % endpoint.endpoint_id
        try:
            self.hardware.validate_endpoint(endpoint)
        except (KeyError, ValueError):
            return "%s_hardware_profile_invalid" % endpoint.endpoint_id
        if require_current_frequency:
            allowed = self.hardware.gpu_type(endpoint.gpu_type).allowed_frequencies_mhz
            if state.freq_mhz not in allowed:
                return "%s_frequency_unsupported" % endpoint.endpoint_id
        return None

    def _latency_safe(self, endpoint: EndpointSpec, now_s: Optional[float] = None) -> bool:
        now_s = self.clock() if now_s is None else now_s
        return self._latency_safety_reason(endpoint, now_s) is None

    def _latency_safety_reason(
        self,
        endpoint: EndpointSpec,
        now_s: float,
    ) -> Optional[str]:
        snapshot = self.telemetry.snapshot(endpoint.endpoint_id, now_s=now_s)
        if snapshot.sample_count < self.config.min_samples:
            return "%s_insufficient_samples" % endpoint.endpoint_id
        metric, slo_ms, _, tail_count, observed_at_s = self._latency_metric(
            endpoint, snapshot
        )
        if observed_at_s is None:
            return "%s_latency_missing" % endpoint.endpoint_id
        if now_s - observed_at_s > self.config.telemetry_max_age_s:
            return "%s_telemetry_stale" % endpoint.endpoint_id
        if metric is None or slo_ms is None:
            return "%s_selected_latency_metric_missing" % endpoint.endpoint_id
        if (
            self._safety_metric_for(endpoint) != LatencySafetyMetric.EWMA
            and tail_count < self.config.min_tail_samples
        ):
            return "%s_insufficient_tail_samples" % endpoint.endpoint_id
        if metric > self.config.safety_fraction * slo_ms:
            return "%s_latency_margin_exceeded" % endpoint.endpoint_id
        return None

    def _safety_metric_for(self, endpoint: EndpointSpec) -> LatencySafetyMetric:
        return (
            self.config.prefill_safety_metric
            if endpoint.role == "prefill"
            else self.config.decode_safety_metric
        )

    def _latency_metric(
        self,
        endpoint: EndpointSpec,
        snapshot: EndpointTelemetrySnapshot,
    ) -> Tuple[Optional[float], Optional[float], str, int, Optional[float]]:
        policy = self._safety_metric_for(endpoint)
        if endpoint.role == "prefill":
            if policy == LatencySafetyMetric.WINDOW_P95:
                return (
                    snapshot.window_ttft_p95_ms,
                    self.config.ttft_slo_ms,
                    "window P95 TTFT",
                    snapshot.window_ttft_count,
                    snapshot.last_ttft_observation_s,
                )
            if policy == LatencySafetyMetric.WINDOW_P99:
                return (
                    snapshot.window_ttft_p99_ms,
                    self.config.ttft_slo_ms,
                    "window P99 TTFT",
                    snapshot.window_ttft_count,
                    snapshot.last_ttft_observation_s,
                )
            return (
                snapshot.ewma_ttft_ms,
                self.config.ttft_slo_ms,
                "EWMA TTFT",
                0,
                snapshot.last_ttft_observation_s,
            )
        if policy == LatencySafetyMetric.WINDOW_P95:
            return (
                snapshot.window_tbt_p95_ms,
                self.config.tbt_slo_ms,
                "window P95 TBT",
                snapshot.window_tbt_count,
                snapshot.last_tbt_observation_s,
            )
        if policy == LatencySafetyMetric.WINDOW_P99:
            return (
                snapshot.window_tbt_p99_ms,
                self.config.tbt_slo_ms,
                "window P99 TBT",
                snapshot.window_tbt_count,
                snapshot.last_tbt_observation_s,
            )
        if self.config.tbt_slo_ms is not None and snapshot.ewma_tbt_ms is not None:
            return (
                snapshot.ewma_tbt_ms,
                self.config.tbt_slo_ms,
                "EWMA TBT",
                0,
                snapshot.last_tbt_observation_s,
            )
        if self.config.tpot_slo_ms is not None and snapshot.ewma_tpot_ms is not None:
            return (
                snapshot.ewma_tpot_ms,
                self.config.tpot_slo_ms,
                "EWMA TPOT",
                0,
                snapshot.last_tpot_observation_s,
            )
        return None, None, "decode latency", 0, None

    @staticmethod
    def _pair_energy_per_request(
        prefill: EndpointTelemetrySnapshot,
        decode: EndpointTelemetrySnapshot,
    ) -> Optional[float]:
        if (
            prefill.ewma_energy_per_request_j is None
            or decode.ewma_energy_per_request_j is None
        ):
            return None
        # KV-transfer energy is intentionally not part of this phase's score.
        return (
            prefill.ewma_energy_per_request_j
            + decode.ewma_energy_per_request_j
        )

    def _fallback_decision(self) -> RoutingDecision:
        if not self.config.has_fallback:
            raise NoEligibleRouteError("no measured-safe compatible pair and no fallback configured")
        prefill = self.registry.get_spec(self.config.fallback_prefill_endpoint_id)
        decode = self.registry.get_spec(self.config.fallback_decode_endpoint_id)
        if prefill.role != "prefill" or decode.role != "decode":
            raise NoEligibleRouteError("fallback endpoint roles are invalid")
        prefill_rejection = self._hard_safety_reason(
            prefill, require_current_frequency=False
        )
        if prefill_rejection is not None:
            raise NoEligibleRouteError(
                "fallback prefill endpoint fails hard safety checks: %s"
                % prefill_rejection
            )
        decode_rejection = self._hard_safety_reason(
            decode, require_current_frequency=False
        )
        if decode_rejection is not None:
            raise NoEligibleRouteError(
                "fallback decode endpoint fails hard safety checks: %s"
                % decode_rejection
            )
        if not self.compatibility.is_compatible(prefill, decode):
            raise NoEligibleRouteError("fallback P/D pair lacks explicit compatibility evidence")
        prefill_allowed = self.hardware.gpu_type(prefill.gpu_type).allowed_frequencies_mhz
        decode_allowed = self.hardware.gpu_type(decode.gpu_type).allowed_frequencies_mhz
        if self.config.fallback_prefill_freq_mhz not in prefill_allowed:
            raise NoEligibleRouteError("fallback prefill frequency is unsupported")
        if self.config.fallback_decode_freq_mhz not in decode_allowed:
            raise NoEligibleRouteError("fallback decode frequency is unsupported")
        return RoutingDecision(
            prefill_endpoint_id=prefill.endpoint_id,
            decode_endpoint_id=decode.endpoint_id,
            prefill_freq_mhz=self.config.fallback_prefill_freq_mhz,
            decode_freq_mhz=self.config.fallback_decode_freq_mhz,
        )

    def last_dvfs_recommendation_time(self, endpoint_id: str) -> Optional[float]:
        return self._last_dvfs_recommendation_s.get(endpoint_id)

    def last_dvfs_actuation_time(self, endpoint_id: str) -> Optional[float]:
        return self._last_dvfs_actuation_s.get(endpoint_id)

    def record_dvfs_actuation(
        self,
        endpoint_id: str,
        observed_freq_mhz: int,
        timestamp_s: Optional[float] = None,
    ) -> None:
        """Record a completed external transition after successful readback.

        This bookkeeping method performs no actuation and does not update
        ``EndpointState``. A future actuator must update the observed state
        from hardware readback before calling it.
        """

        timestamp_s = self.clock() if timestamp_s is None else timestamp_s
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("DVFS actuation timestamp must be finite and non-negative")
        endpoint = self.registry.get_spec(endpoint_id)
        state = self.registry.get_state(endpoint_id)
        self.hardware.validate_endpoint(endpoint)
        allowed = self.hardware.gpu_type(endpoint.gpu_type).allowed_frequencies_mhz
        if observed_freq_mhz not in allowed:
            raise ValueError("actuated frequency is not hardware-supported")
        if state.freq_mhz != observed_freq_mhz:
            raise ValueError(
                "EndpointState frequency must match successful hardware readback"
            )
        previous = self._last_dvfs_actuation_s.get(endpoint_id)
        if previous is not None and timestamp_s < previous:
            raise ValueError("DVFS actuation timestamps must be non-decreasing")
        self._last_dvfs_actuation_s[endpoint_id] = float(timestamp_s)

    def _cooldown_active(self, endpoint_id: str, now_s: float) -> bool:
        previous = self._last_dvfs_actuation_s.get(endpoint_id)
        return (
            previous is not None
            and now_s - previous < self.config.dvfs_min_dwell_s
        )

    def _recommend(
        self,
        endpoint_id: str,
        current_freq_mhz: Optional[int],
        target_freq_mhz: int,
        action: DVFSAction,
        reason: str,
        now_s: float,
    ) -> DVFSRecommendation:
        if action != DVFSAction.HOLD:
            self._last_dvfs_recommendation_s[endpoint_id] = now_s
        return DVFSRecommendation(
            endpoint_id=endpoint_id,
            current_freq_mhz=current_freq_mhz,
            target_freq_mhz=target_freq_mhz,
            action=action,
            reason=reason,
            last_recommendation_time_s=self._last_dvfs_recommendation_s.get(
                endpoint_id
            ),
            last_actuation_time_s=self._last_dvfs_actuation_s.get(endpoint_id),
        )

    def _hold(self, endpoint_id: str, current: int, reason: str) -> DVFSRecommendation:
        return DVFSRecommendation(
            endpoint_id=endpoint_id,
            current_freq_mhz=current,
            target_freq_mhz=current,
            action=DVFSAction.HOLD,
            reason=reason,
            last_recommendation_time_s=self._last_dvfs_recommendation_s.get(
                endpoint_id
            ),
            last_actuation_time_s=self._last_dvfs_actuation_s.get(endpoint_id),
        )
