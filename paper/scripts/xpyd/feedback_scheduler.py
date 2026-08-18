"""Predictor-free XpYd routing and conservative DVFS feedback."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

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


@dataclass(frozen=True)
class DVFSRecommendation:
    endpoint_id: str
    current_freq_mhz: Optional[int]
    target_freq_mhz: int
    action: DVFSAction
    reason: str


@dataclass(frozen=True)
class RouteCandidate:
    prefill_endpoint_id: str
    decode_endpoint_id: str
    energy_per_request_j: Optional[float]
    ranking_basis: str


@dataclass(frozen=True)
class FeedbackSchedulerConfig:
    ttft_slo_ms: float
    tbt_slo_ms: Optional[float] = None
    tpot_slo_ms: Optional[float] = None
    safety_fraction: float = 0.8
    min_samples: int = 1
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
    ) -> None:
        self.registry = registry
        self.compatibility = compatibility
        self.telemetry = telemetry
        self.hardware = hardware
        self.config = config

    def eligible_routes(self) -> Tuple[RouteCandidate, ...]:
        """Return measured-safe compatible pairs in deterministic rank order."""

        prefill_endpoints = sorted(
            self.registry.healthy_active("prefill"),
            key=lambda endpoint: endpoint.endpoint_id,
        )
        decode_endpoints = sorted(
            self.registry.healthy_active("decode"),
            key=lambda endpoint: endpoint.endpoint_id,
        )
        safe_prefill = [
            endpoint for endpoint in prefill_endpoints
            if self._hard_safe(endpoint, require_current_frequency=True)
            and self._latency_safe(endpoint)
        ]
        safe_decode = [
            endpoint for endpoint in decode_endpoints
            if self._hard_safe(endpoint, require_current_frequency=True)
            and self._latency_safe(endpoint)
        ]

        candidates = []
        for prefill in safe_prefill:
            prefill_snapshot = self.telemetry.snapshot(prefill.endpoint_id)
            for decode in safe_decode:
                if not self.compatibility.is_compatible(prefill, decode):
                    continue
                decode_snapshot = self.telemetry.snapshot(decode.endpoint_id)
                pair_energy = self._pair_energy_per_request(
                    prefill_snapshot,
                    decode_snapshot,
                )
                candidates.append(
                    RouteCandidate(
                        prefill_endpoint_id=prefill.endpoint_id,
                        decode_endpoint_id=decode.endpoint_id,
                        energy_per_request_j=pair_energy,
                        ranking_basis=(
                            "measured_energy_per_request"
                            if pair_energy is not None
                            else "deterministic_endpoint_ids"
                        ),
                    )
                )

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

    def choose_route(self) -> RoutingDecision:
        candidates = self.eligible_routes()
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

    def choose_frequency_adjustment(self, endpoint_id: str) -> DVFSRecommendation:
        """Recommend, but never execute, one supported frequency action."""

        endpoint = self.registry.get_spec(endpoint_id)
        state = self.registry.get_state(endpoint_id)
        self.hardware.validate_endpoint(endpoint)
        allowed = tuple(sorted(self.hardware.gpu_type(endpoint.gpu_type).allowed_frequencies_mhz))
        current = state.freq_mhz

        if current not in allowed:
            return DVFSRecommendation(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[-1],
                action=DVFSAction.FALLBACK_MAX,
                reason="observed frequency is absent or unsupported; request a known safe maximum",
            )
        if not state.healthy or state.lifecycle != LifecycleState.ACTIVE:
            return self._hold(endpoint_id, current, "endpoint is not healthy and ACTIVE")

        snapshot = self.telemetry.snapshot(endpoint_id)
        metric, slo_ms, metric_name = self._latency_metric(endpoint, snapshot)
        if snapshot.sample_count < self.config.min_samples or metric is None or slo_ms is None:
            return self._hold(endpoint_id, current, "insufficient measured latency telemetry")

        severe_pressure = (
            state.queue_depth >= self.config.severe_queue_depth
            or state.kv_cache_usage_frac >= self.config.severe_kv_cache_usage_frac
        )
        if severe_pressure and self.config.severe_pressure_to_max:
            return DVFSRecommendation(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[-1],
                action=DVFSAction.FALLBACK_MAX,
                reason="severe queue/KV pressure; request maximum supported frequency",
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
            return DVFSRecommendation(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[current_index + 1],
                action=DVFSAction.STEP_UP,
                reason="%s or runtime pressure crossed the step-up threshold" % metric_name,
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
            return DVFSRecommendation(
                endpoint_id=endpoint_id,
                current_freq_mhz=current,
                target_freq_mhz=allowed[current_index - 1],
                action=DVFSAction.STEP_DOWN,
                reason="large measured SLO headroom and low queue/KV pressure",
            )

        return self._hold(endpoint_id, current, "moderate measured operating region")

    def _hard_safe(self, endpoint: EndpointSpec, require_current_frequency: bool) -> bool:
        state = self.registry.get_state(endpoint.endpoint_id)
        if not state.healthy or state.lifecycle != LifecycleState.ACTIVE:
            return False
        if state.queue_depth > self.config.max_queue_depth:
            return False
        if state.kv_cache_usage_frac > self.config.max_kv_cache_usage_frac:
            return False
        try:
            self.hardware.validate_endpoint(endpoint)
        except (KeyError, ValueError):
            return False
        if require_current_frequency:
            allowed = self.hardware.gpu_type(endpoint.gpu_type).allowed_frequencies_mhz
            if state.freq_mhz not in allowed:
                return False
        return True

    def _latency_safe(self, endpoint: EndpointSpec) -> bool:
        snapshot = self.telemetry.snapshot(endpoint.endpoint_id)
        if snapshot.sample_count < self.config.min_samples:
            return False
        metric, slo_ms, _ = self._latency_metric(endpoint, snapshot)
        return (
            metric is not None
            and slo_ms is not None
            and metric <= self.config.safety_fraction * slo_ms
        )

    def _latency_metric(
        self,
        endpoint: EndpointSpec,
        snapshot: EndpointTelemetrySnapshot,
    ) -> Tuple[Optional[float], Optional[float], str]:
        if endpoint.role == "prefill":
            return snapshot.ewma_ttft_ms, self.config.ttft_slo_ms, "TTFT"
        if self.config.tbt_slo_ms is not None and snapshot.ewma_tbt_ms is not None:
            return snapshot.ewma_tbt_ms, self.config.tbt_slo_ms, "TBT"
        if self.config.tpot_slo_ms is not None and snapshot.ewma_tpot_ms is not None:
            return snapshot.ewma_tpot_ms, self.config.tpot_slo_ms, "TPOT"
        return None, None, "decode latency"

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
        if not self._hard_safe(prefill, require_current_frequency=False):
            raise NoEligibleRouteError("fallback prefill endpoint fails hard safety checks")
        if not self._hard_safe(decode, require_current_frequency=False):
            raise NoEligibleRouteError("fallback decode endpoint fails hard safety checks")
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

    @staticmethod
    def _hold(endpoint_id: str, current: int, reason: str) -> DVFSRecommendation:
        return DVFSRecommendation(
            endpoint_id=endpoint_id,
            current_freq_mhz=current,
            target_freq_mhz=current,
            action=DVFSAction.HOLD,
            reason=reason,
        )
