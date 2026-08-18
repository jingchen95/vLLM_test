"""Runtime telemetry samples and dependency-free in-memory aggregation."""

from dataclasses import dataclass, field
import math
from typing import Dict, Optional


def _require_finite_nonnegative(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric or None" % name)
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("%s must be finite and non-negative" % name)


def _require_optional_count(name: str, value: Optional[int]) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("%s must be a non-negative integer or None" % name)


def _require_count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)


@dataclass(frozen=True)
class EndpointTelemetrySample:
    """One endpoint observation; metrics may be independently unavailable.

    ``completed_requests`` and token counts describe this sample's observation
    interval. ``interval_s`` is optional, but is required to derive service
    rate without guessing elapsed time.
    """

    endpoint_id: str
    timestamp_s: float
    power_w: Optional[float] = None
    energy_j: Optional[float] = None
    queue_depth: Optional[int] = None
    running_requests: Optional[int] = None
    kv_cache_usage_frac: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    completed_requests: int = 0
    interval_s: Optional[float] = None
    ttft_ms: Optional[float] = None
    tbt_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    window_ttft_p95_ms: Optional[float] = None
    window_ttft_p99_ms: Optional[float] = None
    window_tbt_p95_ms: Optional[float] = None
    window_tbt_p99_ms: Optional[float] = None
    window_ttft_count: int = 0
    window_tbt_count: int = 0
    window_prompt_tokens_per_s: Optional[float] = None
    window_generation_tokens_per_s: Optional[float] = None
    window_completed_requests_per_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be non-empty")
        if self.timestamp_s is None:
            raise ValueError("timestamp_s must be present")
        _require_finite_nonnegative("timestamp_s", self.timestamp_s)
        _require_finite_nonnegative("power_w", self.power_w)
        _require_finite_nonnegative("energy_j", self.energy_j)
        _require_finite_nonnegative("ttft_ms", self.ttft_ms)
        _require_finite_nonnegative("tbt_ms", self.tbt_ms)
        _require_finite_nonnegative("tpot_ms", self.tpot_ms)
        for name, value in (
            ("window_ttft_p95_ms", self.window_ttft_p95_ms),
            ("window_ttft_p99_ms", self.window_ttft_p99_ms),
            ("window_tbt_p95_ms", self.window_tbt_p95_ms),
            ("window_tbt_p99_ms", self.window_tbt_p99_ms),
            ("window_prompt_tokens_per_s", self.window_prompt_tokens_per_s),
            ("window_generation_tokens_per_s", self.window_generation_tokens_per_s),
            ("window_completed_requests_per_s", self.window_completed_requests_per_s),
        ):
            _require_finite_nonnegative(name, value)
        _require_optional_count("queue_depth", self.queue_depth)
        _require_optional_count("running_requests", self.running_requests)
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("completed_requests", self.completed_requests),
            ("window_ttft_count", self.window_ttft_count),
            ("window_tbt_count", self.window_tbt_count),
        ):
            _require_count(name, value)
        if self.interval_s is not None:
            _require_finite_nonnegative("interval_s", self.interval_s)
            if self.interval_s == 0:
                raise ValueError("interval_s must be positive when provided")
        if self.kv_cache_usage_frac is not None:
            _require_finite_nonnegative("kv_cache_usage_frac", self.kv_cache_usage_frac)
            if self.kv_cache_usage_frac > 1:
                raise ValueError("kv_cache_usage_frac must be in [0, 1]")


class EWMA:
    """Exponentially weighted moving average over observed values."""

    def __init__(self, alpha: float) -> None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or not 0 < alpha <= 1
        ):
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        self.alpha = float(alpha)
        self._value: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        return self._value

    def update(self, measurement: Optional[float]) -> Optional[float]:
        if measurement is None:
            return self._value
        if isinstance(measurement, bool) or not isinstance(measurement, (int, float)):
            raise ValueError("EWMA measurement must be numeric or None")
        measurement = float(measurement)
        if not math.isfinite(measurement):
            raise ValueError("EWMA measurement must be finite")
        if self._value is None:
            self._value = measurement
        else:
            self._value = self.alpha * measurement + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


@dataclass(frozen=True)
class EndpointTelemetrySnapshot:
    endpoint_id: str
    ewma_ttft_ms: Optional[float]
    ewma_tbt_ms: Optional[float]
    ewma_tpot_ms: Optional[float]
    ewma_power_w: Optional[float]
    ewma_energy_per_request_j: Optional[float]
    ewma_energy_per_output_token_j: Optional[float]
    ewma_service_rate_rps: Optional[float]
    ewma_queue_depth: Optional[float]
    sample_count: int
    last_timestamp_s: Optional[float]
    last_latency_observation_s: Optional[float]
    last_ttft_observation_s: Optional[float]
    last_tbt_observation_s: Optional[float]
    last_tpot_observation_s: Optional[float]
    last_observation_age_s: Optional[float]
    window_ttft_p95_ms: Optional[float]
    window_ttft_p99_ms: Optional[float]
    window_tbt_p95_ms: Optional[float]
    window_tbt_p99_ms: Optional[float]
    window_ttft_count: int
    window_tbt_count: int
    window_prompt_tokens_per_s: Optional[float]
    window_generation_tokens_per_s: Optional[float]
    window_completed_requests_per_s: Optional[float]


@dataclass
class _EndpointAccumulator:
    alpha: float
    sample_count: int = 0
    last_timestamp_s: Optional[float] = None
    last_latency_observation_s: Optional[float] = None
    last_ttft_observation_s: Optional[float] = None
    last_tbt_observation_s: Optional[float] = None
    last_tpot_observation_s: Optional[float] = None
    window_ttft_p95_ms: Optional[float] = None
    window_ttft_p99_ms: Optional[float] = None
    window_tbt_p95_ms: Optional[float] = None
    window_tbt_p99_ms: Optional[float] = None
    window_ttft_count: int = 0
    window_tbt_count: int = 0
    window_prompt_tokens_per_s: Optional[float] = None
    window_generation_tokens_per_s: Optional[float] = None
    window_completed_requests_per_s: Optional[float] = None
    ttft: EWMA = field(init=False)
    tbt: EWMA = field(init=False)
    tpot: EWMA = field(init=False)
    power: EWMA = field(init=False)
    energy_per_request: EWMA = field(init=False)
    energy_per_output_token: EWMA = field(init=False)
    service_rate: EWMA = field(init=False)
    queue_depth: EWMA = field(init=False)

    def __post_init__(self) -> None:
        self.ttft = EWMA(self.alpha)
        self.tbt = EWMA(self.alpha)
        self.tpot = EWMA(self.alpha)
        self.power = EWMA(self.alpha)
        self.energy_per_request = EWMA(self.alpha)
        self.energy_per_output_token = EWMA(self.alpha)
        self.service_rate = EWMA(self.alpha)
        self.queue_depth = EWMA(self.alpha)


class TelemetryAggregator:
    """Maintains endpoint-local EWMAs; it performs no prediction or I/O."""

    def __init__(self, alpha: float = 0.2) -> None:
        # Reuse EWMA validation without retaining the temporary object.
        EWMA(alpha)
        self.alpha = float(alpha)
        self._endpoints: Dict[str, _EndpointAccumulator] = {}

    def observe(self, sample: EndpointTelemetrySample) -> EndpointTelemetrySnapshot:
        accumulator = self._endpoints.setdefault(
            sample.endpoint_id,
            _EndpointAccumulator(self.alpha),
        )
        if (
            accumulator.last_timestamp_s is not None
            and sample.timestamp_s < accumulator.last_timestamp_s
        ):
            raise ValueError("telemetry timestamps must be non-decreasing per endpoint")

        accumulator.ttft.update(sample.ttft_ms)
        accumulator.tbt.update(sample.tbt_ms)
        accumulator.tpot.update(sample.tpot_ms)
        accumulator.power.update(sample.power_w)
        accumulator.queue_depth.update(sample.queue_depth)
        if any(
            value is not None
            for value in (
                sample.ttft_ms,
                sample.tbt_ms,
                sample.tpot_ms,
                sample.window_ttft_p95_ms,
                sample.window_ttft_p99_ms,
                sample.window_tbt_p95_ms,
                sample.window_tbt_p99_ms,
            )
        ):
            accumulator.last_latency_observation_s = float(sample.timestamp_s)
        if any(
            value is not None
            for value in (
                sample.ttft_ms,
                sample.window_ttft_p95_ms,
                sample.window_ttft_p99_ms,
            )
        ):
            accumulator.last_ttft_observation_s = float(sample.timestamp_s)
        if any(
            value is not None
            for value in (
                sample.tbt_ms,
                sample.window_tbt_p95_ms,
                sample.window_tbt_p99_ms,
            )
        ):
            accumulator.last_tbt_observation_s = float(sample.timestamp_s)
        if sample.tpot_ms is not None:
            accumulator.last_tpot_observation_s = float(sample.timestamp_s)

        # Window values describe the most recent interval, so a missing value
        # clears the prior interval rather than silently reusing it.
        accumulator.window_ttft_p95_ms = sample.window_ttft_p95_ms
        accumulator.window_ttft_p99_ms = sample.window_ttft_p99_ms
        accumulator.window_tbt_p95_ms = sample.window_tbt_p95_ms
        accumulator.window_tbt_p99_ms = sample.window_tbt_p99_ms
        accumulator.window_ttft_count = sample.window_ttft_count
        accumulator.window_tbt_count = sample.window_tbt_count
        accumulator.window_prompt_tokens_per_s = sample.window_prompt_tokens_per_s
        accumulator.window_generation_tokens_per_s = sample.window_generation_tokens_per_s
        accumulator.window_completed_requests_per_s = sample.window_completed_requests_per_s

        energy_per_request = None
        if sample.energy_j is not None and sample.completed_requests > 0:
            energy_per_request = sample.energy_j / sample.completed_requests
        accumulator.energy_per_request.update(energy_per_request)

        energy_per_output_token = None
        if sample.energy_j is not None and sample.output_tokens > 0:
            energy_per_output_token = sample.energy_j / sample.output_tokens
        accumulator.energy_per_output_token.update(energy_per_output_token)

        service_rate = None
        if sample.interval_s is not None:
            service_rate = sample.completed_requests / sample.interval_s
        accumulator.service_rate.update(service_rate)

        accumulator.sample_count += 1
        accumulator.last_timestamp_s = float(sample.timestamp_s)
        return self.snapshot(sample.endpoint_id)

    def snapshot(
        self,
        endpoint_id: str,
        now_s: Optional[float] = None,
    ) -> EndpointTelemetrySnapshot:
        accumulator = self._endpoints.get(endpoint_id)
        if accumulator is None:
            return EndpointTelemetrySnapshot(
                endpoint_id=endpoint_id,
                ewma_ttft_ms=None,
                ewma_tbt_ms=None,
                ewma_tpot_ms=None,
                ewma_power_w=None,
                ewma_energy_per_request_j=None,
                ewma_energy_per_output_token_j=None,
                ewma_service_rate_rps=None,
                ewma_queue_depth=None,
                sample_count=0,
                last_timestamp_s=None,
                last_latency_observation_s=None,
                last_ttft_observation_s=None,
                last_tbt_observation_s=None,
                last_tpot_observation_s=None,
                last_observation_age_s=None,
                window_ttft_p95_ms=None,
                window_ttft_p99_ms=None,
                window_tbt_p95_ms=None,
                window_tbt_p99_ms=None,
                window_ttft_count=0,
                window_tbt_count=0,
                window_prompt_tokens_per_s=None,
                window_generation_tokens_per_s=None,
                window_completed_requests_per_s=None,
            )
        observation_age_s = None
        if now_s is not None and accumulator.last_latency_observation_s is not None:
            observation_age_s = max(0.0, now_s - accumulator.last_latency_observation_s)
        return EndpointTelemetrySnapshot(
            endpoint_id=endpoint_id,
            ewma_ttft_ms=accumulator.ttft.value,
            ewma_tbt_ms=accumulator.tbt.value,
            ewma_tpot_ms=accumulator.tpot.value,
            ewma_power_w=accumulator.power.value,
            ewma_energy_per_request_j=accumulator.energy_per_request.value,
            ewma_energy_per_output_token_j=accumulator.energy_per_output_token.value,
            ewma_service_rate_rps=accumulator.service_rate.value,
            ewma_queue_depth=accumulator.queue_depth.value,
            sample_count=accumulator.sample_count,
            last_timestamp_s=accumulator.last_timestamp_s,
            last_latency_observation_s=accumulator.last_latency_observation_s,
            last_ttft_observation_s=accumulator.last_ttft_observation_s,
            last_tbt_observation_s=accumulator.last_tbt_observation_s,
            last_tpot_observation_s=accumulator.last_tpot_observation_s,
            last_observation_age_s=observation_age_s,
            window_ttft_p95_ms=accumulator.window_ttft_p95_ms,
            window_ttft_p99_ms=accumulator.window_ttft_p99_ms,
            window_tbt_p95_ms=accumulator.window_tbt_p95_ms,
            window_tbt_p99_ms=accumulator.window_tbt_p99_ms,
            window_ttft_count=accumulator.window_ttft_count,
            window_tbt_count=accumulator.window_tbt_count,
            window_prompt_tokens_per_s=accumulator.window_prompt_tokens_per_s,
            window_generation_tokens_per_s=accumulator.window_generation_tokens_per_s,
            window_completed_requests_per_s=accumulator.window_completed_requests_per_s,
        )

    def reset(self, endpoint_id: str) -> None:
        self._endpoints.pop(endpoint_id, None)
