"""Deterministic CPU-only example for the predictor-free feedback scheduler."""

from typing import Tuple

from xpyd.compatibility import ConnectorCompatibility
from xpyd.feedback_scheduler import (
    DVFSRecommendation,
    FeedbackScheduler,
    FeedbackSchedulerConfig,
    RouteCandidate,
)
from xpyd.mock_runtime import build_example_runtime
from xpyd.telemetry import EndpointTelemetrySample, TelemetryAggregator
from xpyd.types import RoutingDecision


def build_feedback_example() -> FeedbackScheduler:
    runtime = build_example_runtime()
    # The prior mock already contains cross-TP rows. Add the two equal-TP rows
    # explicitly so every demonstrated pair still has compatibility evidence.
    runtime.compatibility.add(
        ConnectorCompatibility("mock-kv", 1, 1, True, "CPU feedback example")
    )
    runtime.compatibility.add(
        ConnectorCompatibility("mock-kv", 2, 2, True, "CPU feedback example")
    )

    telemetry = TelemetryAggregator(alpha=1.0)
    observations = (
        EndpointTelemetrySample(
            "P0", 1.0, energy_j=2.0, completed_requests=1, ttft_ms=300.0,
        ),
        EndpointTelemetrySample(
            "P2", 1.0, energy_j=2.5, completed_requests=1, ttft_ms=250.0,
        ),
        EndpointTelemetrySample(
            "D0", 1.0, energy_j=5.0, output_tokens=100,
            completed_requests=1, tbt_ms=55.0,
        ),
        EndpointTelemetrySample(
            "D1", 1.0, energy_j=4.0, output_tokens=100,
            completed_requests=1, tbt_ms=92.0,
        ),
        EndpointTelemetrySample(
            "D2", 1.0, energy_j=5.5, output_tokens=100,
            completed_requests=1, tbt_ms=60.0,
        ),
    )
    for observation in observations:
        telemetry.observe(observation)

    return FeedbackScheduler(
        registry=runtime.registry,
        compatibility=runtime.compatibility,
        telemetry=telemetry,
        hardware=runtime.hardware,
        config=FeedbackSchedulerConfig(
            ttft_slo_ms=500.0,
            tbt_slo_ms=100.0,
            safety_fraction=0.8,
            fallback_prefill_endpoint_id="P2",
            fallback_decode_endpoint_id="D0",
            fallback_prefill_freq_mhz=1600,
            fallback_decode_freq_mhz=1600,
        ),
        clock=lambda: 1.0,
    )


def example_feedback_decisions() -> Tuple[
    Tuple[RouteCandidate, ...],
    RoutingDecision,
    DVFSRecommendation,
]:
    scheduler = build_feedback_example()
    return (
        scheduler.eligible_routes(),
        scheduler.choose_route(),
        scheduler.choose_frequency_adjustment("P2"),
    )


if __name__ == "__main__":
    candidates, route, dvfs = example_feedback_decisions()
    print("eligible candidates:")
    for candidate in candidates:
        print(" ", candidate)
    print("D1 filtered: EWMA TBT 92 ms > 0.8 * 100 ms")
    print("selected:", route)
    print("DVFS recommendation:", dvfs)
