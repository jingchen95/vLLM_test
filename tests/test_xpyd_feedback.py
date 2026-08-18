"""CPU-only tests for XpYd telemetry and predictor-free feedback."""

import unittest

from xpyd.compatibility import ConnectorCompatibility
from xpyd.feedback_scheduler import (
    DVFSAction,
    FeedbackScheduler,
    FeedbackSchedulerConfig,
    NoEligibleRouteError,
)
from xpyd.mock_runtime import build_example_runtime
from xpyd.telemetry import EWMA, EndpointTelemetrySample, TelemetryAggregator
from xpyd.types import LifecycleState


class EWMATests(unittest.TestCase):
    def test_first_observation_initializes(self):
        ewma = EWMA(0.2)
        self.assertIsNone(ewma.value)
        self.assertEqual(ewma.update(50.0), 50.0)

    def test_update_math(self):
        ewma = EWMA(0.2)
        ewma.update(50.0)
        self.assertAlmostEqual(ewma.update(90.0), 58.0)

    def test_none_does_not_update(self):
        ewma = EWMA(0.2)
        ewma.update(50.0)
        self.assertEqual(ewma.update(None), 50.0)

    def test_invalid_alpha_rejected(self):
        for alpha in (0, -0.1, 1.1):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                EWMA(alpha)


class TelemetryTests(unittest.TestCase):
    def test_invalid_telemetry_rejected(self):
        invalid_samples = (
            lambda: EndpointTelemetrySample("", 0),
            lambda: EndpointTelemetrySample("P0", -1),
            lambda: EndpointTelemetrySample("P0", 0, power_w=-1),
            lambda: EndpointTelemetrySample("P0", 0, energy_j=-1),
            lambda: EndpointTelemetrySample("P0", None),
            lambda: EndpointTelemetrySample("P0", 0, output_tokens=-1),
            lambda: EndpointTelemetrySample("P0", 0, completed_requests=None),
            lambda: EndpointTelemetrySample("P0", 0, ttft_ms=-1),
            lambda: EndpointTelemetrySample("P0", 0, kv_cache_usage_frac=1.1),
        )
        for build in invalid_samples:
            with self.subTest(build=build), self.assertRaises(ValueError):
                build()

    def test_energy_per_request_requires_positive_request_count(self):
        aggregator = TelemetryAggregator(alpha=1.0)
        valid = aggregator.observe(
            EndpointTelemetrySample("P0", 1, energy_j=40, completed_requests=5)
        )
        self.assertEqual(valid.ewma_energy_per_request_j, 8.0)

        aggregator.reset("P0")
        insufficient = aggregator.observe(
            EndpointTelemetrySample("P0", 2, energy_j=40, completed_requests=0)
        )
        self.assertIsNone(insufficient.ewma_energy_per_request_j)

    def test_energy_per_output_token_requires_positive_token_count(self):
        aggregator = TelemetryAggregator(alpha=1.0)
        valid = aggregator.observe(
            EndpointTelemetrySample("D0", 1, energy_j=20, output_tokens=100)
        )
        self.assertEqual(valid.ewma_energy_per_output_token_j, 0.2)

        aggregator.reset("D0")
        insufficient = aggregator.observe(
            EndpointTelemetrySample("D0", 2, energy_j=20, output_tokens=0)
        )
        self.assertIsNone(insufficient.ewma_energy_per_output_token_j)

    def test_power_does_not_invent_energy(self):
        snapshot = TelemetryAggregator().observe(
            EndpointTelemetrySample("P0", 1, power_w=100)
        )
        self.assertEqual(snapshot.ewma_power_w, 100)
        self.assertIsNone(snapshot.ewma_energy_per_request_j)

    def test_service_rate_requires_explicit_interval(self):
        aggregator = TelemetryAggregator(alpha=1.0)
        unknown = aggregator.observe(
            EndpointTelemetrySample("P0", 1, completed_requests=4)
        )
        self.assertIsNone(unknown.ewma_service_rate_rps)
        known = aggregator.observe(
            EndpointTelemetrySample("P0", 2, completed_requests=4, interval_s=2)
        )
        self.assertEqual(known.ewma_service_rate_rps, 2.0)

    def test_latency_power_and_queue_metrics_are_smoothed_independently(self):
        aggregator = TelemetryAggregator(alpha=0.5)
        aggregator.observe(
            EndpointTelemetrySample(
                "D0", 1, power_w=100, queue_depth=2, tbt_ms=40, tpot_ms=50,
            )
        )
        snapshot = aggregator.observe(
            EndpointTelemetrySample(
                "D0", 2, power_w=120, queue_depth=6, tbt_ms=60,
            )
        )
        self.assertEqual(snapshot.ewma_power_w, 110)
        self.assertEqual(snapshot.ewma_queue_depth, 4)
        self.assertEqual(snapshot.ewma_tbt_ms, 50)
        self.assertEqual(snapshot.ewma_tpot_ms, 50)

    def test_unknown_snapshot_and_reset(self):
        aggregator = TelemetryAggregator()
        self.assertEqual(aggregator.snapshot("missing").sample_count, 0)
        aggregator.observe(EndpointTelemetrySample("P0", 1, ttft_ms=10))
        aggregator.reset("P0")
        self.assertIsNone(aggregator.snapshot("P0").ewma_ttft_ms)

    def test_out_of_order_timestamp_rejected(self):
        aggregator = TelemetryAggregator()
        aggregator.observe(EndpointTelemetrySample("P0", 2, ttft_ms=10))
        with self.assertRaises(ValueError):
            aggregator.observe(EndpointTelemetrySample("P0", 1, ttft_ms=20))


def build_scheduler(
    with_samples=True,
    equal_tp_compatible=True,
    fallback=True,
    min_samples=1,
):
    runtime = build_example_runtime()
    if equal_tp_compatible:
        runtime.compatibility.add(
            ConnectorCompatibility("mock-kv", 1, 1, True, "test evidence")
        )
        runtime.compatibility.add(
            ConnectorCompatibility("mock-kv", 2, 2, True, "test evidence")
        )
    telemetry = TelemetryAggregator(alpha=1.0)
    if with_samples:
        for sample in (
            EndpointTelemetrySample(
                "P0", 1, energy_j=2, completed_requests=1, ttft_ms=200,
            ),
            EndpointTelemetrySample(
                "P2", 1, energy_j=3, completed_requests=1, ttft_ms=200,
            ),
            EndpointTelemetrySample(
                "D0", 1, energy_j=5, completed_requests=1, tbt_ms=20,
            ),
            EndpointTelemetrySample(
                "D1", 1, energy_j=6, completed_requests=1, tbt_ms=20,
            ),
            EndpointTelemetrySample(
                "D2", 1, energy_j=7, completed_requests=1, tbt_ms=20,
            ),
        ):
            telemetry.observe(sample)
    fallback_args = {}
    if fallback:
        fallback_args = {
            "fallback_prefill_endpoint_id": "P2",
            "fallback_decode_endpoint_id": "D0",
            "fallback_prefill_freq_mhz": 1200,
            "fallback_decode_freq_mhz": 800,
        }
    scheduler = FeedbackScheduler(
        runtime.registry,
        runtime.compatibility,
        telemetry,
        runtime.hardware,
        FeedbackSchedulerConfig(
            ttft_slo_ms=500,
            tbt_slo_ms=100,
            safety_fraction=0.8,
            min_samples=min_samples,
            **fallback_args,
        ),
        clock=lambda: 2.0,
    )
    return runtime, telemetry, scheduler


def candidate_ids(scheduler):
    return {
        (candidate.prefill_endpoint_id, candidate.decode_endpoint_id)
        for candidate in scheduler.eligible_routes()
    }


class FeedbackRoutingTests(unittest.TestCase):
    def test_unhealthy_endpoint_rejected(self):
        runtime, _, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.healthy = False
        runtime.registry.update_state(current)
        self.assertFalse(any(pair[0] == "P0" for pair in candidate_ids(scheduler)))

    def test_non_active_endpoint_rejected(self):
        runtime, _, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.lifecycle = LifecycleState.WARM
        runtime.registry.update_state(current)
        self.assertFalse(any(pair[0] == "P0" for pair in candidate_ids(scheduler)))

    def test_incompatible_tp_pair_rejected(self):
        _, _, scheduler = build_scheduler(equal_tp_compatible=False)
        pairs = candidate_ids(scheduler)
        self.assertNotIn(("P0", "D0"), pairs)
        self.assertIn(("P0", "D2"), pairs)
        self.assertIn(("P2", "D0"), pairs)

    def test_excessive_kv_usage_rejected(self):
        runtime, _, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.kv_cache_usage_frac = 0.91
        runtime.registry.update_state(current)
        self.assertFalse(any(pair[0] == "P0" for pair in candidate_ids(scheduler)))

    def test_excessive_queue_pressure_rejected(self):
        runtime, _, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.queue_depth = 9
        runtime.registry.update_state(current)
        self.assertFalse(any(pair[0] == "P0" for pair in candidate_ids(scheduler)))

    def test_endpoint_near_slo_rejected_by_margin(self):
        _, telemetry, scheduler = build_scheduler()
        telemetry.observe(
            EndpointTelemetrySample(
                "P0", 2, energy_j=2, completed_requests=1, ttft_ms=401,
            )
        )
        self.assertFalse(any(pair[0] == "P0" for pair in candidate_ids(scheduler)))

    def test_decode_uses_tpot_when_tbt_is_unavailable(self):
        runtime = build_example_runtime()
        telemetry = TelemetryAggregator(alpha=1.0)
        telemetry.observe(EndpointTelemetrySample("P2", 1, ttft_ms=100))
        telemetry.observe(EndpointTelemetrySample("D0", 1, tpot_ms=70))
        scheduler = FeedbackScheduler(
            runtime.registry,
            runtime.compatibility,
            telemetry,
            runtime.hardware,
            FeedbackSchedulerConfig(
                ttft_slo_ms=500,
                tbt_slo_ms=100,
                tpot_slo_ms=100,
            ),
            clock=lambda: 1.0,
        )
        self.assertIn(("P2", "D0"), candidate_ids(scheduler))

    def test_lowest_energy_safe_pair_selected(self):
        _, _, scheduler = build_scheduler()
        decision = scheduler.choose_route()
        self.assertEqual(
            (decision.prefill_endpoint_id, decision.decode_endpoint_id),
            ("P0", "D0"),
        )
        self.assertEqual(scheduler.eligible_routes()[0].energy_per_request_j, 7.0)

    def test_unknown_telemetry_uses_configured_fallback(self):
        _, _, scheduler = build_scheduler(with_samples=False)
        decision = scheduler.choose_route()
        self.assertEqual(
            (
                decision.prefill_endpoint_id,
                decision.decode_endpoint_id,
                decision.prefill_freq_mhz,
                decision.decode_freq_mhz,
            ),
            ("P2", "D0", 1200, 800),
        )

    def test_insufficient_sample_count_uses_fallback(self):
        _, _, scheduler = build_scheduler(min_samples=2)
        decision = scheduler.choose_route()
        self.assertEqual((decision.prefill_endpoint_id, decision.decode_endpoint_id), ("P2", "D0"))

    def test_no_candidate_and_no_fallback_raises(self):
        _, _, scheduler = build_scheduler(with_samples=False, fallback=False)
        with self.assertRaises(NoEligibleRouteError):
            scheduler.choose_route()

    def test_missing_energy_uses_deterministic_ranking(self):
        runtime, telemetry, scheduler = build_scheduler(with_samples=False)
        for sample in (
            EndpointTelemetrySample("P0", 1, ttft_ms=100),
            EndpointTelemetrySample("P2", 1, ttft_ms=100),
            EndpointTelemetrySample("D0", 1, tbt_ms=20),
            EndpointTelemetrySample("D2", 1, tbt_ms=20),
        ):
            telemetry.observe(sample)
        candidates = scheduler.eligible_routes()
        self.assertEqual(candidates[0].ranking_basis, "deterministic_endpoint_ids")
        self.assertEqual(
            (candidates[0].prefill_endpoint_id, candidates[0].decode_endpoint_id),
            ("P0", "D0"),
        )
        self.assertIsNotNone(runtime.registry.get_state("P0").freq_mhz)


class DVFSFeedbackTests(unittest.TestCase):
    def test_large_slack_steps_down_one_level(self):
        _, _, scheduler = build_scheduler()
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.STEP_DOWN)
        self.assertEqual((recommendation.current_freq_mhz, recommendation.target_freq_mhz), (1600, 1200))

    def test_moderate_region_holds(self):
        _, telemetry, scheduler = build_scheduler()
        telemetry.observe(EndpointTelemetrySample("P0", 2, ttft_ms=300))
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.HOLD)
        self.assertEqual(recommendation.target_freq_mhz, 1600)

    def test_slo_pressure_steps_up_one_level(self):
        runtime, telemetry, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.freq_mhz = 1200
        runtime.registry.update_state(current)
        telemetry.observe(EndpointTelemetrySample("P0", 2, ttft_ms=450))
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.STEP_UP)
        self.assertEqual(recommendation.target_freq_mhz, 1600)

    def test_severe_pressure_falls_back_to_maximum(self):
        runtime, _, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.freq_mhz = 800
        current.queue_depth = 16
        runtime.registry.update_state(current)
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.FALLBACK_MAX)
        self.assertEqual(recommendation.target_freq_mhz, 1600)

    def test_recommended_target_is_always_supported(self):
        runtime, _, scheduler = build_scheduler()
        allowed = runtime.hardware.gpu_type("example_accelerator").allowed_frequencies_mhz
        for endpoint_id in ("P0", "P2", "D0", "D1", "D2"):
            with self.subTest(endpoint_id=endpoint_id):
                target = scheduler.choose_frequency_adjustment(endpoint_id).target_freq_mhz
                self.assertIn(target, allowed)

    def test_minimum_boundary_holds(self):
        runtime, _, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.freq_mhz = 800
        runtime.registry.update_state(current)
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.HOLD)
        self.assertEqual(recommendation.target_freq_mhz, 800)

    def test_maximum_boundary_holds_under_pressure(self):
        _, telemetry, scheduler = build_scheduler()
        telemetry.observe(EndpointTelemetrySample("P0", 2, ttft_ms=450))
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.HOLD)
        self.assertEqual(recommendation.target_freq_mhz, 1600)

    def test_insufficient_telemetry_holds(self):
        _, _, scheduler = build_scheduler(with_samples=False)
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.HOLD)
        self.assertEqual(recommendation.target_freq_mhz, 1600)

    def test_unknown_current_frequency_requests_known_maximum(self):
        runtime, _, scheduler = build_scheduler()
        current = runtime.registry.get_state("P0")
        current.freq_mhz = 1000
        runtime.registry.update_state(current)
        recommendation = scheduler.choose_frequency_adjustment("P0")
        self.assertEqual(recommendation.action, DVFSAction.FALLBACK_MAX)
        self.assertEqual(recommendation.target_freq_mhz, 1600)


if __name__ == "__main__":
    unittest.main()
