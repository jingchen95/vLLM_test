"""CPU-only scheduler refinements and end-to-end dry-run replay tests."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from xpyd.dry_run_controller import build_controller_from_config, run_fixture_replay
from xpyd.feedback_scheduler import DVFSAction
from xpyd.telemetry import EndpointTelemetrySample


FIXTURES = Path(__file__).parent / "fixtures" / "vllm_metrics"
CONFIG_PATH = FIXTURES / "dry_run_config.json"
MANIFEST_PATH = FIXTURES / "dry_run_manifest.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build_controller(config=None):
    return build_controller_from_config(config or load_config(), clock=lambda: 100.0)


def observe_all(controller, timestamp_s=100.0, prefill_p99=200.0,
                decode_p99=100.0, tail_count=20):
    for endpoint in controller.registry.list_endpoints():
        if endpoint.role == "prefill":
            sample = EndpointTelemetrySample(
                endpoint.endpoint_id,
                timestamp_s,
                ttft_ms=prefill_p99,
                window_ttft_p95_ms=prefill_p99,
                window_ttft_p99_ms=prefill_p99,
                window_ttft_count=tail_count,
            )
        else:
            sample = EndpointTelemetrySample(
                endpoint.endpoint_id,
                timestamp_s,
                tbt_ms=decode_p99,
                window_tbt_p95_ms=decode_p99,
                window_tbt_p99_ms=decode_p99,
                window_tbt_count=tail_count,
            )
        controller.telemetry.observe(sample)


def evaluation(controller, prefill_id, decode_id, now_s=100.0):
    return next(
        item for item in controller.scheduler.evaluate_routes(now_s=now_s)
        if item.prefill_endpoint_id == prefill_id
        and item.decode_endpoint_id == decode_id
    )


class TailSafetyTests(unittest.TestCase):
    def test_stale_telemetry_is_rejected(self):
        controller = build_controller()
        observe_all(controller, timestamp_s=1.0)
        item = evaluation(controller, "P0", "D0", now_s=100.0)
        self.assertFalse(item.eligible)
        self.assertEqual(item.reason, "P0_telemetry_stale")

    def test_insufficient_tail_samples_are_rejected(self):
        controller = build_controller()
        observe_all(controller, tail_count=4)
        item = evaluation(controller, "P0", "D0")
        self.assertFalse(item.eligible)
        self.assertEqual(item.reason, "P0_insufficient_tail_samples")

    def test_tail_safe_endpoint_is_accepted(self):
        controller = build_controller()
        observe_all(controller)
        self.assertTrue(evaluation(controller, "P0", "D0").eligible)

    def test_p99_pressure_is_rejected(self):
        controller = build_controller()
        observe_all(controller, prefill_p99=450.0)
        item = evaluation(controller, "P0", "D0")
        self.assertFalse(item.eligible)
        self.assertEqual(item.reason, "P0_latency_margin_exceeded")

    def test_ewma_policy_remains_supported(self):
        config = load_config()
        config["scheduler"]["prefill_safety_metric"] = "ewma"
        config["scheduler"]["decode_safety_metric"] = "ewma"
        controller = build_controller(config)
        observe_all(controller, tail_count=0)
        self.assertTrue(evaluation(controller, "P0", "D0").eligible)

    def test_fresh_other_latency_does_not_refresh_selected_metric(self):
        config = load_config()
        config["scheduler"]["prefill_safety_metric"] = "ewma"
        config["scheduler"]["decode_safety_metric"] = "ewma"
        controller = build_controller(config)
        observe_all(controller, timestamp_s=1.0)
        controller.telemetry.observe(
            EndpointTelemetrySample("P0", 100.0, tbt_ms=10.0)
        )
        item = evaluation(controller, "P0", "D0", now_s=100.0)
        self.assertFalse(item.eligible)
        self.assertEqual(item.reason, "P0_telemetry_stale")


class DVFSStabilityTests(unittest.TestCase):
    def _ewma_controller(self, ttft_ms, freq_mhz=1200, queue_depth=0):
        config = load_config()
        config["scheduler"]["prefill_safety_metric"] = "ewma"
        controller = build_controller(config)
        state = controller.registry.get_state("P0")
        state.freq_mhz = freq_mhz
        state.queue_depth = queue_depth
        controller.registry.update_state(state)
        controller.telemetry.observe(
            EndpointTelemetrySample("P0", 100.0, ttft_ms=ttft_ms)
        )
        return controller

    def test_downshift_threshold(self):
        controller = self._ewma_controller(200.0, freq_mhz=1600)
        result = controller.scheduler.choose_frequency_adjustment("P0", now_s=100.0)
        self.assertEqual(result.action, DVFSAction.STEP_DOWN)
        self.assertEqual(result.target_freq_mhz, 1200)

    def test_hysteresis_middle_region_holds(self):
        controller = self._ewma_controller(350.0)
        result = controller.scheduler.choose_frequency_adjustment("P0", now_s=100.0)
        self.assertEqual(result.action, DVFSAction.HOLD)
        self.assertEqual(result.target_freq_mhz, 1200)

    def test_upshift_threshold(self):
        controller = self._ewma_controller(450.0)
        result = controller.scheduler.choose_frequency_adjustment("P0", now_s=100.0)
        self.assertEqual(result.action, DVFSAction.STEP_UP)
        self.assertEqual(result.target_freq_mhz, 1600)

    def test_cooldown_suppresses_non_emergency_action(self):
        controller = self._ewma_controller(200.0, freq_mhz=1600)
        first = controller.scheduler.choose_frequency_adjustment("P0", now_s=100.0)
        second = controller.scheduler.choose_frequency_adjustment("P0", now_s=105.0)
        self.assertEqual(first.action, DVFSAction.STEP_DOWN)
        self.assertEqual(second.action, DVFSAction.HOLD)
        self.assertIn("dwell", second.reason)

    def test_emergency_bypasses_cooldown(self):
        controller = self._ewma_controller(200.0, freq_mhz=1200)
        controller.scheduler.choose_frequency_adjustment("P0", now_s=100.0)
        state = controller.registry.get_state("P0")
        state.freq_mhz = 800
        state.queue_depth = 16
        controller.registry.update_state(state)
        result = controller.scheduler.choose_frequency_adjustment("P0", now_s=105.0)
        self.assertEqual(result.action, DVFSAction.FALLBACK_MAX)
        self.assertEqual(result.target_freq_mhz, 1600)

    def test_recommendation_does_not_mutate_observed_frequency(self):
        controller = self._ewma_controller(200.0, freq_mhz=1600)
        result = controller.scheduler.choose_frequency_adjustment("P0", now_s=100.0)
        self.assertEqual(result.action, DVFSAction.STEP_DOWN)
        self.assertEqual(controller.registry.get_state("P0").freq_mhz, 1600)


class DryRunIntegrationTests(unittest.TestCase):
    def test_fixture_replay_is_multi_endpoint_and_non_actuating(self):
        controller = build_controller()
        with patch.object(controller.collector, "scrape", side_effect=AssertionError):
            records = run_fixture_replay(controller, MANIFEST_PATH)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[-1].endpoints), 6)
        self.assertEqual(
            {item.http_uri for item in records[-1].endpoints},
            {
                "http://nodeA:8001", "http://nodeA:8002", "http://nodeA:8003",
                "http://nodeB:8101", "http://nodeB:8102", "http://nodeB:8103",
            },
        )
        self.assertTrue(records[0].fallback_used)
        self.assertEqual(
            (
                records[-1].selected_route.prefill_endpoint_id,
                records[-1].selected_route.decode_endpoint_id,
            ),
            ("P0", "D0"),
        )
        reasons = {item.reason for item in records[-1].pair_evaluations}
        self.assertIn("P1_queue_pressure", reasons)
        self.assertIn("D1_latency_margin_exceeded", reasons)
        self.assertFalse(records[-1].actuated)
        self.assertTrue(all(
            controller.registry.get_state(endpoint.endpoint_id).freq_mhz == 1600
            for endpoint in controller.registry.list_endpoints()
        ))

    def test_jsonl_is_compact_and_machine_readable(self):
        controller = build_controller()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry-run.jsonl"
            records = run_fixture_replay(controller, MANIFEST_PATH, output)
            lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        decoded = json.loads(lines[-1])
        self.assertFalse(decoded["actuated"])
        self.assertEqual(decoded["selected_route"]["prefill_endpoint_id"], "P0")
        self.assertNotIn("buckets", lines[-1])
        self.assertEqual(len(records[-1].pair_evaluations), 9)

    def test_scrape_failure_marks_unhealthy_and_preserves_last_good_state(self):
        controller = build_controller()
        run_fixture_replay(controller, MANIFEST_PATH)
        before = controller.registry.get_state("P0")
        preserved = (before.queue_depth, before.running_requests,
                     before.kv_cache_usage_frac, before.last_update_s)

        def provider(endpoint):
            if endpoint.endpoint_id == "P0":
                raise ConnectionError("fixture outage")
            text = (FIXTURES / "normal_t1.prom").read_text(encoding="utf-8")
            return controller.collector.parse_text(endpoint, text, 120.0)

        record = controller.run_once(snapshot_provider=provider, now_s=120.0)
        after = controller.registry.get_state("P0")
        self.assertFalse(after.healthy)
        self.assertEqual(
            (after.queue_depth, after.running_requests,
             after.kv_cache_usage_frac, after.last_update_s),
            preserved,
        )
        endpoint_record = next(item for item in record.endpoints if item.endpoint_id == "P0")
        self.assertFalse(endpoint_record.scrape_ok)
        self.assertIn("fixture outage", endpoint_record.scrape_error)
        self.assertIsNotNone(controller.registry.get_spec("P0"))

    def test_missing_gauges_do_not_overwrite_known_values(self):
        controller = build_controller()
        state = controller.registry.get_state("P0")
        state.queue_depth = 3
        state.kv_cache_usage_frac = 0.5
        controller.registry.update_state(state)
        missing = (FIXTURES / "missing_optional.prom").read_text(encoding="utf-8")
        normal = (FIXTURES / "normal_t0.prom").read_text(encoding="utf-8")

        def provider(endpoint):
            text = missing if endpoint.endpoint_id == "P0" else normal
            return controller.collector.parse_text(endpoint, text, 100.0)

        controller.run_once(snapshot_provider=provider, now_s=100.0)
        after = controller.registry.get_state("P0")
        self.assertEqual(after.running_requests, 0)
        self.assertEqual(after.queue_depth, 3)
        self.assertEqual(after.kv_cache_usage_frac, 0.5)
        self.assertTrue(after.healthy)


if __name__ == "__main__":
    unittest.main()
