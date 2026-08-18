"""CPU-only tests for Phase 3B prerequisite instrumentation."""

import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest

from xpyd.phase3a_observability import (
    FixedPeriodScrapeOrchestrator,
    Phase3AHarness,
    ScheduledScrapeEvent,
    ScrapeAttempt,
)
from xpyd.types import EndpointSpec
from xpyd.vllm_metrics import VLLMMetricsCollector, VLLMRawScrape


def endpoint(endpoint_id, role):
    return EndpointSpec(
        endpoint_id=endpoint_id,
        role=role,
        gpu_type="fixture",
        node=endpoint_id.lower(),
        gpu_ids=(0,),
        tp_degree=1,
        http_uri="http://%s:8000" % endpoint_id.lower(),
        kv_connector="fixture",
    )


ENDPOINTS = (endpoint("P0", "prefill"), endpoint("D0", "decode"))


def harness_config(output_root):
    return {
        "model": "model-a",
        "vllm_version": "0.15.1",
        "output_root": str(output_root),
        "scrape_interval_s": 0.01,
        "scrape_late_tolerance_s": 0.002,
        "client": {"base_url": "http://proxy:8000"},
        "endpoints": [
            {
                "endpoint_id": item.endpoint_id,
                "role": item.role,
                "gpu_type": item.gpu_type,
                "node": item.node,
                "gpu_ids": list(item.gpu_ids),
                "tp_degree": item.tp_degree,
                "http_uri": item.http_uri,
                "kv_connector": item.kv_connector,
            }
            for item in ENDPOINTS
        ],
    }


class FixedPeriodScrapeTests(unittest.TestCase):
    def test_slow_endpoint_does_not_shift_other_endpoint_schedule(self):
        period_s = 0.01
        for slow_endpoint_id in ("P0", "D0"):
            starts = {"P0": [], "D0": []}
            lock = threading.Lock()

            def scrape(item, scheduled_s):
                wall_start = time.time()
                mono_start = time.monotonic()
                with lock:
                    starts[item.endpoint_id].append(mono_start)
                time.sleep(0.032 if item.endpoint_id == slow_endpoint_id else 0.001)
                return ScrapeAttempt(
                    endpoint=item,
                    scheduled_monotonic_s=scheduled_s,
                    actual_wall_start_s=wall_start,
                    actual_wall_finish_s=time.time(),
                    actual_monotonic_start_s=mono_start,
                    actual_monotonic_finish_s=time.monotonic(),
                    raw=None,
                    error=None,
                )

            finish_s = time.monotonic() + 0.085
            events = FixedPeriodScrapeOrchestrator(
                ENDPOINTS, scrape, period_s
            ).run_while(lambda: time.monotonic() < finish_s)
            fast_endpoint_id = "D0" if slow_endpoint_id == "P0" else "P0"
            fast_events = [
                event
                for event in events
                if event.endpoint.endpoint_id == fast_endpoint_id
                and not event.missed
            ]
            slow_misses = [
                event
                for event in events
                if event.endpoint.endpoint_id == slow_endpoint_id and event.missed
            ]
            fast_misses = [
                event
                for event in events
                if event.endpoint.endpoint_id == fast_endpoint_id and event.missed
            ]

            self.assertGreaterEqual(len(fast_events), 6)
            self.assertGreaterEqual(len(slow_misses), 2)
            self.assertEqual(fast_misses, [])
            scheduled = [event.scheduled_monotonic_s for event in fast_events]
            for earlier, later in zip(scheduled, scheduled[1:]):
                self.assertTrue(math.isclose(later - earlier, period_s, abs_tol=1e-9))
            self.assertLess(
                abs(starts["P0"][0] - starts["D0"][0]),
                0.02,
            )

    def test_round_scrapes_endpoints_concurrently_and_preserves_windows(self):
        barrier = threading.Barrier(2)
        collector = VLLMMetricsCollector(model_name="model-a")
        calls = {"P0": 0, "D0": 0}
        lock = threading.Lock()

        def source(item):
            barrier.wait(timeout=0.5)
            with lock:
                calls[item.endpoint_id] += 1
                value = calls[item.endpoint_id]
            text = (
                'vllm:prompt_tokens_total{model_name="model-a"} %d\n'
                'vllm:generation_tokens_total{model_name="model-a"} %d\n'
                'vllm:request_success_total{model_name="model-a"} %d\n'
            ) % (value, value, value)
            snapshot = collector.parse_text(item, text, time.monotonic())
            return VLLMRawScrape(
                metrics_url=collector.metrics_url(item),
                raw_text=text,
                snapshot=snapshot,
            )

        with tempfile.TemporaryDirectory() as temporary:
            harness = Phase3AHarness(
                harness_config(Path(temporary)),
                run_id="concurrent",
                scrape_source=source,
            )
            harness._create_layout()
            harness._scrape_round("first")
            harness._scrape_round("second")
            records = [
                json.loads(line)
                for line in (
                    harness.run_dir / "derived" / "telemetry.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(records), 4)
            self.assertTrue(all(record["status"] == "success" for record in records))
            self.assertTrue(all("scheduled_monotonic_start_s" in record for record in records))
            self.assertTrue(all("scrape_latency_s" in record for record in records))
            second = [record for record in records if record["label"] == "second"]
            self.assertTrue(all(record["window"]["valid"] for record in second))
            self.assertTrue(all(record["window"]["delta_prompt_tokens"] == 1 for record in second))

    def test_late_and_missed_samples_are_explicitly_serialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            harness = Phase3AHarness(
                harness_config(Path(temporary)), run_id="schedule-records"
            )
            harness._create_layout()
            item = ENDPOINTS[0]
            attempt = ScrapeAttempt(
                endpoint=item,
                scheduled_monotonic_s=10.0,
                actual_wall_start_s=100.0,
                actual_wall_finish_s=100.004,
                actual_monotonic_start_s=10.02,
                actual_monotonic_finish_s=10.024,
                raw=None,
                error=OSError("fixture timeout"),
            )
            harness._record_scrape_attempt(
                attempt,
                "load_interval",
                "probe",
                1,
                tolerate_errors=True,
            )
            harness._record_missed_scrape(
                ScheduledScrapeEvent(
                    endpoint=item,
                    schedule_index=2,
                    scheduled_monotonic_s=10.03,
                    detected_monotonic_s=10.05,
                    attempt=None,
                    missed=True,
                    missed_reason="previous_endpoint_scrape_still_in_flight",
                ),
                "load_interval",
                "probe",
                2,
            )
            records = [
                json.loads(line)
                for line in (
                    harness.run_dir / "derived" / "scrapes.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(records[0]["status"], "error")
            self.assertTrue(records[0]["late"])
            self.assertAlmostEqual(records[0]["scheduling_drift_s"], 0.02)
            self.assertAlmostEqual(records[0]["scrape_latency_s"], 0.004)
            self.assertEqual(records[1]["status"], "missed")
            self.assertTrue(records[1]["missed"])
            self.assertIsNone(records[1]["actual_monotonic_start_s"])
            self.assertEqual(
                records[1]["missed_reason"],
                "previous_endpoint_scrape_still_in_flight",
            )


if __name__ == "__main__":
    unittest.main()
