"""CPU-only fixture tests for the Phase 3A observability harness."""

import json
from pathlib import Path
import tempfile
import unittest

from xpyd.phase3a_observability import Phase3AHarness, Phase3AError
from xpyd.types import EndpointSpec
from xpyd.vllm_metrics import VLLMMetricsCollector, VLLMRawScrape


def prom(prompt, generation, requests, *, running=0, waiting=0, kv=0.2):
    return """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="model-a"} %(running)s
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="model-a"} %(waiting)s
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="model-a"} %(kv)s
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="model-a"} %(prompt)s
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="model-a"} %(generation)s
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="model-a",finished_reason="stop"} %(requests)s
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="model-a",le="0.3"} %(requests)s
vllm:time_to_first_token_seconds_bucket{model_name="model-a",le="0.5"} %(requests)s
vllm:time_to_first_token_seconds_bucket{model_name="model-a",le="1.0"} %(requests)s
vllm:time_to_first_token_seconds_bucket{model_name="model-a",le="+Inf"} %(requests)s
vllm:time_to_first_token_seconds_count{model_name="model-a"} %(requests)s
vllm:time_to_first_token_seconds_sum{model_name="model-a"} %(ttft_sum)s
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{model_name="model-a",le="0.1"} %(generation)s
vllm:inter_token_latency_seconds_bucket{model_name="model-a",le="0.2"} %(generation)s
vllm:inter_token_latency_seconds_bucket{model_name="model-a",le="+Inf"} %(generation)s
vllm:inter_token_latency_seconds_count{model_name="model-a"} %(generation)s
vllm:inter_token_latency_seconds_sum{model_name="model-a"} %(tbt_sum)s
""" % {
        "running": running,
        "waiting": waiting,
        "kv": kv,
        "prompt": prompt,
        "generation": generation,
        "requests": requests,
        "ttft_sum": requests * 0.2,
        "tbt_sum": generation * 0.05,
    }


class FixtureHarness(Phase3AHarness):
    def _run_client(self, probe, trace_path):
        summary = {
            "requests_total": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "input_tokens_total": int(probe["input_len"]),
            "requested_output_tokens_total": int(probe["output_len"]),
            "output_tokens_total": int(probe["output_len"]),
            "mean_ttft_ms": 250.0,
            "p99_ttft_ms": 250.0,
            "mean_tpot_ms": 120.0,
            "p99_tpot_ms": 120.0,
            "completion_token_sources": {"server_usage": 1},
            "decode_stream_available_requests": 1,
            "client_ttft_valid_requests": 1,
            "client_tpot_valid_requests": 1,
            "client_itl_valid_requests": 1,
            "logical_request_id_propagated_requests": 1,
        }
        (trace_path.parent / "summary.json").write_text(
            json.dumps(summary) + "\n", encoding="utf-8"
        )
        (trace_path.parent / "requests.jsonl").write_text(
            json.dumps({
                "ok": True,
                "trace_request_id": "fixture-0",
                "logical_request_id": "fixture-0",
                "logical_request_id_propagated": True,
            }) + "\n",
            encoding="utf-8",
        )
        return summary


class FakeProcess:
    def __init__(self, active_polls=1):
        self.poll_count = 0
        self.active_polls = active_polls

    def poll(self):
        self.poll_count += 1
        return None if self.poll_count <= self.active_polls else 0

    def wait(self):
        return 0


class LoadFixtureHarness(FixtureHarness):
    active_polls = 1

    def _start_client(self, probe, trace_path):
        self._run_client(probe, trace_path)
        return FakeProcess(self.active_polls), (trace_path.parent / "fake-stream.log").open("w")


class TrackingFixtureHarness(FixtureHarness):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def _run_client(self, probe, trace_path):
        self.events.append(("client", probe["id"]))
        return super()._run_client(probe, trace_path)

    def _scrape_round(self, label, probe_id=None):
        self.events.append(("scrape", label))
        return super()._scrape_round(label, probe_id)


class ScrapeSequence:
    def __init__(self, frames):
        self.frames = {key: iter(value) for key, value in frames.items()}
        self.collector = VLLMMetricsCollector(model_name="model-a")
        self.timestamp = 0.0

    def __call__(self, endpoint: EndpointSpec):
        self.timestamp += 1.0
        text = next(self.frames[endpoint.endpoint_id])
        if isinstance(text, BaseException):
            raise text
        return VLLMRawScrape(
            metrics_url=self.collector.metrics_url(endpoint),
            raw_text=text,
            snapshot=self.collector.parse_text(endpoint, text, self.timestamp),
        )


def config(output_root):
    return {
        "model": "model-a",
        "tokenizer_model": "model-a",
        "vllm_version": "0.15.1",
        "output_root": str(output_root),
        "scrape_interval_s": 0.01,
        "scrape_late_tolerance_s": 0.002,
        "collector": {"model_name": "model-a"},
        "client": {
            "base_url": "http://fixture-proxy:8000",
            "require_logical_request_id_propagation": True,
        },
        "endpoints": [
            {
                "endpoint_id": "P0", "role": "prefill", "gpu_type": "L40S",
                "node": "p", "gpu_ids": [0], "tp_degree": 1,
                "http_uri": "http://p:8100", "kv_connector": "fixture",
            },
            {
                "endpoint_id": "D0", "role": "decode", "gpu_type": "L4",
                "node": "d", "gpu_ids": [0], "tp_degree": 1,
                "http_uri": "http://d:8200", "kv_connector": "fixture",
            },
        ],
        "semantic_probes": [
            {"id": "one", "input_len": 128, "output_len": 128, "count": 1}
        ],
    }


class Phase3AFixtureTests(unittest.TestCase):
    def test_required_server_usage_rejects_retokenized_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            strict_config = config(Path(temp))
            strict_config["client"]["require_server_token_usage"] = True
            harness = FixtureHarness(strict_config, run_id="strict")
            client_dir = Path(temp) / "client"
            client_dir.mkdir()
            (client_dir / "summary.json").write_text(json.dumps({
                "successful_requests": 1,
                "failed_requests": 0,
                "completion_token_sources": {"retokenized_text_fallback": 1},
            }), encoding="utf-8")

            with self.assertRaisesRegex(Phase3AError, "exact server completion-token usage"):
                harness._validate_client_result(client_dir, 0)

    def test_required_real_streaming_rejects_invalid_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            strict_config = config(Path(temp))
            strict_config["client"]["require_real_decode_streaming"] = True
            harness = FixtureHarness(strict_config, run_id="strict-stream")
            client_dir = Path(temp) / "client"
            client_dir.mkdir()
            (client_dir / "summary.json").write_text(json.dumps({
                "successful_requests": 1,
                "failed_requests": 0,
                "decode_stream_available_requests": 0,
                "client_ttft_valid_requests": 0,
                "client_tpot_valid_requests": 0,
                "client_itl_valid_requests": 0,
            }), encoding="utf-8")

            with self.assertRaisesRegex(Phase3AError, "real decode streaming required"):
                harness._validate_client_result(client_dir, 0)

    def test_phase_warmup_precedes_preflight_and_is_excluded(self):
        baseline = prom(1000, 500, 20)
        source = ScrapeSequence({
            "P0": [baseline, baseline, prom(1128, 501, 21)],
            "D0": [baseline, baseline, prom(1001, 628, 21)],
        })
        with tempfile.TemporaryDirectory() as temp:
            warm_config = config(Path(temp))
            warm_config["phase_warmup"] = {
                "enabled": True,
                "input_len": 128,
                "output_len": 128,
                "count": 1,
                "max_concurrency": 1,
            }
            harness = TrackingFixtureHarness(
                warm_config, run_id="warmup", scrape_source=source,
            )
            run_dir = harness.run(("semantic",))

            self.assertEqual(harness.events[0], ("client", "_phase_warmup"))
            self.assertEqual(harness.events[1], ("scrape", "preflight"))
            warmup = json.loads(
                (run_dir / "derived/phase_warmup.json").read_text(encoding="utf-8")
            )
            self.assertTrue(warmup["excluded_from_measurement_windows"])
            self.assertEqual(warmup["client"]["output_tokens_total"], 128)
            deltas = json.loads(
                (run_dir / "derived/semantic_deltas.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(deltas), 1)

    def test_before_after_pd_deltas_buckets_summary_and_layout(self):
        baseline = prom(1000, 500, 20)
        source = ScrapeSequence({
            "P0": [baseline, baseline, prom(1128, 501, 21)],
            "D0": [baseline, baseline, prom(1001, 628, 21, running=1, waiting=2, kv=0.4)],
        })
        with tempfile.TemporaryDirectory() as temp:
            harness = FixtureHarness(
                config(Path(temp)), run_id="fixture", scrape_source=source,
                wall_clock=lambda: 10.0, monotonic_clock=lambda: 5.0,
            )
            run_dir = harness.run(("semantic",))
            self.assertTrue((run_dir / "metadata.json").is_file())
            self.assertEqual(len(list((run_dir / "P0/raw_metrics").glob("*.prom"))), 3)
            self.assertEqual(len(list((run_dir / "D0/raw_metrics").glob("*.prom"))), 3)
            deltas = json.loads((run_dir / "derived/semantic_deltas.json").read_text())
            p_window = deltas[0]["endpoints"]["P0"]["window"]
            d_window = deltas[0]["endpoints"]["D0"]["window"]
            self.assertEqual(p_window["delta_prompt_tokens"], 128)
            self.assertEqual(p_window["delta_generation_tokens"], 1)
            self.assertEqual(d_window["delta_prompt_tokens"], 1)
            self.assertEqual(d_window["delta_generation_tokens"], 128)
            p_ttft_delta = deltas[0]["endpoints"]["P0"]["histogram_deltas"]["vllm:time_to_first_token_seconds"]
            self.assertEqual(p_ttft_delta["count"], 1)
            summary = json.loads((run_dir / "derived/summary.json").read_text())
            ttft_bounds = summary["latest_endpoint_observations"]["P0"]["bucket_boundaries_seconds"]["vllm:time_to_first_token_seconds"]
            self.assertIn(0.3, ttft_bounds)
            self.assertIn(0.5, ttft_bounds)
            self.assertTrue((run_dir / "derived/summary.md").is_file())
            semantic = json.loads((run_dir / "derived/semantic_summary.json").read_text())
            self.assertEqual(
                semantic["probes"][0]["endpoints"]["P0"]["generation_tokens"]["assessment"],
                "differs",
            )
            self.assertIn(
                semantic["probes"][0]["endpoints"]["D0"]["client_ttft_vs_endpoint_mean"]["assessment"],
                ("approximately_matches", "differs"),
            )
            telemetry = [json.loads(line) for line in (run_dir / "derived/telemetry.jsonl").read_text().splitlines()]
            self.assertTrue(all("central_monotonic_start_s" in row for row in telemetry))
            self.assertTrue(all(row["raw_metrics_path"].endswith(".prom") for row in telemetry))

    def test_missing_metrics_are_recorded_not_filled(self):
        minimal = """vllm:prompt_tokens_total{model_name="model-a"} 1
vllm:generation_tokens_total{model_name="model-a"} 1
vllm:request_success_total{model_name="model-a"} 1
"""
        source = ScrapeSequence({"P0": [minimal, minimal, minimal], "D0": [minimal, minimal, minimal]})
        with tempfile.TemporaryDirectory() as temp:
            run_dir = FixtureHarness(config(Path(temp)), run_id="missing", scrape_source=source).run(("semantic",))
            summary = json.loads((run_dir / "derived/summary.json").read_text())
            self.assertIn(
                "vllm:num_requests_waiting",
                summary["latest_endpoint_observations"]["P0"]["missing_metrics"],
            )
            self.assertIsNone(summary["latest_endpoint_observations"]["P0"]["waiting"])

    def test_counter_reset_is_marked(self):
        high = prom(1000, 500, 20)
        reset = prom(10, 5, 1)
        source = ScrapeSequence({"P0": [high, high, reset], "D0": [high, high, reset]})
        with tempfile.TemporaryDirectory() as temp:
            run_dir = FixtureHarness(config(Path(temp)), run_id="reset", scrape_source=source).run(("semantic",))
            summary = json.loads((run_dir / "derived/summary.json").read_text())
            self.assertEqual(len(summary["reset_or_discontinuity_observations"]), 2)
            delta = json.loads((run_dir / "derived/semantic_deltas.json").read_text())
            self.assertEqual(
                delta[0]["endpoints"]["P0"]["window"]["reason"],
                "counter_or_histogram_reset",
            )

    def test_unreachable_endpoint_fails_loudly_and_records_failure(self):
        class Broken:
            def __call__(self, endpoint):
                raise OSError("fixture offline")

        with tempfile.TemporaryDirectory() as temp:
            harness = FixtureHarness(config(Path(temp)), run_id="broken", scrape_source=Broken())
            with self.assertRaisesRegex(Phase3AError, "metrics scrape failed"):
                harness.run(("semantic",))
            failure = json.loads((Path(temp) / "broken/derived/failure.json").read_text())
            self.assertFalse(failure["completed"])
            self.assertFalse(failure["actuated"])

    def test_short_load_scrapes_while_fixture_client_is_active(self):
        baseline = prom(1000, 500, 20)
        interval = prom(1100, 550, 22, running=2, waiting=1, kv=0.5)
        final = prom(1200, 600, 24)
        source = ScrapeSequence({
            "P0": [baseline, baseline, interval, final],
            "D0": [baseline, baseline, interval, final],
        })
        with tempfile.TemporaryDirectory() as temp:
            load_config = config(Path(temp))
            load_config["load_probes"] = [{
                "id": "load", "input_len": 128, "output_len": 128,
                "rate_rps": 1.0, "duration_s": 1.0, "max_concurrency": 1,
            }]
            harness = LoadFixtureHarness(
                load_config, run_id="load", scrape_source=source, sleeper=lambda _: None,
            )
            run_dir = harness.run(("load",))
            runs = json.loads((run_dir / "derived/load_runs.json").read_text())
            self.assertEqual(runs[0]["probe"]["count"], 1)
            summary = json.loads((run_dir / "derived/summary.json").read_text())
            self.assertEqual(summary["load_probe_count"], 1)
            self.assertEqual(
                summary["endpoint_window_behavior"]["P0"]["max_waiting_requests"], 1
            )
            self.assertTrue(runs[0]["endpoints"]["P0"]["window"]["valid"])
            self.assertTrue(runs[0]["endpoints"]["D0"]["window"]["valid"])
            monitoring = runs[0]["monitoring"]
            self.assertEqual(
                monitoring["sampling_architecture"],
                "independent_fixed_period_endpoint_workers",
            )
            self.assertGreaterEqual(
                monitoring["scheduled_interval_scrapes"]["P0"], 1
            )
            self.assertGreaterEqual(
                monitoring["scheduled_interval_scrapes"]["D0"], 1
            )
            self.assertIn(
                "max_scheduling_drift_s",
                summary["endpoint_window_behavior"]["P0"],
            )

    def test_transient_load_interval_scrape_error_is_recorded_and_tolerated(self):
        baseline = prom(1000, 500, 20)
        interval = prom(1100, 550, 22, running=2, waiting=1, kv=0.5)
        final = prom(1200, 600, 24)
        source = ScrapeSequence({
            "P0": [baseline, baseline, interval, interval, final],
            "D0": [baseline, baseline, OSError("busy"), interval, final],
        })
        with tempfile.TemporaryDirectory() as temp:
            load_config = config(Path(temp))
            # Leave enough wall-clock separation that a briefly descheduled
            # test runner cannot create an unintended third interval scrape.
            load_config["scrape_interval_s"] = 0.1
            load_config["scrape_late_tolerance_s"] = 0.02
            load_config["load_probes"] = [{
                "id": "load", "input_len": 128, "output_len": 128,
                "rate_rps": 1.0, "duration_s": 1.0, "max_concurrency": 1,
            }]
            harness = LoadFixtureHarness(
                load_config, run_id="transient", scrape_source=source,
                sleeper=lambda _: None,
            )
            harness.active_polls = 3
            run_dir = harness.run(("load",))

            errors = [
                json.loads(line) for line in
                (run_dir / "derived/scrape_errors.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(errors), 1)
            self.assertTrue(errors[0]["tolerated"])
            monitoring = json.loads(
                (run_dir / "derived/load_monitoring.json").read_text()
            )[0]
            self.assertEqual(monitoring["failed_interval_scrapes"]["D0"], 1)
            self.assertEqual(monitoring["successful_interval_scrapes"]["D0"], 1)
            summary = json.loads((run_dir / "derived/summary.json").read_text())
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["scrape_error_count"], 1)

    def test_sustained_load_interval_scrape_outage_fails_after_client_finishes(self):
        baseline = prom(1000, 500, 20)
        interval = prom(1100, 550, 22)
        final = prom(1200, 600, 24)
        source = ScrapeSequence({
            "P0": [baseline, baseline, interval, interval, final],
            "D0": [baseline, baseline, OSError("busy-1"), OSError("busy-2"), final],
        })
        with tempfile.TemporaryDirectory() as temp:
            load_config = config(Path(temp))
            load_config["scrape_interval_s"] = 0.1
            load_config["scrape_late_tolerance_s"] = 0.02
            load_config["load_monitoring"] = {
                "minimum_interval_scrapes_per_endpoint": 1,
                "maximum_consecutive_failures_per_endpoint": 1,
            }
            load_config["load_probes"] = [{
                "id": "load", "input_len": 128, "output_len": 128,
                "rate_rps": 1.0, "duration_s": 1.0, "max_concurrency": 1,
            }]
            harness = LoadFixtureHarness(
                load_config, run_id="outage", scrape_source=source,
                sleeper=lambda _: None,
            )
            harness.active_polls = 3
            with self.assertRaisesRegex(Phase3AError, "load monitoring coverage invalid"):
                harness.run(("load",))

            run_dir = Path(temp) / "outage"
            self.assertTrue((run_dir / "client/load/summary.json").is_file())
            self.assertEqual(
                len(list((run_dir / "D0/raw_metrics").glob("*load_after.prom"))), 1
            )
            monitoring = json.loads(
                (run_dir / "derived/load_monitoring.json").read_text()
            )[0]
            self.assertEqual(monitoring["maximum_consecutive_failures"]["D0"], 2)
            self.assertTrue((run_dir / "derived/failure.json").is_file())

    def test_phase3b_auxiliary_scrape_outage_is_reported_not_failed(self):
        baseline = prom(1000, 500, 20)
        interval = prom(1100, 550, 22)
        final = prom(1200, 600, 24)
        source = ScrapeSequence({
            "P0": [baseline, baseline, interval, interval, final],
            "D0": [baseline, baseline, OSError("busy-1"), OSError("busy-2"), final],
        })
        with tempfile.TemporaryDirectory() as temp:
            load_config = config(Path(temp))
            load_config["phase3b_acceptance"] = {
                "prometheus_scrapes_auxiliary": True,
            }
            load_config["scrape_interval_s"] = 0.1
            load_config["scrape_late_tolerance_s"] = 0.02
            load_config["load_monitoring"] = {
                "minimum_interval_scrapes_per_endpoint": 1,
                "maximum_consecutive_failures_per_endpoint": 1,
            }
            load_config["load_probes"] = [{
                "id": "load", "input_len": 128, "output_len": 128,
                "rate_rps": 1.0, "duration_s": 1.0, "max_concurrency": 1,
            }]
            harness = LoadFixtureHarness(
                load_config, run_id="auxiliary-outage", scrape_source=source,
            )
            harness.active_polls = 3
            run_dir = harness.run(("load",))
            monitoring = json.loads(
                (run_dir / "derived/load_monitoring.json").read_text()
            )[0]
            self.assertTrue(monitoring["prometheus_scrapes_auxiliary"])
            self.assertTrue(monitoring["violations"])
            summary = json.loads((run_dir / "derived/summary.json").read_text())
            self.assertTrue(summary["completed"])
            self.assertTrue(summary["phase3b_acceptance"]["prometheus_scrapes_auxiliary"])

    def test_proxy_diagnostics_audit_matches_successful_real_stream_requests(self):
        baseline = prom(1000, 500, 20)
        source = ScrapeSequence({
            "P0": [baseline, baseline, prom(1128, 501, 21)],
            "D0": [baseline, baseline, prom(1128, 628, 21)],
        })
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            diagnostics = temp_path / "proxy.jsonl"
            diagnostics.write_text(json.dumps({
                "request_id": "fixture-0",
                "logical_request_id": "fixture-0",
                "vllm_request_id": "___prefill_addr_p___decode_addr_d_fixture-0",
                "outcome": "completed",
                "incoming_client_stream": True,
                "outgoing_decode_stream": True,
                "decode_stream_available": True,
                "decode_content_type": "text/event-stream; charset=utf-8",
                "client_ttft_valid": True,
                "client_tpot_valid": True,
                "client_itl_valid": True,
                "upstream_chunk_count": 2,
                "upstream_byte_count": 100,
                "timestamps_monotonic_s": {
                    "request_received": 1.0,
                    "prefill_started": 2.0,
                    "prefill_completed": 3.0,
                    "decode_request_started": 4.0,
                    "decode_response_headers_received": 5.0,
                    "decode_first_real_chunk_received": 6.0,
                    "decode_first_real_chunk_forwarded": 7.0,
                    "decode_last_chunk_received": 8.0,
                    "response_completed": 9.0,
                },
                "durations_ms": {
                    "prefill": 1.0,
                    "decode_request_to_headers": 1.0,
                    "decode_request_to_first_real_chunk": 2.0,
                    "first_chunk_forwarding_delay": 1.0,
                    "full_decode_stream": 2.0,
                    "decode_request_to_last_chunk": 4.0,
                    "total_proxy_request": 8.0,
                },
            }) + "\n", encoding="utf-8")
            audit_config = config(temp_path)
            audit_config["client"]["require_real_decode_streaming"] = True
            audit_config["proxy_diagnostics_log"] = str(diagnostics)
            run_dir = FixtureHarness(
                audit_config, run_id="audit", scrape_source=source,
            ).run(("semantic",))

            audit = json.loads(
                (run_dir / "derived/proxy_diagnostics_audit.json").read_text()
            )
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["expected_successful_requests"], 1)
            self.assertEqual(audit["diagnostic_record_count"], 1)
            self.assertTrue(audit["logical_request_ids_exactly_match"])


if __name__ == "__main__":
    unittest.main()
