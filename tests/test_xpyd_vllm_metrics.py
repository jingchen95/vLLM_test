"""CPU-only tests for Prometheus parsing and vLLM window telemetry."""

from pathlib import Path
import math
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from xpyd.prometheus import (
    HistogramDeltaError,
    PrometheusHistogram,
    PrometheusParseError,
    extract_histograms,
    histogram_quantile_upper_bound,
    histogram_violation_fraction_at_boundary,
    parse_prometheus_text,
    subtract_histograms,
)
from xpyd.types import EndpointSpec
from xpyd.vllm_metrics import (
    AmbiguousMetricError,
    INTER_TOKEN,
    QUEUE_TIME,
    TTFT,
    VLLMMetricsCollector,
    VLLMMetricsConnectionError,
    VLLMMetricsHTTPError,
    VLLMRawScrape,
    VLLMWindowTracker,
    window_delta_to_telemetry_sample,
)


FIXTURES = Path(__file__).parent / "fixtures" / "vllm_metrics"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def endpoint(endpoint_id="P0", uri="http://node-a:8001"):
    return EndpointSpec(
        endpoint_id=endpoint_id,
        role="prefill",
        gpu_type="test-gpu",
        node="node-a",
        gpu_ids=(0,),
        tp_degree=1,
        http_uri=uri,
        kv_connector="test-kv",
    )


class PrometheusParserTests(unittest.TestCase):
    def test_gauge_counter_and_labels(self):
        scrape = parse_prometheus_text(
            '# TYPE demo gauge\n'
            'demo{model_name="a",note="quoted\\\" value"} 2.5\n'
            '# TYPE requests_total counter\nrequests_total 4\n'
        )
        self.assertEqual(scrape.type_for("demo"), "gauge")
        self.assertEqual(scrape.find("demo")[0].value, 2.5)
        self.assertEqual(dict(scrape.find("demo")[0].labels)["note"], 'quoted" value')
        self.assertEqual(scrape.find("requests_total")[0].value, 4)

    def test_histogram_reassembled_structurally(self):
        scrape = parse_prometheus_text(fixture("normal_t0.prom"))
        histograms = extract_histograms(scrape, TTFT)
        self.assertEqual(len(histograms), 1)
        histogram = histograms[0].histogram
        self.assertEqual(histogram.count, 20)
        self.assertEqual(histogram.buckets[-1], (math.inf, 20))

    def test_malformed_metric_fails_clearly(self):
        with self.assertRaises(PrometheusParseError):
            parse_prometheus_text('broken{label="unterminated} 1\n')


class VLLMMappingTests(unittest.TestCase):
    def setUp(self):
        self.collector = VLLMMetricsCollector(model_name="model-a")

    def test_running_waiting_kv_counters_and_histograms(self):
        snapshot = self.collector.parse_text(endpoint(), fixture("normal_t0.prom"), 0)
        self.assertEqual(snapshot.num_requests_running, 1)
        self.assertEqual(snapshot.num_requests_waiting, 0)
        self.assertEqual(snapshot.kv_cache_usage_frac, 0.25)
        self.assertEqual(snapshot.prompt_tokens_total, 1000)
        self.assertEqual(snapshot.generation_tokens_total, 500)
        self.assertEqual(snapshot.request_success_total, 20)
        self.assertEqual(snapshot.ttft_histogram.count, 20)
        self.assertEqual(snapshot.inter_token_histogram.count, 50)
        self.assertEqual(snapshot.queue_time_histogram.count, 20)
        self.assertNotIn(TTFT, snapshot.missing_metrics)
        self.assertIn("vllm:request_decode_time_seconds", snapshot.missing_metrics)

    def test_abort_is_not_counted_as_success_by_default(self):
        snapshot = self.collector.parse_text(endpoint(), fixture("normal_t0.prom"), 0)
        self.assertEqual(snapshot.request_success_total, 18 + 2)

    def test_missing_metrics_remain_none_and_are_reported(self):
        snapshot = self.collector.parse_text(endpoint(), fixture("missing_optional.prom"), 0)
        self.assertIsNone(snapshot.num_requests_waiting)
        self.assertIsNone(snapshot.kv_cache_usage_frac)
        self.assertIsNone(snapshot.ttft_histogram)
        self.assertIn("vllm:num_requests_waiting", snapshot.missing_metrics)

    def test_ambiguous_model_requires_selection(self):
        text = (
            'vllm:num_requests_running{model_name="a"} 1\n'
            'vllm:num_requests_running{model_name="b"} 2\n'
        )
        with self.assertRaises(AmbiguousMetricError):
            VLLMMetricsCollector().parse_text(endpoint(), text, 0)
        selected = VLLMMetricsCollector(model_name="b").parse_text(endpoint(), text, 0)
        self.assertEqual(selected.num_requests_running, 2)

    def test_model_is_disambiguated_across_different_metric_names(self):
        text = (
            'vllm:num_requests_running{model_name="a"} 1\n'
            'vllm:num_requests_waiting{model_name="b"} 2\n'
        )
        with self.assertRaises(AmbiguousMetricError):
            VLLMMetricsCollector().parse_text(endpoint(), text, 0)
        selected = VLLMMetricsCollector().parse_text(
            endpoint(), 'vllm:num_requests_running{model_name="a"} 1\n', 0
        )
        self.assertEqual(selected.selected_model_name, "a")

    def test_metrics_uri_appends_once(self):
        self.assertEqual(
            self.collector.metrics_url(endpoint(uri="http://node-a:8001/")),
            "http://node-a:8001/metrics",
        )

    def test_network_and_http_failures_are_explicit(self):
        with patch("xpyd.vllm_metrics.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(VLLMMetricsConnectionError):
                self.collector.scrape(endpoint())
        error = HTTPError("http://node-a:8001/metrics", 503, "busy", {}, None)
        with patch("xpyd.vllm_metrics.urlopen", side_effect=error):
            with self.assertRaises(VLLMMetricsHTTPError):
                self.collector.scrape(endpoint())
        self.assertEqual(
            self.collector.metrics_url(endpoint(uri="http://node-a:8001/metrics")),
            "http://node-a:8001/metrics",
        )

    def test_raw_scrape_retains_exact_payload(self):
        payload = fixture("normal_t0.prom")
        response = unittest.mock.MagicMock()
        response.getcode.return_value = 200
        response.read.return_value = payload.encode("utf-8")
        response.__enter__.return_value = response
        with patch("xpyd.vllm_metrics.urlopen", return_value=response):
            raw = self.collector.scrape_raw(endpoint())
        self.assertIsInstance(raw, VLLMRawScrape)
        self.assertEqual(raw.raw_text, payload)
        self.assertEqual(raw.snapshot.prompt_tokens_total, 1000)
        self.assertEqual(raw.metrics_url, "http://node-a:8001/metrics")


class WindowDeltaTests(unittest.TestCase):
    def setUp(self):
        self.collector = VLLMMetricsCollector(model_name="model-a")
        self.endpoint = endpoint()

    def test_counter_rates_and_histogram_delta(self):
        tracker = VLLMWindowTracker()
        t0 = self.collector.parse_text(self.endpoint, fixture("normal_t0.prom"), 0)
        t1 = self.collector.parse_text(self.endpoint, fixture("normal_t1.prom"), 10)
        self.assertFalse(tracker.observe(t0).valid)
        window = tracker.observe(t1)
        self.assertTrue(window.valid)
        self.assertEqual(window.delta_prompt_tokens, 200)
        self.assertEqual(window.delta_generation_tokens, 120)
        self.assertEqual(window.delta_completed_requests, 8)
        self.assertEqual(window.prompt_tokens_per_s, 20)
        self.assertEqual(window.completed_requests_per_s, 0.8)
        self.assertEqual(dict(window.ttft_histogram.buckets)[0.05], 5)
        self.assertEqual(dict(window.ttft_histogram.buckets)[0.1], 8)
        self.assertEqual(window.ttft_histogram.count, 10)
        self.assertEqual(window.window_ttft_p95_ms, 200)
        self.assertEqual(window.window_ttft_p99_ms, 200)
        self.assertAlmostEqual(window.window_mean_ttft_ms, 70)

    def test_window_bridge_does_not_invent_gpu_energy(self):
        tracker = VLLMWindowTracker()
        tracker.observe(self.collector.parse_text(self.endpoint, fixture("normal_t0.prom"), 0))
        raw = self.collector.parse_text(self.endpoint, fixture("normal_t1.prom"), 10)
        sample = window_delta_to_telemetry_sample(raw, tracker.observe(raw))
        self.assertIsNone(sample.power_w)
        self.assertIsNone(sample.energy_j)
        self.assertEqual(sample.input_tokens, 200)
        self.assertEqual(sample.window_ttft_count, 10)

    def test_counter_or_histogram_reset_invalidates_interval(self):
        tracker = VLLMWindowTracker()
        tracker.observe(self.collector.parse_text(self.endpoint, fixture("normal_t1.prom"), 10))
        reset = tracker.observe(
            self.collector.parse_text(self.endpoint, fixture("counter_reset.prom"), 20)
        )
        self.assertFalse(reset.valid)
        self.assertEqual(reset.reason, "counter_or_histogram_reset")
        self.assertIn("vllm:prompt_tokens_total", reset.reset_metrics)
        self.assertIsNone(reset.delta_prompt_tokens)

    def test_monotonic_values_cannot_establish_process_identity(self):
        tracker = VLLMWindowTracker()
        before = self.collector.parse_text(
            self.endpoint, fixture("normal_t0.prom"), 0
        )
        # These higher values could come from the same process or from a
        # restarted process that already passed the old totals. There is no
        # process identity/start-time evidence in VLLMMetricsSnapshot.
        possibly_restarted = self.collector.parse_text(
            self.endpoint, fixture("normal_t1.prom"), 10
        )
        tracker.observe(before)
        window = tracker.observe(possibly_restarted)
        self.assertTrue(window.valid)
        self.assertEqual(window.reason, "ok")

    def test_zero_new_observations_has_no_quantile(self):
        tracker = VLLMWindowTracker()
        tracker.observe(self.collector.parse_text(self.endpoint, fixture("normal_t0.prom"), 0))
        window = tracker.observe(
            self.collector.parse_text(self.endpoint, fixture("normal_t0.prom"), 10)
        )
        self.assertTrue(window.valid)
        self.assertEqual(window.window_ttft_count, 0)
        self.assertIsNone(window.window_ttft_p95_ms)

    def test_changed_buckets_and_decreased_buckets_rejected(self):
        original = PrometheusHistogram(((0.1, 2.0), (math.inf, 2.0)), 2.0, 0.1)
        changed = PrometheusHistogram(((0.2, 3.0), (math.inf, 3.0)), 3.0, 0.2)
        reset = PrometheusHistogram(((0.1, 1.0), (math.inf, 1.0)), 1.0, 0.05)
        with self.assertRaises(HistogramDeltaError):
            subtract_histograms(changed, original)
        with self.assertRaises(HistogramDeltaError):
            subtract_histograms(reset, original)

    def test_quantile_is_bucket_upper_bound_and_violation_requires_boundary(self):
        histogram = PrometheusHistogram(
            ((0.05, 5.0), (0.1, 8.0), (0.2, 10.0), (math.inf, 10.0)),
            10.0,
            0.7,
        )
        self.assertEqual(histogram_quantile_upper_bound(histogram, 0.95), 0.2)
        self.assertEqual(histogram_quantile_upper_bound(histogram, 0.99), 0.2)
        self.assertEqual(histogram_violation_fraction_at_boundary(histogram, 0.1), 0.2)
        self.assertIsNone(histogram_violation_fraction_at_boundary(histogram, 0.15))


if __name__ == "__main__":
    unittest.main()
