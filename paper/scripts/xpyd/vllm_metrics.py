"""Read-only vLLM 0.15.1 metric collection and window derivation."""

from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from xpyd.prometheus import (
    HistogramDeltaError,
    LabeledHistogram,
    MetricSample,
    PrometheusHistogram,
    PrometheusScrape,
    extract_histograms,
    histogram_quantile_upper_bound,
    parse_prometheus_text,
    subtract_histograms,
)
from xpyd.telemetry import EndpointTelemetrySample
from xpyd.types import EndpointSpec


NUM_RUNNING = "vllm:num_requests_running"
NUM_WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"
PROMPT_TOKENS = "vllm:prompt_tokens_total"
GENERATION_TOKENS = "vllm:generation_tokens_total"
REQUEST_SUCCESS = "vllm:request_success_total"
TTFT = "vllm:time_to_first_token_seconds"
INTER_TOKEN = "vllm:inter_token_latency_seconds"
E2E_LATENCY = "vllm:e2e_request_latency_seconds"
QUEUE_TIME = "vllm:request_queue_time_seconds"
PREFILL_TIME = "vllm:request_prefill_time_seconds"
DECODE_TIME = "vllm:request_decode_time_seconds"

SUPPORTED_METRICS = (
    NUM_RUNNING,
    NUM_WAITING,
    KV_USAGE,
    PROMPT_TOKENS,
    GENERATION_TOKENS,
    REQUEST_SUCCESS,
    TTFT,
    INTER_TOKEN,
    E2E_LATENCY,
    QUEUE_TIME,
    PREFILL_TIME,
    DECODE_TIME,
)


class VLLMMetricsError(RuntimeError):
    pass


class VLLMMetricsConnectionError(VLLMMetricsError):
    pass


class VLLMMetricsHTTPError(VLLMMetricsError):
    pass


class AmbiguousMetricError(VLLMMetricsError):
    pass


@dataclass(frozen=True)
class VLLMMetricsSnapshot:
    endpoint_id: str
    timestamp_s: float
    num_requests_running: Optional[float]
    num_requests_waiting: Optional[float]
    kv_cache_usage_frac: Optional[float]
    prompt_tokens_total: Optional[float]
    generation_tokens_total: Optional[float]
    request_success_total: Optional[float]
    ttft_histogram: Optional[PrometheusHistogram]
    inter_token_histogram: Optional[PrometheusHistogram]
    e2e_latency_histogram: Optional[PrometheusHistogram]
    queue_time_histogram: Optional[PrometheusHistogram]
    prefill_time_histogram: Optional[PrometheusHistogram]
    decode_time_histogram: Optional[PrometheusHistogram]
    missing_metrics: Tuple[str, ...]
    selected_model_name: Optional[str]


@dataclass(frozen=True)
class VLLMRawScrape:
    """One live scrape with the exact Prometheus payload retained."""

    metrics_url: str
    raw_text: str
    snapshot: VLLMMetricsSnapshot


@dataclass(frozen=True)
class VLLMWindowDelta:
    endpoint_id: str
    start_timestamp_s: Optional[float]
    end_timestamp_s: float
    interval_s: Optional[float]
    valid: bool
    reason: str
    reset_metrics: Tuple[str, ...]
    delta_prompt_tokens: Optional[float] = None
    delta_generation_tokens: Optional[float] = None
    delta_completed_requests: Optional[float] = None
    prompt_tokens_per_s: Optional[float] = None
    generation_tokens_per_s: Optional[float] = None
    completed_requests_per_s: Optional[float] = None
    ttft_histogram: Optional[PrometheusHistogram] = None
    inter_token_histogram: Optional[PrometheusHistogram] = None
    window_mean_ttft_ms: Optional[float] = None
    window_ttft_p95_ms: Optional[float] = None
    window_ttft_p99_ms: Optional[float] = None
    window_mean_tbt_ms: Optional[float] = None
    window_tbt_p95_ms: Optional[float] = None
    window_tbt_p99_ms: Optional[float] = None
    window_ttft_count: int = 0
    window_tbt_count: int = 0


def _labels(sample: MetricSample) -> Dict[str, str]:
    return dict(sample.labels)


def _ensure_nonnegative(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise VLLMMetricsError("metric %s must be finite and non-negative" % name)
    return value


class VLLMMetricsCollector:
    """Fetch and map Prometheus text without retries or side effects."""

    def __init__(
        self,
        timeout_s: float = 5.0,
        model_name: Optional[str] = None,
        success_finished_reasons: Iterable[str] = ("stop", "length"),
        clock: Callable[[], float] = time.time,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = float(timeout_s)
        self.model_name = model_name
        self.success_finished_reasons = frozenset(success_finished_reasons)
        self.clock = clock

    @staticmethod
    def metrics_url(endpoint: EndpointSpec) -> str:
        if not endpoint.http_uri:
            raise VLLMMetricsError("endpoint %s has no http_uri" % endpoint.endpoint_id)
        base = endpoint.http_uri.rstrip("/")
        return base if base.endswith("/metrics") else base + "/metrics"

    def scrape(self, endpoint: EndpointSpec) -> VLLMMetricsSnapshot:
        return self.scrape_raw(endpoint).snapshot

    def scrape_raw(self, endpoint: EndpointSpec) -> VLLMRawScrape:
        """Fetch once, retain the raw response, and parse through ``parse_text``."""

        url = self.metrics_url(endpoint)
        request = Request(
            url,
            headers={
                "Accept": "text/plain; version=0.0.4",
                "User-Agent": "xpyd-vllm-metrics/1",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                status = response.getcode()
                if status != 200:
                    raise VLLMMetricsHTTPError("GET %s returned HTTP %s" % (url, status))
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            raise VLLMMetricsHTTPError(
                "GET %s returned HTTP %s" % (url, exc.code)
            ) from exc
        except (URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
            raise VLLMMetricsConnectionError("failed to scrape %s: %s" % (url, exc)) from exc
        snapshot = self.parse_text(endpoint, payload, timestamp_s=self.clock())
        return VLLMRawScrape(
            metrics_url=url,
            raw_text=payload,
            snapshot=snapshot,
        )

    def parse_text(
        self,
        endpoint: EndpointSpec,
        text: str,
        timestamp_s: Optional[float] = None,
    ) -> VLLMMetricsSnapshot:
        """Map fixture or live text through the identical parser path."""

        scrape = parse_prometheus_text(text)
        timestamp_s = self.clock() if timestamp_s is None else timestamp_s
        selected_model_name = self._resolve_model_name(scrape)
        values = {
            NUM_RUNNING: self._unique_scalar(scrape, NUM_RUNNING, selected_model_name),
            NUM_WAITING: self._unique_scalar(scrape, NUM_WAITING, selected_model_name),
            KV_USAGE: self._unique_scalar(scrape, KV_USAGE, selected_model_name),
            PROMPT_TOKENS: self._unique_scalar(scrape, PROMPT_TOKENS, selected_model_name),
            GENERATION_TOKENS: self._unique_scalar(
                scrape, GENERATION_TOKENS, selected_model_name
            ),
        }
        success_total = self._success_total(scrape, selected_model_name)
        histograms = {
            TTFT: self._unique_histogram(scrape, TTFT, selected_model_name),
            INTER_TOKEN: self._unique_histogram(scrape, INTER_TOKEN, selected_model_name),
            E2E_LATENCY: self._unique_histogram(scrape, E2E_LATENCY, selected_model_name),
            QUEUE_TIME: self._unique_histogram(scrape, QUEUE_TIME, selected_model_name),
            PREFILL_TIME: self._unique_histogram(scrape, PREFILL_TIME, selected_model_name),
            DECODE_TIME: self._unique_histogram(scrape, DECODE_TIME, selected_model_name),
        }

        for metric_name, value in values.items():
            values[metric_name] = _ensure_nonnegative(metric_name, value)
        success_total = _ensure_nonnegative(REQUEST_SUCCESS, success_total)
        kv_usage = values[KV_USAGE]
        if kv_usage is not None and kv_usage > 1:
            raise VLLMMetricsError("%s must be a fraction in [0, 1]" % KV_USAGE)

        present = {
            metric_name for metric_name, value in values.items() if value is not None
        }
        if success_total is not None:
            present.add(REQUEST_SUCCESS)
        present.update(
            metric_name for metric_name, value in histograms.items() if value is not None
        )
        return VLLMMetricsSnapshot(
            endpoint_id=endpoint.endpoint_id,
            timestamp_s=float(timestamp_s),
            num_requests_running=values[NUM_RUNNING],
            num_requests_waiting=values[NUM_WAITING],
            kv_cache_usage_frac=kv_usage,
            prompt_tokens_total=values[PROMPT_TOKENS],
            generation_tokens_total=values[GENERATION_TOKENS],
            request_success_total=success_total,
            ttft_histogram=histograms[TTFT],
            inter_token_histogram=histograms[INTER_TOKEN],
            e2e_latency_histogram=histograms[E2E_LATENCY],
            queue_time_histogram=histograms[QUEUE_TIME],
            prefill_time_histogram=histograms[PREFILL_TIME],
            decode_time_histogram=histograms[DECODE_TIME],
            missing_metrics=tuple(name for name in SUPPORTED_METRICS if name not in present),
            selected_model_name=selected_model_name,
        )

    def _resolve_model_name(self, scrape: PrometheusScrape) -> Optional[str]:
        if self.model_name is not None:
            return self.model_name
        model_names = set()
        for sample in scrape.samples:
            base_name = sample.name
            for suffix in ("_bucket", "_count", "_sum"):
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            if base_name not in SUPPORTED_METRICS:
                continue
            model_name = _labels(sample).get("model_name")
            if model_name is not None:
                model_names.add(model_name)
        if len(model_names) > 1:
            raise AmbiguousMetricError(
                "scrape contains multiple model_name series; configure model_name"
            )
        return next(iter(model_names), None)

    def _filter_model_samples(
        self,
        samples: Iterable[MetricSample],
        metric_name: str,
        selected_model_name: Optional[str],
    ) -> Tuple[MetricSample, ...]:
        samples = tuple(samples)
        labeled = tuple(sample for sample in samples if "model_name" in _labels(sample))
        unlabeled = tuple(sample for sample in samples if "model_name" not in _labels(sample))
        if selected_model_name is not None:
            if labeled:
                return tuple(
                    sample for sample in labeled
                    if _labels(sample).get("model_name") == selected_model_name
                )
            return unlabeled
        model_names = {_labels(sample)["model_name"] for sample in labeled}
        if len(model_names) > 1:
            raise AmbiguousMetricError(
                "%s has multiple model_name series; configure model_name" % metric_name
            )
        if labeled and unlabeled:
            raise AmbiguousMetricError("%s mixes labeled and unlabeled series" % metric_name)
        return labeled or unlabeled

    def _unique_scalar(
        self,
        scrape: PrometheusScrape,
        name: str,
        selected_model_name: Optional[str],
    ) -> Optional[float]:
        selected = self._filter_model_samples(
            scrape.find(name), name, selected_model_name
        )
        if not selected:
            return None
        if len(selected) != 1:
            raise AmbiguousMetricError("%s has %d selected series" % (name, len(selected)))
        return selected[0].value

    def _success_total(
        self,
        scrape: PrometheusScrape,
        selected_model_name: Optional[str],
    ) -> Optional[float]:
        selected = self._filter_model_samples(
            scrape.find(REQUEST_SUCCESS), REQUEST_SUCCESS, selected_model_name
        )
        if not selected:
            return None
        with_reason = [sample for sample in selected if "finished_reason" in _labels(sample)]
        without_reason = [sample for sample in selected if "finished_reason" not in _labels(sample)]
        if with_reason and without_reason:
            raise AmbiguousMetricError("request success series mix finish-reason labels")
        if without_reason:
            if len(without_reason) != 1:
                raise AmbiguousMetricError("request success counter is ambiguous")
            return without_reason[0].value
        # Intentionally exclude abort/error reasons by default.
        return sum(
            sample.value for sample in with_reason
            if _labels(sample)["finished_reason"] in self.success_finished_reasons
        )

    def _unique_histogram(
        self,
        scrape: PrometheusScrape,
        name: str,
        selected_model_name: Optional[str],
    ) -> Optional[PrometheusHistogram]:
        histograms = extract_histograms(scrape, name)
        if not histograms:
            return None
        selected = self._filter_model_histograms(
            histograms, name, selected_model_name
        )
        if not selected:
            return None
        if len(selected) != 1:
            raise AmbiguousMetricError("%s has %d selected histograms" % (name, len(selected)))
        return selected[0].histogram

    def _filter_model_histograms(
        self,
        histograms: Iterable[LabeledHistogram],
        metric_name: str,
        selected_model_name: Optional[str],
    ) -> Tuple[LabeledHistogram, ...]:
        histograms = tuple(histograms)
        labeled = tuple(item for item in histograms if "model_name" in dict(item.labels))
        unlabeled = tuple(item for item in histograms if "model_name" not in dict(item.labels))
        if selected_model_name is not None:
            if labeled:
                return tuple(
                    item for item in labeled
                    if dict(item.labels).get("model_name") == selected_model_name
                )
            return unlabeled
        model_names = {dict(item.labels)["model_name"] for item in labeled}
        if len(model_names) > 1:
            raise AmbiguousMetricError(
                "%s has multiple model_name histograms; configure model_name" % metric_name
            )
        if labeled and unlabeled:
            raise AmbiguousMetricError("%s mixes labeled and unlabeled histograms" % metric_name)
        return labeled or unlabeled


class VLLMWindowTracker:
    """Convert process-lifetime counters/histograms into interval deltas.

    Decreases and histogram-layout changes are observable discontinuities and
    invalidate the interval. Without a process identity/start-time metric,
    monotonic values alone cannot prove that the serving process did not
    restart between scrapes.
    """

    def __init__(self) -> None:
        self._previous: Dict[str, VLLMMetricsSnapshot] = {}

    def reset(self, endpoint_id: str) -> None:
        self._previous.pop(endpoint_id, None)

    def observe(self, current: VLLMMetricsSnapshot) -> VLLMWindowDelta:
        previous = self._previous.get(current.endpoint_id)
        self._previous[current.endpoint_id] = current
        if previous is None:
            return VLLMWindowDelta(
                endpoint_id=current.endpoint_id,
                start_timestamp_s=None,
                end_timestamp_s=current.timestamp_s,
                interval_s=None,
                valid=False,
                reason="baseline_initialized",
                reset_metrics=(),
            )
        interval_s = current.timestamp_s - previous.timestamp_s
        if interval_s <= 0:
            return VLLMWindowDelta(
                endpoint_id=current.endpoint_id,
                start_timestamp_s=previous.timestamp_s,
                end_timestamp_s=current.timestamp_s,
                interval_s=None,
                valid=False,
                reason="non_positive_interval",
                reset_metrics=(),
            )

        reset_metrics = []
        deltas = {}
        for name, old, new in (
            (PROMPT_TOKENS, previous.prompt_tokens_total, current.prompt_tokens_total),
            (GENERATION_TOKENS, previous.generation_tokens_total, current.generation_tokens_total),
            (REQUEST_SUCCESS, previous.request_success_total, current.request_success_total),
        ):
            if old is None or new is None:
                deltas[name] = None
            elif new < old:
                reset_metrics.append(name)
                deltas[name] = None
            else:
                deltas[name] = new - old

        window_histograms = {}
        for name, old, new in (
            (TTFT, previous.ttft_histogram, current.ttft_histogram),
            (INTER_TOKEN, previous.inter_token_histogram, current.inter_token_histogram),
        ):
            if old is None or new is None:
                window_histograms[name] = None
                continue
            try:
                window_histograms[name] = subtract_histograms(new, old)
            except HistogramDeltaError:
                reset_metrics.append(name)
                window_histograms[name] = None

        if reset_metrics:
            return VLLMWindowDelta(
                endpoint_id=current.endpoint_id,
                start_timestamp_s=previous.timestamp_s,
                end_timestamp_s=current.timestamp_s,
                interval_s=interval_s,
                valid=False,
                reason="counter_or_histogram_reset",
                reset_metrics=tuple(sorted(set(reset_metrics))),
            )

        ttft = window_histograms[TTFT]
        tbt = window_histograms[INTER_TOKEN]
        return VLLMWindowDelta(
            endpoint_id=current.endpoint_id,
            start_timestamp_s=previous.timestamp_s,
            end_timestamp_s=current.timestamp_s,
            interval_s=interval_s,
            valid=True,
            reason="ok",
            reset_metrics=(),
            delta_prompt_tokens=deltas[PROMPT_TOKENS],
            delta_generation_tokens=deltas[GENERATION_TOKENS],
            delta_completed_requests=deltas[REQUEST_SUCCESS],
            prompt_tokens_per_s=self._rate(deltas[PROMPT_TOKENS], interval_s),
            generation_tokens_per_s=self._rate(deltas[GENERATION_TOKENS], interval_s),
            completed_requests_per_s=self._rate(deltas[REQUEST_SUCCESS], interval_s),
            ttft_histogram=ttft,
            inter_token_histogram=tbt,
            window_mean_ttft_ms=self._mean_ms(ttft),
            window_ttft_p95_ms=self._quantile_ms(ttft, 0.95),
            window_ttft_p99_ms=self._quantile_ms(ttft, 0.99),
            window_mean_tbt_ms=self._mean_ms(tbt),
            window_tbt_p95_ms=self._quantile_ms(tbt, 0.95),
            window_tbt_p99_ms=self._quantile_ms(tbt, 0.99),
            window_ttft_count=self._integer_count(ttft),
            window_tbt_count=self._integer_count(tbt),
        )

    @staticmethod
    def _rate(delta: Optional[float], interval_s: float) -> Optional[float]:
        return None if delta is None else delta / interval_s

    @staticmethod
    def _mean_ms(histogram: Optional[PrometheusHistogram]) -> Optional[float]:
        if histogram is None or histogram.count <= 0 or histogram.sum is None:
            return None
        return 1000.0 * histogram.sum / histogram.count

    @staticmethod
    def _quantile_ms(
        histogram: Optional[PrometheusHistogram],
        quantile: float,
    ) -> Optional[float]:
        if histogram is None:
            return None
        boundary = histogram_quantile_upper_bound(histogram, quantile)
        return None if boundary is None else 1000.0 * boundary

    @staticmethod
    def _integer_count(histogram: Optional[PrometheusHistogram]) -> int:
        if histogram is None or histogram.count <= 0:
            return 0
        if not float(histogram.count).is_integer():
            raise VLLMMetricsError("histogram observation count is not integral")
        return int(histogram.count)


def window_delta_to_telemetry_sample(
    raw: VLLMMetricsSnapshot,
    window: VLLMWindowDelta,
) -> EndpointTelemetrySample:
    """Bridge one valid vLLM interval into the generic telemetry aggregator.

    vLLM supplies no GPU power or energy fields, so both intentionally remain
    absent from the returned sample.
    """

    if not window.valid or window.interval_s is None:
        raise ValueError("only valid window deltas can become telemetry samples")

    def interval_count(value: Optional[float], name: str) -> int:
        if value is None:
            return 0
        if not float(value).is_integer():
            raise VLLMMetricsError("%s interval counter is not integral" % name)
        return int(value)

    def gauge_count(value: Optional[float], name: str) -> Optional[int]:
        if value is None:
            return None
        if not float(value).is_integer():
            raise VLLMMetricsError("%s gauge is not integral" % name)
        return int(value)

    return EndpointTelemetrySample(
        endpoint_id=raw.endpoint_id,
        timestamp_s=raw.timestamp_s,
        queue_depth=gauge_count(raw.num_requests_waiting, NUM_WAITING),
        running_requests=gauge_count(raw.num_requests_running, NUM_RUNNING),
        kv_cache_usage_frac=raw.kv_cache_usage_frac,
        input_tokens=interval_count(window.delta_prompt_tokens, PROMPT_TOKENS),
        output_tokens=interval_count(window.delta_generation_tokens, GENERATION_TOKENS),
        completed_requests=interval_count(window.delta_completed_requests, REQUEST_SUCCESS),
        interval_s=window.interval_s,
        ttft_ms=window.window_mean_ttft_ms,
        tbt_ms=window.window_mean_tbt_ms,
        window_ttft_p95_ms=window.window_ttft_p95_ms,
        window_ttft_p99_ms=window.window_ttft_p99_ms,
        window_tbt_p95_ms=window.window_tbt_p95_ms,
        window_tbt_p99_ms=window.window_tbt_p99_ms,
        window_ttft_count=window.window_ttft_count,
        window_tbt_count=window.window_tbt_count,
        window_prompt_tokens_per_s=window.prompt_tokens_per_s,
        window_generation_tokens_per_s=window.generation_tokens_per_s,
        window_completed_requests_per_s=window.completed_requests_per_s,
    )
