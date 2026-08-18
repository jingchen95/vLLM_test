"""Small standard-library parser for Prometheus text exposition data."""

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, Mapping, Optional, Tuple


_METRIC_NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_SAMPLE_RE = re.compile(
    r"^(?P<name>" + _METRIC_NAME + r")"
    r"(?:\{(?P<labels>.*)\})?"
    r"[ \t]+(?P<value>[^ \t]+)"
    r"(?:[ \t]+(?P<timestamp>-?[0-9]+))?[ \t]*$"
)
_TYPE_RE = re.compile(
    r"^#[ \t]+TYPE[ \t]+(?P<name>" + _METRIC_NAME + r")"
    r"[ \t]+(?P<type>counter|gauge|histogram|summary|untyped)[ \t]*$"
)


class PrometheusParseError(ValueError):
    def __init__(self, line_number: int, message: str) -> None:
        super().__init__("line %d: %s" % (line_number, message))
        self.line_number = line_number


class HistogramDeltaError(ValueError):
    pass


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: Tuple[Tuple[str, str], ...]
    value: float
    timestamp_ms: Optional[int] = None

    def label_map(self) -> Dict[str, str]:
        return dict(self.labels)


@dataclass(frozen=True)
class PrometheusScrape:
    samples: Tuple[MetricSample, ...]
    metric_types: Tuple[Tuple[str, str], ...] = ()

    def find(self, name: str) -> Tuple[MetricSample, ...]:
        return tuple(sample for sample in self.samples if sample.name == name)

    def type_for(self, name: str) -> Optional[str]:
        return dict(self.metric_types).get(name)


@dataclass(frozen=True)
class PrometheusHistogram:
    """One cumulative Prometheus histogram time series."""

    buckets: Tuple[Tuple[float, float], ...]
    count: float
    sum: Optional[float]

    def __post_init__(self) -> None:
        buckets = tuple(self.buckets)
        object.__setattr__(self, "buckets", buckets)
        if not buckets:
            raise ValueError("histogram must contain buckets")
        boundaries = [boundary for boundary, _ in buckets]
        if any(math.isnan(boundary) or boundary == -math.inf for boundary in boundaries):
            raise ValueError("histogram boundaries cannot contain NaN or -Inf")
        if boundaries != sorted(boundaries):
            raise ValueError("histogram boundaries must be strictly ordered")
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("histogram boundaries must be unique")
        previous = -math.inf
        for _, cumulative_count in buckets:
            if not math.isfinite(cumulative_count) or cumulative_count < 0:
                raise ValueError("histogram bucket counts must be finite and non-negative")
            if cumulative_count < previous:
                raise ValueError("histogram bucket counts must be cumulative")
            previous = cumulative_count
        if not math.isfinite(self.count) or self.count < 0:
            raise ValueError("histogram count must be finite and non-negative")
        if buckets[-1][0] == math.inf and buckets[-1][1] != self.count:
            raise ValueError("+Inf bucket must equal histogram count")
        if self.sum is not None and (not math.isfinite(self.sum) or self.sum < 0):
            raise ValueError("histogram sum must be finite and non-negative")


@dataclass(frozen=True)
class LabeledHistogram:
    labels: Tuple[Tuple[str, str], ...]
    histogram: PrometheusHistogram


def _parse_float(raw: str, line_number: int) -> float:
    aliases = {
        "+Inf": math.inf,
        "Inf": math.inf,
        "-Inf": -math.inf,
        "NaN": math.nan,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return float(raw)
    except ValueError as exc:
        raise PrometheusParseError(line_number, "invalid numeric value %r" % raw) from exc


def _parse_labels(raw: str, line_number: int) -> Tuple[Tuple[str, str], ...]:
    labels = []
    seen = set()
    index = 0
    length = len(raw)
    while index < length:
        while index < length and raw[index].isspace():
            index += 1
        match = _LABEL_NAME.match(raw, index)
        if match is None:
            raise PrometheusParseError(line_number, "invalid label name")
        name = match.group(0)
        if name in seen:
            raise PrometheusParseError(line_number, "duplicate label %r" % name)
        seen.add(name)
        index = match.end()
        while index < length and raw[index].isspace():
            index += 1
        if index >= length or raw[index] != "=":
            raise PrometheusParseError(line_number, "expected '=' after label %r" % name)
        index += 1
        while index < length and raw[index].isspace():
            index += 1
        if index >= length or raw[index] != '"':
            raise PrometheusParseError(line_number, "expected quoted label value")
        index += 1
        value_chars = []
        while index < length:
            char = raw[index]
            if char == '"':
                index += 1
                break
            if char == "\\":
                index += 1
                if index >= length:
                    raise PrometheusParseError(line_number, "unterminated label escape")
                escaped = raw[index]
                if escaped == "n":
                    value_chars.append("\n")
                elif escaped in ('"', "\\"):
                    value_chars.append(escaped)
                else:
                    raise PrometheusParseError(
                        line_number,
                        "unsupported label escape \\%s" % escaped,
                    )
                index += 1
                continue
            value_chars.append(char)
            index += 1
        else:
            raise PrometheusParseError(line_number, "unterminated label value")
        labels.append((name, "".join(value_chars)))
        while index < length and raw[index].isspace():
            index += 1
        if index == length:
            break
        if raw[index] != ",":
            raise PrometheusParseError(line_number, "expected ',' between labels")
        index += 1
        if index == length:
            raise PrometheusParseError(line_number, "trailing label comma")
    return tuple(sorted(labels))


def parse_prometheus_text(text: str) -> PrometheusScrape:
    """Parse gauges, counters, and histogram components from text format."""

    samples = []
    metric_types: Dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            type_match = _TYPE_RE.match(line)
            if type_match:
                name = type_match.group("name")
                metric_type = type_match.group("type")
                previous = metric_types.get(name)
                if previous is not None and previous != metric_type:
                    raise PrometheusParseError(line_number, "conflicting TYPE declaration")
                metric_types[name] = metric_type
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            raise PrometheusParseError(line_number, "malformed metric sample")
        value = _parse_float(match.group("value"), line_number)
        timestamp = match.group("timestamp")
        samples.append(
            MetricSample(
                name=match.group("name"),
                labels=_parse_labels(match.group("labels") or "", line_number),
                value=value,
                timestamp_ms=int(timestamp) if timestamp is not None else None,
            )
        )
    return PrometheusScrape(
        samples=tuple(samples),
        metric_types=tuple(sorted(metric_types.items())),
    )


def extract_histograms(
    scrape: PrometheusScrape,
    base_name: str,
) -> Tuple[LabeledHistogram, ...]:
    """Reassemble histogram bucket/count/sum samples by non-``le`` labels."""

    bucket_groups: Dict[Tuple[Tuple[str, str], ...], Dict[float, float]] = {}
    for sample in scrape.find(base_name + "_bucket"):
        labels = sample.label_map()
        if "le" not in labels:
            raise ValueError("histogram bucket is missing the 'le' label")
        boundary = _parse_float(labels.pop("le"), 0)
        key = tuple(sorted(labels.items()))
        group = bucket_groups.setdefault(key, {})
        if boundary in group:
            raise ValueError("duplicate histogram bucket boundary")
        group[boundary] = sample.value

    counts = _unique_labeled_values(scrape.find(base_name + "_count"), "_count")
    sums = _unique_labeled_values(scrape.find(base_name + "_sum"), "_sum")
    histograms = []
    for labels, bucket_map in bucket_groups.items():
        if labels not in counts:
            raise ValueError("histogram buckets are missing a matching _count")
        histograms.append(
            LabeledHistogram(
                labels=labels,
                histogram=PrometheusHistogram(
                    buckets=tuple(sorted(bucket_map.items())),
                    count=counts[labels],
                    sum=sums.get(labels),
                ),
            )
        )
    return tuple(sorted(histograms, key=lambda item: item.labels))


def _unique_labeled_values(
    samples: Iterable[MetricSample],
    component: str,
) -> Dict[Tuple[Tuple[str, str], ...], float]:
    values = {}
    for sample in samples:
        if sample.labels in values:
            raise ValueError("duplicate histogram %s series" % component)
        values[sample.labels] = sample.value
    return values


def subtract_histograms(
    current: PrometheusHistogram,
    previous: PrometheusHistogram,
) -> PrometheusHistogram:
    """Subtract lifetime-cumulative histograms to obtain one window."""

    current_bounds = tuple(boundary for boundary, _ in current.buckets)
    previous_bounds = tuple(boundary for boundary, _ in previous.buckets)
    if current_bounds != previous_bounds:
        raise HistogramDeltaError("histogram bucket layout changed")
    bucket_deltas = []
    for (boundary, current_count), (_, previous_count) in zip(
        current.buckets,
        previous.buckets,
    ):
        delta = current_count - previous_count
        if delta < 0:
            raise HistogramDeltaError("histogram bucket counter reset")
        bucket_deltas.append((boundary, delta))
    count_delta = current.count - previous.count
    if count_delta < 0:
        raise HistogramDeltaError("histogram count reset")
    sum_delta = None
    if current.sum is not None and previous.sum is not None:
        sum_delta = current.sum - previous.sum
        if sum_delta < 0:
            raise HistogramDeltaError("histogram sum reset")
    return PrometheusHistogram(tuple(bucket_deltas), count_delta, sum_delta)


def histogram_quantile_upper_bound(
    histogram: PrometheusHistogram,
    quantile: float,
) -> Optional[float]:
    """Return the first finite bucket bound reaching ``quantile``.

    This is a conservative bucket-resolution approximation, not an exact
    request-level percentile. ``None`` means there were no observations or the
    quantile was reached only by the unbounded ``+Inf`` bucket.
    """

    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if histogram.count <= 0:
        return None
    target = quantile * histogram.count
    for boundary, cumulative_count in histogram.buckets:
        if cumulative_count >= target:
            return boundary if math.isfinite(boundary) else None
    return None


def histogram_violation_fraction_at_boundary(
    histogram: PrometheusHistogram,
    threshold: float,
) -> Optional[float]:
    """Return exact ``fraction(value > threshold)`` only at a bucket boundary."""

    if histogram.count <= 0:
        return None
    for boundary, cumulative_count in histogram.buckets:
        if boundary == threshold:
            return max(0.0, histogram.count - cumulative_count) / histogram.count
    return None
