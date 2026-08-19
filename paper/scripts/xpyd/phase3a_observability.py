"""Phase 3A: read-only P/D observability validation for the existing vLLM path.

This module deliberately contains no routing, model inference, GPU telemetry,
frequency control, or server lifecycle implementation.  It drives the existing
synthetic-trace client, scrapes both vLLM ``/metrics`` endpoints, preserves the
raw text, and derives windows with the canonical XpYd parser/tracker.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from xpyd.dry_run_controller import build_controller_from_config
from xpyd.prometheus import HistogramDeltaError, PrometheusHistogram, subtract_histograms
from xpyd.types import EndpointSpec
from xpyd.vllm_metrics import (
    DECODE_TIME,
    E2E_LATENCY,
    GENERATION_TOKENS,
    INTER_TOKEN,
    PREFILL_TIME,
    PROMPT_TOKENS,
    QUEUE_TIME,
    REQUEST_SUCCESS,
    TTFT,
    VLLMMetricsCollector,
    VLLMMetricsSnapshot,
    VLLMRawScrape,
    VLLMWindowDelta,
    VLLMWindowTracker,
)


HISTOGRAM_FIELDS = {
    TTFT: "ttft_histogram",
    INTER_TOKEN: "inter_token_histogram",
    E2E_LATENCY: "e2e_latency_histogram",
    QUEUE_TIME: "queue_time_histogram",
    PREFILL_TIME: "prefill_time_histogram",
    DECODE_TIME: "decode_time_histogram",
}


class Phase3AError(RuntimeError):
    """An explicit validation failure; never silently downgraded to success."""


@dataclass(frozen=True)
class ScrapeAttempt:
    endpoint: EndpointSpec
    scheduled_monotonic_s: float
    actual_wall_start_s: float
    actual_wall_finish_s: float
    actual_monotonic_start_s: float
    actual_monotonic_finish_s: float
    raw: Optional[VLLMRawScrape]
    error: Optional[BaseException]


@dataclass(frozen=True)
class ScheduledScrapeEvent:
    endpoint: EndpointSpec
    schedule_index: int
    scheduled_monotonic_s: float
    detected_monotonic_s: float
    attempt: Optional[ScrapeAttempt]
    missed: bool
    missed_reason: Optional[str]


class FixedPeriodScrapeOrchestrator:
    """Run one independent fixed-period scrape worker per endpoint.

    Workers never overlap scrapes for the same endpoint.  If an endpoint is
    still busy when one or more of its scheduled slots pass, those slots are
    emitted as explicit missed events.  Other endpoint workers remain
    independent, so a slow P scrape cannot shift D's schedule (or vice versa).
    """

    def __init__(
        self,
        endpoints: Sequence[EndpointSpec],
        scrape: Callable[[EndpointSpec, float], ScrapeAttempt],
        period_s: float,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if period_s <= 0 or not math.isfinite(period_s):
            raise ValueError("period_s must be finite and positive")
        self.endpoints = tuple(endpoints)
        self.scrape = scrape
        self.period_s = period_s
        self.monotonic_clock = monotonic_clock

    def run_while(self, active: Callable[[], bool]) -> list[ScheduledScrapeEvent]:
        base_s = self.monotonic_clock()
        stop = threading.Event()
        events: list[ScheduledScrapeEvent] = []
        lock = threading.Lock()
        stop_requested_at: list[Optional[float]] = [None]

        def emit(event: ScheduledScrapeEvent) -> None:
            with lock:
                events.append(event)

        def worker(endpoint: EndpointSpec) -> None:
            schedule_index = 0
            while not stop.is_set():
                scheduled_s = base_s + schedule_index * self.period_s
                delay_s = scheduled_s - self.monotonic_clock()
                if delay_s > 0 and stop.wait(delay_s):
                    break
                if stop.is_set():
                    break
                attempt = self.scrape(endpoint, scheduled_s)
                detected_s = self.monotonic_clock()
                emit(ScheduledScrapeEvent(
                    endpoint=endpoint,
                    schedule_index=schedule_index,
                    scheduled_monotonic_s=scheduled_s,
                    detected_monotonic_s=detected_s,
                    attempt=attempt,
                    missed=False,
                    missed_reason=None,
                ))
                schedule_index += 1
                coverage_end_s = stop_requested_at[0]
                if coverage_end_s is None:
                    coverage_end_s = detected_s
                while base_s + schedule_index * self.period_s <= coverage_end_s:
                    missed_s = base_s + schedule_index * self.period_s
                    emit(ScheduledScrapeEvent(
                        endpoint=endpoint,
                        schedule_index=schedule_index,
                        scheduled_monotonic_s=missed_s,
                        detected_monotonic_s=detected_s,
                        attempt=None,
                        missed=True,
                        missed_reason="previous_endpoint_scrape_still_in_flight",
                    ))
                    schedule_index += 1

        threads = [
            threading.Thread(
                target=worker,
                args=(endpoint,),
                name="xpyd-scrape-%s" % endpoint.endpoint_id,
                daemon=True,
            )
            for endpoint in self.endpoints
        ]
        for thread in threads:
            thread.start()
        try:
            # Polling only controls when collection stops.  Endpoint cadence is
            # owned by the independent workers above.
            while active():
                stop.wait(min(0.75 * self.period_s, 0.05))
        finally:
            stop_requested_at[0] = self.monotonic_clock()
            stop.set()
            for thread in threads:
                thread.join()
        return sorted(
            events,
            key=lambda event: (
                event.schedule_index,
                event.endpoint.endpoint_id,
                event.missed,
            ),
        )


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return "+Inf" if value > 0 else ("-Inf" if value < 0 else "NaN")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(value), sort_keys=True) + "\n")


def _histogram_dict(histogram: Optional[PrometheusHistogram]) -> Optional[dict]:
    if histogram is None:
        return None
    return {
        "buckets": [
            {"upper_bound_seconds": upper, "cumulative_count": count}
            for upper, count in histogram.buckets
        ],
        "count": histogram.count,
        "sum_seconds": histogram.sum,
    }


def _snapshot_dict(snapshot: VLLMMetricsSnapshot) -> dict:
    result = {
        "endpoint_id": snapshot.endpoint_id,
        "timestamp_s": snapshot.timestamp_s,
        "num_requests_running": snapshot.num_requests_running,
        "num_requests_waiting": snapshot.num_requests_waiting,
        "kv_cache_usage_frac": snapshot.kv_cache_usage_frac,
        "prompt_tokens_total": snapshot.prompt_tokens_total,
        "generation_tokens_total": snapshot.generation_tokens_total,
        "request_success_total": snapshot.request_success_total,
        "missing_metrics": list(snapshot.missing_metrics),
        "selected_model_name": snapshot.selected_model_name,
        "histograms": {},
    }
    for metric, field in HISTOGRAM_FIELDS.items():
        result["histograms"][metric] = _histogram_dict(getattr(snapshot, field))
    return result


def _window_dict(window: VLLMWindowDelta) -> dict:
    result = asdict(window)
    result["ttft_histogram"] = _histogram_dict(window.ttft_histogram)
    result["inter_token_histogram"] = _histogram_dict(window.inter_token_histogram)
    return result


def _clean_metadata(value: Any, key: str = "") -> Any:
    """Reject secret-shaped configuration rather than copying it to results."""

    lowered = key.lower()
    secret_shaped = (
        "password" in lowered
        or "secret" in lowered
        or "authorization" in lowered
        or lowered in ("token", "access_token", "api_token")
    )
    if secret_shaped:
        raise Phase3AError("metadata/config must not contain secret field %r" % key)
    if isinstance(value, dict):
        return {name: _clean_metadata(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_clean_metadata(item, key) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _endpoint_from_config(data: Mapping[str, Any]) -> EndpointSpec:
    return EndpointSpec(
        endpoint_id=str(data["endpoint_id"]),
        role=str(data["role"]),
        gpu_type=str(data["gpu_type"]),
        node=str(data["node"]),
        gpu_ids=tuple(int(item) for item in data.get("gpu_ids", [0])),
        tp_degree=int(data.get("tp_degree", 1)),
        http_uri=str(data["http_uri"]),
        kv_connector=data.get("kv_connector"),
    )


def load_config(path: Path) -> dict:
    config = _clean_metadata(json.loads(path.read_text(encoding="utf-8")))
    required = {"model", "vllm_version", "output_root", "client", "endpoints"}
    missing = required.difference(config)
    if missing:
        raise Phase3AError("config missing required fields: %s" % sorted(missing))
    endpoints = tuple(_endpoint_from_config(item) for item in config["endpoints"])
    if {item.endpoint_id for item in endpoints} != {"P0", "D0"}:
        raise Phase3AError("Phase 3A requires exactly endpoints P0 and D0")
    if {item.role for item in endpoints} != {"prefill", "decode"}:
        raise Phase3AError("Phase 3A requires one prefill and one decode endpoint")
    if str(config["vllm_version"]) != "0.15.1":
        raise Phase3AError("Phase 3A config must explicitly record vLLM 0.15.1")
    return config


class Phase3AHarness:
    def __init__(
        self,
        config: Mapping[str, Any],
        run_id: Optional[str] = None,
        scrape_source: Optional[Callable[[EndpointSpec], VLLMRawScrape]] = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = dict(config)
        self.endpoints = tuple(
            sorted(
                (_endpoint_from_config(item) for item in self.config["endpoints"]),
                key=lambda endpoint: endpoint.endpoint_id,
            )
        )
        self.run_id = run_id or _utc_now_id()
        self.run_dir = Path(self.config["output_root"]) / self.run_id
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self.sleeper = sleeper
        collector_cfg = dict(self.config.get("collector", {}))
        self.collector = VLLMMetricsCollector(**collector_cfg)
        self.scrape_source = scrape_source or self.collector.scrape_raw
        self.tracker = VLLMWindowTracker()
        self.sequence = 0
        self.round_sequence = 0
        self.snapshots: Dict[str, VLLMMetricsSnapshot] = {}
        self.semantic_records: list[dict] = []
        self.load_records: list[dict] = []
        self.load_monitoring_records: list[dict] = []
        self.scrape_errors: list[dict] = []
        self.phase_warmup_record: Optional[dict] = None
        self.proxy_diagnostics_audit: Optional[dict] = None
        self.dry_controller = None
        dry_path = self.config.get("dry_run_config")
        if dry_path:
            dry_config = json.loads(Path(dry_path).read_text(encoding="utf-8"))
            self.dry_controller = build_controller_from_config(dry_config)

    def _create_layout(self) -> None:
        if self.run_dir.exists():
            raise Phase3AError("run directory already exists: %s" % self.run_dir)
        for relative in (
            "client",
            "P0/raw_metrics",
            "D0/raw_metrics",
            "derived",
        ):
            (self.run_dir / relative).mkdir(parents=True, exist_ok=False)

    def _metadata(self, modes: Sequence[str]) -> dict:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unavailable"
        return {
            "schema_version": 1,
            "phase": "3A_observability_validation",
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "modes": list(modes),
            "model": self.config["model"],
            "tokenizer_model": self.config.get("tokenizer_model", self.config["model"]),
            "vllm_version": self.config["vllm_version"],
            "endpoints": [asdict(endpoint) for endpoint in self.endpoints],
            "client": self.config["client"],
            "scrape_interval_s": self.config.get("scrape_interval_s", 1.0),
            "scrape_late_tolerance_s": self.config.get(
                "scrape_late_tolerance_s",
                0.1 * float(self.config.get("scrape_interval_s", 1.0)),
            ),
            "staleness_threshold_s": self.config.get(
                "staleness_threshold_s",
                2.5 * float(self.config.get("scrape_interval_s", 1.0)),
            ),
            "semantic_probes": self.config.get("semantic_probes", []),
            "load_probes": self.config.get("load_probes", []),
            "load_monitoring": self.config.get("load_monitoring", {}),
            "phase_warmup": self.config.get("phase_warmup", {"enabled": False}),
            "physical_context": self.config.get("physical_context", {}),
            "server_logs": self.config.get("server_logs", {}),
            "proxy_diagnostics_log": self.config.get("proxy_diagnostics_log"),
            "dry_run": {
                "enabled": self.dry_controller is not None,
                "actuation_permitted": False,
                "config_reference": self.config.get("dry_run_config"),
            },
            "scope_exclusions": [
                "energy_measurement",
                "GPU telemetry",
                "DVFS or clock commands",
                "routing or lifecycle actuation",
                "model retraining or recalibration",
            ],
        }

    def _perform_scrape(
        self,
        endpoint: EndpointSpec,
        scheduled_monotonic_s: float,
    ) -> ScrapeAttempt:
        wall_start = self.wall_clock()
        mono_start = self.monotonic_clock()
        raw: Optional[VLLMRawScrape] = None
        error: Optional[BaseException] = None
        try:
            raw = self.scrape_source(endpoint)
            if not raw.raw_text.strip():
                raise Phase3AError("empty /metrics response")
            if raw.snapshot.endpoint_id != endpoint.endpoint_id:
                raise Phase3AError("scrape returned wrong endpoint identity")
        except Exception as exc:
            raw = None
            error = exc
        return ScrapeAttempt(
            endpoint=endpoint,
            scheduled_monotonic_s=scheduled_monotonic_s,
            actual_wall_start_s=wall_start,
            actual_wall_finish_s=self.wall_clock(),
            actual_monotonic_start_s=mono_start,
            actual_monotonic_finish_s=self.monotonic_clock(),
            raw=raw,
            error=error,
        )

    def _timing_fields(self, attempt: ScrapeAttempt) -> dict:
        latency_s = max(
            0.0,
            attempt.actual_monotonic_finish_s - attempt.actual_monotonic_start_s,
        )
        drift_s = max(
            0.0,
            attempt.actual_monotonic_start_s - attempt.scheduled_monotonic_s,
        )
        tolerance_s = float(self.config.get(
            "scrape_late_tolerance_s",
            0.1 * float(self.config.get("scrape_interval_s", 1.0)),
        ))
        return {
            "scheduled_monotonic_start_s": attempt.scheduled_monotonic_s,
            "actual_wall_start_s": attempt.actual_wall_start_s,
            "actual_wall_finish_s": attempt.actual_wall_finish_s,
            "actual_monotonic_start_s": attempt.actual_monotonic_start_s,
            "actual_monotonic_finish_s": attempt.actual_monotonic_finish_s,
            # Backward-compatible aliases used by accepted Phase 3A analysis.
            "central_wall_start_s": attempt.actual_wall_start_s,
            "central_wall_end_s": attempt.actual_wall_finish_s,
            "central_monotonic_start_s": attempt.actual_monotonic_start_s,
            "central_monotonic_end_s": attempt.actual_monotonic_finish_s,
            "scrape_latency_s": latency_s,
            "scrape_latency_ms": latency_s * 1000.0,
            "scheduling_drift_s": drift_s,
            "late_tolerance_s": tolerance_s,
            "late": drift_s > tolerance_s,
            "late_by_s": max(0.0, drift_s - tolerance_s),
            "missed": False,
        }

    def _record_scrape_attempt(
        self,
        attempt: ScrapeAttempt,
        label: str,
        probe_id: Optional[str],
        round_id: int,
        *,
        tolerate_errors: bool,
    ) -> Optional[VLLMMetricsSnapshot]:
        self.sequence += 1
        endpoint = attempt.endpoint
        timing = self._timing_fields(attempt)
        common = {
            "sequence": self.sequence,
            "round_sequence": round_id,
            "label": label,
            "probe_id": probe_id,
            "endpoint_id": endpoint.endpoint_id,
            "role": endpoint.role,
            "metrics_uri": self.collector.metrics_url(endpoint),
            **timing,
        }
        if attempt.error is not None:
            error_record = {
                **common,
                "status": "error",
                "error_type": type(attempt.error).__name__,
                "error": str(attempt.error),
                "tolerated": bool(tolerate_errors),
            }
            self.scrape_errors.append(error_record)
            _append_jsonl(
                self.run_dir / "derived" / "scrape_errors.jsonl",
                error_record,
            )
            _append_jsonl(self.run_dir / "derived" / "scrapes.jsonl", error_record)
            if tolerate_errors:
                return None
            raise Phase3AError(
                "metrics scrape failed for %s (%s): %s" %
                (endpoint.endpoint_id, common["metrics_uri"], attempt.error)
            ) from attempt.error

        raw = attempt.raw
        if raw is None:
            raise AssertionError("successful scrape attempt has no raw payload")
        filename = "%06d_%s.prom" % (self.sequence, label.replace("/", "_"))
        raw_path = self.run_dir / endpoint.endpoint_id / "raw_metrics" / filename
        raw_path.write_text(raw.raw_text, encoding="utf-8")
        window = self.tracker.observe(raw.snapshot)
        record = {
            **common,
            "status": "success",
            "metrics_uri": raw.metrics_url,
            "raw_metrics_path": raw_path.relative_to(self.run_dir).as_posix(),
            "snapshot": _snapshot_dict(raw.snapshot),
            "window": _window_dict(window),
        }
        _append_jsonl(self.run_dir / "derived" / "telemetry.jsonl", record)
        _append_jsonl(self.run_dir / "derived" / "scrapes.jsonl", {
            key: value
            for key, value in record.items()
            if key not in ("snapshot", "window")
        })
        self.snapshots[endpoint.endpoint_id] = raw.snapshot
        return raw.snapshot

    def _record_missed_scrape(
        self,
        event: ScheduledScrapeEvent,
        label: str,
        probe_id: Optional[str],
        round_id: int,
    ) -> None:
        self.sequence += 1
        endpoint = event.endpoint
        _append_jsonl(self.run_dir / "derived" / "scrapes.jsonl", {
            "sequence": self.sequence,
            "round_sequence": round_id,
            "label": label,
            "probe_id": probe_id,
            "endpoint_id": endpoint.endpoint_id,
            "role": endpoint.role,
            "metrics_uri": self.collector.metrics_url(endpoint),
            "status": "missed",
            "scheduled_monotonic_start_s": event.scheduled_monotonic_s,
            "missed_detected_monotonic_s": event.detected_monotonic_s,
            "actual_wall_start_s": None,
            "actual_wall_finish_s": None,
            "actual_monotonic_start_s": None,
            "actual_monotonic_finish_s": None,
            "scrape_latency_s": None,
            "scrape_latency_ms": None,
            "scheduling_drift_s": None,
            "late": False,
            "late_by_s": None,
            "missed": True,
            "missed_reason": event.missed_reason,
        })

    def _maybe_run_dry_controller(
        self,
        observed: Mapping[str, VLLMMetricsSnapshot],
        probe_id: Optional[str],
    ) -> None:
        if self.dry_controller is None or len(observed) != len(self.endpoints):
            return

        def provider(endpoint: EndpointSpec) -> VLLMMetricsSnapshot:
            return observed[endpoint.endpoint_id]

        now_s = max(item.timestamp_s for item in observed.values())
        dry_record = self.dry_controller.run_once(
            snapshot_provider=provider,
            now_s=now_s,
            workload_context_key=probe_id,
        )
        dry_dict = dry_record.to_dict()
        if dry_dict.get("actuated") is not False:
            raise Phase3AError("dry-run controller violated non-actuation invariant")
        _append_jsonl(self.run_dir / "derived" / "dry_run.jsonl", dry_dict)

    def _scrape_round(
        self,
        label: str,
        probe_id: Optional[str] = None,
        *,
        tolerate_errors: bool = False,
    ) -> Dict[str, VLLMMetricsSnapshot]:
        self.round_sequence += 1
        round_id = self.round_sequence
        observed: Dict[str, VLLMMetricsSnapshot] = {}
        scheduled_s = self.monotonic_clock()
        with ThreadPoolExecutor(
            max_workers=len(self.endpoints),
            thread_name_prefix="xpyd-scrape-round",
        ) as executor:
            futures = {
                endpoint.endpoint_id: executor.submit(
                    self._perform_scrape, endpoint, scheduled_s
                )
                for endpoint in self.endpoints
            }
            attempts = {
                endpoint_id: future.result()
                for endpoint_id, future in futures.items()
            }
        failures = []
        for endpoint in self.endpoints:
            try:
                snapshot = self._record_scrape_attempt(
                    attempts[endpoint.endpoint_id],
                    label,
                    probe_id,
                    round_id,
                    tolerate_errors=tolerate_errors,
                )
                if snapshot is not None:
                    observed[endpoint.endpoint_id] = snapshot
            except Phase3AError as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        self._maybe_run_dry_controller(observed, probe_id)
        return observed

    def _trace_path(self, probe_id: str) -> Path:
        client_dir = self.run_dir / "client" / probe_id
        client_dir.mkdir(parents=True, exist_ok=False)
        return client_dir / "trace.csv"

    def _write_trace(self, probe: Mapping[str, Any]) -> Path:
        probe_id = str(probe["id"])
        count = int(probe.get("count", 1))
        if count <= 0:
            raise Phase3AError("probe count must be positive")
        rate = float(probe.get("rate_rps", 0.0))
        if count > 1 and rate <= 0:
            raise Phase3AError("multi-request probe requires positive rate_rps")
        path = self._trace_path(probe_id)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("request_id", "arrival_time_s", "input_len", "output_len"),
            )
            writer.writeheader()
            for index in range(count):
                writer.writerow({
                    "request_id": "%s-%04d" % (probe_id, index),
                    "arrival_time_s": 0.0 if index == 0 else index / rate,
                    "input_len": int(probe["input_len"]),
                    "output_len": int(probe["output_len"]),
                })
        return path

    def _client_command(self, probe: Mapping[str, Any], trace_path: Path) -> list[str]:
        client = self.config["client"]
        client_dir = trace_path.parent
        script = Path(client.get("script", "paper/scripts/replay_synthetic_trace.py"))
        command = [
            str(client.get("python", sys.executable)),
            script.as_posix(),
            "--base-url", str(client["base_url"]),
            "--model", str(self.config["model"]),
            "--tokenizer-model", str(self.config.get("tokenizer_model", self.config["model"])),
            "--trace-csv", trace_path.as_posix(),
            "--output-file", (client_dir / "summary.txt").as_posix(),
            "--summary-json", (client_dir / "summary.json").as_posix(),
            "--requests-jsonl", (client_dir / "requests.jsonl").as_posix(),
            "--max-concurrency", str(int(probe.get("max_concurrency", 1))),
            "--num-warmups", str(int(probe.get("num_warmups", 0))),
            "--request-timeout-s", str(float(self.config.get("request_timeout_s", 900))),
            "--fail-on-request-error",
        ]
        _write_json(client_dir / "command.json", {"argv": command})
        return command

    def _validate_client_result(
        self,
        client_dir: Path,
        returncode: int,
        probe: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        if returncode != 0:
            raise Phase3AError("benchmark client failed with exit status %d" % returncode)
        summary_path = client_dir / "summary.json"
        if not summary_path.is_file():
            raise Phase3AError("benchmark produced no client summary: %s" % summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("successful_requests", 0)) <= 0:
            raise Phase3AError("benchmark produced no successful client result")
        if int(summary.get("failed_requests", 0)) != 0:
            raise Phase3AError("benchmark client reported request failures")
        if probe is not None:
            expected = int(probe.get("count", 1))
            requests_total = int(summary.get("requests_total", -1))
            successes = int(summary.get("successful_requests", 0))
            if requests_total != expected or successes != expected:
                raise Phase3AError(
                    "benchmark request cardinality mismatch for %s: "
                    "requests_total=%d, successful_requests=%d, expected=%d" %
                    (probe.get("id"), requests_total, successes, expected)
                )
            expected_input = expected * int(probe["input_len"])
            expected_output = expected * int(probe["output_len"])
            observed_input = int(summary.get("input_tokens_total", -1))
            requested_output = int(summary.get("requested_output_tokens_total", -1))
            measured_output = int(summary.get("output_tokens_total", -1))
            if (
                observed_input != expected_input
                or requested_output != expected_output
                or measured_output != expected_output
            ):
                raise Phase3AError(
                    "benchmark token cardinality mismatch for %s: input=%d/%d, "
                    "requested_output=%d/%d, measured_output=%d/%d" % (
                        probe.get("id"), observed_input, expected_input,
                        requested_output, expected_output,
                        measured_output, expected_output,
                    )
                )
        if bool(self.config.get("client", {}).get("require_server_token_usage", False)):
            successes = int(summary.get("successful_requests", 0))
            sources = summary.get("completion_token_sources", {})
            exact = int(sources.get("server_usage", 0)) if isinstance(sources, dict) else 0
            if exact != successes:
                raise Phase3AError(
                    "exact server completion-token usage required for every successful "
                    "request: got %d of %d" % (exact, successes)
                )
        if bool(self.config.get("client", {}).get("require_real_decode_streaming", False)):
            successes = int(summary.get("successful_requests", 0))
            for field in (
                "decode_stream_available_requests",
                "client_ttft_valid_requests",
                "client_tpot_valid_requests",
                "client_itl_valid_requests",
            ):
                observed = int(summary.get(field, 0))
                if observed != successes:
                    raise Phase3AError(
                        "real decode streaming required for every successful request: "
                        "%s=%d, successful_requests=%d" %
                        (field, observed, successes)
                    )
        if bool(self.config.get("client", {}).get(
            "require_logical_request_id_propagation", False
        )):
            successes = int(summary.get("successful_requests", 0))
            propagated = int(
                summary.get("logical_request_id_propagated_requests", 0)
            )
            if propagated != successes:
                raise Phase3AError(
                    "logical request ID propagation required for every successful "
                    "request: got %d of %d" % (propagated, successes)
                )
        return summary

    def _run_client(self, probe: Mapping[str, Any], trace_path: Path) -> dict:
        command = self._client_command(probe, trace_path)
        client_dir = trace_path.parent
        with (client_dir / "client.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        return self._validate_client_result(client_dir, result.returncode, probe)

    def _run_phase_warmup(self) -> Optional[dict]:
        warmup = dict(self.config.get("phase_warmup", {}))
        if not bool(warmup.get("enabled", False)):
            return None
        probe = {
            "id": "_phase_warmup",
            "input_len": int(warmup["input_len"]),
            "output_len": int(warmup["output_len"]),
            "count": int(warmup.get("count", 1)),
            "max_concurrency": int(warmup.get("max_concurrency", 1)),
        }
        if probe["count"] > 1:
            probe["rate_rps"] = float(warmup.get("rate_rps", 1.0))
        trace = self._write_trace(probe)
        client = self._run_client(probe, trace)
        record = {
            "excluded_from_measurement_windows": True,
            "probe": probe,
            "client": client,
        }
        _write_json(self.run_dir / "derived" / "phase_warmup.json", record)
        return record

    def _start_client(self, probe: Mapping[str, Any], trace_path: Path) -> Tuple[subprocess.Popen, Any]:
        command = self._client_command(probe, trace_path)
        stream = (trace_path.parent / "client.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        except Exception:
            stream.close()
            raise
        return process, stream

    @staticmethod
    def _semantic_delta(
        probe: Mapping[str, Any],
        client: Mapping[str, Any],
        before: Mapping[str, VLLMMetricsSnapshot],
        after: Mapping[str, VLLMMetricsSnapshot],
    ) -> dict:
        endpoints = {}
        for endpoint_id in sorted(before):
            tracker = VLLMWindowTracker()
            tracker.observe(before[endpoint_id])
            window = tracker.observe(after[endpoint_id])
            histograms = {}
            extra_resets = []
            for metric, field in HISTOGRAM_FIELDS.items():
                old = getattr(before[endpoint_id], field)
                new = getattr(after[endpoint_id], field)
                if old is None or new is None:
                    histograms[metric] = None
                    continue
                try:
                    histograms[metric] = _histogram_dict(subtract_histograms(new, old))
                except HistogramDeltaError:
                    histograms[metric] = None
                    extra_resets.append(metric)
            endpoints[endpoint_id] = {
                "window": _window_dict(window),
                "histogram_deltas": histograms,
                "missing_before": list(before[endpoint_id].missing_metrics),
                "missing_after": list(after[endpoint_id].missing_metrics),
                "reset_or_layout_change_metrics": sorted(
                    set(window.reset_metrics).union(extra_resets)
                ),
            }
        return {
            "probe": dict(probe),
            "client": dict(client),
            "endpoints": endpoints,
        }

    def run_semantic(self) -> None:
        probes = self.config.get("semantic_probes", [])
        if not probes:
            raise Phase3AError("semantic mode requested but semantic_probes is empty")
        for probe in probes:
            probe_id = str(probe["id"])
            trace = self._write_trace(probe)
            before = self._scrape_round("semantic_before", probe_id)
            client = self._run_client(probe, trace)
            after = self._scrape_round("semantic_after", probe_id)
            self.semantic_records.append(
                self._semantic_delta(probe, client, before, after)
            )
            _write_json(
                self.run_dir / "derived" / "semantic_deltas.json",
                self.semantic_records,
            )

    def _collect_fixed_period_load_scrapes(
        self,
        process: Any,
        probe_id: str,
        interval_s: float,
        minimum_successes: int,
        maximum_consecutive_failures: int,
    ) -> dict:
        orchestrator = FixedPeriodScrapeOrchestrator(
            endpoints=self.endpoints,
            scrape=self._perform_scrape,
            period_s=interval_s,
            monotonic_clock=self.monotonic_clock,
        )
        events = orchestrator.run_while(lambda: process.poll() is None)
        endpoint_ids = tuple(endpoint.endpoint_id for endpoint in self.endpoints)
        successful = {endpoint_id: 0 for endpoint_id in endpoint_ids}
        failed = {endpoint_id: 0 for endpoint_id in endpoint_ids}
        missed = {endpoint_id: 0 for endpoint_id in endpoint_ids}
        late = {endpoint_id: 0 for endpoint_id in endpoint_ids}
        scheduled = {endpoint_id: 0 for endpoint_id in endpoint_ids}
        consecutive = {endpoint_id: 0 for endpoint_id in endpoint_ids}
        max_consecutive = {endpoint_id: 0 for endpoint_id in endpoint_ids}
        drift_values: Dict[str, list[float]] = {
            endpoint_id: [] for endpoint_id in endpoint_ids
        }
        round_ids = {}
        observed_by_slot: Dict[int, Dict[str, VLLMMetricsSnapshot]] = {}
        for event in events:
            endpoint_id = event.endpoint.endpoint_id
            scheduled[endpoint_id] += 1
            if event.schedule_index not in round_ids:
                self.round_sequence += 1
                round_ids[event.schedule_index] = self.round_sequence
            round_id = round_ids[event.schedule_index]
            if event.missed:
                missed[endpoint_id] += 1
                consecutive[endpoint_id] += 1
                max_consecutive[endpoint_id] = max(
                    max_consecutive[endpoint_id], consecutive[endpoint_id]
                )
                self._record_missed_scrape(
                    event, "load_interval", probe_id, round_id
                )
                continue
            attempt = event.attempt
            if attempt is None:
                raise AssertionError("non-missed schedule event has no attempt")
            timing = self._timing_fields(attempt)
            drift_values[endpoint_id].append(timing["scheduling_drift_s"])
            if timing["late"]:
                late[endpoint_id] += 1
            snapshot = self._record_scrape_attempt(
                attempt,
                "load_interval",
                probe_id,
                round_id,
                tolerate_errors=True,
            )
            if snapshot is None:
                failed[endpoint_id] += 1
                consecutive[endpoint_id] += 1
                max_consecutive[endpoint_id] = max(
                    max_consecutive[endpoint_id], consecutive[endpoint_id]
                )
            else:
                successful[endpoint_id] += 1
                consecutive[endpoint_id] = 0
                observed_by_slot.setdefault(event.schedule_index, {})[
                    endpoint_id
                ] = snapshot

        for schedule_index in sorted(observed_by_slot):
            self._maybe_run_dry_controller(
                observed_by_slot[schedule_index], probe_id
            )

        violations = []
        for endpoint_id in endpoint_ids:
            if successful[endpoint_id] < minimum_successes:
                violations.append(
                    "%s successful interval scrapes %d < %d" % (
                        endpoint_id,
                        successful[endpoint_id],
                        minimum_successes,
                    )
                )
            if max_consecutive[endpoint_id] > maximum_consecutive_failures:
                violations.append(
                    "%s consecutive interval scrape failures/misses %d > %d" % (
                        endpoint_id,
                        max_consecutive[endpoint_id],
                        maximum_consecutive_failures,
                    )
                )
        monitoring = {
            "probe_id": probe_id,
            "sampling_architecture": "independent_fixed_period_endpoint_workers",
            "configured_period_s": interval_s,
            "late_tolerance_s": float(self.config.get(
                "scrape_late_tolerance_s", 0.1 * interval_s
            )),
            "interval_rounds": len(round_ids),
            "scheduled_interval_scrapes": scheduled,
            "successful_interval_scrapes": successful,
            "failed_interval_scrapes": failed,
            "missed_interval_scrapes": missed,
            "late_interval_scrapes": late,
            "maximum_consecutive_failures": max_consecutive,
            "maximum_scheduling_drift_s": {
                endpoint_id: max(drift_values[endpoint_id], default=None)
                for endpoint_id in endpoint_ids
            },
            "mean_scheduling_drift_s": {
                endpoint_id: (
                    sum(drift_values[endpoint_id]) / len(drift_values[endpoint_id])
                    if drift_values[endpoint_id] else None
                )
                for endpoint_id in endpoint_ids
            },
            "minimum_interval_scrapes_per_endpoint": minimum_successes,
            "maximum_consecutive_failures_per_endpoint": maximum_consecutive_failures,
            "tolerated_scrape_error_count": sum(failed.values()),
            "missed_scrape_count": sum(missed.values()),
            "late_scrape_count": sum(late.values()),
            "violations": violations,
        }
        return monitoring

    def run_load(self) -> None:
        probes = self.config.get("load_probes", [])
        if not probes:
            raise Phase3AError("load mode requested but load_probes is empty")
        interval = float(self.config.get("scrape_interval_s", 1.0))
        if not 0 < interval <= 2.0:
            raise Phase3AError("scrape_interval_s must be in (0, 2] for Phase 3A")
        late_tolerance = float(
            self.config.get("scrape_late_tolerance_s", 0.1 * interval)
        )
        if not math.isfinite(late_tolerance) or late_tolerance < 0:
            raise Phase3AError("scrape_late_tolerance_s must be finite and nonnegative")
        monitoring_config = dict(self.config.get("load_monitoring", {}))
        minimum_successes = int(
            monitoring_config.get("minimum_interval_scrapes_per_endpoint", 1)
        )
        maximum_consecutive_failures = int(
            monitoring_config.get("maximum_consecutive_failures_per_endpoint", 3)
        )
        if minimum_successes < 1:
            raise Phase3AError(
                "minimum_interval_scrapes_per_endpoint must be at least 1"
            )
        if maximum_consecutive_failures < 1:
            raise Phase3AError(
                "maximum_consecutive_failures_per_endpoint must be at least 1"
            )
        for probe in probes:
            probe = dict(probe)
            if "count" not in probe:
                duration = float(probe["duration_s"])
                probe["count"] = max(1, int(math.ceil(duration * float(probe["rate_rps"]))))
            probe_id = str(probe["id"])
            trace = self._write_trace(probe)
            before = self._scrape_round("load_before", probe_id)
            process, stream = self._start_client(probe, trace)
            try:
                monitoring = self._collect_fixed_period_load_scrapes(
                    process,
                    probe_id,
                    interval,
                    minimum_successes,
                    maximum_consecutive_failures,
                )
                returncode = process.wait()
            except Exception:
                # Do not let observer failure orphan the workload or cause the
                # outer server cleanup to kill requests still being measured.
                process.wait()
                raise
            finally:
                stream.close()
            after = self._scrape_round("load_after", probe_id)
            client = self._validate_client_result(trace.parent, returncode, probe)
            violations = monitoring["violations"]
            auxiliary_telemetry = bool(
                self.config.get("phase3b_acceptance", {}).get(
                    "prometheus_scrapes_auxiliary", False
                )
            )
            monitoring["prometheus_scrapes_auxiliary"] = auxiliary_telemetry
            monitoring["claim_boundary"] = (
                "auxiliary telemetry; not a Phase 3B hard gate"
                if auxiliary_telemetry else "Phase 3A load-monitoring gate"
            )
            self.load_monitoring_records.append(monitoring)
            _write_json(
                self.run_dir / "derived" / "load_monitoring.json",
                self.load_monitoring_records,
            )
            if violations and not auxiliary_telemetry:
                raise Phase3AError(
                    "load monitoring coverage invalid for %s: %s" %
                    (probe_id, "; ".join(violations))
                )
            delta = self._semantic_delta(probe, client, before, after)
            record = {
                **delta,
                "returncode": returncode,
                "monitoring": monitoring,
            }
            self.load_records.append(record)
            _write_json(self.run_dir / "derived" / "load_runs.json", self.load_records)

    def _copy_server_logs(self, strict: bool = True) -> list[str]:
        errors = []
        for endpoint_id, source_value in self.config.get("server_logs", {}).items():
            if endpoint_id not in {endpoint.endpoint_id for endpoint in self.endpoints}:
                message = "server_logs contains unknown endpoint %s" % endpoint_id
                if strict:
                    raise Phase3AError(message)
                errors.append(message)
                continue
            source = Path(source_value)
            if not source.is_file():
                message = "configured server log is unavailable: %s" % source
                if strict:
                    raise Phase3AError(message)
                errors.append(message)
                continue
            try:
                shutil.copyfile(source, self.run_dir / endpoint_id / "server.log")
            except OSError as exc:
                message = "failed to copy server log %s: %s" % (source, exc)
                if strict:
                    raise Phase3AError(message) from exc
                errors.append(message)
        return errors

    def _copy_proxy_diagnostics(self, strict: bool = True) -> list[str]:
        source_value = self.config.get("proxy_diagnostics_log")
        if not source_value:
            return []
        source = Path(source_value)
        if not source.is_file():
            message = "configured proxy diagnostics log is unavailable: %s" % source
            if strict:
                raise Phase3AError(message)
            return [message]
        try:
            shutil.copyfile(
                source, self.run_dir / "derived" / "proxy_diagnostics.jsonl"
            )
        except OSError as exc:
            message = "failed to copy proxy diagnostics log %s: %s" % (source, exc)
            if strict:
                raise Phase3AError(message) from exc
            return [message]
        return []

    def _audit_proxy_diagnostics(self) -> None:
        path = self.run_dir / "derived" / "proxy_diagnostics.jsonl"
        if not path.is_file():
            self.proxy_diagnostics_audit = None
            return
        records = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase3AError("invalid proxy diagnostics artifact: %s" % exc) from exc
        expected = 0
        if self.phase_warmup_record is not None:
            expected += int(
                self.phase_warmup_record["client"].get("successful_requests", 0)
            )
        expected += sum(
            int(record["client"].get("successful_requests", 0))
            for record in self.semantic_records
        )
        expected += sum(
            int(record["client"].get("successful_requests", 0))
            for record in self.load_records
        )
        violations = []
        if len(records) != expected:
            violations.append(
                "diagnostic record count %d != successful request count %d" %
                (len(records), expected)
            )
        require_streaming = bool(
            self.config.get("client", {}).get("require_real_decode_streaming", False)
        )
        require_logical_id = bool(self.config.get("client", {}).get(
            "require_logical_request_id_propagation", False
        ))
        client_request_ids = []
        for requests_path in sorted((self.run_dir / "client").glob("*/requests.jsonl")):
            for line in requests_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                result = json.loads(line)
                if result.get("ok"):
                    client_request_ids.append(str(
                        result.get("logical_request_id")
                        or result.get("trace_request_id")
                        or ""
                    ))
        diagnostic_request_ids = [
            str(record.get("logical_request_id") or record.get("request_id") or "")
            for record in records
        ]
        if require_logical_id:
            if any(not request_id for request_id in client_request_ids):
                violations.append("successful client result is missing logical request ID")
            if any(not request_id for request_id in diagnostic_request_ids):
                violations.append("proxy diagnostic is missing logical request ID")
            if sorted(client_request_ids) != sorted(diagnostic_request_ids):
                violations.append(
                    "client and proxy logical request ID multisets do not match"
                )
        if require_streaming:
            timestamp_order = (
                "request_received",
                "prefill_started",
                "prefill_completed",
                "decode_request_started",
                "decode_response_headers_received",
                "decode_first_real_chunk_received",
                "decode_first_real_chunk_forwarded",
                "decode_last_chunk_received",
                "response_completed",
            )
            for index, record in enumerate(records):
                content_type = str(record.get("decode_content_type") or "")
                timestamps = record.get("timestamps_monotonic_s") or {}
                timestamp_values = [timestamps.get(key) for key in timestamp_order]
                durations = record.get("durations_ms") or {}
                timing_valid = (
                    all(isinstance(value, (int, float)) for value in timestamp_values)
                    and all(
                        earlier <= later
                        for earlier, later in zip(timestamp_values, timestamp_values[1:])
                    )
                    and durations
                    and all(
                        isinstance(value, (int, float)) and value >= 0
                        for value in durations.values()
                    )
                )
                if not (
                    record.get("outcome") == "completed"
                    and record.get("incoming_client_stream") is True
                    and record.get("outgoing_decode_stream") is True
                    and record.get("decode_stream_available") is True
                    and record.get("client_ttft_valid") is True
                    and record.get("client_tpot_valid") is True
                    and record.get("client_itl_valid") is True
                    and content_type.startswith("text/event-stream")
                    and int(record.get("upstream_chunk_count", 0)) > 0
                    and int(record.get("upstream_byte_count", 0)) > 0
                    and timing_valid
                ):
                    violations.append(
                        "diagnostic record %d is not a completed real SSE path" % index
                    )
        self.proxy_diagnostics_audit = {
            "expected_successful_requests": expected,
            "diagnostic_record_count": len(records),
            "require_real_decode_streaming": require_streaming,
            "require_logical_request_id_propagation": require_logical_id,
            "client_logical_request_id_count": len(client_request_ids),
            "diagnostic_logical_request_id_count": len(diagnostic_request_ids),
            "logical_request_ids_exactly_match": (
                sorted(client_request_ids) == sorted(diagnostic_request_ids)
            ),
            "completed_real_sse_records": sum(
                1 for record in records
                if record.get("outcome") == "completed"
                and record.get("decode_stream_available") is True
            ),
            "valid": not violations,
            "violations": violations,
        }
        _write_json(
            self.run_dir / "derived" / "proxy_diagnostics_audit.json",
            self.proxy_diagnostics_audit,
        )
        if violations:
            raise Phase3AError(
                "proxy diagnostics audit failed: %s" % "; ".join(violations)
            )

    def _summary(self) -> dict:
        latest = {}
        thresholds_s = {
            "ttft": (0.3, 0.5, 1.0),
            "inter_token": (0.1, 0.2),
        }
        for endpoint_id, snapshot in sorted(self.snapshots.items()):
            bounds_by_metric = {
                metric: (
                    [upper for upper, _ in getattr(snapshot, field).buckets]
                    if getattr(snapshot, field) is not None else None
                )
                for metric, field in HISTOGRAM_FIELDS.items()
            }
            latest[endpoint_id] = {
                "missing_metrics": list(snapshot.missing_metrics),
                "running": snapshot.num_requests_running,
                "waiting": snapshot.num_requests_waiting,
                "kv_cache_usage_frac": snapshot.kv_cache_usage_frac,
                "bucket_boundaries_seconds": bounds_by_metric,
                "target_boundary_visibility": {
                    "ttft": {
                        str(target): (
                            None if bounds_by_metric[TTFT] is None
                            else any(math.isclose(target, bound) for bound in bounds_by_metric[TTFT])
                        )
                        for target in thresholds_s["ttft"]
                    },
                    "inter_token": {
                        str(target): (
                            None if bounds_by_metric[INTER_TOKEN] is None
                            else any(math.isclose(target, bound) for bound in bounds_by_metric[INTER_TOKEN])
                        )
                        for target in thresholds_s["inter_token"]
                    },
                },
            }
        resets = []
        endpoint_windows: Dict[str, list[dict]] = {item.endpoint_id: [] for item in self.endpoints}
        telemetry_path = self.run_dir / "derived" / "telemetry.jsonl"
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            endpoint_windows[record["endpoint_id"]].append(record)
            if record["window"]["reason"] == "counter_or_histogram_reset":
                resets.append({
                    "sequence": record["sequence"],
                    "endpoint_id": record["endpoint_id"],
                    "reset_metrics": record["window"]["reset_metrics"],
                })
        scrape_schedule_records: Dict[str, list[dict]] = {
            item.endpoint_id: [] for item in self.endpoints
        }
        scrapes_path = self.run_dir / "derived" / "scrapes.jsonl"
        for line in scrapes_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            scrape_schedule_records[record["endpoint_id"]].append(record)
        behavior = {}
        stale_threshold = float(
            self.config.get(
                "staleness_threshold_s",
                2.5 * float(self.config.get("scrape_interval_s", 1.0)),
            )
        )
        for endpoint_id, records in endpoint_windows.items():
            schedule_records = scrape_schedule_records[endpoint_id]
            snapshots = [record["snapshot"] for record in records]
            valid_windows = [record["window"] for record in records if record["window"]["valid"]]
            mono_starts = [record["central_monotonic_start_s"] for record in records]
            scrape_gaps = [
                later - earlier for earlier, later in zip(mono_starts, mono_starts[1:])
            ]
            behavior[endpoint_id] = {
                "scrape_count": len(records),
                "valid_window_count": len(valid_windows),
                "max_running_requests": max(
                    (item["num_requests_running"] for item in snapshots if item["num_requests_running"] is not None),
                    default=None,
                ),
                "max_waiting_requests": max(
                    (item["num_requests_waiting"] for item in snapshots if item["num_requests_waiting"] is not None),
                    default=None,
                ),
                "max_kv_cache_usage_frac": max(
                    (item["kv_cache_usage_frac"] for item in snapshots if item["kv_cache_usage_frac"] is not None),
                    default=None,
                ),
                "max_prompt_tokens_per_s": max(
                    (item["prompt_tokens_per_s"] for item in valid_windows if item["prompt_tokens_per_s"] is not None),
                    default=None,
                ),
                "max_generation_tokens_per_s": max(
                    (item["generation_tokens_per_s"] for item in valid_windows if item["generation_tokens_per_s"] is not None),
                    default=None,
                ),
                "max_completed_requests_per_s": max(
                    (item["completed_requests_per_s"] for item in valid_windows if item["completed_requests_per_s"] is not None),
                    default=None,
                ),
                "max_window_ttft_p99_ms": max(
                    (item["window_ttft_p99_ms"] for item in valid_windows if item["window_ttft_p99_ms"] is not None),
                    default=None,
                ),
                "max_window_tbt_p99_ms": max(
                    (item["window_tbt_p99_ms"] for item in valid_windows if item["window_tbt_p99_ms"] is not None),
                    default=None,
                ),
                "max_central_scrape_gap_s": max(
                    scrape_gaps,
                    default=None,
                ),
                "staleness_threshold_s": stale_threshold,
                "stale_gap_count": sum(gap > stale_threshold for gap in scrape_gaps),
                "missed_scrape_count": sum(
                    record.get("status") == "missed"
                    for record in schedule_records
                ),
                "late_scrape_count": sum(
                    record.get("late") is True for record in schedule_records
                ),
                "max_scheduling_drift_s": max(
                    (
                        record["scheduling_drift_s"]
                        for record in schedule_records
                        if record.get("scheduling_drift_s") is not None
                    ),
                    default=None,
                ),
                "max_scrape_latency_s": max(
                    (
                        record["scrape_latency_s"]
                        for record in schedule_records
                        if record.get("scrape_latency_s") is not None
                    ),
                    default=None,
                ),
                "missing_metrics_observed": sorted({
                    metric for item in snapshots for metric in item["missing_metrics"]
                }),
            }
        return {
            "run_id": self.run_id,
            "completed": True,
            "actuated": False,
            "latest_endpoint_observations": latest,
            "semantic_probe_count": len(self.semantic_records),
            "load_probe_count": len(self.load_records),
            "scrape_error_count": len(self.scrape_errors),
            "missed_scrape_count": sum(
                behavior[endpoint_id]["missed_scrape_count"]
                for endpoint_id in behavior
            ),
            "late_scrape_count": sum(
                behavior[endpoint_id]["late_scrape_count"]
                for endpoint_id in behavior
            ),
            "load_monitoring": self.load_monitoring_records,
            "proxy_diagnostics_audit": self.proxy_diagnostics_audit,
            "phase3b_acceptance": self.config.get("phase3b_acceptance", {}),
            "reset_or_discontinuity_observations": resets,
            "endpoint_window_behavior": behavior,
            "interpretation_boundary": (
                "Metric deltas are observations from the configured endpoints; "
                "they do not prove internal phase ownership or process continuity."
            ),
        }

    def run(self, modes: Sequence[str]) -> Path:
        normalized = tuple(dict.fromkeys(modes))
        if not normalized or any(mode not in ("semantic", "load") for mode in normalized):
            raise Phase3AError("modes must contain semantic and/or load")
        self._create_layout()
        _write_json(self.run_dir / "metadata.json", self._metadata(normalized))
        try:
            # Initialize lazy cross-node/runtime state before the first metric
            # baseline so warmup requests cannot contaminate measured windows.
            self.phase_warmup_record = self._run_phase_warmup()
            # Preflight is intentionally a real scrape of both endpoints.
            self._scrape_round("preflight")
            if "semantic" in normalized:
                self.run_semantic()
            if "load" in normalized:
                self.run_load()
            self._copy_server_logs()
            self._copy_proxy_diagnostics()
            self._audit_proxy_diagnostics()
            summary = self._summary()
            _write_json(self.run_dir / "derived" / "summary.json", summary)
            write_run_summary_markdown(self.run_dir, summary)
            write_semantic_summary(self.run_dir)
        except Exception as exc:
            log_capture_errors = self._copy_server_logs(strict=False)
            log_capture_errors.extend(self._copy_proxy_diagnostics(strict=False))
            _write_json(self.run_dir / "derived" / "failure.json", {
                "completed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "actuated": False,
                "server_log_capture_errors": log_capture_errors,
            })
            raise
        return self.run_dir


def _comparison(
    observed: Optional[float],
    client_value: Optional[float],
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 1.0,
) -> dict:
    if observed is None or client_value is None:
        return {"assessment": "unavailable", "observed": observed, "client": client_value}
    difference = float(observed) - float(client_value)
    tolerance = max(absolute_tolerance, abs(float(client_value)) * relative_tolerance)
    return {
        "assessment": "approximately_matches" if abs(difference) <= tolerance else "differs",
        "observed": observed,
        "client": client_value,
        "difference": difference,
        "tolerance": tolerance,
    }


def write_semantic_summary(run_dir: Path) -> dict:
    semantic_path = run_dir / "derived" / "semantic_deltas.json"
    records = json.loads(semantic_path.read_text(encoding="utf-8")) if semantic_path.exists() else []
    analyses = []
    for record in records:
        client = record["client"]
        endpoints = {}
        for endpoint_id, endpoint in record["endpoints"].items():
            window = endpoint["window"]
            endpoints[endpoint_id] = {
                "window_valid": window["valid"],
                "window_reason": window["reason"],
                "completed_requests": _comparison(
                    window.get("delta_completed_requests"), client.get("successful_requests"),
                    relative_tolerance=0.0, absolute_tolerance=0.0,
                ),
                "prompt_tokens": _comparison(
                    window.get("delta_prompt_tokens"), client.get("input_tokens_total"),
                    relative_tolerance=0.0, absolute_tolerance=0.0,
                ),
                "generation_tokens": _comparison(
                    window.get("delta_generation_tokens"), client.get("output_tokens_total"),
                    relative_tolerance=0.0, absolute_tolerance=0.0,
                ),
                "client_ttft_vs_endpoint_mean": _comparison(
                    window.get("window_mean_ttft_ms"), client.get("mean_ttft_ms")
                ),
                "client_ttft_p99_vs_endpoint_bucket_upper_bound": _comparison(
                    window.get("window_ttft_p99_ms"), client.get("p99_ttft_ms")
                ),
                "client_tpot_vs_endpoint_inter_token_mean": _comparison(
                    window.get("window_mean_tbt_ms"), client.get("mean_tpot_ms")
                ),
                "client_tpot_p99_vs_endpoint_inter_token_bucket_upper_bound": _comparison(
                    window.get("window_tbt_p99_ms"), client.get("p99_tpot_ms")
                ),
                "missing_metrics": sorted(set(endpoint["missing_before"] + endpoint["missing_after"])),
                "reset_or_layout_change_metrics": endpoint["reset_or_layout_change_metrics"],
            }
        analyses.append({"probe": record["probe"], "endpoints": endpoints})
    result = {
        "run_id": run_dir.name,
        "language_policy": ["observed", "approximately_matches", "differs", "unavailable"],
        "claim_boundary": (
            "These comparisons validate externally visible correlation only; "
            "they do not prove vLLM-internal phase attribution."
        ),
        "probes": analyses,
    }
    _write_json(run_dir / "derived" / "semantic_summary.json", result)
    lines = [
        "# Phase 3A semantic observability summary",
        "",
        result["claim_boundary"],
        "",
    ]
    if not analyses:
        lines.append("Semantic mode was not run; semantic comparisons are unavailable.")
    for analysis in analyses:
        probe = analysis["probe"]
        lines.extend([
            "## %s (%s input / %s output tokens)" %
            (probe["id"], probe["input_len"], probe["output_len"]),
            "",
        ])
        for endpoint_id, endpoint in sorted(analysis["endpoints"].items()):
            lines.append("- %s: requests=%s, prompt_tokens=%s, generation_tokens=%s, "
                         "TTFT(mean)=%s, TTFT(P99/bucket)=%s, "
                         "TPOT-vs-TBT(mean)=%s, TPOT-vs-TBT(P99/bucket)=%s; window=%s." % (
                endpoint_id,
                endpoint["completed_requests"]["assessment"],
                endpoint["prompt_tokens"]["assessment"],
                endpoint["generation_tokens"]["assessment"],
                endpoint["client_ttft_vs_endpoint_mean"]["assessment"],
                endpoint["client_ttft_p99_vs_endpoint_bucket_upper_bound"]["assessment"],
                endpoint["client_tpot_vs_endpoint_inter_token_mean"]["assessment"],
                endpoint["client_tpot_p99_vs_endpoint_inter_token_bucket_upper_bound"]["assessment"],
                endpoint["window_reason"],
            ))
        lines.append("")
    (run_dir / "derived" / "semantic_summary.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
    return result


def write_run_summary_markdown(run_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Phase 3A observability summary",
        "",
        "This is an observational report, not proof of internal phase ownership.",
        "",
        "## Histogram resolution",
        "",
    ]
    for endpoint_id, endpoint in sorted(summary["latest_endpoint_observations"].items()):
        visibility = endpoint["target_boundary_visibility"]
        lines.append("- %s TTFT exact boundaries: 300ms=%s, 500ms=%s, 1000ms=%s; inter-token exact boundaries: 100ms=%s, 200ms=%s." % (
            endpoint_id,
            visibility["ttft"]["0.3"],
            visibility["ttft"]["0.5"],
            visibility["ttft"]["1.0"],
            visibility["inter_token"]["0.1"],
            visibility["inter_token"]["0.2"],
        ))
    lines.extend(["", "## Window behavior", ""])
    for endpoint_id, behavior in sorted(summary["endpoint_window_behavior"].items()):
        lines.append(
            "- %s: scrapes=%s, valid_windows=%s, max_running=%s, max_waiting=%s, "
            "max_KV=%s, max_TTFT_P99_ms=%s, max_TBT_P99_ms=%s, stale_gaps=%s, "
            "late_scrapes=%s, missed_scrapes=%s, max_drift_s=%s, "
            "max_scrape_latency_s=%s." % (
                endpoint_id,
                behavior["scrape_count"],
                behavior["valid_window_count"],
                behavior["max_running_requests"],
                behavior["max_waiting_requests"],
                behavior["max_kv_cache_usage_frac"],
                behavior["max_window_ttft_p99_ms"],
                behavior["max_window_tbt_p99_ms"],
                behavior["stale_gap_count"],
                behavior["late_scrape_count"],
                behavior["missed_scrape_count"],
                behavior["max_scheduling_drift_s"],
                behavior["max_scrape_latency_s"],
            )
        )
    lines.extend([
        "",
        "Exact boundary visibility only describes histogram resolution; it does not enforce an SLO.",
        "",
    ])
    (run_dir / "derived" / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3A P/D observability validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--mode", choices=("semantic", "load", "all"), default="all")
    run.add_argument("--run-id")
    run.add_argument("--semantic-probe-id")
    run.add_argument("--load-probe-id")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "analyze":
        result = write_semantic_summary(args.run_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    config = load_config(args.config)
    if args.semantic_probe_id:
        selected = [
            probe for probe in config.get("semantic_probes", [])
            if probe.get("id") == args.semantic_probe_id
        ]
        if not selected:
            raise Phase3AError(
                "semantic probe ID not found: %s" % args.semantic_probe_id
            )
        config["semantic_probes"] = selected
    if args.load_probe_id:
        selected = [
            probe for probe in config.get("load_probes", [])
            if probe.get("id") == args.load_probe_id
        ]
        if not selected:
            raise Phase3AError("load probe ID not found: %s" % args.load_probe_id)
        config["load_probes"] = selected
    modes = ("semantic", "load") if args.mode == "all" else (args.mode,)
    run_dir = Phase3AHarness(config, run_id=args.run_id).run(modes)
    print(run_dir.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
