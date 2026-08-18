"""Phase 3A: read-only P/D observability validation for the existing vLLM path.

This module deliberately contains no routing, model inference, GPU telemetry,
frequency control, or server lifecycle implementation.  It drives the existing
synthetic-trace client, scrapes both vLLM ``/metrics`` endpoints, preserves the
raw text, and derives windows with the canonical XpYd parser/tracker.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
            "staleness_threshold_s": self.config.get(
                "staleness_threshold_s",
                2.5 * float(self.config.get("scrape_interval_s", 1.0)),
            ),
            "semantic_probes": self.config.get("semantic_probes", []),
            "load_probes": self.config.get("load_probes", []),
            "phase_warmup": self.config.get("phase_warmup", {"enabled": False}),
            "physical_context": self.config.get("physical_context", {}),
            "server_logs": self.config.get("server_logs", {}),
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

    def _scrape_round(self, label: str, probe_id: Optional[str] = None) -> Dict[str, VLLMMetricsSnapshot]:
        self.round_sequence += 1
        round_id = self.round_sequence
        observed: Dict[str, VLLMMetricsSnapshot] = {}
        for endpoint in self.endpoints:
            self.sequence += 1
            wall_start = self.wall_clock()
            mono_start = self.monotonic_clock()
            try:
                raw = self.scrape_source(endpoint)
            except Exception as exc:
                raise Phase3AError(
                    "metrics scrape failed for %s (%s): %s" %
                    (endpoint.endpoint_id, self.collector.metrics_url(endpoint), exc)
                ) from exc
            wall_end = self.wall_clock()
            mono_end = self.monotonic_clock()
            if not raw.raw_text.strip():
                raise Phase3AError("empty /metrics response for %s" % endpoint.endpoint_id)
            if raw.snapshot.endpoint_id != endpoint.endpoint_id:
                raise Phase3AError("scrape returned wrong endpoint identity")
            filename = "%06d_%s.prom" % (self.sequence, label.replace("/", "_"))
            raw_path = self.run_dir / endpoint.endpoint_id / "raw_metrics" / filename
            raw_path.write_text(raw.raw_text, encoding="utf-8")
            window = self.tracker.observe(raw.snapshot)
            record = {
                "sequence": self.sequence,
                "round_sequence": round_id,
                "label": label,
                "probe_id": probe_id,
                "endpoint_id": endpoint.endpoint_id,
                "role": endpoint.role,
                "metrics_uri": raw.metrics_url,
                "central_wall_start_s": wall_start,
                "central_wall_end_s": wall_end,
                "central_monotonic_start_s": mono_start,
                "central_monotonic_end_s": mono_end,
                "raw_metrics_path": raw_path.relative_to(self.run_dir).as_posix(),
                "snapshot": _snapshot_dict(raw.snapshot),
                "window": _window_dict(window),
            }
            _append_jsonl(self.run_dir / "derived" / "telemetry.jsonl", record)
            _append_jsonl(self.run_dir / "derived" / "scrapes.jsonl", {
                key: value for key, value in record.items() if key not in ("snapshot", "window")
            })
            observed[endpoint.endpoint_id] = raw.snapshot
            self.snapshots[endpoint.endpoint_id] = raw.snapshot

        if self.dry_controller is not None:
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

    def _validate_client_result(self, client_dir: Path, returncode: int) -> dict:
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
        if bool(self.config.get("client", {}).get("require_server_token_usage", False)):
            successes = int(summary.get("successful_requests", 0))
            sources = summary.get("completion_token_sources", {})
            exact = int(sources.get("server_usage", 0)) if isinstance(sources, dict) else 0
            if exact != successes:
                raise Phase3AError(
                    "exact server completion-token usage required for every successful "
                    "request: got %d of %d" % (exact, successes)
                )
        return summary

    def _run_client(self, probe: Mapping[str, Any], trace_path: Path) -> dict:
        command = self._client_command(probe, trace_path)
        client_dir = trace_path.parent
        with (client_dir / "client.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        return self._validate_client_result(client_dir, result.returncode)

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

    def run_load(self) -> None:
        probes = self.config.get("load_probes", [])
        if not probes:
            raise Phase3AError("load mode requested but load_probes is empty")
        interval = float(self.config.get("scrape_interval_s", 1.0))
        if not 0 < interval <= 2.0:
            raise Phase3AError("scrape_interval_s must be in (0, 2] for Phase 3A")
        for probe in probes:
            probe = dict(probe)
            if "count" not in probe:
                duration = float(probe["duration_s"])
                probe["count"] = max(1, int(math.ceil(duration * float(probe["rate_rps"]))))
            probe_id = str(probe["id"])
            trace = self._write_trace(probe)
            self._scrape_round("load_before", probe_id)
            process, stream = self._start_client(probe, trace)
            try:
                while process.poll() is None:
                    self._scrape_round("load_interval", probe_id)
                    self.sleeper(interval)
                returncode = process.wait()
            finally:
                stream.close()
            client = self._validate_client_result(trace.parent, returncode)
            self._scrape_round("load_after", probe_id)
            record = {"probe": probe, "client": client, "returncode": returncode}
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
        behavior = {}
        stale_threshold = float(
            self.config.get(
                "staleness_threshold_s",
                2.5 * float(self.config.get("scrape_interval_s", 1.0)),
            )
        )
        for endpoint_id, records in endpoint_windows.items():
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
            self._run_phase_warmup()
            # Preflight is intentionally a real scrape of both endpoints.
            self._scrape_round("preflight")
            if "semantic" in normalized:
                self.run_semantic()
            if "load" in normalized:
                self.run_load()
            self._copy_server_logs()
            summary = self._summary()
            _write_json(self.run_dir / "derived" / "summary.json", summary)
            write_run_summary_markdown(self.run_dir, summary)
            write_semantic_summary(self.run_dir)
        except Exception as exc:
            log_capture_errors = self._copy_server_logs(strict=False)
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
            "max_KV=%s, max_TTFT_P99_ms=%s, max_TBT_P99_ms=%s, stale_gaps=%s." % (
                endpoint_id,
                behavior["scrape_count"],
                behavior["valid_window_count"],
                behavior["max_running_requests"],
                behavior["max_waiting_requests"],
                behavior["max_kv_cache_usage_frac"],
                behavior["max_window_ttft_p99_ms"],
                behavior["max_window_tbt_p99_ms"],
                behavior["stale_gap_count"],
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
    modes = ("semantic", "load") if args.mode == "all" else (args.mode,)
    run_dir = Phase3AHarness(config, run_id=args.run_id).run(modes)
    print(run_dir.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
