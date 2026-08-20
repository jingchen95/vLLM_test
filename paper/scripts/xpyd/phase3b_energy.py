"""Phase 3B read-only GPU-board energy measurement and preflight orchestration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from xpyd.nvml_readonly import (
    NVMLReadOnlyError,
    ReadOnlyNVMLSource,
    configured_cuda_device,
    structured_error,
)
from xpyd.phase3a_observability import Phase3AHarness, load_config as load_phase3a_config


class Phase3BEnergyError(RuntimeError):
    """Explicit Phase 3B measurement or audit failure."""


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(stream: Any, record: Mapping[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()


class AsyncFlushingStream:
    """Keep shared-filesystem flush latency off the sampling deadline path.

    Records are flushed in bounded batches.  Flushing every JSONL record made
    the writer fall minutes behind on Minerva's shared filesystem even though
    sampling itself remained on schedule; shutdown then had to drain that
    backlog.  A batch is still made visible at least four times per second,
    while the sentinel path drains every queued record before returning.
    """

    _MAX_BATCH_RECORDS = 256
    _MAX_BATCH_AGE_S = 0.25

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.items: queue.Queue[Optional[str]] = queue.Queue()
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(
            target=self._write_loop, name="xpyd-energy-writer", daemon=True
        )
        self.thread.start()

    def _write_loop(self) -> None:
        try:
            while True:
                item = self.items.get()
                if item is None:
                    break
                batch = [item]
                deadline = time.monotonic() + self._MAX_BATCH_AGE_S
                closing = False
                while len(batch) < self._MAX_BATCH_RECORDS:
                    timeout = max(0.0, deadline - time.monotonic())
                    try:
                        item = self.items.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if item is None:
                        closing = True
                        break
                    batch.append(item)
                self.stream.write("".join(batch))
                self.stream.flush()
                if closing:
                    break
        except BaseException as exc:
            self.error = exc

    def write(self, text: str) -> None:
        if self.error is not None:
            raise Phase3BEnergyError("energy writer failed: %s" % self.error)
        self.items.put(text)

    def flush(self) -> None:
        # The dedicated writer flushes every record. Sampling never waits for
        # shared-filesystem latency merely to satisfy the stream API.
        return None

    def close(self) -> None:
        self.items.put(None)
        self.thread.join()
        if self.error is not None:
            raise Phase3BEnergyError("energy writer failed: %s" % self.error)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase3BEnergyError("malformed JSONL at %s:%d: %s" % (path, number, exc)) from exc
        if not isinstance(value, dict):
            raise Phase3BEnergyError("non-object JSONL record at %s:%d" % (path, number))
        records.append(value)
    return records


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_wall_s: float
    end_wall_s: float
    included_in_reported_workload: bool
    logical_requests: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class FixedPeriodEnergySampler:
    """One endpoint-local sampler whose deadlines derive from one base time."""

    def __init__(
        self,
        endpoint: str,
        source: Any,
        period_s: float,
        late_tolerance_s: float,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        wait: Optional[Callable[[float], bool]] = None,
    ) -> None:
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("period_s must be finite and positive")
        if not math.isfinite(late_tolerance_s) or late_tolerance_s < 0:
            raise ValueError("late_tolerance_s must be finite and nonnegative")
        self.endpoint = endpoint
        self.source = source
        self.period_s = float(period_s)
        self.late_tolerance_s = float(late_tolerance_s)
        self.monotonic_clock = monotonic_clock
        self.wall_clock = wall_clock
        self._stop = threading.Event()
        self.wait = wait or self._stop.wait

    def stop(self) -> None:
        self._stop.set()

    def run(self, stream: Any, *, duration_s: Optional[float] = None) -> int:
        base = self.monotonic_clock()
        end = None if duration_s is None else base + duration_s
        sequence = 0
        schedule_index = 0
        while not self._stop.is_set():
            scheduled = base + schedule_index * self.period_s
            if end is not None and scheduled > end:
                break
            delay = scheduled - self.monotonic_clock()
            if delay > 0 and self.wait(delay):
                break
            if self._stop.is_set():
                break
            actual_start_mono = self.monotonic_clock()
            actual_start_wall = self.wall_clock()
            error: Optional[BaseException] = None
            payload: dict[str, Any] = {}
            try:
                payload = dict(self.source.query())
            except Exception as exc:
                error = exc
            actual_finish_mono = self.monotonic_clock()
            actual_finish_wall = self.wall_clock()
            drift = max(0.0, actual_start_mono - scheduled)
            sequence += 1
            record = {
                "endpoint": self.endpoint,
                "sequence": sequence,
                "schedule_index": schedule_index,
                "scheduled_local_monotonic_s": scheduled,
                "actual_start_local_monotonic_s": actual_start_mono,
                "actual_finish_local_monotonic_s": actual_finish_mono,
                "actual_start_wall_s": actual_start_wall,
                "actual_finish_wall_s": actual_finish_wall,
                "query_latency_s": max(0.0, actual_finish_mono - actual_start_mono),
                "start_drift_s": drift,
                "late_tolerance_s": self.late_tolerance_s,
                "late": drift > self.late_tolerance_s,
                "missed": False,
                "missed_slot_count": 0,
                "status": "error" if error else "success",
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
                "measurement_source": (
                    "hardware_counter"
                    if self.source.capabilities.total_energy_supported
                    else (
                        "power_integral_estimate"
                        if (self.source.capabilities.total_energy_error or {}).get("not_supported")
                        else "invalid_energy_counter_probe_error"
                    )
                ),
                "power_query_source": getattr(
                    self.source.capabilities, "power_query_source", None
                ),
                "capability_total_energy": self.source.capabilities.total_energy_supported,
                "capability_power": self.source.capabilities.power_supported,
                **payload,
            }
            _append_jsonl(stream, record)
            schedule_index += 1
            detected = self.monotonic_clock()
            while base + schedule_index * self.period_s <= detected:
                sequence += 1
                missed_s = base + schedule_index * self.period_s
                _append_jsonl(stream, {
                    "endpoint": self.endpoint,
                    "sequence": sequence,
                    "schedule_index": schedule_index,
                    "scheduled_local_monotonic_s": missed_s,
                    "actual_start_local_monotonic_s": None,
                    "actual_finish_local_monotonic_s": None,
                    "actual_start_wall_s": None,
                    "actual_finish_wall_s": None,
                    "query_latency_s": None,
                    "start_drift_s": None,
                    "late_tolerance_s": self.late_tolerance_s,
                    "late": False,
                    "missed": True,
                    "missed_slot_count": 1,
                    "status": "missed",
                    "missed_reason": "previous_endpoint_query_still_in_flight",
                    "gpu_uuid": self.source.identity.uuid,
                    "pci_bus_id": self.source.identity.pci_bus_id,
                    "power_w": None,
                    "total_energy_mj": None,
                    "measurement_source": (
                        "hardware_counter"
                        if self.source.capabilities.total_energy_supported
                        else (
                            "power_integral_estimate"
                            if (self.source.capabilities.total_energy_error or {}).get("not_supported")
                            else "invalid_energy_counter_probe_error"
                        )
                    ),
                    "power_query_source": getattr(
                        self.source.capabilities, "power_query_source", None
                    ),
                    "capability_total_energy": self.source.capabilities.total_energy_supported,
                    "capability_power": self.source.capabilities.power_supported,
                    "error_type": None,
                    "error_message": None,
                })
                schedule_index += 1
        return sequence


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def summarize_window(
    endpoint: str,
    records: Sequence[Mapping[str, Any]],
    window: WindowSpec,
    capability: Mapping[str, Any],
    *,
    max_gap_s: float,
    min_coverage_ratio: float,
    max_boundary_gap_s: float,
) -> dict[str, Any]:
    """Summarize a wall-selected window using local monotonic measurement time."""

    all_slots = [
        item for item in records
        if item.get("actual_start_wall_s") is None
        or window.start_wall_s <= float(item["actual_start_wall_s"]) <= window.end_wall_s
    ]
    successful = [
        item for item in records
        if item.get("status") == "success"
        and _finite_number(item.get("actual_start_wall_s"))
        and window.start_wall_s <= float(item["actual_start_wall_s"]) <= window.end_wall_s
    ]
    # Missed records have no wall time, so select them by neighboring scheduled
    # local times only after establishing the successful local-time interval.
    if successful:
        local_lo = float(successful[0]["scheduled_local_monotonic_s"])
        local_hi = float(successful[-1]["scheduled_local_monotonic_s"])
        all_slots = [
            item for item in records
            if _finite_number(item.get("scheduled_local_monotonic_s"))
            and local_lo <= float(item["scheduled_local_monotonic_s"]) <= local_hi
        ]
    else:
        all_slots = []
    reasons: list[str] = []
    requested_duration = window.end_wall_s - window.start_wall_s
    if requested_duration <= 0:
        reasons.append("nonpositive_window_duration")
    identities = {
        (str(item.get("gpu_uuid")), str(item.get("pci_bus_id"))) for item in successful
    }
    expected_identity = capability["identity"]
    expected_pair = (
        str(expected_identity["uuid"]), str(expected_identity["pci_bus_id"])
    )
    if identities and identities != {expected_pair}:
        reasons.append("gpu_identity_changed_or_mismatched")
    if len(successful) < 2:
        reasons.append("fewer_than_two_successful_boundary_samples")

    first = successful[0] if successful else None
    last = successful[-1] if successful else None
    boundary_start_gap = (
        float(first["actual_start_wall_s"]) - window.start_wall_s if first else None
    )
    boundary_end_gap = (
        window.end_wall_s - float(last["actual_start_wall_s"]) if last else None
    )
    if boundary_start_gap is not None and boundary_start_gap > max_boundary_gap_s:
        reasons.append("start_boundary_gap_exceeds_limit")
    if boundary_end_gap is not None and boundary_end_gap > max_boundary_gap_s:
        reasons.append("end_boundary_gap_exceeds_limit")

    monotonic = [float(item["actual_start_local_monotonic_s"]) for item in successful]
    gaps = [b - a for a, b in zip(monotonic, monotonic[1:])]
    max_gap = max(gaps, default=None)
    duration = monotonic[-1] - monotonic[0] if len(monotonic) >= 2 else None
    coverage = (
        min(1.0, max(0.0, duration / requested_duration))
        if duration is not None and requested_duration > 0 else 0.0
    )
    if max_gap is not None and max_gap > max_gap_s:
        reasons.append("sampling_gap_exceeds_limit")
    if coverage < min_coverage_ratio:
        reasons.append("coverage_below_minimum")

    powers = [float(item["power_w"]) for item in successful if _finite_number(item.get("power_w")) and float(item["power_w"]) >= 0]
    power_by_time = [
        (float(item["actual_start_local_monotonic_s"]), float(item["power_w"]))
        for item in successful
        if _finite_number(item.get("power_w")) and float(item["power_w"]) >= 0
    ]
    energy_supported = bool(capability["capabilities"]["total_energy_supported"])
    method = "hardware_counter" if energy_supported else "power_integral_estimate"
    gross_energy: Optional[float] = None
    if energy_supported:
        counter_errors = [
            item for item in successful
            if item.get("field_errors", {}).get("total_energy_mj")
            or not _finite_number(item.get("total_energy_mj"))
        ]
        counters = [float(item["total_energy_mj"]) for item in successful if _finite_number(item.get("total_energy_mj"))]
        if counter_errors:
            reasons.append("energy_counter_error_in_window")
        if any(b < a for a, b in zip(counters, counters[1:])):
            reasons.append("energy_counter_decreased_or_reset")
        if len(counters) == len(successful) and len(counters) >= 2:
            gross_energy = (counters[-1] - counters[0]) / 1000.0
    else:
        energy_error = capability["capabilities"].get("total_energy_error") or {}
        if not energy_error.get("not_supported"):
            reasons.append("energy_counter_unavailable_but_not_explicitly_unsupported")
        if not bool(capability["capabilities"]["power_supported"]):
            reasons.append("power_fallback_unsupported")
        if len(power_by_time) != len(successful):
            reasons.append("missing_or_invalid_power_sample")
        if len(power_by_time) >= 2:
            gross_energy = sum(
                0.5 * (p0 + p1) * (t1 - t0)
                for (t0, p0), (t1, p1) in zip(power_by_time, power_by_time[1:])
            )

    errors = sum(item.get("status") == "error" for item in all_slots)
    missed = sum(item.get("status") == "missed" for item in all_slots)
    late_values = [item for item in all_slots if item.get("late") is True]
    if errors:
        reasons.append("sample_errors_present")
    valid = not reasons and gross_energy is not None
    request_count = window.logical_requests
    output_tokens = window.output_tokens
    drift_values = [
        float(item["start_drift_s"])
        for item in successful if _finite_number(item.get("start_drift_s"))
    ]
    result = {
        "endpoint": endpoint,
        "window": window.name,
        "included_in_reported_workload": window.included_in_reported_workload,
        "requested_boundary_wall_s": {"start": window.start_wall_s, "end": window.end_wall_s},
        "actual_measurement_boundary_wall_s": {
            "start": first.get("actual_start_wall_s") if first else None,
            "end": last.get("actual_start_wall_s") if last else None,
        },
        "actual_measurement_boundary_local_monotonic_s": {
            "start": monotonic[0] if monotonic else None,
            "end": monotonic[-1] if monotonic else None,
        },
        "requested_duration_s": requested_duration,
        "duration_s": duration,
        "method": method,
        "gross_gpu_energy_j": gross_energy if valid else None,
        "unaccepted_candidate_energy_j": gross_energy if not valid else None,
        "mean_power_w": sum(powers) / len(powers) if powers else None,
        "min_power_w": min(powers) if powers else None,
        "max_power_w": max(powers) if powers else None,
        "sample_counts": {
            "scheduled_slots": len(all_slots),
            "successful_slots": len(successful),
            "missed_slots": missed,
            "late_slots": len(late_values),
            "error_slots": errors,
        },
        "drift_s": {
            "maximum": max(drift_values, default=None),
            "mean": sum(drift_values) / len(drift_values) if drift_values else None,
        },
        "maximum_sampling_gap_s": max_gap,
        "coverage_ratio": coverage,
        "boundary_gap_s": {"start": boundary_start_gap, "end": boundary_end_gap},
        "valid": valid,
        "invalidity_reasons": sorted(set(reasons)),
        "logical_request_count": request_count,
        "input_tokens": window.input_tokens,
        "output_tokens": output_tokens,
        "energy_j_per_request": (
            gross_energy / request_count if valid and request_count is not None and request_count > 0 else None
        ),
        "energy_j_per_output_token": (
            gross_energy / output_tokens if valid and output_tokens is not None and output_tokens > 0 else None
        ),
    }
    return result


def add_idle_adjustment(workload: dict[str, Any], idle: Mapping[str, Any]) -> None:
    workload["idle_adjusted_incremental_estimate_j"] = None
    workload["idle_adjustment_reason"] = "idle_or_workload_window_invalid_or_incomparable"
    if not workload.get("valid") or not idle.get("valid"):
        return
    if workload.get("endpoint") != idle.get("endpoint"):
        return
    duration = workload.get("duration_s")
    mean_idle = idle.get("mean_power_w")
    gross = workload.get("gross_gpu_energy_j")
    if all(_finite_number(value) for value in (duration, mean_idle, gross)):
        workload["idle_adjusted_incremental_estimate_j"] = (
            float(gross) - float(mean_idle) * float(duration)
        )
        workload["idle_adjustment_reason"] = None


def aggregate_endpoint_workload(
    windows: Sequence[Mapping[str, Any]],
    *,
    logical_requests: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Limited P0+D0 GPU-only sum without double-counting logical requests."""

    selected = [item for item in windows if item["window"] == "workload"]
    by_endpoint = {str(item["endpoint"]): item for item in selected}
    valid = set(by_endpoint) == {"P0", "D0"} and all(
        item.get("valid") for item in by_endpoint.values()
    )
    energy = (
        sum(float(item["gross_gpu_energy_j"]) for item in by_endpoint.values())
        if valid else None
    )
    return {
        "scope": "P0_plus_D0_GPU_boards_only",
        "valid": valid,
        "invalidity_reasons": [] if valid else ["both_endpoint_workload_windows_must_be_valid"],
        "gross_gpu_energy_j": energy,
        "logical_request_count": logical_requests,
        "output_tokens": output_tokens,
        "energy_j_per_request": energy / logical_requests if energy is not None and logical_requests > 0 else None,
        "energy_j_per_output_token": energy / output_tokens if energy is not None and output_tokens > 0 else None,
        "not_total_system_energy": True,
    }


def _monitor_command(args: argparse.Namespace) -> int:
    output = Path(args.output)
    capability_path = Path(args.capability_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or capability_path.exists():
        raise Phase3BEnergyError("monitor outputs must not already exist")
    source = ReadOnlyNVMLSource(
        args.endpoint,
        args.role,
        configured_cuda_device(args.cuda_visible_device),
        expected_uuid=args.expected_uuid,
        expected_pci_bus_id=args.expected_pci_bus_id,
        expected_gpu_name=args.expected_gpu_name,
    )
    sampler = FixedPeriodEnergySampler(
        args.endpoint, source, args.period_s, args.late_tolerance_s
    )

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        sampler.stop()

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)
    watcher_stop = threading.Event()

    def watch_stop_request() -> None:
        stop_path = Path(args.stop_file) if args.stop_file else None
        stop_stream = sys.stdin if args.stop_stdin else None
        while not watcher_stop.wait(min(0.05, args.period_s / 4.0)):
            if stop_path is not None and stop_path.exists():
                sampler.stop()
                return
            if stop_stream is not None:
                readable, _, _ = select.select([stop_stream], [], [], 0.0)
                if readable:
                    line = stop_stream.readline()
                    if line:
                        sampler.stop()
                        return
                    stop_stream = None

    watcher = threading.Thread(
        target=watch_stop_request, name="xpyd-energy-stop-watcher", daemon=True
    )
    watcher.start()
    try:
        _write_json(capability_path, source.capability_record())
        # Explicit batch flushes above provide visibility; line buffering here
        # would turn one batch back into one shared-filesystem flush per line.
        with output.open("x", encoding="utf-8") as raw_stream:
            stream = AsyncFlushingStream(raw_stream)
            try:
                sampler.run(stream, duration_s=args.duration_s)
            finally:
                stream.close()
    finally:
        watcher_stop.set()
        watcher.join()
        source.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


def _load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    start = value.get("timing_start_unix_s")
    end = value.get("timing_end_unix_s")
    if not _finite_number(start) or not _finite_number(end) or float(end) <= float(start):
        raise Phase3BEnergyError("client summary has invalid timing boundaries: %s" % path)
    return value


def _phase3a_window_specs(run_dir: Path, boundaries: Mapping[str, float]) -> list[WindowSpec]:
    warmup = _load_summary(run_dir / "client" / "_phase_warmup" / "summary.json")
    semantic_dirs = sorted(
        path for path in (run_dir / "client").iterdir()
        if path.is_dir() and path.name.startswith("semantic_")
    )
    load_dirs = sorted(
        path for path in (run_dir / "client").iterdir()
        if path.is_dir() and path.name.startswith("load_")
    )
    if len(semantic_dirs) != 1 or len(load_dirs) != 1:
        raise Phase3BEnergyError("preflight requires exactly one semantic and one load client artifact")
    semantic = _load_summary(semantic_dirs[0] / "summary.json")
    load = _load_summary(load_dirs[0] / "summary.json")

    def spec(name: str, summary: Mapping[str, Any], include: bool) -> WindowSpec:
        return WindowSpec(
            name,
            float(summary["timing_start_unix_s"]),
            float(summary["timing_end_unix_s"]),
            include,
            int(summary["successful_requests"]),
            int(summary["input_tokens_total"]),
            int(summary["output_tokens_total"]),
        )

    return [
        WindowSpec("capability_preflight", boundaries["capability_start"], boundaries["capability_end"], False),
        WindowSpec("idle", boundaries["idle_start"], boundaries["idle_end"], False),
        spec("warmup", warmup, False),
        spec("semantic", semantic, False),
        spec("workload", load, True),
        WindowSpec("cooldown", boundaries["cooldown_start"], boundaries["cooldown_end"], False),
    ]


def _request_token_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations = []
    checks = []
    for record in records:
        client = record["client"]
        requests = int(client["successful_requests"])
        expected = {
            "P0": {
                "delta_completed_requests": requests,
                "delta_prompt_tokens": int(client["input_tokens_total"]) + requests,
                "delta_generation_tokens": requests,
            },
            "D0": {
                "delta_completed_requests": requests,
                "delta_prompt_tokens": int(client["input_tokens_total"]) + requests,
                "delta_generation_tokens": int(client["output_tokens_total"]),
            },
        }
        for endpoint in ("P0", "D0"):
            window = record["endpoints"][endpoint]["window"]
            for field, expected_value in expected[endpoint].items():
                observed = window.get(field)
                valid = bool(window.get("valid")) and observed == expected_value
                checks.append({
                    "probe_id": record["probe"]["id"],
                    "endpoint": endpoint,
                    "field": field,
                    "observed": observed,
                    "expected": expected_value,
                    "valid": valid,
                })
                if not valid:
                    violations.append("%s %s %s=%r expected %r" % (
                        record["probe"]["id"], endpoint, field, observed, expected_value
                    ))
    return {
        "valid": not violations,
        "violations": violations,
        "checks": checks,
        "logical_request_accounting": "client count only; P0 and D0 counts are not added",
    }


def analyze_run(
    run_dir: Path,
    phase3a_run_dir: Path,
    boundaries: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    sampling = config["sampling"]
    window_specs = _phase3a_window_specs(phase3a_run_dir, boundaries)
    summaries: list[dict[str, Any]] = []
    capabilities = {}
    records_by_endpoint = {}
    for endpoint in config["endpoints"]:
        endpoint_id = str(endpoint["endpoint_id"])
        endpoint_dir = run_dir / endpoint_id
        capability = json.loads((endpoint_dir / "capability.json").read_text(encoding="utf-8"))
        records = _read_jsonl(endpoint_dir / "samples.jsonl")
        if not records:
            raise Phase3BEnergyError("%s emitted no energy samples" % endpoint_id)
        capabilities[endpoint_id] = capability
        records_by_endpoint[endpoint_id] = records
        for spec in window_specs:
            summaries.append(summarize_window(
                endpoint_id,
                records,
                spec,
                capability,
                max_gap_s=float(sampling["maximum_gap_s"]),
                min_coverage_ratio=float(sampling["minimum_coverage_ratio"]),
                max_boundary_gap_s=float(sampling["maximum_boundary_gap_s"]),
            ))

    for endpoint_id in ("P0", "D0"):
        idle = next(item for item in summaries if item["endpoint"] == endpoint_id and item["window"] == "idle")
        for item in summaries:
            if item["endpoint"] == endpoint_id and item["window"] in ("semantic", "workload"):
                add_idle_adjustment(item, idle)

    load_summary = next(
        _load_summary(path / "summary.json")
        for path in sorted((phase3a_run_dir / "client").iterdir())
        if path.is_dir() and path.name.startswith("load_")
    )
    aggregate = aggregate_endpoint_workload(
        summaries,
        logical_requests=int(load_summary["successful_requests"]),
        output_tokens=int(load_summary["output_tokens_total"]),
    )
    phase3a_summary = json.loads(
        (phase3a_run_dir / "derived" / "summary.json").read_text(encoding="utf-8")
    )
    phase3a_semantic = json.loads(
        (phase3a_run_dir / "derived" / "semantic_deltas.json").read_text(encoding="utf-8")
    )
    phase3a_load = json.loads(
        (phase3a_run_dir / "derived" / "load_runs.json").read_text(encoding="utf-8")
    )
    request_token_audit = _request_token_audit(phase3a_semantic + phase3a_load)
    diagnostics = _read_jsonl(phase3a_run_dir / "derived" / "proxy_diagnostics.jsonl")
    shared_transport_ids_valid = all(
        str(item.get("logical_request_id") or item.get("request_id") or "")
        and str(item.get("vllm_request_id") or "").endswith(
            str(item.get("logical_request_id") or item.get("request_id") or "")
        )
        for item in diagnostics
    )
    proxy_audit = phase3a_summary.get("proxy_diagnostics_audit") or {}
    workload_windows = [item for item in summaries if item["window"] == "workload"]
    expected_names = {
        str(item["endpoint_id"]): str(item["expected_gpu_name"]).lower()
        for item in config["endpoints"]
    }
    identity_valid = all(
        expected_names[endpoint_id] in str(cap["identity"]["gpu_name"]).lower()
        for endpoint_id, cap in capabilities.items()
    )
    phase3a_valid = bool(
        phase3a_summary.get("completed")
        and proxy_audit.get("valid")
        and proxy_audit.get("logical_request_ids_exactly_match")
        and shared_transport_ids_valid
        and request_token_audit["valid"]
        and all(
            record["endpoints"][endpoint]["window"]["valid"]
            for record in phase3a_semantic + phase3a_load
            for endpoint in ("P0", "D0")
        )
    )
    monitor_errors = {
        endpoint: sum(
            item.get("status") == "error"
            or bool((item.get("field_errors") or {}).get("power_w"))
            or bool((item.get("field_errors") or {}).get("identity"))
            or (
                bool(capabilities[endpoint]["capabilities"]["total_energy_supported"])
                and bool((item.get("field_errors") or {}).get("total_energy_mj"))
            )
            for item in records
        )
        for endpoint, records in records_by_endpoint.items()
    }
    monitor_misses = {
        endpoint: sum(item.get("status") == "missed" for item in records)
        for endpoint, records in records_by_endpoint.items()
    }
    violations = []
    if not identity_valid:
        violations.append("endpoint_to_GPU_identity_mapping_invalid")
    if not phase3a_valid:
        violations.append("Phase_3A_request_proxy_or_metric_audit_invalid")
    if not all(item["valid"] for item in workload_windows):
        violations.append("endpoint_workload_energy_window_invalid")
    if any(monitor_errors.values()):
        violations.append("monitor_sample_errors_present")
    if any(monitor_misses.values()):
        violations.append("monitor_missed_slots_present")
    audit = {
        "valid": not violations,
        "violations": violations,
        "identity_mapping_valid": identity_valid,
        "phase3a_audit_valid": phase3a_valid,
        "logical_request_ids_exactly_match": proxy_audit.get("logical_request_ids_exactly_match"),
        "shared_logical_id_in_vllm_transport_envelope": shared_transport_ids_valid,
        "request_token_audit": request_token_audit,
        "monitor_error_counts": monitor_errors,
        "monitor_missed_slot_counts": monitor_misses,
        "no_gpu_state_mutation": True,
        "no_gpu_state_mutation_evidence": "Phase 3B imports only xpyd.nvml_readonly; source guard forbids NVML setters and state-changing nvidia-smi flags.",
    }
    _write_json(run_dir / "window_summaries.json", summaries)
    _write_json(run_dir / "coverage_audit.json", audit)
    _write_json(run_dir / "phase3a_reference.json", {
        "run_directory": phase3a_run_dir.as_posix(),
        "summary_path": (phase3a_run_dir / "derived" / "summary.json").as_posix(),
        "proxy_diagnostics_audit": proxy_audit,
        "shared_logical_id_in_vllm_transport_envelope": shared_transport_ids_valid,
        "request_token_audit": request_token_audit,
        "semantic_and_load_metric_windows_valid": phase3a_valid,
        "logical_request_count_source": "client summaries, never P0+D0 completion sum",
    })
    _write_json(run_dir / "aggregate.json", aggregate)
    write_report(run_dir, capabilities, summaries, aggregate, audit)
    return audit


def write_report(
    run_dir: Path,
    capabilities: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    lines = [
        "# XpYd Phase 3B minimal energy preflight",
        "",
        "Verdict: **%s**" % ("PASS" if audit["valid"] else "FAIL"),
        "",
        "GPU-board telemetry only; this is not full-system energy and not a performance or energy comparison.",
        "",
        "| Endpoint | GPU | UUID | PCI | Counter | Power |",
        "|---|---|---|---|---:|---:|",
    ]
    for endpoint in ("P0", "D0"):
        cap = capabilities[endpoint]
        identity = cap["identity"]
        abilities = cap["capabilities"]
        lines.append("| %s | %s | `%s` | `%s` | %s | %s |" % (
            endpoint, identity["gpu_name"], identity["uuid"], identity["pci_bus_id"],
            abilities["total_energy_supported"], abilities["power_supported"],
        ))
    lines.extend(["", "| Endpoint | Window | Method | Valid | Energy (J) | Coverage | Missed | Late |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for item in windows:
        counts = item["sample_counts"]
        lines.append("| %s | %s | %s | %s | %s | %.4f | %d | %d |" % (
            item["endpoint"], item["window"], item["method"], item["valid"],
            item["gross_gpu_energy_j"], item["coverage_ratio"], counts["missed_slots"], counts["late_slots"],
        ))
    lines.extend([
        "",
        "Limited P0+D0 workload GPU-board sum: `%s J`; logical requests are counted once (`%s`)." %
        (aggregate.get("gross_gpu_energy_j"), aggregate.get("logical_request_count")),
        "",
        "Request normalization is aggregate window normalization, not exact causal per-request attribution. CPU/node, NIC/network, KV-transfer-specific, cooling, and facility energy are unobserved.",
        "",
    ])
    (run_dir / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"output_root", "phase3a_config", "sampling", "windows", "endpoints"}
    missing = required.difference(config)
    if missing:
        raise Phase3BEnergyError("Phase 3B config missing fields: %s" % sorted(missing))
    endpoints = config["endpoints"]
    if {item.get("endpoint_id") for item in endpoints} != {"P0", "D0"}:
        raise Phase3BEnergyError("Phase 3B requires exactly P0 and D0 endpoints")
    return config


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _monitor_argv(
    endpoint: Mapping[str, Any],
    endpoint_dir: Path,
    sampling: Mapping[str, Any],
) -> list[str]:
    command = [
        sys.executable, "-m", "xpyd.phase3b_energy", "monitor",
        "--endpoint", str(endpoint["endpoint_id"]),
        "--role", str(endpoint["role"]),
        "--cuda-visible-device", str(endpoint["cuda_visible_device"]),
        "--expected-gpu-name", str(endpoint["expected_gpu_name"]),
        "--period-s", str(sampling["period_s"]),
        "--late-tolerance-s", str(sampling["late_tolerance_s"]),
        "--output", (endpoint_dir / "samples.jsonl").as_posix(),
        "--capability-output", (endpoint_dir / "capability.json").as_posix(),
        "--stop-file", (endpoint_dir / "stop.requested").as_posix(),
        "--stop-stdin",
    ]
    if endpoint.get("expected_uuid"):
        command += ["--expected-uuid", str(endpoint["expected_uuid"])]
    if endpoint.get("expected_pci_bus_id"):
        command += ["--expected-pci-bus-id", str(endpoint["expected_pci_bus_id"])]
    return command


def _launch_monitor(
    endpoint: Mapping[str, Any],
    endpoint_dir: Path,
    sampling: Mapping[str, Any],
) -> tuple[subprocess.Popen[Any], Any]:
    command = _monitor_argv(endpoint, endpoint_dir, sampling)
    local_host = socket.gethostname().split(".")[0]
    node = str(endpoint["node"])
    if node != local_host:
        physical_gpu = str(endpoint["cuda_visible_device"])
        if not physical_gpu.isdigit():
            raise Phase3BEnergyError(
                "remote monitor gpu-bind requires a numeric physical GPU index"
            )
        allocation_gpu_count = int(endpoint.get("monitor_gpu_allocation_count", 1))
        gpu_bind = (
            "none" if allocation_gpu_count > 1
            else "map_gpu:%s" % physical_gpu
        )
        command = [
            "srun", "--overlap", "--nodes=1", "--ntasks=1",
            "--nodelist=%s" % node,
            "--gpus-per-node=%d" % allocation_gpu_count,
            "--gpu-bind=%s" % gpu_bind,
            *command,
        ]
    log = (endpoint_dir / "monitor.log").open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=log,
            stderr=subprocess.STDOUT, text=True,
        )
    except Exception:
        log.close()
        raise
    return process, log


def _wait_capabilities(
    processes: Mapping[str, subprocess.Popen[Any]],
    run_dir: Path,
    timeout_s: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(processes)
    while pending and time.monotonic() < deadline:
        for endpoint in tuple(pending):
            process = processes[endpoint]
            capability = run_dir / endpoint / "capability.json"
            samples = run_dir / endpoint / "samples.jsonl"
            if capability.is_file() and capability.stat().st_size and samples.is_file() and samples.stat().st_size:
                pending.remove(endpoint)
            elif process.poll() is not None:
                raise Phase3BEnergyError("%s monitor exited before capability/sample readiness" % endpoint)
        if pending:
            time.sleep(0.1)
    if pending:
        raise Phase3BEnergyError("monitor readiness timeout: %s" % sorted(pending))


def _stop_monitors(
    processes: Mapping[str, subprocess.Popen[Any]],
    logs: Iterable[Any],
    stop_files: Mapping[str, Path],
    *,
    marker_grace_s: float = 2.0,
    shutdown_timeout_s: float = 45.0,
) -> list[str]:
    errors = []
    for endpoint, process in processes.items():
        if process.poll() is None:
            control = process.stdin
            if control is not None:
                try:
                    control.write("stop\n")
                    control.flush()
                    control.close()
                except (BrokenPipeError, OSError):
                    pass
            stop_files[endpoint].write_text("stop\n", encoding="utf-8")
    # A stop file is cheap and sufficient on the local node.  On Minerva the
    # cross-node shared-filesystem attribute cache can hide a newly-created
    # marker for longer than the shutdown timeout, so stdin is the primary
    # cross-node control path.  srun forwards it to the one remote task.
    marker_deadline = time.monotonic() + marker_grace_s
    while any(process.poll() is None for process in processes.values()):
        if time.monotonic() >= marker_deadline:
            break
        time.sleep(0.05)
    # Leave enough time for the asynchronous writer to drain a short shared-FS
    # backlog while remaining bounded if a remote step is genuinely stuck.
    deadline = time.monotonic() + shutdown_timeout_s
    while any(process.poll() is None for process in processes.values()):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    for endpoint, process in processes.items():
        required_signal = False
        if process.poll() is None:
            required_signal = True
            process.terminate()
        try:
            returncode = process.wait(timeout=5)
            if required_signal:
                errors.append("%s monitor ignored graceful stdin/marker stop" % endpoint)
            elif returncode != 0:
                errors.append("%s monitor exit status %s" % (endpoint, returncode))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            errors.append("%s monitor required SIGKILL" % endpoint)
    for log in logs:
        log.close()
    return errors


def run_preflight(config_path: Path, run_id: Optional[str] = None) -> Path:
    config = _load_config(config_path)
    run_id = run_id or _utc_run_id()
    run_dir = Path(os.path.expandvars(str(config["output_root"]))) / run_id
    if run_dir.exists():
        raise Phase3BEnergyError("non-overwriting run directory already exists: %s" % run_dir)
    for relative in ("P0", "D0"):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "metadata.json", {
        "schema_version": 1,
        "phase": "3B_read_only_energy_preflight",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": config,
        "measurement_scope": "P0_and_D0_GPU_boards_only",
        "actuated": False,
        "claim_boundary": "functionality_and_data_integrity_preflight_not_comparison",
    })

    processes: dict[str, subprocess.Popen[Any]] = {}
    stop_files: dict[str, Path] = {}
    logs = []
    boundaries: dict[str, float] = {}
    phase3a_run_dir: Optional[Path] = None
    failure: Optional[BaseException] = None
    stop_errors: list[str] = []
    try:
        for endpoint in config["endpoints"]:
            endpoint_id = str(endpoint["endpoint_id"])
            process, log = _launch_monitor(
                endpoint, run_dir / endpoint_id, config["sampling"]
            )
            processes[endpoint_id] = process
            stop_files[endpoint_id] = run_dir / endpoint_id / "stop.requested"
            logs.append(log)
        _wait_capabilities(
            processes,
            run_dir,
            timeout_s=float(config["sampling"].get("readiness_timeout_s", 30.0)),
        )

        capability_duration = float(config["windows"].get("capability_s", 1.0))
        boundaries["capability_start"] = time.time()
        time.sleep(capability_duration)
        boundaries["capability_end"] = time.time()
        boundaries["idle_start"] = time.time()
        time.sleep(float(config["windows"]["idle_s"]))
        boundaries["idle_end"] = time.time()

        phase3a_config = load_phase3a_config(Path(config["phase3a_config"]))
        phase3a_run_id = "%s-phase3a" % run_id
        phase3a_run_dir = Phase3AHarness(phase3a_config, run_id=phase3a_run_id).run(
            ("semantic", "load")
        )
        boundaries["cooldown_start"] = time.time()
        time.sleep(float(config["windows"]["cooldown_s"]))
        boundaries["cooldown_end"] = time.time()
    except BaseException as exc:
        failure = exc
    finally:
        stop_errors = _stop_monitors(processes, logs, stop_files)
        _write_json(run_dir / "boundaries.json", boundaries)

    if failure is not None:
        _write_json(run_dir / "failure.json", {
            "error": structured_error(failure),
            "monitor_stop_errors": stop_errors,
            "actuated": False,
        })
        raise failure
    if stop_errors:
        raise Phase3BEnergyError("; ".join(stop_errors))
    if phase3a_run_dir is None:
        raise AssertionError("Phase 3A run directory was not assigned")
    audit = analyze_run(run_dir, phase3a_run_dir, boundaries, config)
    if not audit["valid"]:
        raise Phase3BEnergyError(
            "Phase 3B preflight audit failed: %s" % "; ".join(audit["violations"])
        )
    print(run_dir.as_posix())
    return run_dir


def _analyze_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    config = _load_config(Path(args.config))
    boundaries = json.loads((run_dir / "boundaries.json").read_text(encoding="utf-8"))
    phase3a_reference = json.loads(
        (run_dir / "phase3a_reference.json").read_text(encoding="utf-8")
    ) if (run_dir / "phase3a_reference.json").exists() else None
    phase3a_run_dir = Path(args.phase3a_run_dir) if args.phase3a_run_dir else (
        Path(phase3a_reference["run_directory"]) if phase3a_reference else None
    )
    if phase3a_run_dir is None:
        raise Phase3BEnergyError("--phase3a-run-dir is required on first analysis")
    audit = analyze_run(run_dir, phase3a_run_dir, boundaries, config)
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    monitor = subparsers.add_parser("monitor", help="run one node-local read-only monitor")
    # Endpoint identity is generic in ReadOnlyNVMLSource. Phase 3C reuses the
    # accepted monitor for P1/D1 rather than creating a parallel sampler.
    monitor.add_argument("--endpoint", required=True)
    monitor.add_argument("--role", choices=("prefill", "decode"), required=True)
    monitor.add_argument("--cuda-visible-device")
    monitor.add_argument("--expected-uuid")
    monitor.add_argument("--expected-pci-bus-id")
    monitor.add_argument("--expected-gpu-name")
    monitor.add_argument("--period-s", type=float, default=0.2)
    monitor.add_argument("--late-tolerance-s", type=float, default=0.05)
    monitor.add_argument("--duration-s", type=float)
    monitor.add_argument("--output", required=True)
    monitor.add_argument("--capability-output", required=True)
    monitor.add_argument("--stop-file")
    monitor.add_argument("--stop-stdin", action="store_true")
    monitor.set_defaults(handler=_monitor_command)

    preflight = subparsers.add_parser("preflight", help="run the one minimal physical preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--run-id")
    preflight.set_defaults(handler=lambda args: (run_preflight(Path(args.config), args.run_id), 0)[1])

    analyze = subparsers.add_parser("analyze", help="rebuild compact summaries from saved raw samples")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--phase3a-run-dir")
    analyze.set_defaults(handler=_analyze_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
