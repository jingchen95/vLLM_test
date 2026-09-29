"""Stage C2 dynamic shared-pool DVFS with event-driven fast safety.

The efficiency/search path is the unchanged C1c adaptive state machine.  C2
adds only a higher-priority safety path that can restore a reduced-frequency
pool inside an accounting window.  Routing remains fixed four-pair RR and the
controller is never given workload segment labels or transition timestamps.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from xpyd.adaptive_exploration import (
    AdaptiveExplorationConfig,
    AdaptivePoolController,
    EvidenceClass,
    evidence_from_window,
)
from xpyd.phase3b_energy import _read_jsonl
from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    load_config as load_runtime_config,
)
from xpyd.phase3d_control import (
    EndpointClockCapability,
    NvidiaSmiClockBackend,
    PerEndpointClockActuator,
    Phase3DClosedLoopHarness,
)
from xpyd.stagec0_capacity import analyze_point, percentile
from xpyd.stagec1_stationary import (
    ControllerContext,
    POOLS,
    _read_csv,
    _trace_signature,
    _write_csv,
)
from xpyd.workload_harness import build_trace, load_config as load_workload_config


class StageC2Error(RuntimeError):
    """A C2 semantic, runtime, or structural hard gate failed."""


class RecoveredActuationFailure(StageC2Error):
    """A requested transition failed but the affected pool was restored safely."""

    def __init__(self, details: Mapping[str, Any]) -> None:
        super().__init__("actuation/readback failed; pool recovered to validated-safe anchor")
        self.details = dict(details)


POLICIES = (
    "STATIC-HIGH",
    "WINDOW-BOUNDED-FEEDBACK",
    "HYBRID-FAST-SAFETY",
)


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _pool(endpoint_id: str) -> str:
    return "P" if str(endpoint_id).startswith("P") else "D"


def policy_order(repeat: int) -> tuple[str, ...]:
    offset = (int(repeat) - 1) % len(POLICIES)
    return POLICIES[offset:] + POLICIES[:offset]


def derived_context_signature(
    pool: str, requests: Sequence[Mapping[str, Any]], settings: Mapping[str, Any],
) -> str:
    """Derive context from observed tokens; never consume segment labels."""
    if not requests:
        return "%s:unknown" % pool
    if pool == "P":
        boundary = int(settings["context_inference"]["long_input_tokens"])
        fraction = sum(int(row["input_len"]) > boundary for row in requests) / len(requests)
        threshold = float(settings["context_inference"]["prefill_heavy_fraction"])
        return "P:long-input-high" if fraction >= threshold else "P:mixed-input"
    boundary = int(settings["context_inference"]["long_output_tokens"])
    fraction = sum(int(row["requested_output_len"]) > boundary for row in requests) / len(requests)
    threshold = float(settings["context_inference"]["decode_heavy_fraction"])
    return "D:long-output-high" if fraction >= threshold else "D:mixed-output"


class ClockSegmentLedger:
    """Record stable and transition intervals without inventing one window clock."""

    def __init__(self, window_id: str) -> None:
        self.window_id = window_id
        self.rows: list[dict[str, Any]] = []
        self.active: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.window_start_wall_s: Optional[float] = None
        self.window_end_wall_s: Optional[float] = None

    def start(self, frequencies: Mapping[str, int], timestamp_s: float) -> None:
        with self.lock:
            self.window_start_wall_s = float(timestamp_s)
            for endpoint_id, frequency in frequencies.items():
                self.active[endpoint_id] = {
                    "window_id": self.window_id,
                    "endpoint_id": endpoint_id,
                    "pool": _pool(endpoint_id),
                    "segment_kind": "STABLE",
                    "segment_start_wall_s": float(timestamp_s),
                    "requested_frequency_mhz": int(frequency),
                    "observed_readback_frequency_mhz": int(frequency),
                    "reason": "WINDOW_START",
                    "decision_timestamp_s": None,
                    "action_timestamp_s": None,
                    "readback_timestamp_s": float(timestamp_s),
                }

    def transition(
        self, endpoint_id: str, target_mhz: int, *, reason: str,
        decision_timestamp_s: float, action_timestamp_s: float,
        readback_timestamp_s: float, observed_readback_mhz: int,
    ) -> None:
        if readback_timestamp_s < action_timestamp_s:
            raise StageC2Error("readback timestamp precedes physical action")
        with self.lock:
            prior = self.active.pop(endpoint_id)
            prior["segment_end_wall_s"] = float(action_timestamp_s)
            self.rows.append(prior)
            self.rows.append({
                "window_id": self.window_id,
                "endpoint_id": endpoint_id,
                "pool": _pool(endpoint_id),
                "segment_kind": "TRANSITION",
                "segment_start_wall_s": float(action_timestamp_s),
                "segment_end_wall_s": float(readback_timestamp_s),
                "requested_frequency_mhz": int(target_mhz),
                "observed_readback_frequency_mhz": int(observed_readback_mhz),
                "reason": reason,
                "decision_timestamp_s": float(decision_timestamp_s),
                "action_timestamp_s": float(action_timestamp_s),
                "readback_timestamp_s": float(readback_timestamp_s),
            })
            self.active[endpoint_id] = {
                "window_id": self.window_id,
                "endpoint_id": endpoint_id,
                "pool": _pool(endpoint_id),
                "segment_kind": "STABLE",
                "segment_start_wall_s": float(readback_timestamp_s),
                "requested_frequency_mhz": int(target_mhz),
                "observed_readback_frequency_mhz": int(observed_readback_mhz),
                "reason": reason,
                "decision_timestamp_s": float(decision_timestamp_s),
                "action_timestamp_s": float(action_timestamp_s),
                "readback_timestamp_s": float(readback_timestamp_s),
            }

    def finish(self, start_wall_s: float, end_wall_s: float) -> None:
        with self.lock:
            self.window_start_wall_s = float(start_wall_s)
            self.window_end_wall_s = float(end_wall_s)
            for endpoint_id in list(self.active):
                row = self.active.pop(endpoint_id)
                row["segment_end_wall_s"] = float(end_wall_s)
                self.rows.append(row)
            clipped = []
            for row in self.rows:
                start = max(float(row["segment_start_wall_s"]), float(start_wall_s))
                end = min(float(row["segment_end_wall_s"]), float(end_wall_s))
                if end <= start:
                    continue
                item = dict(row)
                item["segment_start_wall_s"] = start
                item["segment_end_wall_s"] = end
                item["duration_s"] = end - start
                item["energy_interval_overlap_j"] = None
                item["energy_interval_overlap_valid"] = False
                clipped.append(item)
            self.rows = sorted(clipped, key=lambda row: (row["endpoint_id"], row["segment_start_wall_s"]))

    def pool_changed(self, pool: str) -> bool:
        return any(row["pool"] == pool and row["segment_kind"] == "TRANSITION" for row in self.rows)


def _segment_energy(records: Sequence[Mapping[str, Any]], start: float, end: float) -> Optional[float]:
    selected = [
        row for row in records
        if row.get("status") == "success" and _finite(row.get("actual_start_wall_s"))
        and start <= float(row["actual_start_wall_s"]) <= end
    ]
    counters = [float(row["total_energy_mj"]) for row in selected if _finite(row.get("total_energy_mj"))]
    if len(counters) == len(selected) and len(counters) >= 2 and all(b >= a for a, b in zip(counters, counters[1:])):
        return (counters[-1] - counters[0]) / 1000.0
    power = [
        (float(row["actual_start_local_monotonic_s"]), float(row["power_w"]))
        for row in selected if _finite(row.get("actual_start_local_monotonic_s")) and _finite(row.get("power_w"))
    ]
    if len(power) < 2:
        return None
    return sum(0.5 * (p0 + p1) * (t1 - t0) for (t0, p0), (t1, p1) in zip(power, power[1:]))


def dynamic_clock_audit(
    endpoint_id: str, records: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]], start: float, end: float,
    memory_target_mhz: int, required_fraction: float = 0.95,
) -> dict[str, Any]:
    selected = [
        row for row in records
        if row.get("status") == "success" and _finite(row.get("actual_start_wall_s"))
        and start <= float(row["actual_start_wall_s"]) <= end
    ]
    stable = [row for row in segments if row["endpoint_id"] == endpoint_id and row["segment_kind"] == "STABLE"]
    matches = 0
    eligible = 0
    segment_rows = []
    for segment in stable:
        samples = [
            row for row in selected
            if float(segment["segment_start_wall_s"]) <= float(row["actual_start_wall_s"]) <= float(segment["segment_end_wall_s"])
            and _finite(row.get("graphics_clock_mhz"))
        ]
        target = int(segment["requested_frequency_mhz"])
        count = sum(math.isclose(float(row["graphics_clock_mhz"]), target, abs_tol=0.5) for row in samples)
        matches += count
        eligible += len(samples)
        segment_rows.append({
            "start_wall_s": segment["segment_start_wall_s"],
            "end_wall_s": segment["segment_end_wall_s"],
            "target_mhz": target,
            "sample_count": len(samples),
            "target_match_count": count,
            "target_match_fraction": count / len(samples) if samples else 0.0,
        })
    graphics = [float(row["graphics_clock_mhz"]) for row in selected if _finite(row.get("graphics_clock_mhz"))]
    memory = [float(row["memory_clock_mhz"]) for row in selected if _finite(row.get("memory_clock_mhz"))]
    memory_matches = sum(math.isclose(value, memory_target_mhz, abs_tol=0.5) for value in memory)
    stable_covered = all(item["sample_count"] > 0 for item in segment_rows)
    match_fraction = matches / eligible if eligible else 0.0
    valid = bool(selected) and stable_covered and match_fraction >= required_fraction and memory_matches / len(memory) >= required_fraction if memory else False
    return {
        "valid": bool(valid),
        "required_match_fraction": required_fraction,
        "energy_sample_count": len(selected),
        "graphics": {
            "target_mhz": None,
            "sample_count": len(graphics),
            "minimum_mhz": min(graphics) if graphics else None,
            "maximum_mhz": max(graphics) if graphics else None,
            "mean_mhz": statistics.fmean(graphics) if graphics else None,
            "target_match_count": matches,
            "target_match_fraction": match_fraction,
        },
        "memory": {
            "target_mhz": memory_target_mhz,
            "sample_count": len(memory),
            "minimum_mhz": min(memory) if memory else None,
            "maximum_mhz": max(memory) if memory else None,
            "mean_mhz": statistics.fmean(memory) if memory else None,
            "target_match_count": memory_matches,
            "target_match_fraction": memory_matches / len(memory) if memory else 0.0,
        },
        "clock_segments": segment_rows,
        "audit_semantics": "stable clock segments; transition intervals excluded from target-match denominator",
    }


def observed_frequency_residency(
    records: Sequence[Mapping[str, Any]], start: float, end: float,
    levels: Mapping[str, int],
) -> dict[str, float]:
    successful = [
        row for row in records
        if row.get("status") == "success" and _finite(row.get("actual_start_wall_s"))
        and _finite(row.get("graphics_clock_mhz"))
        and start <= float(row["actual_start_wall_s"]) <= end
    ]
    duration = {key: 0.0 for key in levels}
    unknown = 0.0
    for left, right in zip(successful, successful[1:]):
        dt = max(0.0, float(right["actual_start_wall_s"]) - float(left["actual_start_wall_s"]))
        matched = next((key for key, value in levels.items() if math.isclose(float(left["graphics_clock_mhz"]), int(value), abs_tol=0.5)), None)
        if matched is None:
            unknown += dt
        else:
            duration[matched] += dt
    total = sum(duration.values()) + unknown
    result = {key: value / total if total else 0.0 for key, value in duration.items()}
    result["UNKNOWN_OR_TRANSITION"] = unknown / total if total else 0.0
    return result


def fast_recovery_recommendation(
    controller: AdaptivePoolController, trigger: str, event_timestamp_s: float,
) -> dict[str, Any]:
    state = controller.state
    current = int(state.current_frequency)
    high = int(controller.frequencies_mhz[-1])
    if current >= high:
        return {"action": "HOLD_HIGH", "target_mhz": high, "prior_mhz": current, "trigger": trigger}
    safe_above = sorted(
        frequency for frequency in state.safe_energy_by_frequency_j
        if int(frequency) > current
    )
    if int(state.known_safe_frequency) > current:
        safe_above.append(int(state.known_safe_frequency))
    target = min(safe_above) if safe_above else high
    current_index = controller.frequencies_mhz.index(current)
    target_index = controller.frequencies_mhz.index(target)
    state.safe_energy_by_frequency_j.pop(current, None)
    state.safe_energy_observations_by_frequency.pop(current, None)
    state.unsafe_index = current_index
    state.unsafe_bound = current
    state.known_safe_frequency = target
    state.last_validated_safe_index = target_index
    state.candidate_frequency = None
    state.search_phase = "RECOVER"
    state.search_status = "RECOVER"
    state.transition_reason = trigger
    state.last_action_reason = "event-driven recovery to previous validated-safe anchor"
    state.converged = target_index == current_index + 1
    if state.converged:
        state.convergence_reason = "fast safety resolved adjacent safe/unsafe boundary"
    remaining = state.safe_energy_by_frequency_j
    if remaining:
        best, energy = min(remaining.items(), key=lambda item: (item[1], item[0]))
        state.best_measured_safe_frequency = int(best)
        state.best_measured_safe_energy_j = float(energy)
    else:
        state.best_measured_safe_frequency = None
        state.best_measured_safe_energy_j = None
    return {
        "action": "FAST_RECOVER",
        "target_mhz": target,
        "prior_mhz": current,
        "trigger": trigger,
        "decision_timestamp_s": float(event_timestamp_s),
    }


class FastSafetyRuntime:
    """Tail live request/load/NVML events and invoke immediate pool recovery."""

    def __init__(
        self, controllers: Mapping[str, AdaptivePoolController],
        recover: Callable[[str, dict[str, Any]], dict[str, Any]],
        settings: Mapping[str, Any], *, policy: str, block_id: str,
        window_index: int, enabled: bool,
    ) -> None:
        self.controllers = dict(controllers)
        self.recover = recover
        self.settings = settings
        self.policy = policy
        self.block_id = block_id
        self.window_index = int(window_index)
        self.enabled = bool(enabled)
        self.event_rows: list[dict[str, Any]] = []
        self.run_dir: Optional[Path] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self.seen = {"safety": 0, "load": 0, "samples": {key: 0 for key in ("P0", "P1", "D0", "D1")}}
        self.last_sample_wall = {key: None for key in ("P0", "P1", "D0", "D1")}
        self.waiting_history: list[int] = []
        self.started_wall_s: Optional[float] = None

    def attach(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def _log_and_maybe_recover(self, pool: str, trigger: str, event: Mapping[str, Any]) -> None:
        timestamp = float(event.get("timestamp_unix_s") or time.time())
        row = {
            "policy": self.policy, "block_id": self.block_id,
            "window_index": self.window_index, "pool": pool,
            "trigger": trigger, "trigger_timestamp_s": timestamp,
            "request_id": event.get("logical_request_id") or event.get("request_id"),
            "workload_class": event.get("workload_class"),
            "workload_segment": event.get("segment"),
            "ttft_ms": event.get("ttft_ms"), "tpot_ms": event.get("tpot_ms"),
            "violation_recorded": trigger in {"HARD_TTFT_VIOLATION", "HARD_TPOT_VIOLATION"},
            "fast_path_enabled": self.enabled,
        }
        if self.enabled:
            recommendation = fast_recovery_recommendation(
                self.controllers[pool], trigger, timestamp
            )
        else:
            current = self.controllers[pool].current_frequency_mhz
            recommendation = {
                "action": "WINDOW_DEFER",
                "target_mhz": current,
                "prior_mhz": current,
                "trigger": trigger,
                "decision_timestamp_s": timestamp,
            }
        row.update(recommendation)
        if self.enabled and recommendation["action"] == "FAST_RECOVER":
            result = self.recover(pool, recommendation)
            row.update(result)
        self.event_rows.append(row)

    def process_safety_event(self, event: Mapping[str, Any]) -> None:
        if event.get("event") == "TTFT_OBSERVED" and _finite(event.get("ttft_ms")):
            if float(event["ttft_ms"]) > float(self.settings["slo"]["ttft_ms"]):
                self._log_and_maybe_recover("P", "HARD_TTFT_VIOLATION", event)
        if event.get("event") == "REQUEST_COMPLETED" and _finite(event.get("tpot_ms")):
            if float(event["tpot_ms"]) > float(self.settings["slo"]["tpot_ms"]):
                self._log_and_maybe_recover("D", "HARD_TPOT_VIOLATION", event)

    def process_load_event(self, event: Mapping[str, Any]) -> None:
        raw_waiting = event.get("client_waiting_requests")
        if raw_waiting is None:
            for pool in ("P", "D"):
                self._log_and_maybe_recover(pool, "INVALID_SAFETY_TELEMETRY", event)
            return
        waiting = int(raw_waiting)
        self.waiting_history.append(waiting)
        self.waiting_history = self.waiting_history[-3:]
        severe = waiting >= int(self.settings["severe_queue_depth"])
        growing = len(self.waiting_history) == 3 and all(b > a for a, b in zip(self.waiting_history, self.waiting_history[1:]))
        if severe or growing:
            trigger = "SEVERE_CLIENT_PRESSURE" if severe else "GROWING_CLIENT_BACKLOG"
            for pool in ("P", "D"):
                self._log_and_maybe_recover(pool, trigger, event)

    def process_internal_failure(self, pool: str, trigger: str, timestamp_s: float) -> None:
        self._log_and_maybe_recover(pool, trigger, {"timestamp_unix_s": timestamp_s})

    def _new_rows(self, path: Path, key: str) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = self._read_live_rows(path)
        start = int(self.seen[key])
        self.seen[key] = len(rows)
        return rows[start:]

    @staticmethod
    def _read_live_rows(path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if not text.endswith("\n") and number == len(text.splitlines()):
                    break
                raise
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _poll_once(self, *, check_staleness: bool = True) -> None:
        assert self.run_dir is not None
        for row in self._new_rows(self.run_dir / "client" / "safety_events.jsonl", "safety"):
            self.process_safety_event(row)
        for row in self._new_rows(self.run_dir / "client" / "live_load_events.jsonl", "load"):
            self.process_load_event(row)
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            key = self.seen["samples"]
            path = self.run_dir / endpoint_id / "samples.jsonl"
            rows = self._read_live_rows(path) if path.is_file() else []
            start = int(key[endpoint_id])
            key[endpoint_id] = len(rows)
            for row in rows[start:]:
                if row.get("status") == "success" and _finite(row.get("actual_start_wall_s")):
                    self.last_sample_wall[endpoint_id] = float(row["actual_start_wall_s"])
        if not check_staleness:
            return
        now = time.time()
        max_age = float(self.settings["fast_safety"]["telemetry_max_age_s"])
        if self.started_wall_s is not None and now - self.started_wall_s >= max_age:
            for pool, endpoint_ids in POOLS.items():
                controller = self.controllers[pool]
                if controller.current_frequency_mhz >= controller.frequencies_mhz[-1]:
                    continue
                stale = any(
                    self.last_sample_wall[item] is None
                    or now - float(self.last_sample_wall[item]) > max_age
                    for item in endpoint_ids
                )
                if stale:
                    self.process_internal_failure(pool, "STALE_SAFETY_TELEMETRY", now)

    def _run(self) -> None:
        assert self.run_dir is not None
        try:
            while not self.stop_event.wait(0.05):
                self._poll_once()
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()

    def start(self) -> None:
        if self.run_dir is None:
            raise StageC2Error("fast safety runtime has no window directory")
        self.started_wall_s = time.time()
        self.thread = threading.Thread(target=self._run, name="xpyd-c2-fast-safety", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30.0)
            if self.thread.is_alive():
                raise StageC2Error("fast safety monitor did not stop")
        # Drain events flushed by the client immediately before it exited.
        # Do not synthesize a new staleness event after measurement ends.
        if self.error is None and self.run_dir is not None:
            self._poll_once(check_staleness=False)
        if self.error is not None:
            raise StageC2Error("fast safety monitor failed: %s" % self.error)


class C2AccountingWindowHarness(Phase3CSubstrateHarness):
    def __init__(
        self, config: Mapping[str, Any], run_id: str,
        trace_rows: Sequence[Mapping[str, Any]], ledger: ClockSegmentLedger,
        safety: Optional[FastSafetyRuntime], levels: Mapping[str, Mapping[str, int]],
    ) -> None:
        super().__init__(config, run_id=run_id, trace_rows=trace_rows)
        self.ledger = ledger
        self.safety = safety
        self.levels = levels

    def _metadata_overrides(self) -> dict[str, Any]:
        return {
            "phase": "Stage_C2_dynamic_shared_pool_DVFS",
            "adaptive_dvfs": True,
            "clock_semantics": "multiple physical clock segments may occur inside one accounting window",
        }

    def _client_extra_args(self) -> list[str]:
        return [
            "--safety-events-jsonl", (self.run_dir / "client" / "safety_events.jsonl").as_posix(),
            "--live-load-events-jsonl", (self.run_dir / "client" / "live_load_events.jsonl").as_posix(),
        ]

    def _scrape(self, endpoint: Any, scheduled: float, label: str = "periodic") -> Any:
        attempt = super()._scrape(endpoint, scheduled, label)
        if (
            self.safety is not None
            and self.safety.started_wall_s is not None
            and not self.safety.stop_event.is_set()
            and attempt.error is not None
        ):
            self.safety.process_internal_failure(
                _pool(endpoint.endpoint_id), "INVALID_SAFETY_TELEMETRY", time.time()
            )
        return attempt

    def _on_client_started(self, client: Any) -> None:
        del client
        frequencies = {
            endpoint_id: int(self.config["fixed_clocks"][endpoint_id]["graphics_mhz"])
            for endpoint_id in ("P0", "P1", "D0", "D1")
        }
        self.ledger.start(frequencies, time.time())
        if self.safety is not None:
            self.safety.attach(self.run_dir)
            self.safety.start()

    def _on_client_finished(self, client: Any) -> None:
        del client
        if self.safety is not None:
            self.safety.stop()
        summary_path = self.run_dir / "client" / "summary.json"
        if summary_path.is_file():
            summary = _read_json(summary_path)
            self.ledger.finish(float(summary["timing_start_unix_s"]), float(summary["timing_end_unix_s"]))
        else:
            start = self.ledger.window_start_wall_s or time.time()
            self.ledger.finish(start, time.time())

    def _clock_audit_for_endpoint(
        self, endpoint_id: str, records: Sequence[Mapping[str, Any]],
        start_wall_s: float, end_wall_s: float,
    ) -> dict[str, Any]:
        target = self.config["fixed_clocks"][endpoint_id]
        return dynamic_clock_audit(
            endpoint_id, records, self.ledger.rows, start_wall_s, end_wall_s,
            int(target["memory_mhz"]), float(target.get("minimum_workload_match_fraction", 0.95)),
        )

    def _clock_hard_gate_name(self) -> str:
        return "dynamic_clock_segments"

    def _claim_boundary(self) -> str:
        return "C2 shared-pool DVFS isolation; no routing, per-endpoint DVFS, TP, model, or request-energy attribution claim"

    def run(self) -> Path:
        path = super().run()
        summary = _read_json(path / "client" / "summary.json")
        start = float(summary["timing_start_unix_s"])
        end = float(summary["timing_end_unix_s"])
        residency = {}
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            samples = _read_jsonl(path / endpoint_id / "samples.jsonl")
            residency[endpoint_id] = observed_frequency_residency(
                samples, start, end, self.levels[endpoint_id]
            )
            for row in self.ledger.rows:
                if row["endpoint_id"] != endpoint_id:
                    continue
                energy = _segment_energy(
                    samples, float(row["segment_start_wall_s"]), float(row["segment_end_wall_s"])
                )
                row["energy_interval_overlap_j"] = energy
                row["energy_interval_overlap_valid"] = energy is not None
        _write_csv(path / "clock_segments.csv", self.ledger.rows)
        _write_json(path / "frequency_residency.json", residency)
        _write_csv(path / "safety_events.csv", self.safety.event_rows if self.safety else [])
        return path


def load_configs(runtime_path: Path, experiment_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_runtime_config(runtime_path)
    experiment = load_workload_config(experiment_path)
    settings = experiment.get("stagec2")
    if not isinstance(settings, Mapping):
        raise StageC2Error("stagec2 settings are required")
    if settings.get("policies") != list(POLICIES):
        raise StageC2Error("C2 compares exactly STATIC, WINDOW-BOUNDED, and HYBRID")
    if settings.get("routing") != "fixed four-pair round-robin" or int(settings.get("max_concurrency", 0)) != 4:
        raise StageC2Error("C2 requires four-pair RR and concurrency 4")
    if settings.get("pool_frequency_state") != {"P": ["P0", "P1"], "D": ["D0", "D1"]}:
        raise StageC2Error("C2 requires shared P and D pool states")
    if settings.get("slo") != {"ttft_ms": 600.0, "tpot_ms": 65.0, "attainment": 0.95}:
        raise StageC2Error("C2 must retain the TIGHT 600/65 contract")
    if any(bool(settings.get(key, False)) for key in ("request_aware_routing", "per_endpoint_dvfs", "predictor", "oracle")):
        raise StageC2Error("C2 isolation forbids routing, asymmetric DVFS, predictors, and oracle input")
    endpoints = {str(item["endpoint_id"]): item for item in runtime["endpoints"]}
    if {key: str(value["node"]) for key, value in endpoints.items()} != {"P0": "neptune", "P1": "neptune", "D0": "europa", "D1": "europa"}:
        raise StageC2Error("C2 topology must remain Neptune L40S + Europa L4")
    runtime["phase3d"] = {
        "minimum_dwell_s": float(settings.get("minimum_dwell_s", 1.0)),
        "feedback": {
            "ttft_slo_ms": 600.0, "tpot_slo_ms": 65.0,
            "telemetry_max_age_s": float(settings["fast_safety"]["telemetry_max_age_s"]),
        },
    }
    return runtime, experiment


class StageC2Harness(Phase3DClosedLoopHarness):
    def __init__(
        self, runtime: Mapping[str, Any], experiment: Mapping[str, Any],
        run_id: Optional[str] = None, backend: Optional[NvidiaSmiClockBackend] = None,
        stage: str = "preflight",
    ) -> None:
        config = copy.deepcopy(dict(runtime))
        config["routing_control_file"] = "/tmp/xpyd_stagec2_unused_routes.json"
        super().__init__(config, run_id=run_id, backend=backend)
        self.experiment = copy.deepcopy(dict(experiment))
        self.settings = self.experiment["stagec2"]
        self.stage = stage
        self.run_dir = Path(os.path.expandvars(str(experiment["output_root"]))) / self.run_id
        self.action_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.safety_rows: list[dict[str, Any]] = []
        self.segment_rows: list[dict[str, Any]] = []
        self.search_rows: list[dict[str, Any]] = []
        self.active_ledger: Optional[ClockSegmentLedger] = None
        self.actuation_lock = threading.RLock()

    @staticmethod
    def _levels(capability: EndpointClockCapability) -> dict[str, int]:
        return {"LOW": capability.selected_low_mhz, "MID": capability.selected_mid_mhz, "HIGH": capability.selected_high_mhz}

    def _controllers(self, capabilities: Mapping[str, EndpointClockCapability]) -> dict[str, AdaptivePoolController]:
        cfg = AdaptiveExplorationConfig.from_mapping(
            self.settings["adaptive_exploration"], ttft_slo_ms=600.0, tpot_slo_ms=65.0
        )
        return {
            "P": AdaptivePoolController("P", "prefill", [1260, 1890, 2520], cfg, slo_regime="TIGHT"),
            "D": AdaptivePoolController("D", "decode", [750, 1125, 1500], cfg, slo_regime="TIGHT"),
        }

    def _actuate_targets(
        self, actuator: PerEndpointClockActuator, context: ControllerContext,
        targets: Mapping[str, int], *, reason: str, policy: str, block_id: str,
        decision_timestamp_s: Optional[float] = None,
        recover_on_failure: bool = False,
    ) -> dict[str, dict[str, Any]]:
        result = {}
        decision_ts = float(decision_timestamp_s or time.time())
        with self.actuation_lock:
            for endpoint_id in sorted(targets):
                target = int(targets[endpoint_id])
                if actuator.requested.get(endpoint_id) != target:
                    row = actuator.actuate(endpoint_id, target, reason)
                    readback_ts = float(row["timestamp_unix_s"]) + float(row["transition_readback_latency_s"])
                    row.update({
                        "policy": policy, "block_id": block_id,
                        "decision_timestamp_s": decision_ts,
                        "readback_timestamp_unix_s": readback_ts,
                    })
                    self.action_rows.append(row)
                    if row["command_status"] != "success" or not row["readback_valid"]:
                        if recover_on_failure and context.adaptive_pools is not None:
                            details = self._recover_actuation_failure(
                                actuator, context, _pool(endpoint_id), row,
                                policy=policy, block_id=block_id,
                            )
                            raise RecoveredActuationFailure(details)
                        raise StageC2Error("physical actuation/readback failed for %s" % endpoint_id)
                    if self.active_ledger is not None:
                        self.active_ledger.transition(
                            endpoint_id, target, reason=reason,
                            decision_timestamp_s=decision_ts,
                            action_timestamp_s=float(row["timestamp_unix_s"]),
                            readback_timestamp_s=readback_ts,
                            observed_readback_mhz=int(row["observed_freq_after_mhz"]),
                        )
                    state = context.registry.get_state(endpoint_id)
                    state.freq_mhz = target
                    context.registry.update_state(state)
                    context.scheduler.record_dvfs_actuation(
                        endpoint_id, observed_freq_mhz=target, timestamp_s=readback_ts
                    )
                    result[endpoint_id] = row
                context.targets[endpoint_id] = target
        return result

    def _recover_actuation_failure(
        self, actuator: PerEndpointClockActuator, context: ControllerContext,
        pool: str, failed_row: Mapping[str, Any], *, policy: str, block_id: str,
    ) -> dict[str, Any]:
        """Fail closed to the prior validated-safe pool anchor."""
        controller = context.adaptive_pools[pool]
        target = int(controller.state.known_safe_frequency)
        detected = float(failed_row["timestamp_unix_s"]) + float(
            failed_row.get("transition_readback_latency_s") or 0.0
        )
        fallback_rows = []
        for endpoint_id in POOLS[pool]:
            row = actuator.actuate(
                endpoint_id, target, "stagec2_actuator_readback_failure_recovery"
            )
            readback_ts = float(row["timestamp_unix_s"]) + float(
                row["transition_readback_latency_s"]
            )
            row.update({
                "policy": policy, "block_id": block_id,
                "decision_timestamp_s": detected,
                "readback_timestamp_unix_s": readback_ts,
            })
            self.action_rows.append(row)
            if row["command_status"] != "success" or not row["readback_valid"]:
                raise StageC2Error(
                    "actuator/readback failure and validated-safe fallback failed for %s"
                    % endpoint_id
                )
            if self.active_ledger is not None:
                self.active_ledger.transition(
                    endpoint_id, target,
                    reason="stagec2_actuator_readback_failure_recovery",
                    decision_timestamp_s=detected,
                    action_timestamp_s=float(row["timestamp_unix_s"]),
                    readback_timestamp_s=readback_ts,
                    observed_readback_mhz=int(row["observed_freq_after_mhz"]),
                )
            state = context.registry.get_state(endpoint_id)
            state.freq_mhz = target
            context.registry.update_state(state)
            context.scheduler.record_dvfs_actuation(
                endpoint_id, observed_freq_mhz=target, timestamp_s=readback_ts
            )
            context.targets[endpoint_id] = target
            fallback_rows.append(row)
        readback_ts = max(float(row["readback_timestamp_unix_s"]) for row in fallback_rows)
        controller.record_actuation(
            target, target, readback_valid=True, timestamp_s=readback_ts
        )
        event = {
            "policy": policy, "block_id": block_id, "pool": pool,
            "trigger": "ACTUATOR_READBACK_FAILURE",
            "trigger_timestamp_s": detected,
            "action": "FAST_RECOVER",
            "prior_mhz": failed_row.get("requested_freq_mhz"),
            "target_mhz": target,
            "action_timestamp_s": min(float(row["timestamp_unix_s"]) for row in fallback_rows),
            "readback_timestamp_s": readback_ts,
            "readback_frequency_mhz": target,
            "readback_valid": True,
            "fast_path_enabled": True,
            "violation_recorded": False,
            "failed_endpoint_id": failed_row.get("endpoint_id"),
            "actuator_error": failed_row.get("error"),
        }
        self.safety_rows.append(event)
        return event

    def _fast_recover(
        self, pool: str, recommendation: dict[str, Any], *,
        actuator: PerEndpointClockActuator, context: ControllerContext,
        policy: str, block_id: str,
    ) -> dict[str, Any]:
        target = int(recommendation["target_mhz"])
        try:
            actions = self._actuate_targets(
                actuator, context, {endpoint: target for endpoint in POOLS[pool]},
                reason="stagec2_%s" % recommendation["trigger"].lower(),
                policy=policy, block_id=block_id,
                decision_timestamp_s=float(recommendation["decision_timestamp_s"]),
                recover_on_failure=True,
            )
        except RecoveredActuationFailure as exc:
            return exc.details
        if len(actions) != 2:
            raise StageC2Error("fast recovery did not actuate both pool endpoints")
        readback_ts = max(float(row["readback_timestamp_unix_s"]) for row in actions.values())
        context.adaptive_pools[pool].record_actuation(
            target, target, readback_valid=True, timestamp_s=readback_ts
        )
        return {
            "action_timestamp_s": min(float(row["timestamp_unix_s"]) for row in actions.values()),
            "readback_timestamp_s": readback_ts,
            "readback_frequency_mhz": target,
            "readback_valid": all(bool(row["readback_valid"]) for row in actions.values()),
        }

    def _efficiency_control(
        self, context: ControllerContext, capabilities: Mapping[str, EndpointClockCapability],
        actuator: PerEndpointClockActuator, metrics: Mapping[str, Any],
        requests: Sequence[Mapping[str, Any]], policy: str, block_id: str,
        before_window_index: int, *, allow_exploration: bool = True,
    ) -> None:
        now = time.time()
        for pool, endpoint_ids in POOLS.items():
            controller = context.adaptive_pools[pool]
            current = {context.targets[item] for item in endpoint_ids}
            if len(current) != 1:
                raise StageC2Error("shared pool became asymmetric")
            current_mhz = current.pop()
            controller.set_current_frequency(current_mhz)
            signature = derived_context_signature(pool, requests, self.settings)
            changed_inside = bool(metrics.get("%s_mid_window_transition" % pool))
            context_changes = controller.state.context_signature not in {"", "stationary", signature}
            if changed_inside and not context_changes:
                controller.state.evidence_since_action += 1
                controller.state.windows_observed += 1
                controller.state.requests_observed += len(requests)
                self.decision_rows.append({
                    "policy": policy, "block_id": block_id, "pool": pool,
                    "control_path": "EFFICIENCY", "decision": "HOLD",
                    "action": "HOLD", "trigger": "MIXED_CLOCK_EVIDENCE",
                    "reason": "mid-window fast recovery already consumed risk; mixed-frequency accounting window is not SAFE efficiency evidence",
                    "decision_timestamp_s": now,
                    "before_window_index": before_window_index,
                    "evidence_window_index": before_window_index - 1,
                    "requested_freq_mhz": current_mhz,
                    "context_signature": signature,
                })
                continue
            evidence_metrics = dict(metrics)
            evidence_metrics["measurement_valid"] = True
            evidence_metrics["telemetry_fresh"] = True
            evidence_metrics["readback_valid"] = True
            evidence = evidence_from_window(
                pool, evidence_metrics, requests, observed_frequency_mhz=current_mhz,
                slo_regime="TIGHT", ttft_slo_ms=600.0, tpot_slo_ms=65.0,
                context_signature=signature, captured_at_s=now,
            )
            decision = controller.decide(evidence, now_s=now, allow_exploration=allow_exploration)
            try:
                actions = self._actuate_targets(
                    actuator, context,
                    {endpoint: int(decision.requested_frequency_mhz) for endpoint in endpoint_ids},
                    reason="stagec2_efficiency_epoch", policy=policy, block_id=block_id,
                    decision_timestamp_s=now, recover_on_failure=True,
                )
            except RecoveredActuationFailure as exc:
                row = decision.as_log()
                row.update({
                    "policy": policy, "block_id": block_id,
                    "control_path": "FAST_SAFETY",
                    "before_window_index": before_window_index,
                    "evidence_window_index": before_window_index - 1,
                    "context_signature": signature,
                    "decision": "RECOVER", "action": "FAST_RECOVER",
                    "trigger": "ACTUATOR_READBACK_FAILURE",
                    "reason": "physical recommendation failed; restored prior validated-safe anchor",
                    "action_timestamp_s": exc.details["action_timestamp_s"],
                    "readback_timestamp_s": exc.details["readback_timestamp_s"],
                })
                self.decision_rows.append(row)
                continue
            target = context.targets[endpoint_ids[0]]
            if target != int(decision.current_frequency_mhz):
                readback_ts = max(float(row["readback_timestamp_unix_s"]) for row in actions.values())
                controller.record_actuation(target, target, readback_valid=True, timestamp_s=readback_ts)
            row = decision.as_log()
            if actions:
                row["action_timestamp_s"] = min(float(item["timestamp_unix_s"]) for item in actions.values())
                row["readback_timestamp_s"] = max(float(item["readback_timestamp_unix_s"]) for item in actions.values())
            row.update({
                "policy": policy, "block_id": block_id, "control_path": "EFFICIENCY",
                "before_window_index": before_window_index,
                "evidence_window_index": before_window_index - 1,
                "context_signature": signature,
                "mid_window_transition": changed_inside,
                "finalization": not allow_exploration,
            })
            self.decision_rows.append(row)

    def _window_config(self, requested: Mapping[str, int], windows_root: Path) -> dict[str, Any]:
        config = copy.deepcopy(self.config)
        config["output_root"] = windows_root.as_posix()
        config["client"]["max_concurrency"] = 4
        config["server_logs"] = {}
        config["coverage_policy"] = {
            "required_endpoint_ids": ["P0", "P1", "D0", "D1"],
            "required_pairs": [["P0", "D0"], ["P0", "D1"], ["P1", "D0"], ["P1", "D1"]],
        }
        for endpoint_id, frequency in requested.items():
            config["fixed_clocks"][endpoint_id]["graphics_mhz"] = int(frequency)
        return config

    def _measure_window(
        self, window_dir: Path, *, block_id: str, policy: str, repeat: int,
        window_index: int, context_id: str, trace_signature: str,
        ledger: ClockSegmentLedger,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        point = {"id": context_id, "offered_rate_rps": float(self.settings["offered_rate_rps"])}
        metrics = analyze_point(point, window_dir, self.settings["slo"], self.settings["trend_thresholds"])
        requests = _read_csv(window_dir / "requests.csv")
        audit = _read_json(window_dir / "audit.json")
        residency = _read_json(window_dir / "frequency_residency.json")
        ttft = [float(row["ttft_ms"]) for row in requests]
        tpot = [float(row["tpot_ms"]) for row in requests]
        row = {key: value for key, value in metrics.items() if key != "endpoint_load"}
        row.update({
            "run_id": self.run_id, "block_id": block_id, "policy": policy,
            "repeat": repeat, "window_index": window_index, "context": context_id,
            "trace_signature": trace_signature,
            "ttft_attainment": sum(value <= 600.0 for value in ttft) / len(ttft),
            "tpot_attainment": sum(value <= 65.0 for value in tpot) / len(tpot),
            "ttft_violation_count": sum(value > 600.0 for value in ttft),
            "tpot_violation_count": sum(value > 65.0 for value in tpot),
            "P_mid_window_transition": ledger.pool_changed("P"),
            "D_mid_window_transition": ledger.pool_changed("D"),
            "endpoint_load_json": json.dumps(metrics["endpoint_load"], sort_keys=True),
            "P_frequency_residency_json": json.dumps(residency["P0"], sort_keys=True),
            "D_frequency_residency_json": json.dumps(residency["D0"], sort_keys=True),
            "clock_segment_count": len(ledger.rows),
            "window_hard_gate_valid": bool(audit["valid"]),
            "window_hard_gates_json": json.dumps(audit["hard_gates"], sort_keys=True),
        })
        for request in requests:
            request.update({
                "block_id": block_id, "policy": policy, "repeat": repeat,
                "window_index": window_index, "context": context_id,
            })
        return row, requests

    def _check_accepted(self, env_name: str, field: str) -> None:
        path = os.environ.get(env_name)
        if not path:
            raise StageC2Error("%s is required" % env_name)
        value = _read_json(Path(path))
        if not value.get(field):
            raise StageC2Error("accepted prerequisite failed: %s" % path)

    def run_calibration(self) -> Path:
        if self.run_dir.exists():
            raise StageC2Error("run directory already exists")
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        actuator = PerEndpointClockActuator(self.backend, capabilities, float(self.config["phase3d"]["minimum_dwell_s"]))
        high = {key: value.selected_high_mhz for key, value in capabilities.items()}
        points = [
            ("HIGH_HIGH", dict(high)),
            ("P_MID_D_HIGH", {**high, "P0": capabilities["P0"].selected_mid_mhz, "P1": capabilities["P1"].selected_mid_mhz}),
        ]
        rows = []
        registry, telemetry, scheduler = self._build_scheduler(capabilities)
        context = ControllerContext(registry, telemetry, scheduler, dict(high), None)
        try:
            for repeat in range(1, int(self.settings["calibration_repetitions"]) + 1):
                for name, frequencies in points:
                    self._actuate_targets(actuator, context, frequencies, reason="stagec2_context_calibration", policy=name, block_id="calibration")
                    trace = build_trace(self.experiment, "context_b", "c2-cal-r%02d" % repeat)
                    window_id = "r%02d-%s" % (repeat, name.lower())
                    config = self._window_config(frequencies, windows_root)
                    Phase3CSubstrateHarness(config, run_id=window_id, trace_rows=trace).run()
                    window_dir = windows_root / window_id
                    requests = _read_csv(window_dir / "requests.csv")
                    summary = _read_json(window_dir / "summary.json")
                    ttft = [float(row["ttft_ms"]) for row in requests]
                    tpot = [float(row["tpot_ms"]) for row in requests]
                    rows.append({
                        "repeat": repeat, "configuration": name,
                        "trace_signature": _trace_signature(trace),
                        "request_count": len(requests),
                        "total_gross_energy_j": summary["energy_j"]["total"],
                        "joules_per_request": summary["joules_per_request"],
                        "ttft_p95_ms": percentile(ttft, 0.95),
                        "ttft_p99_ms": percentile(ttft, 0.99),
                        "ttft_attainment": sum(value <= 600 for value in ttft) / len(ttft),
                        "tpot_p95_ms": percentile(tpot, 0.95),
                        "tpot_attainment": sum(value <= 65 for value in tpot) / len(tpot),
                        "hard_gates_valid": _read_json(window_dir / "audit.json")["valid"],
                    })
        finally:
            self._actuate_targets(actuator, context, high, reason="stagec2_calibration_restore_high", policy="RESTORE", block_id="final")
        _write_csv(self.run_dir / "c2_context_calibration.csv", rows)
        high_rows = [row for row in rows if row["configuration"] == "HIGH_HIGH"]
        mid_rows = [row for row in rows if row["configuration"] == "P_MID_D_HIGH"]
        high_safe = bool(high_rows) and all(_bool(row["hard_gates_valid"]) and float(row["ttft_attainment"]) >= 0.95 and float(row["tpot_attainment"]) >= 0.95 for row in high_rows)
        mid_unsafe = bool(mid_rows) and all(float(row["ttft_attainment"]) < 0.95 for row in mid_rows)
        matched = all(
            len({row["trace_signature"] for row in rows if int(row["repeat"]) == repeat}) == 1
            for repeat in range(1, int(self.settings["calibration_repetitions"]) + 1)
        )
        summary = {
            "phase": "Stage_C2_context_calibration", "valid": all(_bool(row["hard_gates_valid"]) for row in rows),
            "context_pair_established": high_safe and mid_unsafe and matched,
            "context_a_reference": "accepted C1c P=MID/D=HIGH safe result",
            "context_b": "100% long-input with unchanged 70/30 short/long output mix at 0.20 req/s",
            "high_safe": high_safe, "p_mid_unsafe": mid_unsafe, "matched_trace_audit": matched,
            "rows": rows,
        }
        _write_json(self.run_dir / "c2_context_calibration_summary.json", summary)
        lines = [
            "# XPYD Stage C2 context calibration", "",
            "Context pair established: **%s**" % ("YES" if summary["context_pair_established"] else "NO"), "",
            "| Repeat | Configuration | TTFT p95 | TTFT attainment | TPOT attainment | Hard gates |",
            "|---:|---|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append("| %d | %s | %.1f | %.1f%% | %.1f%% | %s |" % (
                row["repeat"], row["configuration"], row["ttft_p95_ms"],
                100 * row["ttft_attainment"], 100 * row["tpot_attainment"], row["hard_gates_valid"],
            ))
        (self.run_dir / "c2_context_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(self.run_dir)
        return self.run_dir

    def run_dynamic(self) -> Path:
        self._check_accepted("XPYD_STAGEC2_ACCEPTED_CALIBRATION", "context_pair_established")
        if self.stage == "full":
            self._check_accepted("XPYD_STAGEC2_ACCEPTED_PREFLIGHT", "preflight_pass")
        if self.run_dir.exists():
            raise StageC2Error("run directory already exists")
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        expected = self.settings["frequency_levels_mhz"]
        if {key: self._levels(value) for key, value in capabilities.items()} != expected:
            raise StageC2Error("discovered clocks differ from frozen C2 action space")
        _write_json(self.run_dir / "capabilities.json", {"endpoints": {key: asdict(value) for key, value in capabilities.items()}})
        actuator = PerEndpointClockActuator(self.backend, capabilities, float(self.config["phase3d"]["minimum_dwell_s"]))
        high = {key: value.selected_high_mhz for key, value in capabilities.items()}
        sequence = list(self.settings["preflight_sequence" if self.stage == "preflight" else "full_sequence"])
        repetitions = 1 if self.stage == "preflight" else int(self.settings["repetitions"])
        rows: list[dict[str, Any]] = []
        requests_all: list[dict[str, Any]] = []
        restoration_valid = False
        run_error: Optional[str] = None
        try:
            for repeat in range(1, repetitions + 1):
                for policy in policy_order(repeat):
                    block_id = "r%02d-%s" % (repeat, policy.lower())
                    registry, telemetry, scheduler = self._build_scheduler(capabilities)
                    controllers = None if policy == "STATIC-HIGH" else self._controllers(capabilities)
                    context = ControllerContext(registry, telemetry, scheduler, dict(high), controllers)
                    self._actuate_targets(actuator, context, high, reason="stagec2_cold_start_high", policy=policy, block_id=block_id)
                    previous = None
                    previous_requests: list[dict[str, Any]] = []
                    for window_index, context_id in enumerate(sequence, 1):
                        if controllers is not None and previous is not None:
                            self._efficiency_control(
                                context, capabilities, actuator, previous, previous_requests,
                                policy, block_id, window_index,
                            )
                        scenario = "context_a" if context_id == "A" else "context_b"
                        trace = build_trace(self.experiment, scenario, "c2-r%02d-w%02d" % (repeat, window_index))
                        signature = _trace_signature(trace)
                        window_id = "%s-w%02d-%s" % (block_id, window_index, context_id.lower())
                        ledger = ClockSegmentLedger(window_id)
                        self.active_ledger = ledger
                        safety = None
                        if controllers is not None:
                            safety = FastSafetyRuntime(
                                controllers,
                                lambda pool, recommendation, a=actuator, c=context, p=policy, b=block_id: self._fast_recover(
                                    pool, recommendation, actuator=a, context=c, policy=p, block_id=b
                                ),
                                self.settings, policy=policy, block_id=block_id,
                                window_index=window_index,
                                enabled=policy == "HYBRID-FAST-SAFETY",
                            )
                        config = self._window_config(context.targets, windows_root)
                        window_dir = C2AccountingWindowHarness(
                            config, window_id, trace, ledger, safety, expected
                        ).run()
                        self.active_ledger = None
                        if safety is not None:
                            self.safety_rows.extend(safety.event_rows)
                        for segment in ledger.rows:
                            segment.update({"policy": policy, "block_id": block_id, "repeat": repeat, "window_index": window_index, "context": context_id})
                        self.segment_rows.extend(ledger.rows)
                        row, request_rows = self._measure_window(
                            window_dir, block_id=block_id, policy=policy, repeat=repeat,
                            window_index=window_index, context_id=context_id,
                            trace_signature=signature, ledger=ledger,
                        )
                        rows.append(row)
                        requests_all.extend(request_rows)
                        previous, previous_requests = row, request_rows
                    if controllers is not None and previous is not None:
                        self._efficiency_control(
                            context, capabilities, actuator, previous, previous_requests,
                            policy, block_id, len(sequence) + 1, allow_exploration=False,
                        )
                        for pool, controller in controllers.items():
                            state = controller.state
                            self.search_rows.append({
                                "policy": policy, "block_id": block_id,
                                "repeat": repeat, "pool": pool,
                                "current_frequency_mhz": state.current_frequency,
                                "final_validated_safe_frequency_mhz": state.known_safe_frequency,
                                "unsafe_bound_mhz": state.unsafe_bound,
                                "best_measured_safe_frequency_mhz": state.best_measured_safe_frequency,
                                "best_measured_safe_energy_j": state.best_measured_safe_energy_j,
                                "windows_observed": state.windows_observed,
                                "requests_observed": state.requests_observed,
                                "converged": state.converged,
                                "convergence_reason": state.convergence_reason,
                                "windows_until_convergence": state.convergence_window_count,
                                "requests_until_convergence": state.convergence_request_count,
                            })
        except BaseException as exc:
            run_error = "%s: %s" % (type(exc).__name__, exc)
            raise
        finally:
            self.active_ledger = None
            if "actuator" in locals() and "context" in locals():
                self._actuate_targets(actuator, context, high, reason="stagec2_final_restore_high", policy="RESTORE", block_id="final")
                restoration_valid = all(actuator.read(key).graphics_clock_mhz == value for key, value in high.items())
            if self.run_dir.exists():
                self._write_reports(
                    rows, requests_all, restoration_valid, sequence,
                    repetitions, run_error,
                )
        print(self.run_dir)
        return self.run_dir

    def _write_reports(
        self, rows: Sequence[Mapping[str, Any]], requests: Sequence[Mapping[str, Any]],
        restoration_valid: bool, sequence: Sequence[str], repetitions: int,
        run_error: Optional[str] = None,
    ) -> None:
        first_b = next((index for index, value in enumerate(sequence, 1) if value == "B"), None)
        last_b = max((index for index, value in enumerate(sequence, 1) if value == "B"), default=None)
        for block_id in sorted({str(row["block_id"]) for row in rows}):
            cumulative_energy = 0.0
            cumulative_requests = 0
            for row in sorted(
                (item for item in rows if item["block_id"] == block_id),
                key=lambda item: int(item["window_index"]),
            ):
                index = int(row["window_index"])
                row["transition_phase"] = (
                    "A_BEFORE" if first_b is None or index < first_b
                    else ("B_PRESSURE" if last_b is not None and index <= last_b else "A_AFTER")
                )
                cumulative_energy += float(row["total_gross_energy_j"])
                cumulative_requests += int(row["request_count"])
                row["cumulative_energy_j"] = cumulative_energy
                row["cumulative_requests"] = cumulative_requests
                row["cumulative_joules_per_request"] = cumulative_energy / cumulative_requests
        _write_csv(self.run_dir / "stagec2_measurements.csv", rows)
        _write_csv(self.run_dir / "stagec2_requests.csv", requests)
        _write_csv(self.run_dir / "controller_decisions.csv", self.decision_rows)
        _write_csv(self.run_dir / "safety_events.csv", self.safety_rows)
        _write_csv(self.run_dir / "clock_actions.csv", self.action_rows)
        _write_csv(self.run_dir / "clock_segments.csv", self.segment_rows)
        _write_csv(self.run_dir / "stagec2_search_state.csv", self.search_rows)
        signatures: dict[tuple[int, int], set[str]] = {}
        for row in rows:
            signatures.setdefault((int(row["repeat"]), int(row["window_index"])), set()).add(str(row["trace_signature"]))
        matched = bool(signatures) and all(len(value) == 1 for value in signatures.values())
        hard_windows = bool(rows) and all(_bool(row["window_hard_gate_valid"]) for row in rows)
        segment_valid = bool(self.segment_rows) and all(float(row["duration_s"]) > 0 for row in self.segment_rows)
        fast_rows = [row for row in self.safety_rows if row.get("fast_path_enabled") and row.get("action") == "FAST_RECOVER"]
        mid_window_fast = bool(fast_rows) and all(float(row["readback_timestamp_s"]) >= float(row["action_timestamp_s"]) > float(row["trigger_timestamp_s"]) for row in fast_rows)
        repeat_summaries = []
        adaptation_rows = []
        for repeat in range(1, repetitions + 1):
            for policy in POLICIES:
                selected = [
                    row for row in rows
                    if row["policy"] == policy and int(row["repeat"]) == repeat
                ]
                selected_requests = [
                    row for row in requests
                    if row["policy"] == policy and int(row["repeat"]) == repeat
                ]
                if not selected:
                    continue
                ttft = [float(row["ttft_ms"]) for row in selected_requests]
                tpot = [float(row["tpot_ms"]) for row in selected_requests]
                energy = sum(float(row["total_gross_energy_j"]) for row in selected)
                phase_energy = {
                    phase: sum(
                        float(row["total_gross_energy_j"])
                        for row in selected if row.get("transition_phase") == phase
                    )
                    for phase in ("A_BEFORE", "B_PRESSURE", "A_AFTER")
                }
                policy_segments = [
                    row for row in self.segment_rows
                    if row.get("policy") == policy and int(row.get("repeat", 0)) == repeat
                    and row.get("segment_kind") == "STABLE"
                ]
                residency = {}
                for pool in ("P", "D"):
                    pool_segments = [row for row in policy_segments if row["pool"] == pool]
                    total_duration = sum(float(row["duration_s"]) for row in pool_segments)
                    residency[pool] = {
                        str(frequency): (
                            sum(float(row["duration_s"]) for row in pool_segments
                                if int(row["observed_readback_frequency_mhz"]) == frequency)
                            / total_duration if total_duration else 0.0
                        )
                        for frequency in sorted({
                            int(row["observed_readback_frequency_mhz"])
                            for row in pool_segments
                        })
                    }
                repeat_summaries.append({
                    "repeat": repeat, "policy": policy,
                    "request_count": len(selected_requests),
                    "total_gross_energy_j": energy,
                    "joules_per_request": energy / len(selected_requests),
                    "ttft_p95_ms": percentile(ttft, 0.95),
                    "ttft_p99_ms": percentile(ttft, 0.99),
                    "ttft_attainment": sum(value <= 600 for value in ttft) / len(ttft),
                    "ttft_violation_count": sum(value > 600 for value in ttft),
                    "tpot_p95_ms": percentile(tpot, 0.95),
                    "tpot_p99_ms": percentile(tpot, 0.99),
                    "tpot_attainment": sum(value <= 65 for value in tpot) / len(tpot),
                    "tpot_violation_count": sum(value > 65 for value in tpot),
                    "slo_contract_pass": (
                        sum(value <= 600 for value in ttft) / len(ttft) >= 0.95
                        and sum(value <= 65 for value in tpot) / len(tpot) >= 0.95
                    ),
                    "completed_rate_rps_mean": statistics.fmean(float(row["completed_rate_rps"]) for row in selected),
                    "admission_delay_p95_ms": percentile([float(row["client_admission_delay_ms"]) for row in selected_requests], 0.95),
                    "client_waiting_max": max(float(row["client_waiting_max"]) for row in selected),
                    "backlog_growing_windows": sum(_bool(row["backlog_growing"]) for row in selected),
                    "probe_count": sum(
                        row.get("policy") == policy and row.get("block_id") in {item["block_id"] for item in selected}
                        and row.get("decision") == "EXPLORE_DOWN"
                        for row in self.decision_rows
                    ),
                    "fast_recovery_count": sum(
                        row.get("policy") == policy and row.get("block_id") in {item["block_id"] for item in selected}
                        and row.get("action") == "FAST_RECOVER"
                        for row in self.safety_rows
                    ),
                    "energy_a_before_j": phase_energy["A_BEFORE"],
                    "energy_b_pressure_j": phase_energy["B_PRESSURE"],
                    "energy_a_after_j": phase_energy["A_AFTER"],
                    "frequency_residency_json": json.dumps(residency, sort_keys=True),
                })

                if first_b is not None:
                    first_b_requests = [
                        row for row in selected_requests if int(row["window_index"]) == first_b
                    ]
                    transition_start = min((
                        value for value in (
                            _float_or_none(row.get("send_unix_s"))
                            for row in first_b_requests
                        ) if value is not None
                    ), default=None)
                    recovery_candidates = []
                    recovery_candidates.extend(
                        row for row in self.safety_rows
                        if row.get("policy") == policy and int(row.get("window_index", 0)) == first_b
                        and row.get("action") == "FAST_RECOVER" and row.get("pool") == "P"
                    )
                    recovery_candidates.extend(
                        row for row in self.decision_rows
                        if row.get("policy") == policy and row.get("block_id") in {item["block_id"] for item in selected}
                        and int(row.get("before_window_index", 0)) > first_b
                        and row.get("pool") == "P" and row.get("action") in {"RECOVER_HIGH", "STEP_UP"}
                    )
                    recovery_candidates = [row for row in recovery_candidates if row.get("action_timestamp_s") is not None]
                    recovery = min(recovery_candidates, key=lambda row: float(row["action_timestamp_s"]), default=None)
                    action_time = float(recovery["action_timestamp_s"]) if recovery else None
                    trigger_time = float(recovery.get("trigger_timestamp_s") or action_time) if recovery else None
                    violating_before_recovery = sum(
                        (float(row["ttft_ms"]) > 600.0
                         and _float_or_none(row.get("first_token_unix_s")) is not None
                         and (action_time is None or float(row["first_token_unix_s"]) <= action_time))
                        or (float(row["tpot_ms"]) > 65.0
                            and _float_or_none(row.get("complete_unix_s")) is not None
                            and (action_time is None or float(row["complete_unix_s"]) <= action_time))
                        for row in first_b_requests
                    )
                    return_index = next(
                        (index for index, value in enumerate(sequence, 1)
                         if last_b is not None and index > last_b and value == "A"), None
                    )
                    return_start = None
                    rediscovery = None
                    if return_index is not None:
                        return_requests = [row for row in selected_requests if int(row["window_index"]) == return_index]
                        return_start = min((float(row["send_unix_s"]) for row in return_requests), default=None)
                        rediscovery = min((
                            row for row in self.decision_rows
                            if row.get("policy") == policy
                            and row.get("block_id") in {item["block_id"] for item in selected}
                            and int(row.get("before_window_index", 0)) >= return_index
                            and row.get("pool") == "P" and row.get("decision") == "EXPLORE_DOWN"
                            and row.get("action_timestamp_s") is not None
                        ), key=lambda row: float(row["action_timestamp_s"]), default=None)
                    adaptation_rows.append({
                        "repeat": repeat, "policy": policy,
                        "pressure_transition_window": first_b,
                        "recovery_action": recovery.get("action") if recovery else None,
                        "recovery_trigger": recovery.get("trigger") if recovery else None,
                        "transition_to_recovery_s": (
                            action_time - transition_start
                            if action_time is not None and transition_start is not None else None
                        ),
                        "trigger_to_readback_s": (
                            float(recovery.get("readback_timestamp_s")) - trigger_time
                            if recovery and recovery.get("readback_timestamp_s") is not None else None
                        ),
                        "requests_admitted_before_recovery": sum(
                            action_time is not None and float(row["send_unix_s"]) <= action_time
                            for row in first_b_requests
                        ),
                        "violating_requests_before_recovery": violating_before_recovery,
                        "return_to_a_window": return_index,
                        "return_to_downward_rediscovery_s": (
                            float(rediscovery["action_timestamp_s"]) - return_start
                            if rediscovery is not None and return_start is not None else None
                        ),
                        "rediscovery_action_frequency_mhz": (
                            rediscovery.get("requested_freq_mhz") if rediscovery else None
                        ),
                    })
        static_by_repeat = {
            int(row["repeat"]): row for row in repeat_summaries if row["policy"] == "STATIC-HIGH"
        }
        for row in repeat_summaries:
            baseline = static_by_repeat.get(int(row["repeat"]))
            if baseline and row["policy"] != "STATIC-HIGH":
                saving = (
                    float(baseline["joules_per_request"]) - float(row["joules_per_request"])
                ) / float(baseline["joules_per_request"])
                row["paired_energy_saving_fraction_raw"] = saving
                row["paired_energy_saving_fraction"] = saving if row["slo_contract_pass"] else None
        _write_csv(self.run_dir / "stagec2_repeat_summary.csv", repeat_summaries)
        _write_csv(self.run_dir / "stagec2_adaptation.csv", adaptation_rows)

        aggregates = []
        for policy in POLICIES:
            selected = [row for row in rows if row["policy"] == policy]
            selected_requests = [row for row in requests if row["policy"] == policy]
            if not selected:
                continue
            energy = sum(float(row["total_gross_energy_j"]) for row in selected)
            ttft = [float(row["ttft_ms"]) for row in selected_requests]
            tpot = [float(row["tpot_ms"]) for row in selected_requests]
            aggregates.append({
                "policy": policy, "request_count": len(selected_requests),
                "total_gross_energy_j": energy, "joules_per_request": energy / len(selected_requests),
                "ttft_p95_ms": percentile(ttft, 0.95), "ttft_p99_ms": percentile(ttft, 0.99),
                "ttft_attainment": sum(value <= 600 for value in ttft) / len(ttft),
                "ttft_violation_count": sum(value > 600 for value in ttft),
                "tpot_p95_ms": percentile(tpot, 0.95), "tpot_p99_ms": percentile(tpot, 0.99),
                "tpot_attainment": sum(value <= 65 for value in tpot) / len(tpot),
                "tpot_violation_count": sum(value > 65 for value in tpot),
                "slo_contract_pass": (
                    sum(value <= 600 for value in ttft) / len(ttft) >= 0.95
                    and sum(value <= 65 for value in tpot) / len(tpot) >= 0.95
                ),
                "completed_rate_rps_mean": statistics.fmean(float(row["completed_rate_rps"]) for row in selected),
                "admission_delay_p95_ms": percentile([float(row["client_admission_delay_ms"]) for row in selected_requests], 0.95),
                "fast_recovery_count": sum(row.get("policy") == policy and row.get("action") == "FAST_RECOVER" for row in self.safety_rows),
                "probe_count": sum(row.get("policy") == policy and row.get("decision") == "EXPLORE_DOWN" for row in self.decision_rows),
                "unsafe_probe_count": sum(
                    row.get("policy") == policy and row.get("trigger") in {"SLO_RISK", "HARD_TTFT_VIOLATION", "HARD_TPOT_VIOLATION"}
                    for row in [*self.decision_rows, *self.safety_rows]
                ),
                "paired_energy_saving_mean_fraction_raw": statistics.fmean([
                    float(row["paired_energy_saving_fraction_raw"])
                    for row in repeat_summaries
                    if row["policy"] == policy and row.get("paired_energy_saving_fraction_raw") is not None
                ]) if policy != "STATIC-HIGH" else None,
            })
        _write_csv(self.run_dir / "stagec2_policy_summary.csv", aggregates)
        preflight_pass = (
            run_error is None and (self.stage != "preflight" or (
                matched and hard_windows and segment_valid and mid_window_fast and restoration_valid
                and any(row.get("P_mid_window_transition") or row.get("D_mid_window_transition") for row in rows if row["policy"] == "HYBRID-FAST-SAFETY")
            ))
        )
        summary = {
            "schema_version": 1, "phase": "Stage_C2_dynamic_shared_pool_DVFS",
            "stage": self.stage,
            "valid": run_error is None and matched and hard_windows and segment_valid and restoration_valid,
            "error": run_error,
            "preflight_pass": preflight_pass,
            "matched_trace_audit": matched, "window_hard_gate_audit": hard_windows,
            "clock_segment_audit": segment_valid, "mid_window_fast_recovery_audit": mid_window_fast,
            "final_high_restoration_audit": restoration_valid,
            "policies": list(POLICIES), "sequence": list(sequence),
            "repetitions": repetitions, "policy_results": aggregates,
            "repeat_results": repeat_summaries,
            "adaptation_results": adaptation_rows,
            "search_state": self.search_rows,
            "claim_boundary": "shared-pool C2 fast-safety isolation; no routing, per-endpoint DVFS, TP, model, oracle, or physical request-energy attribution",
        }
        _write_json(self.run_dir / "stagec2_summary.json", summary)
        lines = [
            "# XPYD Stage C2 dynamic shared-pool DVFS", "",
            "Structural verdict: **%s**" % ("PASS" if summary["valid"] else "FAIL"), "",
            "| Policy | J/request | TTFT p95 / attainment / violations | TPOT p95 / attainment / violations | Fast recoveries |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in aggregates:
            lines.append("| %s | %.2f | %.1f / %.1f%% / %d | %.2f / %.1f%% / %d | %d |" % (
                row["policy"], row["joules_per_request"], row["ttft_p95_ms"], 100 * row["ttft_attainment"], row["ttft_violation_count"],
                row["tpot_p95_ms"], 100 * row["tpot_attainment"], row["tpot_violation_count"], row["fast_recovery_count"],
            ))
        (self.run_dir / "stagec2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.stage == "preflight" and run_error is None and not preflight_pass:
            raise StageC2Error("C2 preflight structural audit failed")

    def run(self) -> Path:
        if self.stage == "calibration":
            return self.run_calibration()
        return self.run_dynamic()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("calibration", "preflight", "full"), required=True)
    args = parser.parse_args()
    runtime, experiment = load_configs(args.runtime_config, args.experiment_config)
    StageC2Harness(runtime, experiment, args.run_id, stage=args.stage).run()


if __name__ == "__main__":
    main()
