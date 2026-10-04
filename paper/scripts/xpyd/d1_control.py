"""Per-endpoint control primitives for XPYD Stage D1."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional, Sequence

from xpyd.adaptive_exploration import AdaptivePoolController
from xpyd.stagec2_dynamic import FastSafetyRuntime, _finite, fast_recovery_recommendation


ENDPOINT_PHASE = {"P0": "P", "P1": "P", "D0": "D", "D1": "D"}
PHASE_ENDPOINTS = {"P": ("P0", "P1"), "D": ("D0", "D1")}


def active_candidate(controller: AdaptivePoolController) -> bool:
    candidate = controller.state.candidate_frequency
    return (
        candidate is not None
        and int(candidate) == controller.current_frequency_mhz
    )


def exploration_permissions(
    controllers: Mapping[str, AdaptivePoolController],
    phase: str,
) -> dict[str, bool]:
    """Allow at most one same-phase unvalidated probe with a safe fallback."""
    endpoints = PHASE_ENDPOINTS[phase]
    active = [item for item in endpoints if active_candidate(controllers[item])]
    if len(active) > 1:
        raise ValueError("simultaneous unvalidated same-phase probes are forbidden")
    if active:
        return {item: False for item in endpoints}
    safe_fallbacks = [
        item for item in endpoints
        if int(controllers[item].state.known_safe_frequency)
        == controllers[item].current_frequency_mhz
    ]
    if not safe_fallbacks:
        return {item: False for item in endpoints}
    # Deterministic alternation based only on online evidence counts.
    chosen = min(
        endpoints,
        key=lambda item: (
            controllers[item].state.windows_observed,
            controllers[item].state.requests_observed,
            item,
        ),
    )
    if chosen in safe_fallbacks and len(safe_fallbacks) == 1:
        chosen = next(item for item in endpoints if item != chosen)
    return {item: item == chosen for item in endpoints}


class EndpointFastSafetyRuntime(FastSafetyRuntime):
    """C2a fast safety scoped to the exact endpoint used by a request."""

    def __init__(
        self,
        controllers: Mapping[str, AdaptivePoolController],
        recover: Callable[[str, dict[str, Any]], dict[str, Any]],
        settings: Mapping[str, Any],
        *,
        policy: str,
        block_id: str,
        window_index: int,
        enabled: bool,
    ) -> None:
        # The parent supplies file tailing/thread lifecycle; its pool-oriented
        # event handlers are all overridden below.
        super().__init__(
            {}, recover, settings, policy=policy, block_id=block_id,
            window_index=window_index, enabled=enabled,
        )
        self.controllers = dict(controllers)
        self.endpoint_evidence: dict[str, dict[str, Any]] = {}
        for endpoint_id, controller in self.controllers.items():
            state = controller.state
            seeded = (
                controller.is_frequency_validated_safe(
                    controller.current_frequency_mhz
                )
                and state.recent_evidence_frequency
                == controller.current_frequency_mhz
            )
            self.endpoint_evidence[endpoint_id] = {
                "valid_request_count": (
                    int(state.recent_valid_request_count) if seeded else 0
                ),
                "phase_pass_count": (
                    int(state.recent_phase_pass_count) if seeded else 0
                ),
                "phase_token_coverage": (
                    int(state.recent_phase_token_coverage) if seeded else 0
                ),
                "first_timestamp_s": (
                    float(state.recent_evidence_captured_at_s)
                    if seeded and state.recent_evidence_captured_at_s > 0
                    else None
                ),
            }

    def _log_endpoint(
        self,
        endpoint_id: str,
        trigger: str,
        event: Mapping[str, Any],
        *,
        candidate_required: bool = False,
    ) -> None:
        timestamp = float(event.get("timestamp_unix_s") or time.time())
        controller = self.controllers[endpoint_id]
        row = {
            "policy": self.policy,
            "block_id": self.block_id,
            "window_index": self.window_index,
            "endpoint_id": endpoint_id,
            "pool": ENDPOINT_PHASE[endpoint_id],
            "trigger": trigger,
            "trigger_timestamp_s": timestamp,
            "request_id": event.get("logical_request_id") or event.get("request_id"),
            "workload_class": event.get("workload_class"),
            "ttft_ms": event.get("ttft_ms"),
            "tpot_ms": event.get("tpot_ms"),
            "violation_recorded": trigger in {
                "HARD_TTFT_VIOLATION", "HARD_TPOT_VIOLATION"
            },
            "fast_path_enabled": self.enabled,
            "evidence_valid_request_count": event.get(
                "evidence_valid_request_count"
            ),
            "evidence_phase_pass_count": event.get(
                "evidence_phase_pass_count"
            ),
            "evidence_attainment": event.get("evidence_attainment"),
            "evidence_phase_token_coverage": event.get(
                "evidence_phase_token_coverage"
            ),
        }
        if self.enabled and candidate_required and not active_candidate(controller):
            recommendation = {
                "action": "LOG_ONLY_NO_ACTIVE_CANDIDATE",
                "target_mhz": controller.current_frequency_mhz,
                "prior_mhz": controller.current_frequency_mhz,
                "trigger": trigger,
                "decision_timestamp_s": timestamp,
            }
        elif self.enabled:
            recommendation = fast_recovery_recommendation(
                controller, trigger, timestamp
            )
        else:
            recommendation = {
                "action": "WINDOW_DEFER",
                "target_mhz": controller.current_frequency_mhz,
                "prior_mhz": controller.current_frequency_mhz,
                "trigger": trigger,
                "decision_timestamp_s": timestamp,
            }
        row.update(recommendation)
        if self.enabled and recommendation["action"] == "FAST_RECOVER":
            row.update(self.recover(endpoint_id, recommendation))
        self.event_rows.append(row)

    def _update_validated_endpoint_evidence(
        self, endpoint_id: str, event: Mapping[str, Any],
    ) -> None:
        controller = self.controllers[endpoint_id]
        if not self.enabled or active_candidate(controller):
            return
        current = controller.current_frequency_mhz
        if current >= controller.frequencies_mhz[-1]:
            return
        if not controller.is_frequency_validated_safe(current):
            return
        phase = ENDPOINT_PHASE[endpoint_id]
        latency_key = "ttft_ms" if phase == "P" else "tpot_ms"
        if not _finite(event.get(latency_key)):
            return
        if phase == "D" and event.get("ok") is not True:
            return
        token_key = "input_len" if phase == "P" else "actual_output_len"
        raw_tokens = event.get(token_key)
        if raw_tokens is None and phase == "D":
            # Older CPU fixtures may not carry actual output; never substitute
            # realized work with requested budget for safety eligibility.
            return
        try:
            tokens = int(raw_tokens)
        except (TypeError, ValueError):
            return
        if tokens <= 0:
            return
        evidence = self.endpoint_evidence[endpoint_id]
        timestamp = float(event.get("timestamp_unix_s") or time.time())
        if evidence["first_timestamp_s"] is None:
            evidence["first_timestamp_s"] = timestamp
        evidence["valid_request_count"] += 1
        evidence["phase_token_coverage"] += tokens
        slo = (
            float(controller.config.ttft_slo_ms)
            if phase == "P" else float(controller.config.tpot_slo_ms)
        )
        if float(event[latency_key]) <= slo:
            evidence["phase_pass_count"] += 1
        count = int(evidence["valid_request_count"])
        attainment = float(evidence["phase_pass_count"]) / count
        duration = max(
            0.0, timestamp - float(evidence["first_timestamp_s"])
        )
        enough = (
            count >= int(controller.config.min_valid_requests)
            and int(evidence["phase_token_coverage"]) >= int(
                controller.config.min_input_tokens
                if phase == "P" else controller.config.min_output_tokens
            )
            and (
                duration >= float(controller.config.min_observation_duration_s)
                or controller.state.recent_valid_request_count > 0
            )
        )
        if not enough or attainment >= float(controller.config.attainment_gate):
            return
        enriched = dict(event)
        enriched.update({
            "evidence_valid_request_count": count,
            "evidence_phase_pass_count": int(evidence["phase_pass_count"]),
            "evidence_attainment": attainment,
            "evidence_phase_token_coverage": int(
                evidence["phase_token_coverage"]
            ),
        })
        self._log_endpoint(
            endpoint_id, "ENDPOINT_EVIDENCE_SLO_RISK", enriched
        )

    def process_safety_event(self, event: Mapping[str, Any]) -> None:
        if self.measurement_started_monotonic_s is None:
            self.measurement_started_monotonic_s = time.monotonic()
        if event.get("event") == "TTFT_OBSERVED" and _finite(event.get("ttft_ms")):
            if float(event["ttft_ms"]) > float(self.settings["slo"]["ttft_ms"]):
                endpoint = str(event.get("selected_prefill_endpoint_id") or "")
                if endpoint in PHASE_ENDPOINTS["P"]:
                    candidate = active_candidate(self.controllers[endpoint])
                    self._log_endpoint(
                        endpoint, "HARD_TTFT_VIOLATION", event,
                        candidate_required=True,
                    )
                    if candidate:
                        return
            endpoint = str(event.get("selected_prefill_endpoint_id") or "")
            if endpoint in PHASE_ENDPOINTS["P"]:
                self._update_validated_endpoint_evidence(endpoint, event)
        if event.get("event") == "REQUEST_COMPLETED" and _finite(event.get("tpot_ms")):
            if float(event["tpot_ms"]) > float(self.settings["slo"]["tpot_ms"]):
                endpoint = str(event.get("selected_decode_endpoint_id") or "")
                if endpoint in PHASE_ENDPOINTS["D"]:
                    candidate = active_candidate(self.controllers[endpoint])
                    self._log_endpoint(
                        endpoint, "HARD_TPOT_VIOLATION", event,
                        candidate_required=True,
                    )
                    if candidate:
                        return
            endpoint = str(event.get("selected_decode_endpoint_id") or "")
            if endpoint in PHASE_ENDPOINTS["D"]:
                self._update_validated_endpoint_evidence(endpoint, event)

    def process_load_event(self, event: Mapping[str, Any]) -> None:
        if self.measurement_started_monotonic_s is None:
            self.measurement_started_monotonic_s = time.monotonic()
        raw_waiting = event.get("client_waiting_requests")
        if raw_waiting is None:
            for endpoint_id, controller in self.controllers.items():
                if active_candidate(controller):
                    self._log_endpoint(
                        endpoint_id, "INVALID_SAFETY_TELEMETRY", event
                    )
            return
        waiting = int(raw_waiting)
        self.waiting_history.append(waiting)
        self.waiting_history = self.waiting_history[-3:]
        severe = waiting >= int(self.settings["severe_queue_depth"])
        growing = (
            len(self.waiting_history) == 3
            and all(b > a for a, b in zip(
                self.waiting_history, self.waiting_history[1:]
            ))
        )
        if severe or growing:
            trigger = "SEVERE_CLIENT_PRESSURE" if severe else "GROWING_CLIENT_BACKLOG"
            for endpoint_id, controller in self.controllers.items():
                if active_candidate(controller):
                    self._log_endpoint(endpoint_id, trigger, event)

    def process_internal_failure(
        self, endpoint_id: str, trigger: str, timestamp_s: float,
    ) -> None:
        if (
            trigger in {"STALE_SAFETY_TELEMETRY", "INVALID_SAFETY_TELEMETRY"}
            and self.measurement_started_monotonic_s is None
        ):
            return
        self._log_endpoint(
            endpoint_id, trigger, {"timestamp_unix_s": timestamp_s}
        )

    def _poll_once(self, *, check_staleness: bool = True) -> None:
        assert self.run_dir is not None
        for row in self._new_rows(
            self.run_dir / "client" / "safety_events.jsonl", "safety"
        ):
            self.process_safety_event(row)
        for row in self._new_rows(
            self.run_dir / "client" / "live_load_events.jsonl", "load"
        ):
            self.process_load_event(row)
        for endpoint_id in self.controllers:
            key = self.seen["samples"]
            path = self.run_dir / endpoint_id / "samples.jsonl"
            rows = self._read_live_rows(path) if path.is_file() else []
            start = int(key[endpoint_id])
            key[endpoint_id] = len(rows)
            for row in rows[start:]:
                if row.get("status") == "success" and _finite(
                    row.get("actual_start_wall_s")
                ):
                    self.last_sample_wall[endpoint_id] = float(
                        row["actual_start_wall_s"]
                    )
                    self.last_sample_observed_monotonic_s[endpoint_id] = time.monotonic()
        if not check_staleness:
            return
        now_mono = time.monotonic()
        max_age = float(self.settings["fast_safety"]["telemetry_max_age_s"])
        if (
            self.measurement_started_monotonic_s is not None
            and now_mono - self.measurement_started_monotonic_s >= max_age
        ):
            for endpoint_id, controller in self.controllers.items():
                if controller.current_frequency_mhz >= controller.frequencies_mhz[-1]:
                    continue
                seen = self.last_sample_observed_monotonic_s[endpoint_id]
                if seen is None or now_mono - float(seen) > max_age:
                    self.process_internal_failure(
                        endpoint_id, "STALE_SAFETY_TELEMETRY", time.time()
                    )


__all__ = [
    "ENDPOINT_PHASE",
    "PHASE_ENDPOINTS",
    "EndpointFastSafetyRuntime",
    "active_candidate",
    "exploration_permissions",
]
