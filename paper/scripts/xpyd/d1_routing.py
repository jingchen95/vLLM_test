"""Model-free, pair-aware request routing for XPYD Stage D1.

The router deliberately uses a lexicographic hierarchy instead of a weighted
score: compatibility/health, SLO safety, live pressure, measured energy, then
controlled exploration/tie breaking.  All learned state is supplied by the
runtime controller file; no offline profile is embedded here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence


class D1RoutingError(ValueError):
    """The D1 control state cannot support a safe routing decision."""


@dataclass(frozen=True)
class RequestRoutingContext:
    request_id: str
    input_tokens: int
    requested_output_tokens: int
    request_class: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise D1RoutingError("routing context requires request_id")
        if self.input_tokens <= 0 or self.requested_output_tokens <= 0:
            raise D1RoutingError("routing token counts must be positive")
        if self.request_class not in {"SS", "SL", "LS", "LL"}:
            raise D1RoutingError("routing class must be SS, SL, LS, or LL")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class D1RoutingDecision:
    pair: tuple[str, str]
    reason: str
    context: RequestRoutingContext
    candidate_states: tuple[dict[str, Any], ...]
    selected_state: dict[str, Any]
    control_generation: Optional[str]
    decision_timestamp_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair": list(self.pair),
            "reason": self.reason,
            "context": self.context.as_dict(),
            "candidate_states": [dict(item) for item in self.candidate_states],
            "selected_state": dict(self.selected_state),
            "control_generation": self.control_generation,
            "decision_timestamp_s": self.decision_timestamp_s,
            "score_semantics": (
                "lexicographic compatibility/health -> safety -> pressure -> "
                "measured window-amortized energy -> exploration/tie"
            ),
        }


StateLoader = Callable[[], Mapping[str, Any]]


@contextmanager
def routing_state_lock(control_file: Path | str, *, exclusive: bool):
    """Serialize routing decisions with controller state publication.

    The lock file is stable across atomic replacements of the JSON control
    file.  Readers hold a shared lock for the complete routing decision;
    publishers hold an exclusive lock while committing a new visible state.
    """
    lock_path = Path(str(control_file) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class D1PairRouter:
    """Choose one complete compatible P->D pair from fresh runtime state."""

    REASONS = {
        "COLD_START_RR",
        "ONLY_SAFE_PAIR",
        "LOWER_PRESSURE",
        "ENERGY_PREFERENCE",
        "EXPLORATION",
        "FALLBACK",
        "FIXED_RR",
    }

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        state_loader: StateLoader,
        *,
        maximum_age_s: float = 30.0,
        cold_start_requests: int = 4,
        candidate_probe_interval_requests: int = 2,
        exploration_interval_requests: int = 8,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not pairs or len(set(pairs)) != len(pairs):
            raise D1RoutingError("D1 requires unique compatible pairs")
        if maximum_age_s <= 0 or cold_start_requests < 0:
            raise D1RoutingError("invalid D1 routing configuration")
        self.pairs = tuple(pairs)
        self.state_loader = state_loader
        self.maximum_age_s = float(maximum_age_s)
        self.cold_start_requests = int(cold_start_requests)
        self.candidate_probe_interval_requests = max(
            1, int(candidate_probe_interval_requests)
        )
        self.exploration_interval_requests = max(1, int(exploration_interval_requests))
        self.wall_clock = wall_clock
        self._lock = threading.Lock()
        self._index = 0
        self._generation: Optional[str] = None
        self._generation_decisions = 0
        self._pair_selections = {pair: 0 for pair in self.pairs}

    def _fresh_state(self) -> Mapping[str, Any]:
        value = self.state_loader()
        updated = value.get("updated_unix_s")
        if not _finite(updated):
            raise D1RoutingError("D1 control state lacks freshness timestamp")
        age = self.wall_clock() - float(updated)
        if age < -5.0 or age > self.maximum_age_s:
            raise D1RoutingError("D1 control state is stale")
        return value

    @staticmethod
    def _endpoint_snapshot(
        endpoint_id: str, state: Mapping[str, Any], load: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = dict((state.get("endpoints") or {}).get(endpoint_id) or {})
        active = (load.get("active_requests_by_endpoint") or {}).get(endpoint_id, 0)
        input_work = (load.get("active_input_tokens_by_endpoint") or {}).get(endpoint_id, 0)
        output_work = (load.get("active_output_tokens_by_endpoint") or {}).get(endpoint_id, 0)
        return {
            "endpoint_id": endpoint_id,
            "healthy": raw.get("healthy") is True,
            "routing_eligible": raw.get("routing_eligible", True) is True,
            "telemetry_fresh": raw.get("telemetry_fresh") is True,
            "safety_status": str(raw.get("safety_status", "UNKNOWN")),
            "requested_frequency_mhz": raw.get("requested_frequency_mhz"),
            "observed_frequency_mhz": raw.get("observed_frequency_mhz"),
            "validated_safe_frequency_mhz": raw.get("validated_safe_frequency_mhz"),
            "candidate_frequency_mhz": raw.get("candidate_frequency_mhz"),
            "recent_latency_ms": raw.get("recent_latency_ms"),
            "recent_energy_j_per_assigned_request": raw.get(
                "recent_energy_j_per_assigned_request"
            ),
            "active_requests": int(active),
            "active_input_tokens": int(input_work),
            "active_output_tokens": int(output_work),
        }

    def _pair_snapshot(
        self,
        pair: tuple[str, str],
        state: Mapping[str, Any],
        load: Mapping[str, Any],
        context: RequestRoutingContext,
    ) -> dict[str, Any]:
        p = self._endpoint_snapshot(pair[0], state, load)
        d = self._endpoint_snapshot(pair[1], state, load)
        by_class = state.get("class_pair_feedback") or {}
        feedback = dict((by_class.get(context.request_class) or {}).get(
            "%s->%s" % pair
        ) or {})
        endpoint_safe = (
            p["healthy"] and d["healthy"]
            and p["routing_eligible"] and d["routing_eligible"]
            and p["telemetry_fresh"] and d["telemetry_fresh"]
            and p["safety_status"] == "VALIDATED_SAFE"
            and d["safety_status"] == "VALIDATED_SAFE"
        )
        known_safe = endpoint_safe and feedback.get("slo_safe", True) is not False
        candidate_probe = (
            p["healthy"] and d["healthy"]
            and p["routing_eligible"] and d["routing_eligible"]
            and p["telemetry_fresh"] and d["telemetry_fresh"]
            and p["safety_status"] in {"VALIDATED_SAFE", "CANDIDATE_PROBE"}
            and d["safety_status"] in {"VALIDATED_SAFE", "CANDIDATE_PROBE"}
            and "UNSAFE" not in {p["safety_status"], d["safety_status"]}
            and feedback.get("slo_safe", True) is not False
        )
        pressure = p["active_requests"] + d["active_requests"]
        token_pressure = p["active_input_tokens"] + d["active_output_tokens"]
        energy_values = [
            p["recent_energy_j_per_assigned_request"],
            d["recent_energy_j_per_assigned_request"],
        ]
        energy_known = all(_finite(item) for item in energy_values)
        return {
            "pair": list(pair),
            "prefill": p,
            "decode": d,
            "healthy": p["healthy"] and d["healthy"],
            "routing_eligible": (
                p["routing_eligible"] and d["routing_eligible"]
            ),
            "endpoint_safe": endpoint_safe,
            "known_safe": known_safe,
            "candidate_probe": candidate_probe,
            "pressure": pressure,
            "token_pressure": token_pressure,
            "measured_energy_j_per_assigned_request": (
                sum(float(item) for item in energy_values) if energy_known else None
            ),
            "class_observations": int(feedback.get("observations", 0)),
            "class_recent_ttft_ms": feedback.get("recent_ttft_ms"),
            "class_recent_tpot_ms": feedback.get("recent_tpot_ms"),
            "class_slo_safe": feedback.get("slo_safe"),
        }

    def _rr(self, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
        allowed = {tuple(item["pair"]): item for item in candidates}
        for offset in range(len(self.pairs)):
            pair = self.pairs[(self._index + offset) % len(self.pairs)]
            if pair in allowed:
                self._index = (self.pairs.index(pair) + 1) % len(self.pairs)
                return allowed[pair]
        raise D1RoutingError("no compatible pair available")

    def choose(
        self,
        context: RequestRoutingContext,
        load_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> D1RoutingDecision:
        with self._lock:
            state = self._fresh_state()
            generation = str(state.get("generation", "default"))
            if generation != self._generation:
                self._generation = generation
                self._generation_decisions = 0
                self._pair_selections = {pair: 0 for pair in self.pairs}
            load = dict(load_snapshot or {})
            snapshots = [
                self._pair_snapshot(pair, state, load, context) for pair in self.pairs
            ]
            eligible = [
                item for item in snapshots
                if item["healthy"] and item["routing_eligible"]
            ]
            if not eligible:
                raise D1RoutingError("no routing-eligible compatible D1 pair")
            mode = str(state.get("routing_mode", "RUNTIME_FEEDBACK"))
            if mode == "FIXED_RR":
                selected, reason = self._rr(eligible), "FIXED_RR"
            elif self._generation_decisions < self.cold_start_requests:
                # Cold start is the sole exception to known-safe preference. It
                # is allowed only over healthy, fresh endpoints and supplies the
                # observations that the controller did not have a priori.
                fresh = [
                    item for item in eligible
                    if item["prefill"]["telemetry_fresh"]
                    and item["decode"]["telemetry_fresh"]
                    and item["prefill"]["safety_status"] != "UNSAFE"
                    and item["decode"]["safety_status"] != "UNSAFE"
                ]
                if not fresh:
                    raise D1RoutingError("cold start has no fresh non-unsafe pair")
                selected, reason = self._rr(fresh), "COLD_START_RR"
            else:
                safe = [item for item in eligible if item["known_safe"]]
                probes = [
                    item for item in eligible
                    if item["candidate_probe"] and not item["known_safe"]
                ]
                if not safe:
                    fallback = [item for item in eligible if item["endpoint_safe"]]
                    if not fallback:
                        raise D1RoutingError(
                            "no known-safe pair; unknown/stale candidates fail closed"
                        )
                    minimum_pressure = min(item["pressure"] for item in fallback)
                    selected = self._rr([
                        item for item in fallback
                        if item["pressure"] == minimum_pressure
                    ])
                    reason = "FALLBACK"
                    pair = tuple(selected["pair"])
                    self._generation_decisions += 1
                    self._pair_selections[pair] += 1
                    return D1RoutingDecision(
                        pair=pair,
                        reason=reason,
                        context=context,
                        candidate_states=tuple(snapshots),
                        selected_state=dict(selected),
                        control_generation=self._generation,
                        decision_timestamp_s=self.wall_clock(),
                    )
                minimum_pressure = min(item["pressure"] for item in safe)
                pressure_candidates = [
                    item for item in safe if item["pressure"] == minimum_pressure
                ]
                pressure_redirected = len(pressure_candidates) < len(safe)
                probe_due = (
                    probes
                    and self._generation_decisions
                    % self.candidate_probe_interval_requests == 0
                    and min(item["pressure"] for item in probes) <= minimum_pressure
                )
                refresh_due = (
                    self._generation_decisions % self.exploration_interval_requests == 0
                    and len(pressure_candidates) > 1
                )
                if probe_due:
                    probe_pressure = min(item["pressure"] for item in probes)
                    eligible_probes = [
                        item for item in probes if item["pressure"] == probe_pressure
                    ]
                    minimum = min(item["class_observations"] for item in eligible_probes)
                    selected = self._rr([
                        item for item in eligible_probes
                        if item["class_observations"] == minimum
                    ])
                    reason = "EXPLORATION"
                elif len(safe) == 1:
                    selected, reason = safe[0], "ONLY_SAFE_PAIR"
                elif pressure_redirected:
                    selected, reason = self._rr(pressure_candidates), "LOWER_PRESSURE"
                elif refresh_due:
                    least = min(
                        self._pair_selections[tuple(item["pair"])]
                        for item in pressure_candidates
                    )
                    selected = self._rr([
                        item for item in pressure_candidates
                        if self._pair_selections[tuple(item["pair"])] == least
                    ])
                    reason = "EXPLORATION"
                else:
                    energies = [
                        item for item in pressure_candidates
                        if _finite(item["measured_energy_j_per_assigned_request"])
                    ]
                    if energies:
                        minimum_energy = min(
                            float(item["measured_energy_j_per_assigned_request"])
                            for item in energies
                        )
                        best = [
                            item for item in energies
                            if math.isclose(
                                float(item["measured_energy_j_per_assigned_request"]),
                                minimum_energy,
                                rel_tol=1e-9,
                                abs_tol=1e-9,
                            )
                        ]
                        selected, reason = self._rr(best), "ENERGY_PREFERENCE"
                    else:
                        # Runtime-only feedback has not separated the choices
                        # yet; controlled RR keeps them observed.
                        selected, reason = self._rr(pressure_candidates), "EXPLORATION"
            pair = tuple(selected["pair"])
            self._generation_decisions += 1
            self._pair_selections[pair] += 1
            return D1RoutingDecision(
                pair=pair,
                reason=reason,
                context=context,
                candidate_states=tuple(snapshots),
                selected_state=dict(selected),
                control_generation=self._generation,
                decision_timestamp_s=self.wall_clock(),
            )
