"""Minimal context-aware safe probing for measured route costs.

The cost is a control-window statistic, not physical request-level energy:
total gross energy of every GPU board divided by logical requests completed in
the window.  This module performs no I/O, prediction, or actuation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Optional, Sequence


Route = tuple[str, str]


def probe_safe_routes(
    context: Any, evaluations: Sequence[Any], now_s: float,
    headroom_fraction: float,
) -> tuple[list[list[str]], dict[str, list[str]]]:
    """Apply the validated strict probe gate to route evaluations.

    This remains separate from route ranking: compatibility/eligibility,
    endpoint health, queue/KV pressure, fresh role latency, and SLO headroom
    must all pass before a route may be explored.
    """
    safe: list[list[str]] = []
    reasons: dict[str, list[str]] = {}
    scheduler = context.scheduler
    for evaluation in evaluations:
        route = [evaluation.prefill_endpoint_id, evaluation.decode_endpoint_id]
        key = "%s->%s" % tuple(route)
        failures: list[str] = []
        if not evaluation.eligible:
            failures.append(str(evaluation.reason))
        for endpoint_id in route:
            spec = context.registry.get_spec(endpoint_id)
            state = context.registry.get_state(endpoint_id)
            snapshot = context.telemetry.snapshot(endpoint_id, now_s=now_s)
            if not state.healthy:
                failures.append("%s_unhealthy" % endpoint_id)
            if (
                not state.queue_depth_observed
                or state.queue_depth > scheduler.config.dvfs_low_queue_depth
            ):
                failures.append("%s_queue_not_low" % endpoint_id)
            if (
                not state.kv_cache_usage_observed
                or state.kv_cache_usage_frac
                > scheduler.config.dvfs_low_kv_cache_usage_frac
            ):
                failures.append("%s_kv_not_low" % endpoint_id)
            if spec.role == "prefill":
                metric = snapshot.ewma_ttft_ms
                observed_at = snapshot.last_ttft_observation_s
                slo = scheduler.config.ttft_slo_ms
            else:
                metric = snapshot.ewma_tpot_ms
                observed_at = snapshot.last_tpot_observation_s
                slo = scheduler.config.tpot_slo_ms
            if (
                observed_at is None
                or now_s - observed_at > scheduler.config.telemetry_max_age_s
            ):
                failures.append("%s_latency_missing_or_stale" % endpoint_id)
            if metric is None or slo is None or metric > headroom_fraction * slo:
                failures.append("%s_probe_headroom_insufficient" % endpoint_id)
        reasons[key] = sorted(set(failures))
        if not failures:
            safe.append(route)
    return safe, reasons


@dataclass(frozen=True)
class RouteCostSnapshot:
    context_key: str
    prefill_endpoint_id: str
    decode_endpoint_id: str
    ewma_system_joules_per_request: float
    sample_count: int
    last_observation_s: float
    last_observation_sequence: int
    age_s: float
    age_windows: int
    fresh: bool


@dataclass(frozen=True)
class RouteProbeDecision:
    route: Optional[Route]
    mode: str
    probe: bool
    freeze_dvfs: bool
    reason: str
    cost_joules_per_request: Optional[float]


@dataclass
class _RouteCostState:
    value: float
    sample_count: int
    last_observation_s: float
    last_observation_sequence: int


class ContextRouteCostStore:
    """EWMA route costs isolated by explicit workload context."""

    def __init__(
        self, alpha: float = 0.5, maximum_age_s: float = 600.0,
        maximum_age_windows: int = 4,
    ) -> None:
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not math.isfinite(maximum_age_s) or maximum_age_s <= 0.0:
            raise ValueError("maximum_age_s must be positive")
        if maximum_age_windows <= 0:
            raise ValueError("maximum_age_windows must be positive")
        self.alpha = float(alpha)
        self.maximum_age_s = float(maximum_age_s)
        self.maximum_age_windows = int(maximum_age_windows)
        self._states: dict[tuple[str, Route], _RouteCostState] = {}

    @staticmethod
    def _normalize_route(route: Sequence[str]) -> Route:
        if len(route) != 2 or not all(str(item) for item in route):
            raise ValueError("route must contain non-empty P and D endpoint IDs")
        return str(route[0]), str(route[1])

    def observe(
        self, context_key: str, route: Sequence[str], *,
        total_system_gross_energy_j: float, logical_requests: int,
        timestamp_s: float, sequence: int,
    ) -> RouteCostSnapshot:
        if not str(context_key):
            raise ValueError("context_key must be non-empty")
        if logical_requests <= 0:
            raise ValueError("logical_requests must be positive")
        if not math.isfinite(total_system_gross_energy_j) or total_system_gross_energy_j < 0:
            raise ValueError("total_system_gross_energy_j must be finite and nonnegative")
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if sequence <= 0:
            raise ValueError("sequence must be positive")
        normalized = self._normalize_route(route)
        key = (str(context_key), normalized)
        measurement = float(total_system_gross_energy_j) / int(logical_requests)
        previous = self._states.get(key)
        if previous is not None:
            if timestamp_s < previous.last_observation_s:
                raise ValueError("route-cost timestamps must be non-decreasing")
            if sequence <= previous.last_observation_sequence:
                raise ValueError("route-cost sequences must increase")
            value = self.alpha * measurement + (1.0 - self.alpha) * previous.value
            count = previous.sample_count + 1
        else:
            value = measurement
            count = 1
        self._states[key] = _RouteCostState(
            value=value,
            sample_count=count,
            last_observation_s=float(timestamp_s),
            last_observation_sequence=int(sequence),
        )
        return self.snapshot(str(context_key), normalized, timestamp_s, sequence)  # type: ignore[return-value]

    def snapshot(
        self, context_key: str, route: Sequence[str], now_s: float,
        sequence: int,
    ) -> Optional[RouteCostSnapshot]:
        normalized = self._normalize_route(route)
        state = self._states.get((str(context_key), normalized))
        if state is None:
            return None
        age_s = max(0.0, float(now_s) - state.last_observation_s)
        age_windows = max(0, int(sequence) - state.last_observation_sequence)
        return RouteCostSnapshot(
            context_key=str(context_key),
            prefill_endpoint_id=normalized[0],
            decode_endpoint_id=normalized[1],
            ewma_system_joules_per_request=state.value,
            sample_count=state.sample_count,
            last_observation_s=state.last_observation_s,
            last_observation_sequence=state.last_observation_sequence,
            age_s=age_s,
            age_windows=age_windows,
            fresh=(
                age_s <= self.maximum_age_s
                and age_windows <= self.maximum_age_windows
            ),
        )

    def context_snapshots(
        self, context_key: str, routes: Iterable[Sequence[str]],
        now_s: float, sequence: int,
    ) -> tuple[RouteCostSnapshot, ...]:
        values = [
            self.snapshot(context_key, route, now_s, sequence) for route in routes
        ]
        return tuple(item for item in values if item is not None)

    def as_dict(
        self, context_key: str, routes: Iterable[Sequence[str]],
        now_s: float, sequence: int,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for route in routes:
            normalized = self._normalize_route(route)
            snapshot = self.snapshot(context_key, normalized, now_s, sequence)
            result["%s->%s" % normalized] = (
                asdict(snapshot) if snapshot is not None else None
            )
        return result


class SafeRouteProber:
    """Probe unseen/stale safe routes, otherwise exploit recent measured cost."""

    def __init__(
        self, costs: ContextRouteCostStore, minimum_probe_interval_windows: int = 1,
        fallback_route: Route = ("P0", "D0"),
    ) -> None:
        if minimum_probe_interval_windows <= 0:
            raise ValueError("minimum_probe_interval_windows must be positive")
        self.costs = costs
        self.minimum_probe_interval_windows = int(minimum_probe_interval_windows)
        self.fallback_route = costs._normalize_route(fallback_route)
        self._last_probe_sequence: dict[str, int] = {}

    @staticmethod
    def _routes(values: Iterable[Sequence[str]]) -> tuple[Route, ...]:
        return tuple(sorted({ContextRouteCostStore._normalize_route(item) for item in values}))

    def _endpoint_observation_counts(
        self, context_key: str, routes: Sequence[Route], now_s: float,
        sequence: int,
    ) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for route in routes:
            snapshot = self.costs.snapshot(context_key, route, now_s, sequence)
            if snapshot is None:
                continue
            counts[route[0]] = counts.get(route[0], 0) + snapshot.sample_count
            counts[route[1]] = counts.get(route[1], 0) + snapshot.sample_count
        return counts

    def choose(
        self, context_key: str, *, compatible_routes: Iterable[Sequence[str]],
        eligible_routes: Iterable[Sequence[str]],
        probe_safe_routes: Iterable[Sequence[str]], now_s: float,
        sequence: int,
    ) -> RouteProbeDecision:
        compatible = self._routes(compatible_routes)
        eligible = self._routes(eligible_routes)
        probe_safe = set(self._routes(probe_safe_routes))
        eligible_set = set(eligible)
        unseen: list[tuple[Route, Optional[RouteCostSnapshot]]] = []
        stale: list[tuple[Route, Optional[RouteCostSnapshot]]] = []
        for route in compatible:
            if route not in eligible_set or route not in probe_safe:
                continue
            snapshot = self.costs.snapshot(context_key, route, now_s, sequence)
            if snapshot is None:
                unseen.append((route, snapshot))
            elif not snapshot.fresh:
                stale.append((route, snapshot))
        last_probe = self._last_probe_sequence.get(str(context_key))
        probe_due = (
            last_probe is None
            or sequence - last_probe >= self.minimum_probe_interval_windows
        )
        # The accepted all-route safety warmup gives unseen routes a short,
        # bounded opportunity to obtain their first observation.  Once every
        # route has been seen, stale refreshes obey the probe interval so an
        # exploitation/DVFS window separates recurring probes.
        refresh = unseen if unseen else (stale if probe_due else [])
        if refresh:
            endpoint_counts = self._endpoint_observation_counts(
                context_key, compatible, now_s, sequence
            )

            def refresh_key(item: tuple[Route, Optional[RouteCostSnapshot]]) -> tuple[object, ...]:
                route, snapshot = item
                is_unseen = snapshot is None
                if is_unseen:
                    balance = endpoint_counts.get(route[0], 0) + endpoint_counts.get(route[1], 0)
                    return (0, balance, route)
                return (1, snapshot.last_observation_sequence, route)

            route, snapshot = min(refresh, key=refresh_key)
            self._last_probe_sequence[str(context_key)] = int(sequence)
            return RouteProbeDecision(
                route=route,
                mode="PROBE_UNSEEN" if snapshot is None else "PROBE_STALE",
                probe=True,
                freeze_dvfs=True,
                reason=(
                    "compatible route has no context-specific system-cost observation"
                    if snapshot is None
                    else "context-specific route cost exceeded freshness limit"
                ),
                cost_joules_per_request=(
                    snapshot.ewma_system_joules_per_request if snapshot else None
                ),
            )

        measured = []
        for route in eligible:
            snapshot = self.costs.snapshot(context_key, route, now_s, sequence)
            if snapshot is not None and snapshot.fresh:
                measured.append((snapshot.ewma_system_joules_per_request, route))
        if measured:
            cost, route = min(measured, key=lambda item: (item[0], item[1]))
            return RouteProbeDecision(
                route=route, mode="EXPLOIT_RECENT", probe=False,
                freeze_dvfs=False,
                reason="minimum recent context-specific measured system cost",
                cost_joules_per_request=cost,
            )
        if self.fallback_route in eligible_set:
            return RouteProbeDecision(
                route=self.fallback_route, mode="SAFE_FALLBACK_NO_PROBE",
                probe=False, freeze_dvfs=False,
                reason="probing unsafe and no recent route cost; use explicit safe fallback",
                cost_joules_per_request=None,
            )
        return RouteProbeDecision(
            route=None, mode="NO_ELIGIBLE_ROUTE", probe=False,
            freeze_dvfs=False,
            reason="no compatible measured-safe route",
            cost_joules_per_request=None,
        )
