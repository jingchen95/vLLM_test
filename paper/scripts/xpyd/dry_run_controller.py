"""Non-actuating orchestration for live or fixture-backed XpYd dry runs."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from xpyd.compatibility import CompatibilityTable, ConnectorCompatibility
from xpyd.feedback_scheduler import (
    DVFSAction,
    DVFSRecommendation,
    FeedbackScheduler,
    FeedbackSchedulerConfig,
    NoEligibleRouteError,
    RouteEvaluation,
)
from xpyd.hardware import GPUTypeProfile, HardwareProfile, NodeProfile
from xpyd.registry import EndpointRegistry
from xpyd.telemetry import TelemetryAggregator
from xpyd.types import EndpointSpec, EndpointState, LifecycleState, RoutingDecision
from xpyd.vllm_metrics import (
    VLLMMetricsCollector,
    VLLMMetricsSnapshot,
    VLLMWindowDelta,
    VLLMWindowTracker,
    window_delta_to_telemetry_sample,
)


SnapshotProvider = Callable[[EndpointSpec], VLLMMetricsSnapshot]


@dataclass(frozen=True)
class EndpointDryRunRecord:
    endpoint_id: str
    role: str
    tp_degree: int
    node: str
    http_uri: Optional[str]
    scrape_ok: bool
    scrape_error: Optional[str]
    missing_metrics: Tuple[str, ...]
    queue_depth: int
    queue_depth_observed: bool
    running_requests: int
    kv_cache_usage_frac: float
    kv_cache_usage_observed: bool
    healthy: bool
    lifecycle: str
    observed_freq_mhz: Optional[int]
    state_last_update_s: float
    telemetry_fresh: bool
    last_observation_age_s: Optional[float]
    ewma_ttft_ms: Optional[float]
    ewma_tbt_ms: Optional[float]
    ewma_tpot_ms: Optional[float]
    window_ttft_p95_ms: Optional[float]
    window_ttft_p99_ms: Optional[float]
    window_tbt_p95_ms: Optional[float]
    window_tbt_p99_ms: Optional[float]
    window_ttft_count: int
    window_tbt_count: int
    window_valid: bool
    window_reason: str


@dataclass(frozen=True)
class DryRunDecisionRecord:
    timestamp_s: float
    endpoints: Tuple[EndpointDryRunRecord, ...]
    pair_evaluations: Tuple[RouteEvaluation, ...]
    selected_route: Optional[RoutingDecision]
    route_error: Optional[str]
    fallback_used: bool
    dvfs_recommendations: Tuple[DVFSRecommendation, ...]
    actuated: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s,
            "endpoints": [asdict(endpoint) for endpoint in self.endpoints],
            "pair_evaluations": [asdict(pair) for pair in self.pair_evaluations],
            "selected_route": (
                asdict(self.selected_route) if self.selected_route is not None else None
            ),
            "route_error": self.route_error,
            "fallback_used": self.fallback_used,
            "dvfs_recommendations": [
                {
                    **asdict(recommendation),
                    "action": recommendation.action.value,
                }
                for recommendation in self.dvfs_recommendations
            ],
            "actuated": False,
        }


class DryRunController:
    """Scrape, derive, evaluate, and log; never route or actuate."""

    def __init__(
        self,
        registry: EndpointRegistry,
        telemetry: TelemetryAggregator,
        scheduler: FeedbackScheduler,
        collector: VLLMMetricsCollector,
        window_tracker: Optional[VLLMWindowTracker] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.registry = registry
        self.telemetry = telemetry
        self.scheduler = scheduler
        self.collector = collector
        self.window_tracker = window_tracker or VLLMWindowTracker()
        self.clock = clock

    def run_once(
        self,
        snapshot_provider: Optional[SnapshotProvider] = None,
        now_s: Optional[float] = None,
        workload_context_key: Optional[str] = None,
    ) -> DryRunDecisionRecord:
        now_s = self.clock() if now_s is None else float(now_s)
        provider = snapshot_provider or self.collector.scrape
        scrape_status: Dict[str, Tuple[bool, Optional[str], Tuple[str, ...]]] = {}
        window_status: Dict[str, VLLMWindowDelta] = {}

        for endpoint in sorted(
            self.registry.list_endpoints(),
            key=lambda item: item.endpoint_id,
        ):
            try:
                raw = provider(endpoint)
                if raw.endpoint_id != endpoint.endpoint_id:
                    raise ValueError("snapshot endpoint_id does not match requested endpoint")
                self._synchronize_state(raw)
                window = self.window_tracker.observe(raw)
                window_status[endpoint.endpoint_id] = window
                if window.valid:
                    self.telemetry.observe(window_delta_to_telemetry_sample(raw, window))
                scrape_status[endpoint.endpoint_id] = (True, None, raw.missing_metrics)
            except Exception as exc:  # collection boundary: record and continue
                self._mark_unhealthy(endpoint.endpoint_id)
                scrape_status[endpoint.endpoint_id] = (
                    False,
                    "%s: %s" % (type(exc).__name__, exc),
                    (),
                )
                window_status[endpoint.endpoint_id] = VLLMWindowDelta(
                    endpoint_id=endpoint.endpoint_id,
                    start_timestamp_s=None,
                    end_timestamp_s=now_s,
                    interval_s=None,
                    valid=False,
                    reason="scrape_failed",
                    reset_metrics=(),
                )

        pair_evaluations = self.scheduler.evaluate_routes(
            now_s=now_s,
            workload_context_key=workload_context_key,
        )
        eligible = tuple(pair for pair in pair_evaluations if pair.eligible)
        route = None
        route_error = None
        fallback_used = False
        try:
            route = self.scheduler.choose_route(
                now_s=now_s,
                workload_context_key=workload_context_key,
            )
            fallback_used = not bool(eligible)
        except (NoEligibleRouteError, KeyError, ValueError) as exc:
            route_error = "%s: %s" % (type(exc).__name__, exc)

        recommendations = []
        for endpoint in sorted(
            self.registry.list_endpoints(),
            key=lambda item: item.endpoint_id,
        ):
            try:
                recommendations.append(
                    self.scheduler.choose_frequency_adjustment(
                        endpoint.endpoint_id,
                        now_s=now_s,
                    )
                )
            except (KeyError, ValueError) as exc:
                state = self.registry.get_state(endpoint.endpoint_id)
                if state.freq_mhz is not None:
                    recommendations.append(
                        DVFSRecommendation(
                            endpoint_id=endpoint.endpoint_id,
                            current_freq_mhz=state.freq_mhz,
                            target_freq_mhz=state.freq_mhz,
                            action=DVFSAction.HOLD,
                            reason="recommendation_error: %s" % exc,
                        )
                    )

        endpoint_records = tuple(
            self._endpoint_record(
                endpoint,
                now_s,
                scrape_status[endpoint.endpoint_id],
                window_status[endpoint.endpoint_id],
            )
            for endpoint in sorted(
                self.registry.list_endpoints(),
                key=lambda item: item.endpoint_id,
            )
        )
        return DryRunDecisionRecord(
            timestamp_s=now_s,
            endpoints=endpoint_records,
            pair_evaluations=pair_evaluations,
            selected_route=route,
            route_error=route_error,
            fallback_used=fallback_used,
            dvfs_recommendations=tuple(recommendations),
            actuated=False,
        )

    def append_jsonl(self, path: Path, record: DryRunDecisionRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def _synchronize_state(self, raw: VLLMMetricsSnapshot) -> None:
        state = self.registry.get_state(raw.endpoint_id)
        state.healthy = True
        state.last_update_s = raw.timestamp_s
        if raw.num_requests_waiting is not None:
            state.queue_depth = self._integer_gauge(
                raw.num_requests_waiting,
                "vllm:num_requests_waiting",
            )
            state.queue_depth_observed = True
        if raw.num_requests_running is not None:
            state.running_requests = self._integer_gauge(
                raw.num_requests_running,
                "vllm:num_requests_running",
            )
        if raw.kv_cache_usage_frac is not None:
            state.kv_cache_usage_frac = raw.kv_cache_usage_frac
            state.kv_cache_usage_observed = True
        # Missing gauges intentionally preserve prior values and their
        # observed/unknown status.
        self.registry.update_state(state)

    def _mark_unhealthy(self, endpoint_id: str) -> None:
        state = self.registry.get_state(endpoint_id)
        state.healthy = False
        # last_update_s remains the last successful scrape time.
        self.registry.update_state(state)

    @staticmethod
    def _integer_gauge(value: float, name: str) -> int:
        if value < 0 or not float(value).is_integer():
            raise ValueError("%s must be a non-negative integer gauge" % name)
        return int(value)

    def _endpoint_record(
        self,
        endpoint: EndpointSpec,
        now_s: float,
        scrape_status: Tuple[bool, Optional[str], Tuple[str, ...]],
        window: VLLMWindowDelta,
    ) -> EndpointDryRunRecord:
        state = self.registry.get_state(endpoint.endpoint_id)
        snapshot = self.telemetry.snapshot(endpoint.endpoint_id, now_s=now_s)
        fresh = (
            snapshot.last_observation_age_s is not None
            and snapshot.last_observation_age_s
            <= self.scheduler.config.telemetry_max_age_s
        )
        return EndpointDryRunRecord(
            endpoint_id=endpoint.endpoint_id,
            role=endpoint.role,
            tp_degree=endpoint.tp_degree,
            node=endpoint.node,
            http_uri=endpoint.http_uri,
            scrape_ok=scrape_status[0],
            scrape_error=scrape_status[1],
            missing_metrics=scrape_status[2],
            queue_depth=state.queue_depth,
            queue_depth_observed=state.queue_depth_observed,
            running_requests=state.running_requests,
            kv_cache_usage_frac=state.kv_cache_usage_frac,
            kv_cache_usage_observed=state.kv_cache_usage_observed,
            healthy=state.healthy,
            lifecycle=state.lifecycle.value,
            observed_freq_mhz=state.freq_mhz,
            state_last_update_s=state.last_update_s,
            telemetry_fresh=fresh,
            last_observation_age_s=snapshot.last_observation_age_s,
            ewma_ttft_ms=snapshot.ewma_ttft_ms,
            ewma_tbt_ms=snapshot.ewma_tbt_ms,
            ewma_tpot_ms=snapshot.ewma_tpot_ms,
            window_ttft_p95_ms=snapshot.window_ttft_p95_ms,
            window_ttft_p99_ms=snapshot.window_ttft_p99_ms,
            window_tbt_p95_ms=snapshot.window_tbt_p95_ms,
            window_tbt_p99_ms=snapshot.window_tbt_p99_ms,
            window_ttft_count=snapshot.window_ttft_count,
            window_tbt_count=snapshot.window_tbt_count,
            window_valid=window.valid,
            window_reason=window.reason,
        )


def build_controller_from_config(
    config: Mapping[str, object],
    clock: Callable[[], float] = time.time,
) -> DryRunController:
    """Build arbitrary multi-node/multi-endpoint inventory from JSON data."""

    hardware = HardwareProfile(
        gpu_types=(GPUTypeProfile(**item) for item in config["gpu_types"]),
        nodes=(NodeProfile(**item) for item in config["nodes"]),
    )
    registry = EndpointRegistry(
        allow_gpu_overlap=bool(config.get("allow_gpu_overlap", False))
    )
    for item in config["endpoints"]:
        item = dict(item)
        queue_depth_present = "queue_depth" in item
        kv_cache_usage_present = "kv_cache_usage_frac" in item
        queue_depth_observed = bool(
            item.pop("queue_depth_observed", queue_depth_present)
        )
        kv_cache_usage_observed = bool(
            item.pop("kv_cache_usage_observed", kv_cache_usage_present)
        )
        state = EndpointState(
            endpoint_id=item["endpoint_id"],
            freq_mhz=item.pop("freq_mhz", None),
            lifecycle=LifecycleState(item.pop("lifecycle", "ACTIVE")),
            healthy=bool(item.pop("healthy", True)),
            queue_depth=int(item.pop("queue_depth", 0)),
            running_requests=int(item.pop("running_requests", 0)),
            kv_cache_usage_frac=float(item.pop("kv_cache_usage_frac", 0.0)),
            queue_depth_observed=queue_depth_observed,
            kv_cache_usage_observed=kv_cache_usage_observed,
            last_update_s=float(item.pop("last_update_s", 0.0)),
        )
        registry.register(EndpointSpec(**item), state)
    compatibility = CompatibilityTable(
        ConnectorCompatibility(**item) for item in config["compatibility"]
    )
    telemetry = TelemetryAggregator(alpha=float(config.get("telemetry_alpha", 0.2)))
    scheduler_config = FeedbackSchedulerConfig(**dict(config["scheduler"]))
    scheduler = FeedbackScheduler(
        registry,
        compatibility,
        telemetry,
        hardware,
        scheduler_config,
        clock=clock,
    )
    collector_config = dict(config.get("collector", {}))
    collector = VLLMMetricsCollector(clock=clock, **collector_config)
    return DryRunController(
        registry,
        telemetry,
        scheduler,
        collector,
        clock=clock,
    )


def _fixture_frames(path: Path) -> Iterable[Tuple[float, Dict[str, Path]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for frame in data["frames"]:
        yield (
            float(frame["timestamp_s"]),
            {
                endpoint_id: (path.parent / relative_path).resolve()
                for endpoint_id, relative_path in frame["metrics"].items()
            },
        )


def run_fixture_replay(
    controller: DryRunController,
    manifest_path: Path,
    output_path: Optional[Path] = None,
) -> Tuple[DryRunDecisionRecord, ...]:
    """Replay local text fixtures through the same collector mapping path."""

    records = []
    for timestamp_s, metrics_paths in _fixture_frames(manifest_path):
        def provider(endpoint: EndpointSpec) -> VLLMMetricsSnapshot:
            if endpoint.endpoint_id not in metrics_paths:
                raise ValueError("fixture frame missing endpoint %s" % endpoint.endpoint_id)
            text = metrics_paths[endpoint.endpoint_id].read_text(encoding="utf-8")
            return controller.collector.parse_text(endpoint, text, timestamp_s)

        record = controller.run_once(snapshot_provider=provider, now_s=timestamp_s)
        records.append(record)
        if output_path is not None:
            controller.append_jsonl(output_path, record)
    return tuple(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-actuating XpYd dry-run controller")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.interval <= 0 or args.duration <= 0:
        raise SystemExit("--interval and --duration must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    controller = build_controller_from_config(config)
    if args.fixture_manifest is not None:
        records = run_fixture_replay(controller, args.fixture_manifest, args.output)
        print(json.dumps(records[-1].to_dict(), indent=2, sort_keys=True))
        return 0

    deadline = time.time() + args.duration
    while True:
        record = controller.run_once()
        controller.append_jsonl(args.output, record)
        print(json.dumps(record.to_dict(), sort_keys=True))
        if time.time() + args.interval > deadline:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
