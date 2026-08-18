"""Architecture-only primitives for future XpYd disaggregated serving."""

from xpyd.compatibility import CompatibilityTable, ConnectorCompatibility
from xpyd.feedback_scheduler import (
    DVFSAction,
    DVFSRecommendation,
    FeedbackScheduler,
    FeedbackSchedulerConfig,
    NoEligibleRouteError,
    RouteCandidate,
)
from xpyd.hardware import (
    GPUTypeProfile,
    HardwareProfile,
    NodeProfile,
    from_current_gpu_specs,
    from_gpu_specs,
)
from xpyd.registry import (
    DuplicateEndpointError,
    EndpointRegistry,
    GPUAllocationConflictError,
)
from xpyd.telemetry import (
    EWMA,
    EndpointTelemetrySample,
    EndpointTelemetrySnapshot,
    TelemetryAggregator,
)
from xpyd.types import (
    EndpointRole,
    EndpointSpec,
    EndpointState,
    LifecycleState,
    RoutingDecision,
)

__all__ = [
    "CompatibilityTable",
    "ConnectorCompatibility",
    "DVFSAction",
    "DVFSRecommendation",
    "DuplicateEndpointError",
    "EndpointRegistry",
    "EndpointRole",
    "EndpointSpec",
    "EndpointState",
    "EndpointTelemetrySample",
    "EndpointTelemetrySnapshot",
    "EWMA",
    "FeedbackScheduler",
    "FeedbackSchedulerConfig",
    "GPUAllocationConflictError",
    "GPUTypeProfile",
    "HardwareProfile",
    "LifecycleState",
    "NodeProfile",
    "NoEligibleRouteError",
    "RouteCandidate",
    "RoutingDecision",
    "TelemetryAggregator",
    "from_current_gpu_specs",
    "from_gpu_specs",
]
