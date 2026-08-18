"""Architecture-only primitives for future XpYd disaggregated serving."""

from xpyd.compatibility import CompatibilityTable, ConnectorCompatibility
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
    "DuplicateEndpointError",
    "EndpointRegistry",
    "EndpointRole",
    "EndpointSpec",
    "EndpointState",
    "GPUAllocationConflictError",
    "GPUTypeProfile",
    "HardwareProfile",
    "LifecycleState",
    "NodeProfile",
    "RoutingDecision",
    "from_current_gpu_specs",
    "from_gpu_specs",
]
