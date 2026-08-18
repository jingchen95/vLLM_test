"""Core value types for an endpoint-oriented XpYd runtime."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple


EndpointRole = Literal["prefill", "decode"]


class LifecycleState(str, Enum):
    """Lifecycle of a pre-instantiated serving endpoint."""

    ACTIVE = "ACTIVE"
    WARM = "WARM"
    COLD = "COLD"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class EndpointSpec:
    """Immutable placement and launch-time identity for one endpoint."""

    endpoint_id: str
    role: EndpointRole
    gpu_type: str
    node: str
    gpu_ids: Tuple[int, ...]
    tp_degree: int
    http_uri: Optional[str] = None
    kv_connector: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_ids", tuple(self.gpu_ids))
        if not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be non-empty")
        if self.role not in ("prefill", "decode"):
            raise ValueError("role must be 'prefill' or 'decode'")
        if not self.gpu_type.strip():
            raise ValueError("gpu_type must be non-empty")
        if not self.node.strip():
            raise ValueError("node must be non-empty")
        if not isinstance(self.tp_degree, int) or isinstance(self.tp_degree, bool) or self.tp_degree <= 0:
            raise ValueError("tp_degree must be a positive integer")
        if len(self.gpu_ids) != self.tp_degree:
            raise ValueError("len(gpu_ids) must equal tp_degree")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must be unique within an endpoint")
        if any(not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids must contain non-negative integers")


@dataclass
class EndpointState:
    """Mutable observations for an endpoint; it contains no transition policy.

    Queue/KV numeric defaults are not safety evidence unless their matching
    ``*_observed`` flag is true.
    """

    endpoint_id: str
    freq_mhz: Optional[int]
    lifecycle: LifecycleState
    healthy: bool
    queue_depth: int = 0
    running_requests: int = 0
    kv_cache_usage_frac: float = 0.0
    queue_depth_observed: bool = False
    kv_cache_usage_observed: bool = False
    last_update_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be non-empty")
        if self.freq_mhz is not None and (
            not isinstance(self.freq_mhz, int)
            or isinstance(self.freq_mhz, bool)
            or self.freq_mhz <= 0
        ):
            raise ValueError("freq_mhz must be a positive integer or None")
        if not isinstance(self.lifecycle, LifecycleState):
            raise ValueError("lifecycle must be a LifecycleState")
        if self.queue_depth < 0 or self.running_requests < 0:
            raise ValueError("queue and running request counts cannot be negative")
        if not 0.0 <= self.kv_cache_usage_frac <= 1.0:
            raise ValueError("kv_cache_usage_frac must be in [0, 1]")
        if not isinstance(self.queue_depth_observed, bool):
            raise ValueError("queue_depth_observed must be a boolean")
        if not isinstance(self.kv_cache_usage_observed, bool):
            raise ValueError("kv_cache_usage_observed must be a boolean")
        if self.last_update_s < 0:
            raise ValueError("last_update_s cannot be negative")


@dataclass(frozen=True)
class RoutingDecision:
    """Fast-path choice of independent P/D endpoints and DVFS setpoints."""

    prefill_endpoint_id: str
    decode_endpoint_id: str
    prefill_freq_mhz: int
    decode_freq_mhz: int

    def __post_init__(self) -> None:
        if not self.prefill_endpoint_id.strip() or not self.decode_endpoint_id.strip():
            raise ValueError("both endpoint IDs must be non-empty")
        if self.prefill_freq_mhz <= 0 or self.decode_freq_mhz <= 0:
            raise ValueError("both frequency setpoints must be positive")
