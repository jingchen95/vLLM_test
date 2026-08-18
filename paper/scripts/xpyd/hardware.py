"""Hardware inventory abstractions independent of concrete GPU product names."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple

from xpyd.types import EndpointSpec


@dataclass(frozen=True)
class GPUTypeProfile:
    gpu_type: str
    allowed_frequencies_mhz: Tuple[int, ...]
    max_frequency_mhz: int
    nominal_tdp_w: float
    idle_power_w: float
    supported_tp_degrees: Tuple[int, ...]

    def __post_init__(self) -> None:
        frequencies = tuple(self.allowed_frequencies_mhz)
        tp_degrees = tuple(self.supported_tp_degrees)
        object.__setattr__(self, "allowed_frequencies_mhz", frequencies)
        object.__setattr__(self, "supported_tp_degrees", tp_degrees)
        if not self.gpu_type.strip():
            raise ValueError("gpu_type must be non-empty")
        if not frequencies or any(freq <= 0 for freq in frequencies):
            raise ValueError("allowed frequencies must be non-empty and positive")
        if len(set(frequencies)) != len(frequencies):
            raise ValueError("allowed frequencies must be unique")
        if self.max_frequency_mhz != max(frequencies):
            raise ValueError("max_frequency_mhz must equal the largest allowed frequency")
        if self.nominal_tdp_w <= 0 or self.idle_power_w < 0:
            raise ValueError("nominal TDP must be positive and idle power non-negative")
        if not tp_degrees or any(tp <= 0 for tp in tp_degrees):
            raise ValueError("supported TP degrees must be non-empty and positive")
        if len(set(tp_degrees)) != len(tp_degrees):
            raise ValueError("supported TP degrees must be unique")


@dataclass(frozen=True)
class NodeProfile:
    node: str
    gpu_type: str
    gpu_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        gpu_ids = tuple(self.gpu_ids)
        object.__setattr__(self, "gpu_ids", gpu_ids)
        if not self.node.strip() or not self.gpu_type.strip():
            raise ValueError("node and gpu_type must be non-empty")
        if not gpu_ids:
            raise ValueError("a node profile must contain at least one GPU")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("node GPU IDs must be unique")
        if any(not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError("node GPU IDs must be non-negative integers")


class HardwareProfile:
    """Validated, read-only GPU-type and node inventory."""

    def __init__(
        self,
        gpu_types: Iterable[GPUTypeProfile],
        nodes: Iterable[NodeProfile] = (),
    ) -> None:
        type_index = {}
        for profile in gpu_types:
            if profile.gpu_type in type_index:
                raise ValueError("duplicate GPU type profile: %s" % profile.gpu_type)
            type_index[profile.gpu_type] = profile

        node_index = {}
        for node in nodes:
            if node.node in node_index:
                raise ValueError("duplicate node profile: %s" % node.node)
            if node.gpu_type not in type_index:
                raise ValueError("node %s references unknown GPU type %s" % (node.node, node.gpu_type))
            node_index[node.node] = node

        self._gpu_types = MappingProxyType(type_index)
        self._nodes = MappingProxyType(node_index)

    @property
    def gpu_types(self) -> Mapping[str, GPUTypeProfile]:
        return self._gpu_types

    @property
    def nodes(self) -> Mapping[str, NodeProfile]:
        return self._nodes

    def gpu_type(self, name: str) -> GPUTypeProfile:
        return self._gpu_types[name]

    def node(self, name: str) -> NodeProfile:
        return self._nodes[name]

    def validate_endpoint(self, endpoint: "EndpointSpec") -> None:
        gpu_profile = self.gpu_type(endpoint.gpu_type)
        node_profile = self.node(endpoint.node)
        if node_profile.gpu_type != endpoint.gpu_type:
            raise ValueError("endpoint GPU type does not match its node profile")
        if not set(endpoint.gpu_ids).issubset(node_profile.gpu_ids):
            raise ValueError("endpoint claims GPUs outside its node profile")
        if endpoint.tp_degree not in gpu_profile.supported_tp_degrees:
            raise ValueError("endpoint TP degree is not supported by its GPU type")


def from_gpu_specs(
    gpu_specs: Mapping[str, Mapping[str, object]],
    nodes: Optional[Iterable[NodeProfile]] = None,
) -> HardwareProfile:
    """Adapt a GPU_SPECS-shaped mapping without importing the legacy stack."""

    profiles = []
    for gpu_type, spec in gpu_specs.items():
        profiles.append(
            GPUTypeProfile(
                gpu_type=gpu_type,
                allowed_frequencies_mhz=tuple(spec["frequencies"]),
                max_frequency_mhz=int(spec["max_freq_mhz"]),
                nominal_tdp_w=float(spec["tdp_w"]),
                idle_power_w=float(spec["idle_power_w"]),
                supported_tp_degrees=tuple(spec["tp_degrees"]),
            )
        )
    return HardwareProfile(profiles, nodes or ())


def from_current_gpu_specs(
    nodes: Optional[Iterable[NodeProfile]] = None,
) -> HardwareProfile:
    """Optional bridge to the current simulator inventory."""

    from jsep_cluster import GPU_SPECS

    return from_gpu_specs(GPU_SPECS, nodes)
