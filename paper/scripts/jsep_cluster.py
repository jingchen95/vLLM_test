"""
JSEP Phase B: Heterogeneous GPU cluster state model.

Models a cluster with L40S and L4 GPU pools, tracking active instances,
TP degree, frequency, and power consumption.
"""

from dataclasses import dataclass, field
from typing import Dict
import copy


@dataclass
class GPUPool:
    gpu_type: str          # "l40s" or "l4"
    total_gpus: int        # physical GPUs available
    tp_degree: int = 1     # current TP degree
    freq_mhz: int = 0     # current SM frequency (0 = max)
    active_instances: int = 0  # active TP-group instances

    @property
    def max_instances(self) -> int:
        return self.total_gpus // self.tp_degree

    @property
    def active_gpus(self) -> int:
        return self.active_instances * self.tp_degree

    @property
    def idle_gpus(self) -> int:
        return self.total_gpus - self.active_gpus


# Hardware constants
GPU_SPECS = {
    "l40s": {
        "total_gpus": 4,
        "tdp_w": 350.0,
        "max_freq_mhz": 2520,
        "idle_power_w": 90.0,   # per-GPU power when not serving (estimated)
        "tp_degrees": [1, 2, 4],
        "frequencies": [210, 480, 735, 990, 1245, 1500, 1755, 2010, 2265, 2520],
    },
    "l4": {
        "total_gpus": 8,
        "tdp_w": 72.0,
        "max_freq_mhz": 2040,
        "idle_power_w": 18.0,
        "tp_degrees": [1, 2, 4, 8],
        "frequencies": [210, 360, 570, 780, 990, 1200, 1410, 1620, 1830, 2040],
    },
}


def set_idle_power_w(l40s_idle_w: float, l4_idle_w: float) -> None:
    """Override per-GPU idle power for sensitivity analysis.

    Call once before creating any ClusterState objects.
    Default: L40S=90 W, L4=18 W (measured hardware standby).
    """
    GPU_SPECS["l40s"]["idle_power_w"] = l40s_idle_w
    GPU_SPECS["l4"]["idle_power_w"]   = l4_idle_w


class ClusterState:
    """Full heterogeneous cluster: L40S pool + L4 pool."""

    def __init__(self):
        self.pools: Dict[str, GPUPool] = {}
        for gpu_type, spec in GPU_SPECS.items():
            self.pools[gpu_type] = GPUPool(
                gpu_type=gpu_type,
                total_gpus=spec["total_gpus"],
                tp_degree=1,
                freq_mhz=spec["max_freq_mhz"],
                active_instances=spec["total_gpus"],  # all active by default
            )

    def reconfigure(self, gpu_type: str, tp_degree: int, freq_mhz: int,
                    active_instances: int):
        """Apply new configuration to a GPU pool."""
        pool = self.pools[gpu_type]
        pool.tp_degree = tp_degree
        pool.freq_mhz = freq_mhz
        max_inst = pool.total_gpus // tp_degree
        pool.active_instances = min(active_instances, max_inst)

    def total_power(self, active_power: Dict[str, float]) -> float:
        """Compute total cluster power.

        Args:
            active_power: {gpu_type: total power for all active instances (W)}
                          e.g., {"l40s": 520.0, "l4": 180.0}
        Returns:
            Total cluster power including idle GPU power.
        """
        total = 0.0
        for gpu_type, pool in self.pools.items():
            total += active_power.get(gpu_type, 0.0)
            total += pool.idle_gpus * GPU_SPECS[gpu_type]["idle_power_w"]
        return total

    def idle_power(self) -> float:
        """Total power when no requests are being served."""
        return sum(
            spec["total_gpus"] * spec["idle_power_w"]
            for spec in GPU_SPECS.values()
        )

    def copy(self) -> "ClusterState":
        """Deep copy for what-if analysis."""
        return copy.deepcopy(self)

    def summary(self) -> str:
        parts = []
        for gpu_type, pool in self.pools.items():
            if pool.active_instances > 0:
                parts.append(
                    f"{gpu_type.upper()}: {pool.active_instances}×TP{pool.tp_degree}"
                    f"@{pool.freq_mhz}MHz"
                )
            else:
                parts.append(f"{gpu_type.upper()}: OFF")
        return " | ".join(parts)
