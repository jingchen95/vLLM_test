"""CPU-only example of endpoint registration and P/D route selection."""

from typing import Tuple

from xpyd.compatibility import CompatibilityTable, ConnectorCompatibility
from xpyd.hardware import GPUTypeProfile, HardwareProfile, NodeProfile
from xpyd.registry import EndpointRegistry
from xpyd.types import EndpointSpec, EndpointState, LifecycleState, RoutingDecision


class MockXpYdRuntime:
    """Validation-only facade; it performs no lifecycle, network, or DVFS action."""

    def __init__(
        self,
        hardware: HardwareProfile,
        registry: EndpointRegistry,
        compatibility: CompatibilityTable,
    ) -> None:
        self.hardware = hardware
        self.registry = registry
        self.compatibility = compatibility

    def select(
        self,
        prefill_endpoint_id: str,
        decode_endpoint_id: str,
        prefill_freq_mhz: int,
        decode_freq_mhz: int,
    ) -> RoutingDecision:
        prefill = self.registry.get_spec(prefill_endpoint_id)
        decode = self.registry.get_spec(decode_endpoint_id)
        if prefill.role != "prefill" or decode.role != "decode":
            raise ValueError("selection must pair a prefill endpoint with a decode endpoint")
        for endpoint, freq_mhz in (
            (prefill, prefill_freq_mhz),
            (decode, decode_freq_mhz),
        ):
            state = self.registry.get_state(endpoint.endpoint_id)
            if state.lifecycle != LifecycleState.ACTIVE or not state.healthy:
                raise ValueError("endpoint is not healthy and active: %s" % endpoint.endpoint_id)
            self.hardware.validate_endpoint(endpoint)
            if freq_mhz not in self.hardware.gpu_type(endpoint.gpu_type).allowed_frequencies_mhz:
                raise ValueError("frequency is not allowed for endpoint %s" % endpoint.endpoint_id)
        if not self.compatibility.is_compatible(prefill, decode):
            raise ValueError("P/D connector and TP combination is not explicitly compatible")
        return RoutingDecision(
            prefill_endpoint_id=prefill.endpoint_id,
            decode_endpoint_id=decode.endpoint_id,
            prefill_freq_mhz=prefill_freq_mhz,
            decode_freq_mhz=decode_freq_mhz,
        )


def build_example_runtime() -> MockXpYdRuntime:
    """Build P0/P1/P2 and D0/D1/D2 over eight fake GPUs."""

    hardware = HardwareProfile(
        gpu_types=(
            GPUTypeProfile(
                gpu_type="example_accelerator",
                allowed_frequencies_mhz=(800, 1200, 1600),
                max_frequency_mhz=1600,
                nominal_tdp_w=250.0,
                idle_power_w=35.0,
                supported_tp_degrees=(1, 2),
            ),
        ),
        nodes=(
            NodeProfile(
                node="example-node",
                gpu_type="example_accelerator",
                gpu_ids=tuple(range(8)),
            ),
        ),
    )
    registry = EndpointRegistry()
    placements = (
        ("P0", "prefill", (0,)),
        ("P1", "prefill", (1,)),
        ("P2", "prefill", (2, 3)),
        ("D0", "decode", (4,)),
        ("D1", "decode", (5,)),
        ("D2", "decode", (6, 7)),
    )
    for endpoint_id, role, gpu_ids in placements:
        registry.register(
            EndpointSpec(
                endpoint_id=endpoint_id,
                role=role,
                gpu_type="example_accelerator",
                node="example-node",
                gpu_ids=gpu_ids,
                tp_degree=len(gpu_ids),
                kv_connector="mock-kv",
            ),
            EndpointState(
                endpoint_id=endpoint_id,
                freq_mhz=1600,
                lifecycle=LifecycleState.ACTIVE,
                healthy=True,
                queue_depth_observed=True,
                kv_cache_usage_observed=True,
            ),
        )

    compatibility = CompatibilityTable(
        (
            ConnectorCompatibility(
                connector="mock-kv",
                prefill_tp=2,
                decode_tp=1,
                supported=True,
                reason="explicit mock validation for P2 -> D0",
            ),
            ConnectorCompatibility(
                connector="mock-kv",
                prefill_tp=1,
                decode_tp=2,
                supported=True,
                reason="explicit mock validation for P0 -> D2",
            ),
        )
    )
    return MockXpYdRuntime(hardware, registry, compatibility)


def example_decisions() -> Tuple[RoutingDecision, RoutingDecision]:
    runtime = build_example_runtime()
    return (
        runtime.select("P2", "D0", 1200, 800),
        runtime.select("P0", "D2", 800, 1200),
    )


if __name__ == "__main__":
    for decision in example_decisions():
        print(decision)
