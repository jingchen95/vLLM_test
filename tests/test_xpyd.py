"""Unit tests for the architecture-only XpYd package."""

import unittest

from xpyd.compatibility import (
    CompatibilityTable, ConnectorCompatibility, EndpointPairCompatibility,
)
from xpyd.hardware import GPUTypeProfile, HardwareProfile, NodeProfile, from_gpu_specs
from xpyd.mock_runtime import build_example_runtime, example_decisions
from xpyd.registry import EndpointRegistry, GPUAllocationConflictError
from xpyd.types import EndpointSpec, EndpointState, LifecycleState, RoutingDecision


def endpoint(
    endpoint_id="P0",
    role="prefill",
    gpu_ids=(0,),
    tp_degree=1,
    connector="test-kv",
):
    return EndpointSpec(
        endpoint_id=endpoint_id,
        role=role,
        gpu_type="test-accelerator",
        node="node-a",
        gpu_ids=gpu_ids,
        tp_degree=tp_degree,
        kv_connector=connector,
    )


def state(endpoint_id, healthy=True, lifecycle=LifecycleState.ACTIVE):
    return EndpointState(
        endpoint_id=endpoint_id,
        freq_mhz=1200,
        lifecycle=lifecycle,
        healthy=healthy,
    )


class EndpointValidationTests(unittest.TestCase):
    def test_endpoint_spec_validation(self):
        with self.assertRaises(ValueError):
            endpoint(endpoint_id="")
        with self.assertRaises(ValueError):
            endpoint(role="other")
        with self.assertRaises(ValueError):
            endpoint(tp_degree=0, gpu_ids=())
        with self.assertRaises(ValueError):
            endpoint(tp_degree=2, gpu_ids=(0,))
        with self.assertRaises(ValueError):
            endpoint(tp_degree=2, gpu_ids=(0, 0))

    def test_endpoint_spec_is_immutable(self):
        spec = endpoint()
        with self.assertRaises(AttributeError):
            spec.tp_degree = 2


class InventoryTests(unittest.TestCase):
    def test_mixed_tp_inventory(self):
        registry = EndpointRegistry()
        registry.register(endpoint("P0", "prefill", (0,), 1), state("P0"))
        registry.register(endpoint("P2", "prefill", (1, 2), 2), state("P2"))
        registry.register(endpoint("D0", "decode", (3,), 1), state("D0"))
        self.assertEqual([spec.tp_degree for spec in registry.list_by_role("prefill")], [1, 2])
        self.assertEqual([spec.endpoint_id for spec in registry.list_by_role("decode")], ["D0"])

    def test_duplicate_active_gpu_detection(self):
        registry = EndpointRegistry()
        registry.register(endpoint("P0", "prefill", (0,), 1), state("P0"))
        with self.assertRaises(GPUAllocationConflictError):
            registry.register(endpoint("D0", "decode", (0,), 1), state("D0"))

        overlap_allowed = EndpointRegistry(allow_gpu_overlap=True)
        overlap_allowed.register(endpoint("P0", "prefill", (0,), 1), state("P0"))
        overlap_allowed.register(endpoint("D0", "decode", (0,), 1), state("D0"))
        self.assertEqual(len(overlap_allowed.list_endpoints()), 2)

    def test_health_and_active_filtering(self):
        registry = EndpointRegistry()
        registry.register(endpoint("P0", "prefill", (0,), 1), state("P0"))
        registry.register(
            endpoint("P1", "prefill", (1,), 1),
            state("P1", healthy=False),
        )
        registry.register(
            endpoint("P2", "prefill", (2,), 1),
            state("P2", healthy=True, lifecycle=LifecycleState.WARM),
        )
        self.assertEqual(
            [spec.endpoint_id for spec in registry.healthy_active("prefill")],
            ["P0"],
        )


class RoutingAndCompatibilityTests(unittest.TestCase):
    def test_independent_prefill_decode_decision(self):
        decision = RoutingDecision("P2", "D0", 1200, 800)
        self.assertEqual(decision.prefill_endpoint_id, "P2")
        self.assertEqual(decision.decode_endpoint_id, "D0")
        self.assertNotEqual(decision.prefill_freq_mhz, decision.decode_freq_mhz)

    def test_prefill_tp_can_differ_from_decode_tp(self):
        first, second = example_decisions()
        self.assertEqual((first.prefill_endpoint_id, first.decode_endpoint_id), ("P2", "D0"))
        self.assertEqual((second.prefill_endpoint_id, second.decode_endpoint_id), ("P0", "D2"))

        runtime = build_example_runtime()
        self.assertEqual(runtime.registry.get_spec("P2").tp_degree, 2)
        self.assertEqual(runtime.registry.get_spec("D0").tp_degree, 1)

    def test_unknown_compatibility_fails_closed(self):
        table = CompatibilityTable()
        self.assertFalse(table.is_compatible(endpoint("P0"), endpoint("D0", "decode", (1,))))

        table_with_other_pair = CompatibilityTable(
            (
                ConnectorCompatibility(
                    connector="test-kv",
                    prefill_tp=2,
                    decode_tp=1,
                    supported=True,
                    reason="measured pair",
                ),
            )
        )
        self.assertFalse(
            table_with_other_pair.is_compatible(
                endpoint("P0", "prefill", (0,), 1),
                endpoint("D2", "decode", (1, 2), 2),
            )
        )

    def test_mock_runtime_rejects_unlisted_tp_pair(self):
        runtime = build_example_runtime()
        with self.assertRaises(ValueError):
            runtime.select("P0", "D0", 800, 800)

    def test_exact_pair_evidence_fails_closed_for_unlisted_same_tp_pair(self):
        p0 = endpoint("P0", "prefill", (0,), 1)
        p1 = endpoint("P1", "prefill", (1,), 1)
        d0 = endpoint("D0", "decode", (2,), 1)
        table = CompatibilityTable(
            entries=(ConnectorCompatibility("test-kv", 1, 1, True, "generic"),),
            endpoint_pairs=(EndpointPairCompatibility(
                "P0", "D0", "test-kv", 1, 1, True, "physical smoke",
            ),),
        )
        self.assertTrue(table.is_compatible(p0, d0))
        self.assertFalse(table.is_compatible(p1, d0))


class HardwareIndependenceTests(unittest.TestCase):
    def test_arbitrary_gpu_profile(self):
        profile = HardwareProfile(
            (
                GPUTypeProfile(
                    gpu_type="future-chip-z",
                    allowed_frequencies_mhz=(700, 1100),
                    max_frequency_mhz=1100,
                    nominal_tdp_w=180.0,
                    idle_power_w=20.0,
                    supported_tp_degrees=(1, 3),
                ),
            ),
            (NodeProfile("future-node", "future-chip-z", (10, 11, 12)),),
        )
        self.assertEqual(profile.gpu_type("future-chip-z").supported_tp_degrees, (1, 3))
        self.assertNotIn("l40s", {name.lower() for name in profile.gpu_types})
        self.assertNotIn("l4", {name.lower() for name in profile.gpu_types})

    def test_generic_gpu_specs_adapter(self):
        profile = from_gpu_specs(
            {
                "custom-device": {
                    "frequencies": [500, 900],
                    "max_freq_mhz": 900,
                    "tdp_w": 100,
                    "idle_power_w": 12,
                    "tp_degrees": [1, 2],
                }
            }
        )
        self.assertEqual(profile.gpu_type("custom-device").max_frequency_mhz, 900)


if __name__ == "__main__":
    unittest.main()
