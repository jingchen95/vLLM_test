"""CPU-only Phase 3D actuator and live-route safety tests."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from xpyd.compatibility import CompatibilityTable, EndpointPairCompatibility
from xpyd.disagg_proxy import FileControlledCompatiblePairs
from xpyd.phase3d_control import (
    EndpointClockCapability,
    GPUReadback,
    NvidiaSmiClockBackend,
    PerEndpointClockActuator,
    Phase3DError,
    audit_connector_concurrency,
    select_validation_frequencies,
)
from xpyd.registry import EndpointRegistry
from xpyd.types import EndpointSpec, EndpointState, LifecycleState


class FakeClockBackend:
    def __init__(self, capabilities, frequencies=None):
        self.capabilities = capabilities
        self.frequencies = frequencies or {
            endpoint_id: capability.selected_high_mhz
            for endpoint_id, capability in capabilities.items()
        }
        self.fail = False

    def read(self, endpoint_id, node, gpu_id, expected_mhz=None):
        capability = self.capabilities[endpoint_id]
        return GPUReadback(
            endpoint_id, node, gpu_id, capability.gpu_name,
            capability.gpu_uuid, capability.pci_bus_id,
            self.frequencies[endpoint_id], 10.0,
        )

    def set_graphics_clock(self, capability, target_mhz):
        if self.fail:
            raise RuntimeError("fixture failure")
        self.frequencies[capability.endpoint_id] = target_mhz
        return self.read(capability.endpoint_id, capability.node, capability.gpu_id)


def capabilities():
    return {
        endpoint_id: EndpointClockCapability(
            endpoint_id=endpoint_id,
            node="pnode" if endpoint_id.startswith("P") else "dnode",
            gpu_id=index % 2,
            gpu_name="fixture",
            gpu_uuid="uuid-%s" % endpoint_id,
            pci_bus_id="0000:%02d:00.0" % index,
            supported_graphics_mhz=(600, 900, 1200),
            selected_low_mhz=600,
            selected_mid_mhz=900,
            selected_high_mhz=1200,
        )
        for index, endpoint_id in enumerate(("P0", "P1", "D0", "D1"))
    }


class ActuatorPrimitiveTests(unittest.TestCase):
    def test_closed_loop_concurrency_cannot_exceed_validated_connector_limit(self):
        valid = audit_connector_concurrency([
            {"id": "light", "max_concurrency": 1},
            {"id": "moderate", "max_concurrency": 1},
        ], 1)
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["observed_max_concurrency"], 1)

        invalid = audit_connector_concurrency([
            {"id": "light", "max_concurrency": 1},
            {"id": "moderate", "max_concurrency": 4},
        ], 1)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["violating_workload_ids"], ["moderate"])

    def test_validation_states_are_supported_and_bounded_by_safe_high(self):
        self.assertEqual(
            select_validation_frequencies((300, 600, 900, 1200, 1500), 1200),
            (600, 900, 1200),
        )
        with self.assertRaises(Phase3DError):
            select_validation_frequencies((600, 900, 1200), 1100)

    def test_actuation_keeps_requested_and_observed_state_separate(self):
        caps = capabilities()
        backend = FakeClockBackend(caps)
        actuator = PerEndpointClockActuator(backend, caps, 0.0)
        actuator.requested.update({key: 1200 for key in caps})
        row = actuator.actuate("P0", 900, "test")
        self.assertEqual(row["command_status"], "success")
        self.assertEqual(row["previous_requested_freq_mhz"], 1200)
        self.assertEqual(row["observed_freq_after_mhz"], 900)
        self.assertEqual(backend.frequencies["P1"], 1200)

    def test_failed_command_never_updates_requested_state(self):
        caps = capabilities()
        backend = FakeClockBackend(caps)
        actuator = PerEndpointClockActuator(backend, caps, 0.0)
        actuator.requested["P0"] = 1200
        backend.fail = True
        row = actuator.actuate("P0", 900, "test_failure")
        self.assertEqual(row["command_status"], "failed")
        self.assertFalse(row["readback_valid"])
        self.assertEqual(actuator.requested["P0"], 1200)

    def test_minimum_dwell_starts_at_successful_actuation(self):
        caps = capabilities()
        backend = FakeClockBackend(caps)
        now = [0.0]
        slept = []

        def sleep(seconds):
            slept.append(seconds)
            now[0] += seconds

        actuator = PerEndpointClockActuator(
            backend, caps, 2.0, sleep=sleep,
            monotonic=lambda: now[0], wall_clock=lambda: now[0],
        )
        actuator.requested["P0"] = 1200
        self.assertEqual(actuator.actuate("P0", 900, "down")["command_status"], "success")
        self.assertEqual(actuator.actuate("P0", 600, "down")["command_status"], "success")
        self.assertEqual(slept, [2.0])

    def test_overlapping_physical_gpu_mapping_is_rejected(self):
        caps = capabilities()
        duplicate = caps["P1"]
        caps["P1"] = EndpointClockCapability(
            **{
                **duplicate.__dict__,
                "node": caps["P0"].node,
                "gpu_id": caps["P0"].gpu_id,
            }
        )
        with self.assertRaises(Phase3DError):
            PerEndpointClockActuator(FakeClockBackend(caps), caps, 0.0)

    def test_node_backend_requires_fresh_matching_readback(self):
        output = (
            "[NVGPUFREQ][INFO] fixture plugin message\n"
            "IDENTITY=NVIDIA L4, GPU-a, 00000000:01:00.0, 1500\n"
        )

        def runner(command):
            return subprocess.CompletedProcess(command, 0, output)

        backend = NvidiaSmiClockBackend(runner=runner, clock=lambda: 4.0)
        result = backend.read("D0", "io", 0)
        self.assertEqual(result.graphics_clock_mhz, 1500)
        self.assertEqual(result.gpu_uuid, "GPU-a")

    def test_bounded_poll_uses_freshest_matching_actual_clock(self):
        output = (
            "IDENTITY=NVIDIA L4, GPU-a, 00000000:01:00.0, 210\n"
            "IDENTITY=NVIDIA L4, GPU-a, 00000000:01:00.0, 1125\n"
        )

        def runner(command):
            return subprocess.CompletedProcess(command, 0, output)

        backend = NvidiaSmiClockBackend(runner=runner, clock=lambda: 5.0)
        result = backend.read("D0", "io", 0, expected_mhz=1125)
        self.assertEqual(result.graphics_clock_mhz, 1125)


def route_fixture():
    registry = EndpointRegistry()
    specs = {}
    for endpoint_id, role, node in (
        ("P0", "prefill", "pnode"), ("P1", "prefill", "pnode"),
        ("D0", "decode", "dnode"), ("D1", "decode", "dnode"),
    ):
        spec = EndpointSpec(
            endpoint_id, role, "fixture", node,
            (0 if endpoint_id.endswith("0") else 1,), 1,
            kv_connector="fixture-kv",
        )
        specs[endpoint_id] = spec
        registry.register(spec, EndpointState(
            endpoint_id, 1200, LifecycleState.ACTIVE, True,
        ))
    pairs = [("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1")]
    table = CompatibilityTable(endpoint_pairs=[
        EndpointPairCompatibility(p, d, "fixture-kv", 1, 1, True, "fixture")
        for p, d in pairs
    ])
    return registry, table, pairs


class LiveRouteControlTests(unittest.TestCase):
    def test_control_file_changes_next_physical_pair(self):
        registry, table, pairs = route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.json"
            path.write_text(json.dumps({
                "updated_unix_s": 10.0,
                "pairs": [["P1", "D0"]],
            }), encoding="utf-8")
            selector = FileControlledCompatiblePairs(
                registry, table, pairs, str(path), 30.0, wall_clock=lambda: 11.0,
            )
            self.assertEqual(selector.choose(), ("P1", "D0"))

    def test_missing_stale_or_incompatible_control_fails_closed(self):
        registry, table, pairs = route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.json"
            selector = FileControlledCompatiblePairs(
                registry, table, pairs, str(path), 5.0, wall_clock=lambda: 20.0,
            )
            with self.assertRaises(ValueError):
                selector.choose()
            path.write_text(json.dumps({
                "updated_unix_s": 1.0, "pairs": [["P0", "D0"]],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                selector.choose()
            path.write_text(json.dumps({
                "updated_unix_s": 20.0, "pairs": [["P9", "D9"]],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                selector.choose()


if __name__ == "__main__":
    unittest.main()
