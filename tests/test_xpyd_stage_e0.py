"""CPU-only tests for the Stage E0 regime diagnosis."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from xpyd.phase3d_control import (
    EndpointClockCapability,
    GPUReadback,
    PerEndpointClockActuator,
    Phase3DError,
)
from xpyd.stage_e0_regime import (
    DEFAULT,
    DEFAULT_TARGET_MHZ,
    E0WindowHarness,
    GovernorClockBackend,
    StageE0Error,
    StageE0RegimeHarness,
    _decision_rules,
    default_governor_clock_audit,
    idle_window_summary,
    load_e0_configs,
)

ENDPOINTS = ("P0", "P1", "D0", "D1")
LEVELS = {"P": (1260, 1890, 2520), "D": (750, 1125, 1500)}


def capability(endpoint_id):
    low, mid, high = LEVELS[endpoint_id[0]]
    return EndpointClockCapability(
        endpoint_id=endpoint_id, node="neptune" if endpoint_id[0] == "P" else "europa",
        gpu_id=int(endpoint_id[1]), gpu_name="L40S" if endpoint_id[0] == "P" else "L4",
        gpu_uuid="uuid-%s" % endpoint_id, pci_bus_id="0000:0%s:00.0" % endpoint_id[1],
        supported_graphics_mhz=(low, mid, high), selected_low_mhz=low,
        selected_mid_mhz=mid, selected_high_mhz=high,
    )


class FakeGovernorBackend:
    def __init__(self):
        self.caps = {endpoint_id: capability(endpoint_id) for endpoint_id in ENDPOINTS}
        self.graphics = {key: value.selected_high_mhz for key, value in self.caps.items()}
        self.memory_locked = {key: True for key in self.caps}
        self.calls = []

    def discover(self, endpoint_id, node, gpu_id, safe_high_mhz):
        return self.caps[endpoint_id]

    def read(self, endpoint_id, node, gpu_id, expected_mhz=None):
        cap = self.caps[endpoint_id]
        return GPUReadback(endpoint_id, cap.node, cap.gpu_id, cap.gpu_name, cap.gpu_uuid,
                           cap.pci_bus_id, self.graphics[endpoint_id], 1.0)

    def set_graphics_clock(self, cap, target_mhz):
        self.calls.append(("lgc", cap.endpoint_id, target_mhz))
        self.graphics[cap.endpoint_id] = target_mhz
        return self.read(cap.endpoint_id, cap.node, cap.gpu_id)

    def reset_clocks(self, cap):
        self.calls.append(("reset", cap.endpoint_id))
        self.graphics[cap.endpoint_id] = 210
        self.memory_locked[cap.endpoint_id] = False
        return self.read(cap.endpoint_id, cap.node, cap.gpu_id)

    def lock_memory_clock(self, cap, memory_mhz):
        self.calls.append(("lmc", cap.endpoint_id, memory_mhz))
        self.memory_locked[cap.endpoint_id] = True
        return memory_mhz


class StageE0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def load(self):
        old = dict(os.environ)
        os.environ.update({
            "XPYD_PROXY_URI": "http://127.0.0.1:8090",
            "XPYD_PROXY_DIAGNOSTICS_LOG": "/tmp/proxy.jsonl",
            "XPYD_P0_HTTP_HOST": "127.0.0.1", "XPYD_P1_HTTP_HOST": "127.0.0.1",
            "XPYD_D0_HTTP_HOST": "127.0.0.1", "XPYD_D1_HTTP_HOST": "127.0.0.1",
            "XPYD_P_ADDR_HOST": "10.0.0.1", "XPYD_D_ADDR_HOST": "10.0.0.2",
        })
        try:
            return load_e0_configs(
                self.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json",
                self.root / "paper/configs/xpyd_stage_e0_regime_neptune_europa.json",
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

    def harness(self, output_root, backend=None):
        runtime, experiment = self.load()
        experiment["output_root"] = str(output_root)
        backend = backend or FakeGovernorBackend()
        harness = StageE0RegimeHarness(
            runtime, experiment, run_id="fixture", backend=backend,
            sleep=lambda seconds: None,
        )
        harness._make_actuator = lambda caps: PerEndpointClockActuator(
            backend, caps, 0.0, sleep=lambda seconds: None,
        )
        return harness, backend

    # ------------------------------------------------------------ config
    def test_config_freezes_matrix_slo_and_trace(self):
        runtime, experiment = self.load()
        settings = experiment["stagec1"]
        self.assertEqual((settings["slo"]["ttft_ms"], settings["slo"]["tpot_ms"]), (600.0, 65.0))
        self.assertEqual(settings["max_concurrency"], 4)
        self.assertEqual(
            [point["id"] for point in experiment["stage_e0"]["points"]],
            ["HIGH_HIGH", "P_MID_D_HIGH", "P_DEFAULT_D_HIGH", "DEFAULT_DEFAULT"],
        )
        self.assertIn(DEFAULT, experiment["stage_e0"]["idle_ladder"]["states"])
        self.assertEqual(experiment["stage_e0"]["decision_rules"]["p_default_idle_break_even_w"], 64.6)

    def test_config_rejects_changed_point_matrix(self):
        runtime, experiment = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            experiment["stage_e0"]["points"] = experiment["stage_e0"]["points"][:2]
            path = Path(tmp) / "e0.json"
            path.write_text(json.dumps(experiment), encoding="utf-8")
            old = dict(os.environ)
            os.environ.update({
                "XPYD_PROXY_URI": "x", "XPYD_PROXY_DIAGNOSTICS_LOG": "/tmp/p.jsonl",
                "XPYD_P0_HTTP_HOST": "h", "XPYD_P1_HTTP_HOST": "h",
                "XPYD_D0_HTTP_HOST": "h", "XPYD_D1_HTTP_HOST": "h",
                "XPYD_P_ADDR_HOST": "h", "XPYD_D_ADDR_HOST": "h",
            })
            try:
                with self.assertRaises(StageE0Error):
                    load_e0_configs(
                        self.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json", path,
                    )
            finally:
                os.environ.clear()
                os.environ.update(old)

    # ----------------------------------------------------------- targets
    def test_level_targets_and_default_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness, _ = self.harness(tmp)
            self.assertEqual(harness._level_targets("MID", "HIGH"),
                             {"P0": 1890, "P1": 1890, "D0": 1500, "D1": 1500})
            self.assertEqual(harness._level_targets(DEFAULT, "HIGH"),
                             {"P0": DEFAULT_TARGET_MHZ, "P1": DEFAULT_TARGET_MHZ, "D0": 1500, "D1": 1500})

    def test_run_order_rotates_and_covers_every_point(self):
        points = [{"id": name} for name in ("A", "B", "C", "D")]
        orders = [[item["id"] for item in StageE0RegimeHarness.run_order(points, r)] for r in (1, 2, 3)]
        self.assertEqual(orders[0], ["A", "B", "C", "D"])
        self.assertEqual(orders[1], ["B", "C", "D", "A"])
        self.assertEqual(len({order[0] for order in orders}), 3)
        self.assertTrue(all(sorted(order) == ["A", "B", "C", "D"] for order in orders))

    # ---------------------------------------------------------- actuation
    def test_default_reset_then_relock_memory_and_graphics(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness, backend = self.harness(tmp)
            caps = backend.caps
            actuator = harness._make_actuator(caps)
            high = {key: value.selected_high_mhz for key, value in caps.items()}
            harness._apply(actuator, caps, high, "init")
            harness._apply(actuator, caps, harness._level_targets(DEFAULT, "HIGH"), "p_default")
            self.assertEqual(harness.default_endpoints, {"P0", "P1"})
            self.assertNotIn("P0", actuator.requested)
            self.assertFalse(backend.memory_locked["P0"])
            self.assertTrue(backend.memory_locked["D0"])
            # Re-applying the same DEFAULT state must not reset again.
            resets = sum(call[0] == "reset" for call in backend.calls)
            harness._apply(actuator, caps, harness._level_targets(DEFAULT, "HIGH"), "again")
            self.assertEqual(sum(call[0] == "reset" for call in backend.calls), resets)
            backend.calls.clear()
            harness._apply(actuator, caps, high, "restore")
            self.assertEqual(harness.default_endpoints, set())
            self.assertTrue(all(backend.memory_locked.values()))
            p0_calls = [call for call in backend.calls if call[1] == "P0"]
            self.assertEqual(p0_calls, [("lmc", "P0", 9001), ("lgc", "P0", 2520)])
            self.assertEqual(backend.graphics, high)

    def test_window_rejects_default_set_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness, _ = self.harness(tmp)
            with self.assertRaises(StageE0Error):
                harness._run_window("w", [], harness._level_targets(DEFAULT, "HIGH"), Path(tmp))

    # ------------------------------------------------------------- audits
    def test_default_audit_describes_without_target(self):
        records = [
            {"status": "success", "actual_start_wall_s": 10.0 + i, "graphics_clock_mhz": clock,
             "memory_clock_mhz": 405.0 if clock < 1000 else 9001.0}
            for i, clock in enumerate((210.0, 2520.0, 2490.0, 210.0))
        ]
        audit = default_governor_clock_audit(records, 10.0, 13.0)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["mode"], DEFAULT)
        self.assertEqual(audit["graphics"]["minimum_mhz"], 210.0)
        self.assertEqual(audit["graphics"]["maximum_mhz"], 2520.0)
        self.assertEqual(audit["graphics"]["distinct_values"], 3)
        self.assertFalse(default_governor_clock_audit(records, 100.0, 101.0)["valid"])

    def test_window_harness_dispatches_default_audit_only_for_default_endpoints(self):
        fake = SimpleNamespace(default_endpoints={"P0"})
        records = [{"status": "success", "actual_start_wall_s": 1.0, "graphics_clock_mhz": 300.0}]
        audit = E0WindowHarness._clock_audit_for_endpoint(fake, "P0", records, 0.0, 2.0)
        self.assertEqual(audit["mode"], DEFAULT)

    def test_idle_summary_uses_energy_counter(self):
        records = [
            {"status": "success", "actual_start_wall_s": 100.0 + 0.2 * i,
             "total_energy_mj": 1_000_000 + 13_000 * i, "power_w": 65.0,
             "graphics_clock_mhz": 210.0, "memory_clock_mhz": 405.0,
             "gpu_utilization_pct": 0.0, "clock_throttle_reasons": ["gpu_idle"]}
            for i in range(51)
        ]
        row = idle_window_summary(records, 100.0, 110.0, 0.2, 0.9)
        self.assertTrue(row["valid"])
        self.assertAlmostEqual(row["idle_power_w"], 65.0, places=6)
        self.assertEqual(row["throttle_reasons"], "gpu_idle")
        busy = [dict(item, gpu_utilization_pct=5.0) for item in records]
        self.assertFalse(idle_window_summary(busy, 100.0, 110.0, 0.2, 0.9)["valid"])
        self.assertFalse(idle_window_summary(records[:10], 100.0, 110.0, 0.2, 0.9)["valid"])

    # ------------------------------------------------------------ backend
    def test_backend_reset_and_memory_relock_commands(self):
        commands = []

        def runner(command):
            commands.append(command[-1])
            if "-lmc" in command[-1]:
                return subprocess.CompletedProcess(command, 0, "MEMORY=405\nMEMORY=9001\n")
            return subprocess.CompletedProcess(command, 0, "IDENTITY=NVIDIA L40S, uuid-P0, 0000:00:00.0, 210\n")

        backend = GovernorClockBackend(runner=runner, clock=lambda: 1.0)
        readback = backend.reset_clocks(capability("P0"))
        self.assertEqual(readback.graphics_clock_mhz, 210)
        self.assertIn("-rgc", commands[0])
        self.assertIn("-rmc", commands[0])
        self.assertEqual(backend.lock_memory_clock(capability("P0"), 9001), 9001)
        self.assertIn("-lmc 9001,9001", commands[-1])

    def test_backend_memory_relock_mismatch_fails_closed(self):
        def runner(command):
            return subprocess.CompletedProcess(command, 0, "MEMORY=405\n")

        with self.assertRaises(Phase3DError):
            GovernorClockBackend(runner=runner).lock_memory_clock(capability("P0"), 9001)

    # -------------------------------------------------------------- rules
    def test_decision_rules(self):
        def point(point_id, jpr, ok):
            return {"point_id": point_id, "load_point": "MEDIUM",
                    "joules_per_request": jpr, "every_repeat_slo_pass": ok}

        points = [point("HIGH_HIGH", 2600, True), point("P_MID_D_HIGH", 2330, True),
                  point("P_DEFAULT_D_HIGH", 2100, True), point("DEFAULT_DEFAULT", 2000, False)]
        idle = [{"ladder": "pre", "state": DEFAULT, "pool": "P", "mean_idle_power_w_per_gpu": 40.0}]
        rules = _decision_rules(points, idle, {"p_default_idle_break_even_w": 64.6})
        medium = rules[0]
        self.assertFalse(medium["default_meets_slo_every_repeat"])
        self.assertTrue(medium["default_energy_le_tuned"])
        self.assertTrue(medium["p_default_meets_slo_every_repeat"])
        self.assertTrue(rules[-1]["p_default_idle_below_break_even"])

    # --------------------------------------------------------- full flow
    def test_run_flow_with_fakes_writes_summary_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness, backend = self.harness(tmp)
            windows = []

            def run_window(window_id, trace, targets, root):
                self.assertEqual(
                    {key for key, value in targets.items() if value == DEFAULT_TARGET_MHZ},
                    harness.default_endpoints,
                )
                windows.append((window_id, dict(targets)))
                return Path(root) / window_id

            def measurement(point, window_dir, *, load_point, repeat, context, signature, window_id):
                default_p = point["P_level"] == DEFAULT
                energy = 30000.0 if default_p else 40000.0
                row = {
                    "window_id": window_id, "point_id": point["id"], "P_level": point["P_level"],
                    "D_level": point["D_level"], "repeat": repeat, "load_point": load_point["id"],
                    "trace_signature": signature, "total_gross_energy_j": energy,
                    "P0_energy_j": energy / 4, "P1_energy_j": energy / 4,
                    "D0_energy_j": energy / 4, "D1_energy_j": energy / 4,
                    "completed_rate_rps": 0.1, "admission_delay_p95_ms": 0.1,
                    "client_waiting_max": 1.0, "backlog_behavior": "drains",
                    "measurement_valid": True,
                }
                for endpoint_id in ENDPOINTS:
                    row.update({"%s_clock_valid" % endpoint_id: True,
                                "%s_energy_valid" % endpoint_id: True,
                                "%s_throttle_valid" % endpoint_id: True})
                ttft = 650.0 if default_p else 500.0
                requests = [{"ttft_ms": ttft, "tpot_ms": 60.0, "repeat": repeat,
                             "point_id": point["id"], "load_point": load_point["id"]}
                            for _ in range(20)]
                return row, requests

            def measure_idle(label, state):
                for endpoint_id in ENDPOINTS:
                    harness.idle_rows.append({
                        "ladder": label, "state": state, "endpoint_id": endpoint_id,
                        "pool": endpoint_id[0], "idle_power_w": 40.0 if state == DEFAULT else 70.0,
                        "valid": True,
                    })

            harness._run_window = run_window
            harness._point_measurement = measurement
            harness._measure_idle = measure_idle
            run_dir = harness.run()

            self.assertEqual(len(windows), 4 * 2 * 3)
            self.assertEqual(harness.default_endpoints, set())
            self.assertEqual(backend.graphics, {"P0": 2520, "P1": 2520, "D0": 1500, "D1": 1500})
            self.assertTrue(all(backend.memory_locked.values()))
            summary = json.loads((run_dir / "e0_summary.json").read_text())
            self.assertTrue(summary["restoration_valid"])
            self.assertTrue(summary["run_completed"])
            self.assertEqual({row["ladder"] for row in summary["idle_ladder"]}, {"pre", "post"})
            default = next(row for row in summary["points"]
                           if row["point_id"] == "DEFAULT_DEFAULT" and row["load_point"] == "MEDIUM")
            self.assertFalse(default["every_repeat_slo_pass"])
            self.assertAlmostEqual(default["raw_saving_vs_high"], 0.25)
            self.assertTrue((run_dir / "e0_summary.md").is_file())
            self.assertTrue((run_dir / "e0_point_table.md").is_file())
            self.assertTrue((run_dir / "idle_ladder.csv").is_file())


if __name__ == "__main__":
    unittest.main()
