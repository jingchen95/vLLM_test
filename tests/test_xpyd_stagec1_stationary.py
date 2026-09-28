"""CPU-only semantic tests for Stage C1."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from xpyd.feedback_scheduler import DVFSAction
from xpyd.stagec1_stationary import (
    POLICIES,
    StageC1Harness,
    admission_risk,
    coordinate_pool,
    frequency_residency,
    load_configs,
    oscillation_count,
    policy_order_for_repeat,
)
from xpyd.stagec1_headroom import _level_frequency
from xpyd.workload_harness import build_trace


def recommendation(endpoint, action, target, reason="test"):
    return SimpleNamespace(
        endpoint_id=endpoint, action=action, target_freq_mhz=target, reason=reason
    )


class StageC1Tests(unittest.TestCase):
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
            return load_configs(
                self.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json",
                self.root / "paper/configs/xpyd_stagec1_stationary_neptune_europa.json",
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_config_fixes_c1_semantics_and_aligned_fast_recovery(self):
        runtime, experiment = self.load()
        settings = experiment["stagec1"]
        self.assertEqual(tuple(settings["policies"]), POLICIES)
        self.assertEqual(settings["repetitions"], 3)
        self.assertEqual(settings["windows_per_block"], 4)
        self.assertEqual(settings["max_concurrency"], 4)
        self.assertEqual(
            [point["offered_rate_rps"] for point in settings["load_points"]],
            [0.10, 0.20],
        )
        self.assertEqual(runtime["phase3d"]["feedback"]["ttft_slo_ms"], 600.0)
        self.assertEqual(runtime["phase3d"]["feedback"]["tpot_slo_ms"], 65.0)
        self.assertTrue(runtime["phase3d"]["feedback"]["latency_risk_to_max"])
        self.assertFalse(settings["request_aware_routing"])
        self.assertFalse(settings["oracle"])

    def test_three_policy_order_is_balanced_and_cyclic(self):
        self.assertEqual(policy_order_for_repeat(1), POLICIES)
        self.assertEqual(
            policy_order_for_repeat(2),
            ("ONE-STEP-FEEDBACK", "ADAPTIVE-SAFE-SEARCH", "STATIC-HIGH"),
        )
        self.assertEqual(
            policy_order_for_repeat(3),
            ("ADAPTIVE-SAFE-SEARCH", "STATIC-HIGH", "ONE-STEP-FEEDBACK"),
        )

    def test_one_step_and_adaptive_share_gates_but_differ_in_jump_limit(self):
        runtime, experiment = self.load()
        harness = StageC1Harness(runtime, experiment, run_id="cpu-only")
        capabilities = {
            endpoint: SimpleNamespace(
                selected_low_mhz=1260 if endpoint.startswith("P") else 750,
                selected_mid_mhz=1890 if endpoint.startswith("P") else 1125,
                selected_high_mhz=2520 if endpoint.startswith("P") else 1500,
            )
            for endpoint in ("P0", "P1", "D0", "D1")
        }
        one_step = harness._adaptive_pools(capabilities, "ONE-STEP-FEEDBACK")
        adaptive = harness._adaptive_pools(capabilities, "ADAPTIVE-SAFE-SEARCH")
        self.assertIsNotNone(one_step)
        self.assertIsNotNone(adaptive)
        for pool in ("P", "D"):
            left, right = one_step[pool].config, adaptive[pool].config
            self.assertEqual(left.ttft_slo_ms, right.ttft_slo_ms)
            self.assertEqual(left.tpot_slo_ms, right.tpot_slo_ms)
            self.assertEqual(left.attainment_gate, right.attainment_gate)
            self.assertEqual(left.min_valid_requests, right.min_valid_requests)
            self.assertEqual(left.headroom_for_one_step, right.headroom_for_one_step)
            self.assertEqual(left.large_jump_levels, 1)
            self.assertEqual(right.large_jump_levels, 2)

    def test_matched_trace_is_exact_35_15_35_15(self):
        _, experiment = self.load()
        left = build_trace(experiment, "stationary_medium", "static")
        right = build_trace(experiment, "stationary_medium", "feedback")
        counts = {key: sum(row["workload_class"] == key for row in left) for key in ("SS", "SL", "LS", "LL")}
        self.assertEqual(counts, {"SS": 7, "SL": 3, "LS": 7, "LL": 3})
        self.assertEqual(
            [(row["arrival_time_s"], row["input_len"], row["output_len"]) for row in left],
            [(row["arrival_time_s"], row["input_len"], row["output_len"]) for row in right],
        )

    def test_pool_downshift_requires_both_endpoints(self):
        mixed = [
            recommendation("P0", DVFSAction.STEP_DOWN, 1890),
            recommendation("P1", DVFSAction.HOLD, 2520),
        ]
        self.assertEqual(
            coordinate_pool("P", ("P0", "P1"), mixed, 2520, 2520, False)["action"],
            "HOLD",
        )
        agreed = [
            recommendation("P0", DVFSAction.STEP_DOWN, 1890),
            recommendation("P1", DVFSAction.STEP_DOWN, 1890),
        ]
        result = coordinate_pool("P", ("P0", "P1"), agreed, 2520, 2520, False)
        self.assertEqual((result["action"], result["target_freq_mhz"]), ("STEP_DOWN", 1890))

    def test_fresh_latency_or_admission_risk_recovers_pool_high(self):
        recommendations = [
            recommendation("P0", DVFSAction.RECOVER_HIGH, 2520),
            recommendation("P1", DVFSAction.HOLD, 1260),
        ]
        result = coordinate_pool("P", ("P0", "P1"), recommendations, 1260, 2520, False)
        self.assertEqual((result["action"], result["target_freq_mhz"]), ("RECOVER_HIGH", 2520))
        holds = [
            recommendation("P0", DVFSAction.HOLD, 1260),
            recommendation("P1", DVFSAction.HOLD, 1260),
        ]
        result = coordinate_pool("P", ("P0", "P1"), holds, 1260, 2520, True)
        self.assertEqual(result["target_freq_mhz"], 2520)
        self.assertTrue(admission_risk({"client_waiting_at_offered_close": 1}, {"admission_risk_p95_ms": 250}))

    def test_launcher_has_dedicated_c1_mode(self):
        source = (self.root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("XPYD_STAGEC1_RUNTIME_CONFIG", source)
        self.assertIn("-m xpyd.stagec1_stationary", source)

    def test_headroom_config_starts_high_and_changes_one_axis(self):
        _, experiment = load_configs(
            self.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json",
            self.root / "paper/configs/xpyd_stagec1_headroom_neptune_europa.json",
        )
        points = experiment["stagec1_headroom"]["diagnostic_points"]
        self.assertEqual((points[0]["P_level"], points[0]["D_level"]), ("HIGH", "HIGH"))
        self.assertEqual(
            [(point["P_level"], point["D_level"]) for point in points[1:]],
            [("HIGH", "MID"), ("HIGH", "LOW"), ("MID", "HIGH"), ("LOW", "HIGH")],
        )
        levels = experiment["stagec1"]["frequency_levels_mhz"]
        self.assertEqual(_level_frequency(levels, "P", "LOW"), 1260)
        self.assertEqual(_level_frequency(levels, "D", "MID"), 1125)

    def test_headroom_does_not_enable_controller_or_oracle(self):
        _, experiment = load_configs(
            self.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json",
            self.root / "paper/configs/xpyd_stagec1_headroom_neptune_europa.json",
        )
        settings = experiment["stagec1"]
        self.assertFalse(settings["request_aware_routing"])
        self.assertFalse(settings["probing"])
        self.assertFalse(settings["predictor"])
        self.assertFalse(settings["oracle"])

    def test_actuation_records_after_registry_readback_state_update(self):
        class State:
            freq_mhz = 1500

        class Registry:
            def __init__(self):
                self.state = State()

            def get_state(self, _endpoint):
                return self.state

            def update_state(self, state):
                self.state = state

        class Scheduler:
            def __init__(self, registry):
                self.registry = registry
                self.observed = []

            def record_dvfs_actuation(self, endpoint, observed_freq_mhz, timestamp_s):
                self.assert_state = self.registry.get_state(endpoint).freq_mhz
                self.observed.append((endpoint, observed_freq_mhz, timestamp_s))

        class Actuator:
            requested = {"D0": 1500}

            def actuate(self, endpoint, target, reason):
                self.requested[endpoint] = target
                return {
                    "endpoint_id": endpoint, "requested_freq_mhz": target,
                    "command_status": "success", "readback_valid": True,
                    "timestamp_unix_s": 100.0,
                    "transition_readback_latency_s": 0.5,
                }

        registry = Registry()
        scheduler = Scheduler(registry)
        harness = StageC1Harness.__new__(StageC1Harness)
        harness.action_rows = []
        harness._actuate_targets(
            Actuator(), type("Context", (), {
                "registry": registry, "scheduler": scheduler,
                "targets": {"D0": 1500},
            })(), {"D0": 1125}, "test", "HEADROOM", "point",
        )
        self.assertEqual(scheduler.assert_state, 1125)
        self.assertEqual(scheduler.observed[0][1], 1125)
        self.assertEqual(scheduler.observed[0][2], 100.5)

    def test_residency_and_oscillation_reporting(self):
        rows = [
            {"P0_freq_mhz": 2520, "observation_duration_s": 3},
            {"P0_freq_mhz": 1890, "observation_duration_s": 1},
        ]
        residency = frequency_residency(
            rows, "P0", {"LOW": 1260, "MID": 1890, "HIGH": 2520}
        )
        self.assertEqual(residency, {"LOW": 0.0, "MID": 0.25, "HIGH": 0.75})
        self.assertEqual(
            oscillation_count([
                {"action": "STEP_DOWN"}, {"action": "HOLD"},
                {"action": "RECOVER_HIGH"}, {"action": "STEP_DOWN"},
            ]),
            2,
        )

    def test_final_controller_state_is_exported_to_convergence_report(self):
        harness = StageC1Harness.__new__(StageC1Harness)
        harness.decision_rows = [{
            "block_id": "b", "pool": "P", "decision": "EXPLORE_DOWN",
            "evidence_classification": "SAFE", "candidate_freq_mhz": 1260,
        }]
        harness.block_search_rows = []
        state = SimpleNamespace(
            convergence_window_count=4, convergence_request_count=80,
            converged=True, convergence_reason="no lower supported frequency remains",
            known_safe_frequency=1260, best_measured_safe_frequency=1890,
            current_frequency=1890,
            best_measured_safe_energy_j=70.0,
            safe_energy_by_frequency_j={2520: 90.0, 1890: 70.0, 1260: 75.0},
        )
        context = SimpleNamespace(
            adaptive_pools={"P": SimpleNamespace(state=state)},
            targets={"P0": 1260, "P1": 1260},
        )
        harness._record_block_search_summary(
            context, policy="ADAPTIVE-SAFE-SEARCH", block_id="b",
            repeat=1, load_point="MEDIUM",
        )
        row = harness.block_search_rows[0]
        self.assertEqual(row["final_validated_safe_frequency_mhz"], 1260)
        self.assertEqual(row["best_measured_safe_frequency_mhz"], 1890)
        self.assertEqual(row["windows_until_convergence"], 4)
        self.assertEqual(row["requests_until_convergence"], 80)


if __name__ == "__main__":
    unittest.main()
