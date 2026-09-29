"""CPU-only contract tests for the Stage C2a stationary experiment."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

from xpyd.adaptive_exploration import AdaptiveExplorationConfig, AdaptivePoolController
from xpyd.stagec2a_fast_safety import (
    POLICIES,
    StageC2AFastSafetyHarness,
    load_configs,
    policy_order,
    segment_residency,
)
from xpyd.stagec2_dynamic import RecoveredActuationFailure


class StageC2AContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.runtime_path = cls.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json"
        cls.experiment_path = cls.root / "paper/configs/xpyd_stagec2a_fast_safety_neptune_europa.json"

    def test_config_reuses_exact_c1_stationary_contract(self):
        _, experiment = load_configs(self.runtime_path, self.experiment_path)
        settings = experiment["stagec2a"]
        self.assertEqual(tuple(settings["policies"]), POLICIES)
        self.assertEqual(settings["slo"], {
            "ttft_ms": 600.0, "tpot_ms": 65.0, "attainment": 0.95,
        })
        self.assertEqual(
            [(row["id"], row["offered_rate_rps"]) for row in settings["load_points"]],
            [("MEDIUM", 0.10), ("HIGH-SUSTAINABLE", 0.20)],
        )
        self.assertEqual(settings["frequency_levels_mhz"]["P0"], {
            "LOW": 1260, "MID": 1890, "HIGH": 2520,
        })
        self.assertEqual(settings["frequency_levels_mhz"]["D0"], {
            "LOW": 750, "MID": 1125, "HIGH": 1500,
        })
        self.assertIsNone(settings["fast_safety"]["risk_fraction_trigger"])
        self.assertNotIn("context_b", json.dumps(experiment).lower())

    def test_policy_order_is_cyclic_and_balanced(self):
        self.assertEqual(policy_order(1), POLICIES)
        self.assertEqual(policy_order(2)[0], "WINDOW-BOUNDED-FEEDBACK")
        self.assertEqual(policy_order(3)[0], "FAST-SAFETY-FEEDBACK")

    def test_segment_residency_uses_actual_stable_clock_intervals(self):
        segments = [
            {"endpoint_id": "P0", "segment_kind": "STABLE", "duration_s": 2.0,
             "observed_readback_frequency_mhz": 1260},
            {"endpoint_id": "P0", "segment_kind": "TRANSITION", "duration_s": 0.2,
             "observed_readback_frequency_mhz": 1890},
            {"endpoint_id": "P0", "segment_kind": "STABLE", "duration_s": 6.0,
             "observed_readback_frequency_mhz": 1890},
        ]
        self.assertEqual(segment_residency(segments, "P0"), {
            "1260": 0.25, "1890": 0.75,
        })

    def test_unsafe_exposure_distinguishes_window_and_fast_recovery(self):
        harness = StageC2AFastSafetyHarness.__new__(StageC2AFastSafetyHarness)
        harness.decision_rows = [
            {"decision": "EXPLORE_DOWN", "policy": "WINDOW-BOUNDED-FEEDBACK",
             "block_id": "window", "pool": "P", "before_window_index": 3,
             "requested_freq_mhz": 1260},
            {"policy": "WINDOW-BOUNDED-FEEDBACK", "block_id": "window",
             "pool": "P", "evidence_window_index": 3,
             "evidence_classification": "UNSAFE", "action": "STEP_UP",
             "requested_freq_mhz": 1890, "trigger": "SLO_RISK",
             "readback_timestamp_s": 40.0},
            {"decision": "EXPLORE_DOWN", "policy": "FAST-SAFETY-FEEDBACK",
             "block_id": "fast", "pool": "P", "before_window_index": 3,
             "requested_freq_mhz": 1260},
        ]
        harness.safety_rows = [{
            "policy": "FAST-SAFETY-FEEDBACK", "block_id": "fast",
            "window_index": 3, "pool": "P", "trigger": "HARD_TTFT_VIOLATION",
            "action": "FAST_RECOVER", "target_mhz": 1890,
            "request_id": "fast-1", "readback_timestamp_s": 22.0,
        }]
        harness.segment_rows = [
            {"block_id": block, "window_index": 3, "endpoint_id": "P0",
             "segment_kind": "STABLE", "requested_frequency_mhz": 1260,
             "segment_start_wall_s": 10.0}
            for block in ("window", "fast")
        ]
        rows = [
            {"block_id": block, "window_index": 3, "repeat": 1,
             "load_point": "HIGH-SUSTAINABLE", "window_start_wall_s": 10.0}
            for block in ("window", "fast")
        ]
        requests = []
        for block in ("window", "fast"):
            for index in range(4):
                requests.append({
                    "block_id": block, "window_index": 3,
                    "send_unix_s": str(10 + index * 5),
                    "first_token_unix_s": str(15 + index * 5),
                    "complete_unix_s": str(16 + index * 5),
                    "ttft_ms": "650" if index == 1 else "500",
                    "tpot_ms": "60",
                })
        exposure = harness._unsafe_exposure_rows(rows, requests)
        by_policy = {row["policy"]: row for row in exposure}
        self.assertEqual(
            by_policy["WINDOW-BOUNDED-FEEDBACK"]["requests_exposed_before_readback"], 4
        )
        self.assertEqual(
            by_policy["FAST-SAFETY-FEEDBACK"]["requests_exposed_before_readback"], 3
        )
        self.assertLess(
            by_policy["FAST-SAFETY-FEEDBACK"]["unsafe_probe_duration_s"],
            by_policy["WINDOW-BOUNDED-FEEDBACK"]["unsafe_probe_duration_s"],
        )

    def test_launcher_has_explicit_c2a_modes(self):
        source = (self.root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn('"c2a-preflight"', source)
        self.assertIn('"c2a-full"', source)
        self.assertIn("-m xpyd.stagec2a_fast_safety", source)

    def test_physical_readback_failure_invokes_validated_safe_fallback(self):
        config = AdaptiveExplorationConfig(
            ttft_slo_ms=600, tpot_slo_ms=65, min_valid_requests=1,
            min_input_tokens=1, min_output_tokens=1,
            min_observation_duration_s=0,
        )
        p = AdaptivePoolController("P", "prefill", [100, 200, 300], config)
        d = AdaptivePoolController("D", "decode", [100, 200, 300], config)

        class Registry:
            def __init__(self):
                self.states = {
                    key: SimpleNamespace(freq_mhz=300)
                    for key in ("P0", "P1", "D0", "D1")
                }

            def get_state(self, endpoint_id):
                return self.states[endpoint_id]

            def update_state(self, state):
                return state

        class Scheduler:
            def record_dvfs_actuation(self, *args, **kwargs):
                return None

        class Actuator:
            def __init__(self):
                self.requested = {key: 300 for key in ("P0", "P1", "D0", "D1")}
                self.now = 10.0

            def actuate(self, endpoint_id, target, reason):
                self.now += 0.1
                failed = target == 200
                if not failed:
                    self.requested[endpoint_id] = target
                return {
                    "timestamp_unix_s": self.now,
                    "transition_readback_latency_s": 0.01,
                    "endpoint_id": endpoint_id,
                    "requested_freq_mhz": target,
                    "previous_requested_freq_mhz": 300,
                    "observed_freq_after_mhz": None if failed else target,
                    "command_status": "failed" if failed else "success",
                    "readback_valid": not failed,
                    "reason": reason,
                    "error": "mismatch" if failed else None,
                }

        harness = StageC2AFastSafetyHarness.__new__(StageC2AFastSafetyHarness)
        harness.action_rows = []
        harness.safety_rows = []
        harness.active_ledger = None
        harness.actuation_lock = threading.RLock()
        context = SimpleNamespace(
            adaptive_pools={"P": p, "D": d}, registry=Registry(),
            scheduler=Scheduler(), targets={key: 300 for key in ("P0", "P1", "D0", "D1")},
        )
        with self.assertRaises(RecoveredActuationFailure):
            harness._actuate_targets(
                Actuator(), context, {"P0": 200, "P1": 200},
                reason="probe", policy="FAST-SAFETY-FEEDBACK",
                block_id="fixture", recover_on_failure=True,
            )
        self.assertEqual(context.targets["P0"], 300)
        self.assertEqual(context.targets["P1"], 300)
        self.assertEqual(harness.safety_rows[0]["trigger"], "ACTUATOR_READBACK_FAILURE")
        self.assertTrue(harness.safety_rows[0]["readback_valid"])


if __name__ == "__main__":
    unittest.main()
