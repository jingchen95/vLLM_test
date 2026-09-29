"""CPU-only regression tests for Stage C2 hybrid fast safety."""

from __future__ import annotations

import unittest

from xpyd.adaptive_exploration import (
    AdaptiveExplorationConfig,
    AdaptivePoolController,
    PoolEvidence,
)
from xpyd.stagec2_dynamic import (
    ClockSegmentLedger,
    FastSafetyRuntime,
    derived_context_signature,
    dynamic_clock_audit,
    fast_recovery_recommendation,
    observed_frequency_residency,
    policy_order,
)


class StageC2DynamicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AdaptiveExplorationConfig(
            ttft_slo_ms=600.0,
            tpot_slo_ms=65.0,
            min_valid_requests=8,
            min_input_tokens=128,
            min_output_tokens=128,
            min_observation_duration_s=1.0,
            max_evidence_timeout_s=120.0,
            headroom_for_one_step=0.01,
            headroom_for_large_jump=0.10,
            large_jump_levels=1,
        )
        self.settings = {
            "slo": {"ttft_ms": 600.0, "tpot_ms": 65.0, "attainment": 0.95},
            "fast_safety": {"telemetry_max_age_s": 1.0},
            "severe_queue_depth": 4,
            "context_inference": {
                "long_input_tokens": 512,
                "long_output_tokens": 256,
                "prefill_heavy_fraction": 0.75,
                "decode_heavy_fraction": 0.50,
            },
        }

    def controller(self, pool: str) -> AdaptivePoolController:
        phase = "prefill" if pool == "P" else "decode"
        return AdaptivePoolController(pool, phase, [100, 200, 300], self.config)

    def evidence(
        self, controller: AdaptivePoolController, *, latency: float,
        context: str = "context-a", count: int = 10,
    ) -> PoolEvidence:
        ttft = latency if controller.pool == "P" else 500.0
        tpot = latency if controller.pool == "D" else 50.0
        slo = 600.0 if controller.pool == "P" else 65.0
        passed = count if latency <= slo else count - 1
        return PoolEvidence(
            pool=controller.pool,
            phase=controller.phase,
            observed_frequency_mhz=controller.current_frequency_mhz,
            observation_duration_s=10.0,
            valid_request_count=count,
            input_token_coverage=count * 128,
            output_token_coverage=count * 128,
            ttft_statistic_ms=ttft,
            tpot_statistic_ms=tpot,
            ttft_pass_count=passed if controller.pool == "P" else count,
            ttft_violation_count=count - passed if controller.pool == "P" else 0,
            tpot_pass_count=passed if controller.pool == "D" else count,
            tpot_violation_count=count - passed if controller.pool == "D" else 0,
            backlog_stable=True,
            telemetry_fresh=True,
            health_valid=True,
            readback_valid=True,
            request_accounting_valid=True,
            pressure_signal="stable",
            gross_energy_j=100.0,
            context_signature=context,
        )

    def activate_mid(self, controller: AdaptivePoolController, *, context: str = "context-a") -> None:
        latency = 570.0 if controller.pool == "P" else 60.0
        decision = controller.decide(self.evidence(controller, latency=latency, context=context), now_s=1.0)
        self.assertEqual(decision.requested_frequency_mhz, 200)
        controller.record_actuation(200, 200, readback_valid=True, timestamp_s=1.1)

    def runtime(self, controllers, recover, *, enabled=True):
        return FastSafetyRuntime(
            controllers, recover, self.settings, policy="HYBRID-FAST-SAFETY",
            block_id="r01-hybrid", window_index=2, enabled=enabled,
        )

    def test_ttft_violation_recovers_p_before_next_evidence_epoch(self):
        p, d = self.controller("P"), self.controller("D")
        self.activate_mid(p)
        recovered = []

        def recover(pool, recommendation):
            recovered.append((pool, recommendation))
            controllers[pool].record_actuation(
                recommendation["target_mhz"], recommendation["target_mhz"],
                readback_valid=True, timestamp_s=2.1,
            )
            return {"action_timestamp_s": 2.0, "readback_timestamp_s": 2.1,
                    "readback_frequency_mhz": recommendation["target_mhz"], "readback_valid": True}

        controllers = {"P": p, "D": d}
        runtime = self.runtime(controllers, recover)
        runtime.process_safety_event({
            "event": "TTFT_OBSERVED", "timestamp_unix_s": 1.9,
            "logical_request_id": "req-1", "ttft_ms": 601.0,
        })
        self.assertEqual(recovered[0][0], "P")
        self.assertEqual(p.current_frequency_mhz, 300)
        self.assertEqual(runtime.event_rows[0]["action"], "FAST_RECOVER")

    def test_tpot_violation_recovers_d_before_next_evidence_epoch(self):
        p, d = self.controller("P"), self.controller("D")
        self.activate_mid(d)
        controllers = {"P": p, "D": d}

        def recover(pool, recommendation):
            controllers[pool].record_actuation(
                recommendation["target_mhz"], recommendation["target_mhz"],
                readback_valid=True, timestamp_s=2.1,
            )
            return {"action_timestamp_s": 2.0, "readback_timestamp_s": 2.1,
                    "readback_frequency_mhz": recommendation["target_mhz"], "readback_valid": True}

        runtime = self.runtime(controllers, recover)
        runtime.process_safety_event({
            "event": "REQUEST_COMPLETED", "timestamp_unix_s": 1.9,
            "logical_request_id": "req-2", "tpot_ms": 65.1,
        })
        self.assertEqual(d.current_frequency_mhz, 300)
        self.assertEqual(runtime.event_rows[0]["pool"], "D")

    def test_recovery_uses_previous_validated_safe_anchor(self):
        controller = self.controller("P")
        self.activate_mid(controller)
        safe_mid = controller.decide(
            self.evidence(controller, latency=570.0), now_s=2.0
        )
        self.assertEqual(safe_mid.requested_frequency_mhz, 100)
        controller.record_actuation(100, 100, readback_valid=True, timestamp_s=2.1)
        recommendation = fast_recovery_recommendation(
            controller, "HARD_TTFT_VIOLATION", 3.0
        )
        self.assertEqual(recommendation["target_mhz"], 200)
        self.assertEqual(controller.state.known_safe_frequency, 200)

    def test_normal_downward_exploration_still_waits_for_evidence(self):
        controller = self.controller("P")
        decision = controller.decide(
            self.evidence(controller, latency=500.0, count=1), now_s=1.0
        )
        self.assertEqual((decision.action, decision.trigger), ("HOLD", "INSUFFICIENT_EVIDENCE"))
        self.assertEqual(controller.current_frequency_mhz, 300)

    def test_triggering_violation_remains_logged_after_recovery(self):
        p, d = self.controller("P"), self.controller("D")
        self.activate_mid(p)
        controllers = {"P": p, "D": d}

        def recover(pool, recommendation):
            controllers[pool].record_actuation(300, 300, readback_valid=True, timestamp_s=2.1)
            return {"action_timestamp_s": 2.0, "readback_timestamp_s": 2.1,
                    "readback_frequency_mhz": 300, "readback_valid": True}

        runtime = self.runtime(controllers, recover)
        runtime.process_safety_event({
            "event": "TTFT_OBSERVED", "timestamp_unix_s": 1.9,
            "logical_request_id": "violating-request", "ttft_ms": 700.0,
        })
        self.assertEqual(len(runtime.event_rows), 1)
        row = runtime.event_rows[0]
        self.assertTrue(row["violation_recorded"])
        self.assertEqual(row["request_id"], "violating-request")
        self.assertEqual(row["ttft_ms"], 700.0)

    def test_mid_window_transition_and_residency_are_segmented(self):
        ledger = ClockSegmentLedger("w1")
        ledger.start({"P0": 200}, 1.0)
        ledger.transition(
            "P0", 300, reason="hard violation", decision_timestamp_s=1.8,
            action_timestamp_s=2.0, readback_timestamp_s=2.1,
            observed_readback_mhz=300,
        )
        ledger.finish(1.0, 3.0)
        self.assertEqual([row["segment_kind"] for row in ledger.rows],
                         ["STABLE", "TRANSITION", "STABLE"])
        samples = [
            {"status": "success", "actual_start_wall_s": 1.2,
             "actual_start_local_monotonic_s": 1.2, "graphics_clock_mhz": 200,
             "memory_clock_mhz": 5000, "power_w": 100},
            {"status": "success", "actual_start_wall_s": 2.2,
             "actual_start_local_monotonic_s": 2.2, "graphics_clock_mhz": 300,
             "memory_clock_mhz": 5000, "power_w": 110},
            {"status": "success", "actual_start_wall_s": 2.8,
             "actual_start_local_monotonic_s": 2.8, "graphics_clock_mhz": 300,
             "memory_clock_mhz": 5000, "power_w": 110},
        ]
        audit = dynamic_clock_audit("P0", samples, ledger.rows, 1.0, 3.0, 5000)
        self.assertTrue(audit["valid"])
        self.assertEqual(len(audit["clock_segments"]), 2)
        residency = observed_frequency_residency(
            samples, 1.0, 3.0, {"MID": 200, "HIGH": 300}
        )
        self.assertGreater(residency["MID"], 0.0)
        self.assertGreater(residency["HIGH"], 0.0)

    def test_recommendation_action_and_readback_timestamps_are_distinct(self):
        ledger = ClockSegmentLedger("w1")
        ledger.start({"P0": 200}, 1.0)
        ledger.transition(
            "P0", 300, reason="hard violation", decision_timestamp_s=1.7,
            action_timestamp_s=1.9, readback_timestamp_s=2.0,
            observed_readback_mhz=300,
        )
        ledger.finish(1.0, 2.5)
        transition = next(row for row in ledger.rows if row["segment_kind"] == "TRANSITION")
        self.assertLess(transition["decision_timestamp_s"], transition["action_timestamp_s"])
        self.assertLess(transition["action_timestamp_s"], transition["readback_timestamp_s"])

    def test_exploration_can_resume_after_recovery_and_context_relaxes(self):
        controller = self.controller("P")
        self.activate_mid(controller, context="context-b")
        recommendation = fast_recovery_recommendation(
            controller, "HARD_TTFT_VIOLATION", 2.0
        )
        controller.record_actuation(
            recommendation["target_mhz"], recommendation["target_mhz"],
            readback_valid=True, timestamp_s=2.1,
        )
        context_change = controller.decide(
            self.evidence(controller, latency=570.0, context="context-a"), now_s=3.0
        )
        self.assertEqual(context_change.trigger, "CONTEXT_CHANGE")
        resumed = controller.decide(
            self.evidence(controller, latency=570.0, context="context-a"), now_s=4.0
        )
        self.assertEqual((resumed.decision, resumed.requested_frequency_mhz),
                         ("EXPLORE_DOWN", 200))

    def test_window_bounded_policy_logs_but_does_not_mutate_mid_window(self):
        p, d = self.controller("P"), self.controller("D")
        self.activate_mid(p)
        calls = []
        runtime = self.runtime({"P": p, "D": d}, lambda *args: calls.append(args), enabled=False)
        runtime.process_safety_event({
            "event": "TTFT_OBSERVED", "timestamp_unix_s": 2.0,
            "logical_request_id": "req", "ttft_ms": 700.0,
        })
        self.assertEqual(p.current_frequency_mhz, 200)
        self.assertEqual(runtime.event_rows[0]["action"], "WINDOW_DEFER")
        self.assertEqual(calls, [])

    def test_invalid_required_telemetry_triggers_fail_closed_recovery(self):
        p, d = self.controller("P"), self.controller("D")
        self.activate_mid(p)
        controllers = {"P": p, "D": d}

        def recover(pool, recommendation):
            controllers[pool].record_actuation(300, 300, readback_valid=True, timestamp_s=2.1)
            return {"action_timestamp_s": 2.0, "readback_timestamp_s": 2.1,
                    "readback_frequency_mhz": 300, "readback_valid": True}

        runtime = self.runtime(controllers, recover)
        runtime.process_internal_failure("P", "INVALID_SAFETY_TELEMETRY", 1.9)
        self.assertEqual(p.current_frequency_mhz, 300)
        self.assertEqual(runtime.event_rows[0]["trigger"], "INVALID_SAFETY_TELEMETRY")

    def test_readback_failure_fails_closed_to_safe_anchor(self):
        p, d = self.controller("P"), self.controller("D")
        self.activate_mid(p)
        controllers = {"P": p, "D": d}

        def recover(pool, recommendation):
            controllers[pool].record_actuation(300, 300, readback_valid=True, timestamp_s=2.1)
            return {"action_timestamp_s": 2.0, "readback_timestamp_s": 2.1,
                    "readback_frequency_mhz": 300, "readback_valid": True}

        runtime = self.runtime(controllers, recover)
        runtime.process_internal_failure("P", "ACTUATOR_READBACK_FAILURE", 1.9)
        self.assertEqual(p.current_frequency_mhz, 300)
        self.assertEqual(runtime.event_rows[0]["action"], "FAST_RECOVER")
        self.assertLess(runtime.event_rows[0]["readback_timestamp_s"], 100.0)

    def test_hard_violation_does_not_reclassify_a_validated_non_candidate(self):
        p, d = self.controller("P"), self.controller("D")
        p.set_current_frequency(200)
        p.state.known_safe_frequency = 200
        p.state.last_validated_safe_index = 1
        calls = []
        runtime = self.runtime({"P": p, "D": d}, lambda *args: calls.append(args))
        runtime.process_safety_event({
            "event": "TTFT_OBSERVED", "timestamp_unix_s": 2.0,
            "logical_request_id": "noise", "ttft_ms": 601.0,
        })
        self.assertEqual(runtime.event_rows[0]["action"], "LOG_ONLY_NO_ACTIVE_CANDIDATE")
        self.assertEqual(calls, [])

    def test_context_signature_uses_tokens_and_policy_order_is_balanced(self):
        requests = [
            {"input_len": 2048, "requested_output_len": 128, "segment": "secret-a"}
            for _ in range(8)
        ] + [
            {"input_len": 128, "requested_output_len": 512, "segment": "secret-b"}
            for _ in range(2)
        ]
        self.assertEqual(
            derived_context_signature("P", requests, self.settings), "P:long-input-high"
        )
        self.assertEqual(policy_order(1)[0], "STATIC-HIGH")
        self.assertEqual(policy_order(2)[0], "WINDOW-BOUNDED-FEEDBACK")
        self.assertEqual(policy_order(3)[0], "HYBRID-FAST-SAFETY")


if __name__ == "__main__":
    unittest.main()
