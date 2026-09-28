"""CPU-only deterministic tests for the C1 adaptive safe-search controller."""

from __future__ import annotations

from dataclasses import replace
import unittest

from xpyd.adaptive_exploration import (
    AdaptiveExplorationConfig,
    AdaptivePoolController,
    EvidenceClass,
    PoolEvidence,
    evidence_from_window,
)


class AdaptiveExplorationTests(unittest.TestCase):
    def setUp(self):
        self.config = AdaptiveExplorationConfig(
            ttft_slo_ms=600.0,
            tpot_slo_ms=65.0,
            min_valid_requests=8,
            min_input_tokens=128,
            min_output_tokens=128,
            min_observation_duration_s=1.0,
            max_evidence_timeout_s=10.0,
            headroom_for_one_step=0.01,
            headroom_for_large_jump=0.10,
            large_jump_levels=2,
        )

    def evidence(self, controller, *, frequency=None, latency=None, passed=10,
                 count=10, backlog_stable=True, fresh=True, health=True,
                 readback=True, accounting=True, energy=100.0):
        frequency = controller.current_frequency_mhz if frequency is None else frequency
        if controller.phase == "prefill":
            ttft = 500.0 if latency is None else latency
            tpot = 40.0
            ttft_pass, ttft_bad = passed, count - passed
            tpot_pass, tpot_bad = count, 0
        else:
            ttft = 500.0
            tpot = 55.0 if latency is None else latency
            ttft_pass, ttft_bad = count, 0
            tpot_pass, tpot_bad = passed, count - passed
        return PoolEvidence(
            pool=controller.pool,
            phase=controller.phase,
            observed_frequency_mhz=frequency,
            observation_duration_s=2.0,
            valid_request_count=count,
            input_token_coverage=count * 128,
            output_token_coverage=count * 128,
            ttft_statistic_ms=ttft,
            tpot_statistic_ms=tpot,
            ttft_pass_count=ttft_pass,
            ttft_violation_count=ttft_bad,
            tpot_pass_count=tpot_pass,
            tpot_violation_count=tpot_bad,
            backlog_stable=backlog_stable,
            telemetry_fresh=fresh,
            health_valid=health,
            readback_valid=readback,
            request_accounting_valid=accounting,
            pressure_signal="stable" if backlog_stable else "growing",
            gross_energy_j=energy,
            context_signature="stationary-test",
        )

    def controller(self, pool="P", phase="prefill"):
        return AdaptivePoolController(pool, phase, [100, 200, 300], self.config)

    def one_step_controller(self, pool="P", phase="prefill"):
        return AdaptivePoolController(
            pool, phase, [100, 200, 300],
            replace(self.config, large_jump_levels=1),
        )

    def apply(self, controller, decision, now=1.0):
        if decision.requested_frequency_mhz != decision.current_frequency_mhz:
            controller.record_actuation(
                decision.requested_frequency_mhz,
                decision.requested_frequency_mhz,
                readback_valid=True,
                timestamp_s=now,
            )

    def test_a_headroom_explores_large_jump_then_retains_safe_candidate(self):
        controller = self.controller()
        first = controller.decide(self.evidence(controller, latency=450), now_s=1.0)
        self.assertEqual((first.decision, first.action, first.requested_frequency_mhz),
                         ("EXPLORE_DOWN", "STEP_DOWN", 100))
        self.apply(controller, first)
        second = controller.decide(self.evidence(controller, frequency=100, latency=500), now_s=2.0)
        self.assertEqual((second.evidence_classification, second.decision, second.action),
                         (EvidenceClass.SAFE.value, "ACCEPT_SAFE", "HOLD"))
        self.assertEqual(controller.state.known_safe_frequency, 100)

    def test_b_slo_violation_rolls_back_immediately(self):
        controller = self.controller()
        first = controller.decide(self.evidence(controller, latency=500), now_s=1.0)
        self.apply(controller, first)
        unsafe = controller.decide(
            self.evidence(controller, frequency=100, latency=650, passed=7), now_s=2.0
        )
        self.assertEqual(unsafe.evidence_classification, EvidenceClass.UNSAFE.value)
        self.assertEqual((unsafe.decision, unsafe.action, unsafe.requested_frequency_mhz),
                         ("RECOVER", "RECOVER_HIGH", 300))

    def test_c_insufficient_evidence_does_not_become_safe(self):
        controller = self.controller()
        insufficient = self.evidence(controller, count=1, passed=1)
        result = controller.decide(insufficient, now_s=1.0)
        self.assertEqual((result.decision, result.action, result.evidence_classification),
                         ("HOLD", "HOLD", EvidenceClass.INSUFFICIENT_EVIDENCE.value))
        self.assertEqual(controller.state.known_safe_frequency, 300)

    def test_d_backlog_growth_recovers_even_with_good_latency(self):
        controller = self.controller()
        result = controller.decide(self.evidence(controller, backlog_stable=False), now_s=1.0)
        self.assertEqual((result.trigger, result.decision, result.requested_frequency_mhz),
                         (EvidenceClass.PRESSURE_RISK.value, "HOLD", 300))

    def test_e_stale_telemetry_is_conservative(self):
        controller = self.controller()
        result = controller.decide(self.evidence(controller, fresh=False), now_s=1.0)
        self.assertEqual((result.evidence_classification, result.decision),
                         (EvidenceClass.STALE_TELEMETRY.value, "HOLD"))
        self.assertEqual(controller.state.search_phase, "RECOVER")

    def test_f_unsafe_lower_bound_is_refined_without_restart(self):
        controller = self.controller()
        first = controller.decide(self.evidence(controller, latency=450), now_s=1.0)
        self.apply(controller, first)
        unsafe = controller.decide(
            self.evidence(controller, frequency=100, latency=650, passed=7), now_s=2.0
        )
        self.apply(controller, unsafe, now=2.0)
        refined = controller.decide(self.evidence(controller, frequency=300, latency=500), now_s=3.0)
        self.assertEqual((refined.trigger, refined.decision, refined.requested_frequency_mhz),
                         ("REFINE", "EXPLORE_DOWN", 200))

    def test_g_p_and_d_search_state_is_independent(self):
        p = self.controller("P", "prefill")
        d = self.controller("D", "decode")
        p_first = p.decide(self.evidence(p, latency=500), now_s=1.0)
        d_first = d.decide(self.evidence(d, latency=50), now_s=1.0)
        self.apply(p, p_first)
        self.apply(d, d_first)
        p_safe = p.decide(self.evidence(p, frequency=100, latency=500), now_s=2.0)
        d_unsafe = d.decide(self.evidence(d, frequency=100, latency=75, passed=7), now_s=2.0)
        self.assertEqual(p_safe.evidence_classification, EvidenceClass.SAFE.value)
        self.assertEqual(d_unsafe.evidence_classification, EvidenceClass.UNSAFE.value)
        self.assertEqual(p.state.known_safe_frequency, 100)
        self.assertEqual(d.state.known_safe_frequency, 300)

    def test_h_readback_mismatch_is_never_validated_safe(self):
        controller = self.controller()
        result = controller.decide(
            self.evidence(controller, frequency=200, readback=False), now_s=1.0
        )
        self.assertEqual(result.evidence_classification, EvidenceClass.READBACK_FAILURE.value)
        self.assertEqual(result.decision, "HOLD")
        self.assertEqual(controller.state.known_safe_frequency, 300)

    def test_window_adapter_keeps_phase_token_and_request_audits(self):
        metrics = {
            "measurement_valid": True,
            "observation_duration_s": 2.0,
            "ttft_p95_ms": 500.0,
            "tpot_p95_ms": 55.0,
            "P0_energy_j": 12.0,
            "P1_energy_j": 13.0,
            "backlog_growing": False,
            "client_waiting_at_offered_close": 0,
        }
        requests = [
            {"real_sse": "True", "logical_request_id_propagated": "True",
             "input_len": "128", "observed_output_tokens": "64",
             "ttft_ms": "500", "tpot_ms": "55"},
            {"real_sse": "False", "logical_request_id_propagated": "True",
             "input_len": "128", "observed_output_tokens": "64",
             "ttft_ms": "500", "tpot_ms": "55"},
        ]
        evidence = evidence_from_window(
            "P", metrics, requests, observed_frequency_mhz=300,
            slo_regime="TIGHT", ttft_slo_ms=600.0, tpot_slo_ms=65.0,
            context_signature="stationary:MEDIUM", captured_at_s=4.0,
        )
        self.assertEqual((evidence.valid_request_count, evidence.input_token_coverage), (1, 128))
        self.assertFalse(evidence.request_accounting_valid)

    def test_known_unsafe_adjacent_level_is_never_reprobed(self):
        controller = self.one_step_controller()
        probe_mid = controller.decide(self.evidence(controller, latency=500), now_s=1.0)
        self.assertEqual(probe_mid.requested_frequency_mhz, 200)
        self.apply(controller, probe_mid)
        unsafe_mid = controller.decide(
            self.evidence(controller, frequency=200, latency=650, passed=7), now_s=2.0
        )
        self.assertEqual(unsafe_mid.requested_frequency_mhz, 300)
        self.apply(controller, unsafe_mid, now=2.0)
        safe_high = controller.decide(
            self.evidence(controller, frequency=300, latency=500), now_s=3.0
        )
        self.assertEqual((safe_high.action, safe_high.requested_frequency_mhz), ("HOLD", 300))
        self.assertEqual(controller.state.unsafe_bound, 200)
        self.assertTrue(controller.state.converged)

    def test_headroom_thresholds_control_hold_one_step_and_coarse_jump(self):
        low_headroom = self.controller()
        hold = low_headroom.decide(self.evidence(low_headroom, latency=595), now_s=1.0)
        self.assertEqual((hold.action, hold.requested_frequency_mhz), ("HOLD", 300))

        moderate = self.controller()
        one_step = moderate.decide(self.evidence(moderate, latency=570), now_s=1.0)
        self.assertEqual((one_step.action, one_step.requested_frequency_mhz), ("STEP_DOWN", 200))

        strong = self.controller()
        coarse = strong.decide(self.evidence(strong, latency=500), now_s=1.0)
        self.assertEqual((coarse.action, coarse.requested_frequency_mhz), ("STEP_DOWN", 100))

    def test_final_window_candidate_is_classified_without_unmeasured_probe(self):
        controller = self.controller()
        probe = controller.decide(self.evidence(controller, latency=500, energy=100), now_s=1.0)
        self.apply(controller, probe)
        final = controller.decide(
            self.evidence(controller, frequency=100, latency=500, energy=80),
            now_s=2.0, allow_exploration=False,
        )
        self.assertEqual((final.trigger, final.action), ("FINAL_EVIDENCE", "HOLD"))
        self.assertEqual(controller.state.known_safe_frequency, 100)
        self.assertEqual(controller.state.best_measured_safe_frequency, 100)
        self.assertTrue(controller.state.converged)
        self.assertEqual(controller.state.convergence_window_count, 2)
        self.assertEqual(controller.state.convergence_request_count, 20)

    def test_missing_endpoint_pressure_is_unknown_and_explicit_zero_is_safe(self):
        requests = [
            {"real_sse": "True", "logical_request_id_propagated": "True",
             "input_len": "128", "observed_output_tokens": "128",
             "ttft_ms": "500", "tpot_ms": "55"}
            for _ in range(10)
        ]
        metrics = {
            "measurement_valid": True, "observation_duration_s": 2.0,
            "ttft_p95_ms": 500.0, "tpot_p95_ms": 55.0,
            "P0_energy_j": 10.0, "P1_energy_j": 10.0,
            "backlog_growing": False, "client_waiting_at_offered_close": 0,
            "endpoint_load": {
                "P0": {"waiting_mean": 0.0, "waiting_max": 0.0},
                "P1": {"waiting_mean": 0.0, "waiting_max": 0.0},
            },
        }
        explicit_zero = evidence_from_window(
            "P", metrics, requests, observed_frequency_mhz=300,
            slo_regime="TIGHT", ttft_slo_ms=600, tpot_slo_ms=65,
        )
        self.assertTrue(explicit_zero.backlog_stable)
        controller = self.controller()
        self.assertEqual(
            controller.decide(explicit_zero, now_s=1.0).evidence_classification,
            EvidenceClass.SAFE.value,
        )

        missing = dict(metrics)
        missing["endpoint_load"] = {
            "P0": {"waiting_mean": 0.0, "waiting_max": 0.0},
            "P1": {"waiting_mean": None, "waiting_max": None},
        }
        unknown = evidence_from_window(
            "P", missing, requests, observed_frequency_mhz=300,
            slo_regime="TIGHT", ttft_slo_ms=600, tpot_slo_ms=65,
        )
        self.assertIsNone(unknown.backlog_stable)
        other = self.controller()
        self.assertEqual(
            other.decide(unknown, now_s=1.0).evidence_classification,
            EvidenceClass.PRESSURE_RISK.value,
        )

    def test_best_measured_safe_energy_point_not_lowest_frequency(self):
        controller = self.controller()
        probe = controller.decide(self.evidence(controller, latency=500, energy=80), now_s=1.0)
        self.apply(controller, probe)
        select = controller.decide(
            self.evidence(controller, frequency=100, latency=500, energy=120), now_s=2.0
        )
        self.assertEqual(controller.state.safe_energy_by_frequency_j, {300: 80.0, 100: 120.0})
        self.assertEqual(controller.state.best_measured_safe_frequency, 300)
        self.assertEqual((select.decision, select.requested_frequency_mhz), ("SELECT_BEST_SAFE", 300))

    def test_recommendation_does_not_change_last_actual_action_time(self):
        controller = self.controller()
        decision = controller.decide(self.evidence(controller, latency=500), now_s=10.0)
        self.assertEqual(decision.action_timestamp_s, None)
        self.assertEqual(controller.state.last_action_timestamp_s, 0.0)
        controller.record_actuation(
            decision.requested_frequency_mhz, decision.requested_frequency_mhz,
            readback_valid=True, timestamp_s=11.5,
        )
        self.assertEqual(controller.state.last_action_timestamp_s, 11.5)


if __name__ == "__main__":
    unittest.main()
