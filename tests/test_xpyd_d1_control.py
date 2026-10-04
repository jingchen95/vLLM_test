"""CPU-only tests for D1 per-endpoint control and safety isolation."""

import unittest

from xpyd.adaptive_exploration import AdaptiveExplorationConfig, AdaptivePoolController
from xpyd.d1_control import EndpointFastSafetyRuntime, exploration_permissions


def controllers():
    config = AdaptiveExplorationConfig(
        ttft_slo_ms=600, tpot_slo_ms=65, min_valid_requests=1,
        min_input_tokens=1, min_output_tokens=1, min_observation_duration_s=0,
    )
    result = {
        "P0": AdaptivePoolController("P", "prefill", [1260, 1890, 2520], config),
        "P1": AdaptivePoolController("P", "prefill", [1260, 1890, 2520], config),
        "D0": AdaptivePoolController("D", "decode", [750, 1125, 1500], config),
        "D1": AdaptivePoolController("D", "decode", [750, 1125, 1500], config),
    }
    return result


def runtime(values, recovered):
    return EndpointFastSafetyRuntime(
        values,
        lambda endpoint, recommendation: recovered.append(
            (endpoint, recommendation["target_mhz"])
        ) or {"readback_valid": True},
        {
            "slo": {"ttft_ms": 600.0, "tpot_ms": 65.0},
            "fast_safety": {"telemetry_max_age_s": 300.0},
            "severe_queue_depth": 4,
        },
        policy="JOINT-ROUTING-PER-ENDPOINT", block_id="fixture",
        window_index=1, enabled=True,
    )


class D1ControlTests(unittest.TestCase):
    def test_ttft_violation_recovers_only_selected_p_endpoint(self):
        values = controllers()
        values["P0"].state.candidate_frequency = 1890
        values["P0"].set_current_frequency(1890)
        recovered = []
        monitor = runtime(values, recovered)
        monitor.process_safety_event({
            "event": "TTFT_OBSERVED", "ttft_ms": 650.0,
            "selected_prefill_endpoint_id": "P0",
            "selected_decode_endpoint_id": "D1",
            "logical_request_id": "r1", "timestamp_unix_s": 1.0,
        })
        self.assertEqual(recovered, [("P0", 2520)])
        self.assertEqual(monitor.event_rows[0]["endpoint_id"], "P0")

    def test_tpot_violation_recovers_only_selected_d_endpoint(self):
        values = controllers()
        values["D1"].state.candidate_frequency = 1125
        values["D1"].set_current_frequency(1125)
        recovered = []
        monitor = runtime(values, recovered)
        monitor.process_safety_event({
            "event": "REQUEST_COMPLETED", "tpot_ms": 70.0,
            "selected_prefill_endpoint_id": "P0",
            "selected_decode_endpoint_id": "D1",
            "logical_request_id": "r2", "timestamp_unix_s": 2.0,
        })
        self.assertEqual(recovered, [("D1", 1500)])

    def test_previous_validated_safe_anchor_not_unconditional_high(self):
        values = controllers()
        values["P1"].state.known_safe_frequency = 1890
        values["P1"].state.last_validated_safe_index = 1
        values["P1"].state.safe_energy_by_frequency_j[1890] = 10.0
        values["P1"].state.candidate_frequency = 1260
        values["P1"].set_current_frequency(1260)
        recovered = []
        monitor = runtime(values, recovered)
        monitor.process_safety_event({
            "event": "TTFT_OBSERVED", "ttft_ms": 601.0,
            "selected_prefill_endpoint_id": "P1",
            "logical_request_id": "r3", "timestamp_unix_s": 3.0,
        })
        self.assertEqual(recovered, [("P1", 1890)])

    def test_p0_p1_and_d0_d1_hold_independent_clocks(self):
        values = controllers()
        values["P0"].set_current_frequency(1890)
        values["P1"].set_current_frequency(2520)
        values["D0"].set_current_frequency(1125)
        values["D1"].set_current_frequency(1500)
        self.assertEqual(
            [values[key].current_frequency_mhz for key in ("P0", "P1", "D0", "D1")],
            [1890, 2520, 1125, 1500],
        )

    def test_exploration_guard_never_allows_two_same_phase_probes(self):
        values = controllers()
        permissions = exploration_permissions(values, "P")
        self.assertEqual(sum(permissions.values()), 1)
        chosen = next(key for key, allowed in permissions.items() if allowed)
        values[chosen].state.candidate_frequency = 1890
        values[chosen].set_current_frequency(1890)
        self.assertEqual(
            exploration_permissions(values, "P"), {"P0": False, "P1": False}
        )

    def test_readback_or_stale_failure_is_scoped_to_affected_endpoint(self):
        values = controllers()
        values["D0"].state.candidate_frequency = 1125
        values["D0"].set_current_frequency(1125)
        recovered = []
        monitor = runtime(values, recovered)
        monitor.measurement_started_monotonic_s = 1.0
        monitor.process_internal_failure("D0", "STALE_SAFETY_TELEMETRY", 4.0)
        self.assertEqual(recovered, [("D0", 1500)])
        self.assertEqual(monitor.event_rows[0]["trigger"], "STALE_SAFETY_TELEMETRY")

        values = controllers()
        values["P0"].state.candidate_frequency = 1890
        values["P0"].set_current_frequency(1890)
        recovered = []
        monitor = runtime(values, recovered)
        monitor.measurement_started_monotonic_s = 1.0
        monitor.process_internal_failure("P0", "ACTUATOR_READBACK_FAILURE", 5.0)
        self.assertEqual(recovered, [("P0", 2520)])
        self.assertEqual(monitor.event_rows[0]["endpoint_id"], "P0")

    def test_single_outlier_does_not_revoke_validated_point_with_safe_attainment(self):
        values = controllers()
        d0 = values["D0"]
        d0.set_current_frequency(1125)
        d0.state.known_safe_frequency = 1125
        d0.state.validated_safe_frequencies.add(1125)
        d0.state.recent_evidence_frequency = 1125
        d0.state.recent_valid_request_count = 100
        d0.state.recent_phase_pass_count = 100
        d0.state.recent_phase_token_coverage = 12800
        d0.state.recent_evidence_captured_at_s = 1.0
        recovered = []
        monitor = runtime(values, recovered)
        monitor.process_safety_event({
            "event": "REQUEST_COMPLETED", "ok": True,
            "tpot_ms": 65.1, "actual_output_len": 128,
            "selected_decode_endpoint_id": "D0",
            "logical_request_id": "isolated", "timestamp_unix_s": 2.0,
        })
        self.assertEqual(recovered, [])
        self.assertTrue(d0.is_frequency_validated_safe(1125))
        self.assertEqual(
            monitor.event_rows[0]["action"], "LOG_ONLY_NO_ACTIVE_CANDIDATE"
        )

    def test_sufficient_recent_evidence_revokes_validated_point_and_recovers(self):
        values = controllers()
        d0 = values["D0"]
        d0.set_current_frequency(1125)
        d0.state.known_safe_frequency = 1125
        d0.state.validated_safe_frequencies.add(1125)
        d0.state.safe_energy_by_frequency_j[1125] = 42.0
        d0.state.safe_energy_observations_by_frequency[1125] = 1
        d0.state.recent_evidence_frequency = 1125
        d0.state.recent_valid_request_count = 10
        d0.state.recent_phase_pass_count = 10
        d0.state.recent_phase_token_coverage = 1280
        d0.state.recent_evidence_captured_at_s = 1.0
        recovered = []
        monitor = runtime(values, recovered)
        monitor.process_safety_event({
            "event": "REQUEST_COMPLETED", "ok": True,
            "tpot_ms": 65.1, "actual_output_len": 128,
            "selected_decode_endpoint_id": "D0",
            "logical_request_id": "sufficient", "timestamp_unix_s": 2.0,
        })
        self.assertEqual(recovered, [("D0", 1500)])
        self.assertFalse(d0.is_frequency_validated_safe(1125))
        self.assertEqual(d0.state.known_safe_frequency, 1500)
        self.assertNotIn(1125, d0.state.safe_energy_by_frequency_j)
        self.assertEqual(
            d0.state.revoked_safe_energy_history[-1]["energy_j"], 42.0
        )
        self.assertEqual(
            monitor.event_rows[-1]["trigger"],
            "ENDPOINT_EVIDENCE_SLO_RISK",
        )


if __name__ == "__main__":
    unittest.main()
