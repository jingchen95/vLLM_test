"""CPU-only tests for Stage D1 request-level pair routing."""

import unittest

from xpyd.d1_routing import D1PairRouter, D1RoutingError, RequestRoutingContext


PAIRS = (("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"))


def endpoint(status="VALIDATED_SAFE", energy=10.0, fresh=True, healthy=True):
    return {
        "healthy": healthy,
        "telemetry_fresh": fresh,
        "safety_status": status,
        "requested_frequency_mhz": 1500,
        "observed_frequency_mhz": 1500,
        "validated_safe_frequency_mhz": 1500,
        "candidate_frequency_mhz": None,
        "recent_latency_ms": 50.0,
        "recent_energy_j_per_assigned_request": energy,
    }


def state(mode="RUNTIME_FEEDBACK"):
    return {
        "updated_unix_s": 100.0,
        "generation": "block-1",
        "routing_mode": mode,
        "endpoints": {
            "P0": endpoint(energy=4.0),
            "P1": endpoint(energy=8.0),
            "D0": endpoint(energy=3.0),
            "D1": endpoint(energy=6.0),
        },
        "class_pair_feedback": {},
    }


def context(name="SS", request_id="r1"):
    return RequestRoutingContext(request_id, 128, 128, name)


class D1RoutingTests(unittest.TestCase):
    def router(self, value, **kwargs):
        return D1PairRouter(
            PAIRS, lambda: value, wall_clock=lambda: 100.0,
            maximum_age_s=10.0, **kwargs,
        )

    def test_cold_start_is_safe_exact_pair_round_robin(self):
        router = self.router(state(), cold_start_requests=4)
        chosen = [router.choose(context(request_id=f"r{i}")).pair for i in range(4)]
        self.assertEqual(chosen, list(PAIRS))

    def test_fixed_rr_comparator_remains_exact_rr(self):
        router = self.router(state("FIXED_RR"), cold_start_requests=0)
        chosen = [router.choose(context(request_id=f"r{i}")).pair for i in range(8)]
        self.assertEqual(chosen, list(PAIRS) * 2)
        self.assertTrue(all(
            router.choose(context(request_id=f"x{i}")).reason == "FIXED_RR"
            for i in range(4)
        ))

    def test_request_class_is_consumed_before_pair_selection(self):
        value = state()
        value["class_pair_feedback"] = {
            "SS": {"P0->D0": {"slo_safe": False, "observations": 3}},
            "LL": {"P1->D1": {"slo_safe": False, "observations": 3}},
        }
        ss = self.router(value, cold_start_requests=0).choose(context("SS"))
        ll = self.router(value, cold_start_requests=0).choose(context("LL"))
        self.assertNotEqual(ss.pair, ("P0", "D0"))
        self.assertNotEqual(ll.pair, ("P1", "D1"))
        self.assertEqual(ss.context.request_class, "SS")

    def test_unknown_or_stale_endpoint_is_not_energy_preferred(self):
        value = state()
        value["endpoints"]["P0"] = endpoint(
            status="UNKNOWN", energy=0.1, fresh=False
        )
        decision = self.router(value, cold_start_requests=0).choose(context())
        self.assertEqual(decision.pair[0], "P1")
        self.assertNotEqual(decision.reason, "ENERGY_PREFERENCE" if decision.pair[0] == "P0" else "")

    def test_rising_pressure_spills_to_another_safe_endpoint(self):
        load = {
            "active_requests_by_endpoint": {"P0": 2, "D0": 2},
            "active_input_tokens_by_endpoint": {"P0": 4096},
            "active_output_tokens_by_endpoint": {"D0": 1024},
        }
        decision = self.router(state(), cold_start_requests=0).choose(context(), load)
        self.assertEqual(decision.pair, ("P1", "D1"))
        self.assertEqual(decision.reason, "LOWER_PRESSURE")

    def test_pair_aware_pressure_uses_full_pair_not_independent_choices(self):
        load = {
            "active_requests_by_endpoint": {"P0": 1, "P1": 0, "D0": 1, "D1": 0}
        }
        decision = self.router(state(), cold_start_requests=0).choose(context(), load)
        self.assertEqual(decision.pair, ("P1", "D1"))
        selected = decision.selected_state
        self.assertEqual(selected["pressure"], 0)
        self.assertEqual(selected["pair"], ["P1", "D1"])

    def test_candidate_is_only_used_as_controlled_exploration(self):
        value = state()
        value["endpoints"]["P0"] = endpoint(
            status="CANDIDATE_PROBE", energy=0.1
        )
        router = self.router(
            value, cold_start_requests=0, exploration_interval_requests=2
        )
        first = router.choose(context(request_id="r0"))
        second = router.choose(context(request_id="r1"))
        self.assertEqual(first.reason, "EXPLORATION")
        self.assertEqual(first.pair[0], "P0")
        self.assertEqual(second.pair[0], "P1")

    def test_no_known_safe_pair_fails_closed(self):
        value = state()
        for key in value["endpoints"]:
            value["endpoints"][key] = endpoint(status="UNKNOWN")
        with self.assertRaisesRegex(D1RoutingError, "no known-safe pair"):
            self.router(value, cold_start_requests=0).choose(context())


if __name__ == "__main__":
    unittest.main()
