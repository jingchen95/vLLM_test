"""CPU-only tests for Phase 4B.1 context-aware safe route probing."""

import unittest
import json
from pathlib import Path
import tempfile
import time

from xpyd.phase3d_control import EndpointClockCapability, GPUReadback
from xpyd.phase4b1_evaluation import Phase4B1Harness, load_config
from xpyd.phase4b_evaluation import Phase4BEvaluationHarness
from xpyd.route_probing import ContextRouteCostStore, SafeRouteProber
from xpyd.telemetry import EndpointTelemetrySample


ROUTES = (("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"))


class ContextRouteCostTests(unittest.TestCase):
    def test_phase4b1_config_is_small_and_context_complete(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json"
        )
        settings = config["phase4b1"]
        self.assertEqual(settings["policies"], ["PASSIVE_FULL", "ACTIVE_FULL"])
        self.assertEqual(settings["smoke_contexts"], ["small_light"])
        self.assertEqual(set(settings["contexts"]), {
            "small_light", "prefill_heavy", "decode_heavy", "both_heavy",
        })
        self.assertEqual(settings["smoke_windows_per_context"], 7)
        self.assertLess(
            settings["route_cost_maximum_age_s"],
            config["phase3d"]["feedback"]["telemetry_max_age_s"],
        )
        self.assertEqual(settings["route_cost_maximum_age_windows"], 4)
        self.assertEqual(settings["minimum_probe_interval_windows"], 2)

    def test_launcher_has_separate_phase4b1_entry(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_disagg_benchmark.sh").read_text()
        self.assertIn('XPYD_PHASE4B1_CONFIG="${XPYD_PHASE4B1_CONFIG:-}"', source)
        self.assertIn("-m xpyd.phase4b1_evaluation", source)

    def test_system_window_cost_and_context_ewma_are_isolated(self):
        costs = ContextRouteCostStore(alpha=0.5)
        first = costs.observe(
            "small", ROUTES[0], total_system_gross_energy_j=200.0,
            logical_requests=2, timestamp_s=10.0, sequence=1,
        )
        self.assertEqual(first.ewma_system_joules_per_request, 100.0)
        second = costs.observe(
            "small", ROUTES[0], total_system_gross_energy_j=400.0,
            logical_requests=2, timestamp_s=20.0, sequence=2,
        )
        self.assertEqual(second.ewma_system_joules_per_request, 150.0)
        other = costs.observe(
            "decode", ROUTES[0], total_system_gross_energy_j=800.0,
            logical_requests=2, timestamp_s=20.0, sequence=2,
        )
        self.assertEqual(other.ewma_system_joules_per_request, 400.0)
        self.assertEqual(
            costs.snapshot("small", ROUTES[0], 20.0, 2).sample_count, 2
        )

    def test_unseen_probes_balance_endpoints_then_exploit(self):
        costs = ContextRouteCostStore(alpha=1.0, maximum_age_windows=4)
        prober = SafeRouteProber(costs, minimum_probe_interval_windows=2)
        observed = []
        for sequence in range(1, 5):
            decision = prober.choose(
                "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
                probe_safe_routes=ROUTES, now_s=float(sequence), sequence=sequence,
            )
            self.assertTrue(decision.probe)
            self.assertTrue(decision.freeze_dvfs)
            observed.append(decision.route)
            costs.observe(
                "small", decision.route,
                total_system_gross_energy_j=200.0 + sequence,
                logical_requests=2, timestamp_s=float(sequence), sequence=sequence,
            )
        self.assertEqual(observed, [ROUTES[0], ROUTES[3], ROUTES[1], ROUTES[2]])
        decision = prober.choose(
            "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
            probe_safe_routes=ROUTES, now_s=5.0, sequence=5,
        )
        self.assertFalse(decision.probe)
        self.assertEqual(decision.mode, "EXPLOIT_RECENT")

    def test_stale_probe_waits_for_interval_after_unseen_coverage(self):
        costs = ContextRouteCostStore(
            alpha=1.0, maximum_age_s=2.0, maximum_age_windows=4,
        )
        prober = SafeRouteProber(costs, minimum_probe_interval_windows=2)
        for sequence in range(1, 5):
            decision = prober.choose(
                "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
                probe_safe_routes=ROUTES, now_s=float(sequence), sequence=sequence,
            )
            self.assertEqual(decision.mode, "PROBE_UNSEEN")
            costs.observe(
                "small", decision.route, total_system_gross_energy_j=200.0,
                logical_requests=2, timestamp_s=float(sequence), sequence=sequence,
            )
        decision = prober.choose(
            "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
            probe_safe_routes=ROUTES, now_s=5.0, sequence=5,
        )
        self.assertEqual(decision.mode, "EXPLOIT_RECENT")
        self.assertFalse(decision.probe)
        decision = prober.choose(
            "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
            probe_safe_routes=ROUTES, now_s=6.0, sequence=6,
        )
        self.assertEqual(decision.mode, "PROBE_STALE")
        self.assertTrue(decision.probe)

    def test_stale_route_is_refreshed_after_exploitation(self):
        costs = ContextRouteCostStore(alpha=1.0, maximum_age_windows=4)
        prober = SafeRouteProber(costs)
        for sequence, route in enumerate(ROUTES, 1):
            costs.observe(
                "small", route, total_system_gross_energy_j=200.0 + sequence,
                logical_requests=2, timestamp_s=float(sequence), sequence=sequence,
            )
        # Keep the cheapest first route fresh through two exploit windows.
        costs.observe(
            "small", ROUTES[0], total_system_gross_energy_j=100.0,
            logical_requests=2, timestamp_s=5.0, sequence=5,
        )
        costs.observe(
            "small", ROUTES[0], total_system_gross_energy_j=100.0,
            logical_requests=2, timestamp_s=6.0, sequence=6,
        )
        decision = prober.choose(
            "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
            probe_safe_routes=ROUTES, now_s=7.0, sequence=7,
        )
        self.assertTrue(decision.probe)
        self.assertEqual(decision.mode, "PROBE_STALE")
        self.assertEqual(decision.route, ROUTES[1])

    def test_pressure_suppresses_probe_and_uses_recent_safe_route(self):
        costs = ContextRouteCostStore(alpha=1.0)
        costs.observe(
            "small", ROUTES[0], total_system_gross_energy_j=200.0,
            logical_requests=2, timestamp_s=1.0, sequence=1,
        )
        prober = SafeRouteProber(costs)
        decision = prober.choose(
            "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
            probe_safe_routes=(), now_s=2.0, sequence=2,
        )
        self.assertFalse(decision.probe)
        self.assertFalse(decision.freeze_dvfs)
        self.assertEqual(decision.mode, "EXPLOIT_RECENT")
        self.assertEqual(decision.route, ROUTES[0])

    def test_pressure_with_no_cost_fails_to_explicit_safe_route(self):
        prober = SafeRouteProber(ContextRouteCostStore())
        decision = prober.choose(
            "small", compatible_routes=ROUTES, eligible_routes=ROUTES,
            probe_safe_routes=(), now_s=1.0, sequence=1,
        )
        self.assertFalse(decision.probe)
        self.assertEqual(decision.mode, "SAFE_FALLBACK_NO_PROBE")
        self.assertEqual(decision.route, ("P0", "D0"))

    def test_cpu_only_harness_exercises_full_probe_audit(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json"
        )
        levels = {
            "P0": (1260, 1890, 2520), "P1": (1260, 1890, 2520),
            "D0": (750, 1125, 1500), "D1": (750, 1125, 1500),
        }
        capabilities = {
            endpoint_id: EndpointClockCapability(
                endpoint_id=endpoint_id,
                node="neptune" if endpoint_id.startswith("P") else "io",
                gpu_id=int(endpoint_id[-1]), gpu_name="fixture",
                gpu_uuid="uuid-" + endpoint_id,
                pci_bus_id="0000:0%s:00.0" % endpoint_id[-1],
                supported_graphics_mhz=values,
                selected_low_mhz=values[0], selected_mid_mhz=values[1],
                selected_high_mhz=values[2],
            )
            for endpoint_id, values in levels.items()
        }

        class Backend:
            def __init__(self):
                self.frequencies = {
                    endpoint_id: values[-1] for endpoint_id, values in levels.items()
                }

            def discover(self, endpoint_id, node, gpu_id, maximum_mhz):
                return capabilities[endpoint_id]

            def read(self, endpoint_id, node, gpu_id, expected_mhz=None):
                capability = capabilities[endpoint_id]
                return GPUReadback(
                    endpoint_id, node, gpu_id, capability.gpu_name,
                    capability.gpu_uuid, capability.pci_bus_id,
                    self.frequencies[endpoint_id], 1.0,
                )

            def set_graphics_clock(self, capability, target_mhz):
                self.frequencies[capability.endpoint_id] = target_mhz
                return self.read(
                    capability.endpoint_id, capability.node, capability.gpu_id
                )

        class Harness(Phase4B1Harness):
            def _reference(self):
                return {
                    "path": "fixture", "sha256": "fixture",
                    "summary": {"valid": True},
                    "oracles": {
                        item: {"J_per_request": 90.0}
                        for item in ("small_light", "prefill_heavy", "decode_heavy", "both_heavy")
                    },
                    "near_signatures": {},
                }

            def _accepted_frequencies(self, reference):
                return {
                    endpoint_id: {"LOW": values[0], "MID": values[1], "HIGH": values[2]}
                    for endpoint_id, values in levels.items()
                }

            def _discover_exact(self, accepted):
                return capabilities

            def _write_routes(self, pairs, reason):
                self.last_pairs = [list(item) for item in pairs]

            def _run_window(
                self, window_id, workload, count, requested, pairs,
                windows_root, *, require_all_pairs=False,
            ):
                directory = windows_root / window_id
                directory.mkdir(parents=True)
                (directory / "audit.json").write_text(
                    json.dumps({"valid": True}), encoding="utf-8"
                )
                return directory

            def _measurement_row(
                self, window_id, policy, workload, repeat, context,
                accepted, reference, window_dir,
            ):
                route = context.selected_pairs[0]
                route_energy = {
                    "P0->D0": 200.0, "P0->D1": 220.0,
                    "P1->D0": 210.0, "P1->D1": 180.0,
                }["%s->%s" % tuple(route)]
                now = time.time()
                gates = {
                    "logical_request_id": True,
                    "requested_output_tokens": True,
                    "endpoint_assignment": True,
                    "explicit_compatibility": True,
                    "nvml_energy_windows": True,
                    "fixed_clocks": True,
                    "no_invalidating_thermal_or_hw_slowdown": True,
                }
                row = {
                    "run_id": self.run_id, "window_id": window_id,
                    "policy": policy, "workload": workload["id"],
                    "repeat": repeat, "timing_start_unix_s": now - 1.0,
                    "timing_end_unix_s": now, "duration_s": 1.0,
                    "completed_requests": 2, "output_tokens": 256,
                    "measurement_valid": True, "slo_pass": True,
                    "total_gpu_gross_energy_j": route_energy,
                    "joules_per_request": route_energy / 2.0,
                    "mean_ttft_ms": 100.0, "mean_tpot_ms": 30.0,
                    "selected_pairs_json": json.dumps(context.selected_pairs),
                    "hard_gates_json": json.dumps(gates),
                }
                requests = [{
                    "window_id": window_id, "policy": policy,
                    "workload": workload["id"], "repeat": repeat,
                    "request_id": "%s-%d" % (window_id, index),
                    "prefill_endpoint_id": route[0],
                    "decode_endpoint_id": route[1],
                    "ttft_ms": 100.0, "tpot_ms": 30.0,
                    "mean_itl_ms": 30.0, "e2e_latency_ms": 4000.0,
                } for index in range(2)]
                return row, requests

            def _observe_window(self, window_id, window_dir, registry, telemetry):
                selected = {endpoint for pair in self.last_pairs for endpoint in pair}
                observed_at = time.time()
                for endpoint_id in capabilities:
                    state = registry.get_state(endpoint_id)
                    state.queue_depth = 0
                    state.queue_depth_observed = True
                    state.kv_cache_usage_frac = 0.1
                    state.kv_cache_usage_observed = True
                    state.healthy = True
                    state.last_update_s = observed_at
                    registry.update_state(state)
                    role = registry.get_spec(endpoint_id).role
                    telemetry.observe(EndpointTelemetrySample(
                        endpoint_id, observed_at,
                        energy_j=10.0, completed_requests=1 if endpoint_id in selected else 0,
                        interval_s=1.0, queue_depth=0, kv_cache_usage_frac=0.1,
                        ttft_ms=100.0 if role == "prefill" and endpoint_id in selected else None,
                        tpot_ms=30.0 if role == "decode" and endpoint_id in selected else None,
                    ))

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            accepted_audit = temporary / "stationary.json"
            accepted_audit.write_text(
                json.dumps({"valid": True, "smoke": False}), encoding="utf-8"
            )
            config["phase4b1"]["output_root"] = str(temporary / "out")
            config["phase4b1"]["accepted_stationary_audit"] = str(accepted_audit)
            config["routing_control_file"] = str(temporary / "routes.json")
            result = Harness(config, run_id="fixture", backend=Backend()).run()
            summary = json.loads((result / "phase4b1_summary.json").read_text())
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["decision_gate"], "READY_FOR_DYNAMIC_EVALUATION")
            self.assertEqual(len(summary["active_observed_routes"]), 4)
            self.assertGreater(summary["active_probe_window_count"], 4)
            self.assertGreater(summary["active_exploit_window_count"], 0)

            active_smoke_audit = temporary / "active-smoke.json"
            active_smoke_audit.write_text(
                json.dumps({"valid": True, "smoke": True}), encoding="utf-8"
            )
            dynamic_root = temporary / "dynamic"
            config["phase4b"]["dynamic"]["output_root"] = str(dynamic_root)
            config["phase4b"]["dynamic"]["accepted_stationary_audit"] = str(
                accepted_audit
            )
            config["phase4b"]["dynamic"]["accepted_active_smoke_audit"] = str(
                active_smoke_audit
            )
            dynamic = Harness(config, run_id="dynamic-fixture", backend=Backend())
            dynamic.stage = "dynamic"
            dynamic.smoke = False
            dynamic.run_dir = dynamic_root / "dynamic-fixture"
            result = Phase4BEvaluationHarness.run_dynamic(dynamic)
            dynamic_summary = json.loads(
                (result / "phase4b_summary.json").read_text()
            )
            self.assertTrue(dynamic_summary["valid"])
            self.assertEqual(dynamic_summary["dynamic_window_count"], 162)
            self.assertEqual(
                {item["policy"] for item in dynamic_summary["policy_aggregates"]},
                {"STATIC", "PASSIVE_FULL", "ACTIVE_FULL"},
            )
            self.assertTrue(
                dynamic_summary["hard_gates"][
                    "active_context_route_system_cost_formula"
                ]
            )


if __name__ == "__main__":
    unittest.main()
