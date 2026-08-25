"""CPU-only Phase 4B policy-isolation, oracle, and reporting tests."""

import json
from pathlib import Path
import tempfile
import time
import unittest

from xpyd.phase4b_evaluation import (
    POLICIES,
    Phase4BEvaluationHarness,
    classify_oracle_gap,
    counterbalanced_blocks,
    exact_near_optimal_match,
    load_config,
    load_phase4a_reference,
    relative_energy_savings,
    summarize_dynamic_adaptation,
    summarize_stationary,
)
from xpyd.phase3d_control import (
    EndpointClockCapability,
    GPUReadback,
    PerEndpointClockActuator,
)
from xpyd.telemetry import EndpointTelemetrySample


WORKLOADS = (
    {"id": "small_light"},
    {"id": "prefill_heavy"},
    {"id": "decode_heavy"},
    {"id": "both_heavy"},
)


def reference():
    return {
        "oracles": {
            item["id"]: {"workload": item["id"], "J_per_request": 100.0}
            for item in WORKLOADS
        },
        "near_signatures": {
            item["id"]: [{
                "config_id": "p0_d0_ll",
                "route": ["P0", "D0"],
                "frequencies": {"P0": 1260, "P1": 1260, "D0": 750, "D1": 750},
            }]
            for item in WORKLOADS
        },
    }


class Phase4BPolicyTests(unittest.TestCase):
    def test_relative_energy_savings_rejects_missing_or_zero_baseline(self):
        self.assertIsNone(relative_energy_savings(None, 90.0))
        self.assertIsNone(relative_energy_savings(0.0, 0.0))
        self.assertIsNone(relative_energy_savings(-1.0, 0.0))
        self.assertAlmostEqual(relative_energy_savings(100.0, 90.0), 0.1)

    def test_checked_in_config_fixes_neptune_io_and_four_policies(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json"
        )
        self.assertEqual(tuple(config["phase4b"]["policies"]), POLICIES)
        self.assertEqual(len(config["workloads"]), 4)
        self.assertEqual(
            tuple(config["phase4b"]["dynamic"]["policies"]),
            ("STATIC", "PASSIVE_FULL", "ACTIVE_FULL"),
        )
        self.assertEqual(config["phase4b"]["dynamic"]["repeats"], 3)
        self.assertIn(
            "accepted_active_smoke_audit", config["phase4b"]["dynamic"]
        )
        self.assertEqual(len(config["phase4b"]["dynamic"]["traces"]), 2)
        endpoints = {item["endpoint_id"]: item["node"] for item in config["endpoints"]}
        self.assertEqual(endpoints, {
            "P0": "neptune", "P1": "neptune", "D0": "io", "D1": "io",
        })

    def test_seeded_blocks_cover_each_policy_workload_once(self):
        first = counterbalanced_blocks(POLICIES, WORKLOADS, 41)
        second = counterbalanced_blocks(POLICIES, WORKLOADS, 41)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(
            {(policy, workload["id"]) for policy, workload in first},
            {(policy, workload["id"]) for policy in POLICIES for workload in WORKLOADS},
        )

    def test_oracle_gap_boundaries_include_below_oracle_noise(self):
        self.assertEqual(classify_oracle_gap(-0.02), "<=5%")
        self.assertEqual(classify_oracle_gap(0.05), "<=5%")
        self.assertEqual(classify_oracle_gap(0.10), "5-10%")
        self.assertEqual(classify_oracle_gap(0.10001), ">10%")

    def test_near_optimal_mapping_requires_one_exact_runtime_state(self):
        ref = reference()
        frequencies = {"P0": 1260, "P1": 1260, "D0": 750, "D1": 750}
        self.assertTrue(exact_near_optimal_match(
            "small_light", [["P0", "D0"]], frequencies, ref
        ))
        self.assertFalse(exact_near_optimal_match(
            "small_light", [["P1", "D0"]], frequencies, ref
        ))
        self.assertIsNone(exact_near_optimal_match(
            "small_light", [["P0", "D0"], ["P1", "D1"]], frequencies, ref
        ))

    def test_loader_freezes_sha_and_rejects_model_use(self):
        summary = {
            "valid": True,
            "ready_for_phase4b": True,
            "models_trained_or_used": [],
            "oracles": [
                {"workload": item["id"], "J_per_request": 100.0}
                for item in WORKLOADS
            ],
            "configuration_aggregates": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase4a_summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            loaded = load_phase4a_reference(path)
            self.assertEqual(len(loaded["sha256"]), 64)
            summary["models_trained_or_used"] = ["forbidden"]
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "model use"):
                load_phase4a_reference(path)

    def test_stationary_summary_reports_variance_slo_and_oracle_gap(self):
        rows = []
        requests = []
        controls = []
        for policy in POLICIES:
            for workload in WORKLOADS:
                for repeat, energy in enumerate((110.0, 120.0, 130.0), 1):
                    rows.append({
                        "policy": policy, "workload": workload["id"],
                        "measurement_valid": True, "duration_s": 10.0,
                        "total_gpu_gross_energy_j": energy,
                        "joules_per_request": energy,
                        "joules_per_output_token": energy / 128.0,
                        "mean_ttft_ms": 500.0, "mean_tpot_ms": 60.0,
                        "mean_itl_ms": 60.0, "mean_e2e_latency_ms": 8000.0,
                        "throughput_requests_s": 0.2,
                        "phase4a_near_optimal_match": True,
                        "P0_requested_level": "LOW", "P1_requested_level": "LOW",
                        "D0_requested_level": "LOW", "D1_requested_level": "LOW",
                    })
                    requests.append({
                        "policy": policy, "workload": workload["id"],
                        "prefill_endpoint_id": "P0", "decode_endpoint_id": "D0",
                        "ttft_ms": 500.0, "tpot_ms": 60.0,
                    })
                    controls.append({
                        "policy": policy, "workload": workload["id"],
                        "measured_iteration": True, "dvfs_action_count": 1,
                        "routing_changed": False, "fallback_reason": None,
                        "telemetry_fresh": True,
                    })
        summary = summarize_stationary(
            rows, requests, controls, WORKLOADS, POLICIES, reference(),
            3, 1000.0, 80.0,
        )
        self.assertEqual(len(summary), 16)
        first = summary[0]
        self.assertTrue(first["complete"])
        self.assertAlmostEqual(first["joules_per_request_mean"], 120.0)
        self.assertAlmostEqual(first["joules_per_request_std"], 10.0)
        self.assertAlmostEqual(first["oracle_gap"], 0.20)
        self.assertEqual(first["oracle_gap_class"], ">10%")
        self.assertTrue(first["slo_pass"])
        self.assertEqual(first["dvfs_action_count"], 3)

    def test_launcher_has_phase4b_entrypoint_and_oracle_environment(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn('XPYD_PHASE4B_CONFIG="${XPYD_PHASE4B_CONFIG:-}"', source)
        self.assertIn('args=(-m xpyd.phase4b_evaluation --config "${XPYD_PHASE4B_CONFIG}")', source)
        self.assertIn("export XPYD_PHASE4B_ORACLE_SUMMARY", source)
        self.assertIn('args+=(--stage "${XPYD_PHASE4B_STAGE}")', source)
        self.assertIn("export XPYD_PHASE4B_ACCEPTED_STATIONARY_AUDIT", source)
        self.assertIn("export XPYD_PHASE4B_ACCEPTED_ACTIVE_SMOKE_AUDIT", source)

    def test_dynamic_reaction_requires_new_state_observation(self):
        signatures = ("A", "B")
        rows = []
        controls = []
        requests = []
        sequence = 0
        for state_index, state_signatures in enumerate((
            ("A", "A", "A"),
            ("A", "B", "B"),
            ("B", "B", "B"),
        )):
            for window_in_state, signature in enumerate(state_signatures, 1):
                sequence += 1
                control_id = "c%d" % sequence
                rows.append({
                    "trace_id": "trace", "policy": "FULL_FEEDBACK",
                    "sequence_index": sequence, "state_index": state_index,
                    "window_in_state": window_in_state,
                    "transition_timestamp_unix_s": state_index * 10.0,
                    "control_timestamp_unix_s": state_index * 10.0 + window_in_state * 2.5,
                    "control_state_signature": signature,
                    "control_iteration_id": control_id,
                    "total_gpu_gross_energy_j": 100.0,
                })
                controls.append({
                    "control_iteration_id": control_id,
                    "dvfs_action_count": int(signature == signatures[1]),
                    "routing_changed": False, "fallback_reason": None,
                    "telemetry_fresh": True,
                })
                requests.append({
                    "trace_id": "trace", "policy": "FULL_FEEDBACK",
                    "state_index": state_index,
                    "ttft_ms": 500.0, "tpot_ms": 60.0,
                })
        summary = summarize_dynamic_adaptation(
            rows, controls, requests,
            [{"id": "trace", "states": ["light", "heavy", "light"]}],
            ["FULL_FEEDBACK"], 1000.0, 80.0,
        )
        self.assertEqual(len(summary), 2)
        first = summary[0]
        self.assertTrue(first["reaction_observed"])
        self.assertAlmostEqual(first["time_to_first_reaction_s"], 5.0)
        self.assertAlmostEqual(first["settling_time_s"], 5.0)
        second = summary[1]
        self.assertFalse(second["reaction_observed"])
        self.assertEqual(second["settling_time_s"], 0.0)

    def test_dynamic_stationary_audit_path_uses_environment_expansion(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "paper/scripts/xpyd/phase4b_evaluation.py").read_text()
        self.assertIn(
            'Path(os.path.expandvars(str(dynamic["accepted_stationary_audit"])))',
            source,
        )

    def test_missing_feedback_fails_closed_without_oracle_input(self):
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
                self.frequencies = {key: value[-1] for key, value in levels.items()}

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

        with tempfile.TemporaryDirectory() as directory:
            config["routing_control_file"] = str(Path(directory) / "route.json")
            backend = Backend()
            harness = Phase4BEvaluationHarness(config, "fixture", backend=backend)
            context = harness._new_context(capabilities)
            actuator = PerEndpointClockActuator(backend, capabilities, 0.0)
            actuator.requested.update(backend.frequencies)
            pairs = [["P0", "D0"], ["P0", "D1"], ["P1", "D0"], ["P1", "D1"]]
            selected = harness._decide(
                "FULL_FEEDBACK", "small_light", "control", context,
                capabilities, actuator, pairs, measured_iteration=True,
            )
            self.assertEqual(selected, pairs)
            self.assertEqual(
                harness.control_rows[-1]["fallback_reason"],
                "missing_or_stale_feedback_use_safe_high",
            )
            self.assertFalse(harness.control_rows[-1]["telemetry_fresh"])
            self.assertNotIn("oracle", harness.control_rows[-1])

            observed_at = time.time()
            for endpoint_id in capabilities:
                state = context.registry.get_state(endpoint_id)
                state.queue_depth = 0
                state.queue_depth_observed = True
                state.kv_cache_usage_frac = 0.1
                state.kv_cache_usage_observed = True
                context.registry.update_state(state)
            context.telemetry.observe(EndpointTelemetrySample(
                "P0", observed_at, energy_j=10.0, completed_requests=1,
                interval_s=1.0, queue_depth=0, kv_cache_usage_frac=0.1,
                ttft_ms=100.0,
            ))
            context.telemetry.observe(EndpointTelemetrySample(
                "D0", observed_at, energy_j=20.0, completed_requests=1,
                interval_s=1.0, queue_depth=0, kv_cache_usage_frac=0.1,
                tpot_ms=30.0,
            ))
            selected = harness._decide(
                "FULL_FEEDBACK", "small_light", "partial", context,
                capabilities, actuator, pairs, measured_iteration=True,
            )
            self.assertEqual(selected, [["P0", "D0"]])
            self.assertIsNone(harness.control_rows[-1]["fallback_reason"])
            self.assertFalse(harness.control_rows[-1]["telemetry_fresh"])
            self.assertEqual(
                json.loads(harness.control_rows[-1]["stale_endpoint_ids_json"]),
                ["D1", "P1"],
            )

            context.telemetry.observe(EndpointTelemetrySample(
                "P1", observed_at, energy_j=11.0, completed_requests=1,
                interval_s=1.0, queue_depth=0, kv_cache_usage_frac=0.1,
                ttft_ms=110.0,
            ))
            context.telemetry.observe(EndpointTelemetrySample(
                "D1", observed_at, energy_j=21.0, completed_requests=1,
                interval_s=1.0, queue_depth=0, kv_cache_usage_frac=0.1,
                tpot_ms=31.0,
            ))
            before = dict(backend.frequencies)
            selected = harness._decide(
                "ACTIVE_FULL", "small_light", "probe", context,
                capabilities, actuator, pairs, measured_iteration=True,
                route_override=["P1", "D1"], freeze_dvfs=True,
                routing_decision_mode="PROBE_UNSEEN",
                routing_score_basis="context_route_total_four_gpu_window_joules_per_request",
            )
            self.assertEqual(selected, [["P1", "D1"]])
            self.assertEqual(backend.frequencies, before)
            self.assertTrue(harness.control_rows[-1]["route_probe"])
            self.assertTrue(harness.control_rows[-1]["dvfs_frozen_for_probe"])
            self.assertEqual(harness.control_rows[-1]["dvfs_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
