"""CPU-only tests for Phase 4C stress design and reporting logic."""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from xpyd.phase3d_control import EndpointClockCapability, GPUReadback
from xpyd.phase4c_stress import (
    POLICIES,
    Phase4CStressHarness,
    classify_robustness,
    load_config,
    oracle_reference_status,
    paired_energy_comparisons,
)
from xpyd.telemetry import EndpointTelemetrySample


class Phase4CStressTests(unittest.TestCase):
    def test_checked_in_design_is_small_harder_and_connector_safe(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json")
        settings = config["phase4c"]
        self.assertEqual(tuple(settings["policies"]), POLICIES)
        self.assertEqual(settings["repeats"], 3)
        self.assertEqual(len(settings["traces"]), 3)
        self.assertEqual(settings["windows_per_state"], 3)
        self.assertEqual(config["client"]["max_concurrency"], 1)
        self.assertEqual(config["phase3d"]["validated_connector_max_concurrency"], 1)
        self.assertLess(settings["slo"]["ttft_ms"], config["phase4b"]["slo"]["ttft_ms"])
        self.assertLess(settings["slo"]["tpot_ms"], config["phase4b"]["slo"]["tpot_ms"])
        self.assertEqual(max(item["rate_rps"] for item in settings["stress_workloads"]), 2.0)
        self.assertEqual(
            {item["node"] for item in config["endpoints"] if item["endpoint_id"].startswith("P")},
            {"neptune"},
        )
        self.assertEqual(
            {item["node"] for item in config["endpoints"] if item["endpoint_id"].startswith("D")},
            {"io"},
        )

    def test_loader_does_not_require_irrelevant_active_smoke_audit(self):
        root = Path(__file__).resolve().parents[1]
        with mock.patch.dict("os.environ", {
            "XPYD_PHASE4B_ACCEPTED_ACTIVE_SMOKE_AUDIT": "",
        }):
            config = load_config(
                root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json"
            )
        self.assertEqual(config["phase4c"]["policies"], ["STATIC", "PASSIVE_FULL"])

    def test_oracle_is_withheld_outside_rate_or_tighter_slo_coverage(self):
        oracle = {"TTFT": 480.0, "TPOT": 59.0}
        valid, reason = oracle_reference_status(
            {"rate_rps": 0.5}, oracle, reference_rate_rps=0.5,
            ttft_slo_ms=550.0, tpot_slo_ms=65.0,
        )
        self.assertTrue(valid)
        self.assertIn("exact", reason)
        valid, reason = oracle_reference_status(
            {"rate_rps": 2.0}, oracle, reference_rate_rps=0.5,
            ttft_slo_ms=550.0, tpot_slo_ms=65.0,
        )
        self.assertFalse(valid)
        self.assertIn("outside", reason)
        valid, reason = oracle_reference_status(
            {"rate_rps": 0.5}, {"TTFT": 560.0, "TPOT": 59.0},
            reference_rate_rps=0.5, ttft_slo_ms=550.0, tpot_slo_ms=65.0,
        )
        self.assertFalse(valid)
        self.assertIn("fails", reason)

    def test_paired_energy_comparison_retains_repeat_variance(self):
        rows = []
        for repeat, static, passive in ((1, 100.0, 90.0), (2, 110.0, 99.0), (3, 90.0, 81.0)):
            rows.extend([
                {"trace_id": "stress", "repeat": repeat, "policy": "STATIC", "total_gpu_gross_energy_j": static},
                {"trace_id": "stress", "repeat": repeat, "policy": "PASSIVE_FULL", "total_gpu_gross_energy_j": passive},
            ])
        result = paired_energy_comparisons(rows, ["stress"])[0]
        self.assertEqual(result["paired_repeat_count"], 3)
        self.assertAlmostEqual(result["passive_energy_saving_vs_static_mean"], 0.1)
        self.assertAlmostEqual(result["passive_energy_saving_vs_static_std"], 0.0)

    def test_decision_gate_classifies_robust_passive(self):
        comparisons = [
            {"paired_repeat_count": 3, "passive_energy_saving_vs_static_mean": 0.08}
            for _ in range(3)
        ]
        passive = [{
            "measurement_valid": True, "request_count": 27, "slo_violation_count": 0,
            "stale_telemetry_event_count": 0, "fallback_event_count": 0,
            "window_count": 9, "oracle_reference_available": False, "oracle_gap": None,
        } for _ in range(9)]
        classification, reasons = classify_robustness(comparisons, passive)
        self.assertEqual(classification, "A. PASSIVE ROBUSTLY SUFFICIENT")
        self.assertTrue(reasons)

    def test_launcher_exposes_phase4c_without_active_policy(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn('XPYD_PHASE4C_CONFIG="${XPYD_PHASE4C_CONFIG:-}"', source)
        self.assertIn('args=(-m xpyd.phase4c_stress --config "${XPYD_PHASE4C_CONFIG}")', source)
        self.assertIn("export XPYD_PHASE4C_ACCEPTED_DYNAMIC_AUDIT", source)
        config = json.loads((root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json").read_text())
        self.assertEqual(config["phase4c"]["policies"], ["STATIC", "PASSIVE_FULL"])

    def test_full_mocked_plan_emits_required_outputs_and_gates(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json")
        levels = {
            "P0": (1260, 1890, 2520), "P1": (1260, 1890, 2520),
            "D0": (750, 1125, 1500), "D1": (750, 1125, 1500),
        }
        capabilities = {
            endpoint: EndpointClockCapability(
                endpoint_id=endpoint,
                node="neptune" if endpoint.startswith("P") else "io",
                gpu_id=int(endpoint[-1]), gpu_name="fixture", gpu_uuid="uuid-" + endpoint,
                pci_bus_id="0000:0%s:00.0" % endpoint[-1], supported_graphics_mhz=values,
                selected_low_mhz=values[0], selected_mid_mhz=values[1],
                selected_high_mhz=values[2],
            ) for endpoint, values in levels.items()
        }

        class Backend:
            def __init__(self):
                self.frequencies = {key: values[-1] for key, values in levels.items()}

            def discover(self, endpoint_id, node, gpu_id, configured_high_mhz):
                return capabilities[endpoint_id]

            def read(self, endpoint_id, node, gpu_id, expected_mhz=None):
                value = capabilities[endpoint_id]
                return GPUReadback(
                    endpoint_id, node, gpu_id, value.gpu_name, value.gpu_uuid,
                    value.pci_bus_id, self.frequencies[endpoint_id], 1.0,
                )

            def set_graphics_clock(self, capability, target_mhz):
                self.frequencies[capability.endpoint_id] = target_mhz
                return self.read(capability.endpoint_id, capability.node, capability.gpu_id)

        class Harness(Phase4CStressHarness):
            def _reference(self):
                oracles = {
                    workload: {"J_per_request": 100.0, "TTFT": 480.0, "TPOT": 59.0}
                    for workload in ("small_light", "prefill_heavy", "decode_heavy", "both_heavy")
                }
                return {
                    "path": "fixture", "sha256": "fixture", "oracles": oracles,
                    "near_signatures": {},
                    "summary": {
                        "valid": True,
                        "frequency_levels_mhz": {
                            endpoint: {"LOW": values[0], "MID": values[1], "HIGH": values[2]}
                            for endpoint, values in levels.items()
                        },
                    },
                }

            def _write_routes(self, pairs, reason):
                self.last_pairs = [list(pair) for pair in pairs]

            def _run_window(self, window_id, workload, count, requested, pairs, windows_root, *, require_all_pairs=False):
                directory = windows_root / window_id
                directory.mkdir(parents=True)
                (directory / "audit.json").write_text(json.dumps({"valid": True}))
                return directory

            def _measurement_row(self, window_id, policy, workload, repeat, context, accepted, reference, window_dir):
                now = time.time()
                energy = 270.0 if policy == "STATIC" else 240.0
                route = context.selected_pairs[0]
                gates = {
                    "logical_request_id": True, "requested_output_tokens": True,
                    "endpoint_assignment": True, "explicit_compatibility": True,
                    "nvml_energy_windows": True,
                }
                row = {
                    "run_id": self.run_id, "window_id": window_id, "policy": policy,
                    "workload": workload["id"], "input_len": workload["input_len"],
                    "output_len": workload["output_len"], "rate_rps": workload["rate_rps"],
                    "repeat": repeat, "timing_start_unix_s": now - 1.0,
                    "timing_end_unix_s": now, "duration_s": 1.0,
                    "completed_requests": 3, "output_tokens": 384,
                    "total_gpu_gross_energy_j": energy, "joules_per_request": energy / 3,
                    "joules_per_output_token": energy / 384, "mean_ttft_ms": 200.0,
                    "p99_ttft_ms": 210.0, "mean_tpot_ms": 55.0, "p99_tpot_ms": 56.0,
                    "mean_itl_ms": 55.0, "mean_e2e_latency_ms": 7200.0,
                    "throughput_requests_s": 3.0, "slo_pass": True,
                    "measurement_valid": True, "route_distribution_json": "{}",
                    "selected_pairs_json": json.dumps(context.selected_pairs),
                    "phase4a_oracle_joules_per_request": 100.0, "oracle_gap": -0.1,
                    "oracle_gap_class": "<=5%", "phase4a_near_optimal_match": False,
                    "hard_gates_json": json.dumps(gates),
                }
                for endpoint, frequencies in accepted.items():
                    target = context.target_frequencies[endpoint]
                    row[endpoint + "_requested_freq_mhz"] = target
                    row[endpoint + "_requested_level"] = next(level for level, value in frequencies.items() if value == target)
                    row[endpoint + "_observed_freq_mhz"] = target
                    row[endpoint + "_frequency_match_fraction"] = 1.0
                    row[endpoint + "_gross_energy_j"] = energy / 4
                requests = [{
                    "window_id": window_id, "policy": policy, "workload": workload["id"],
                    "repeat": repeat, "request_id": "%s-%d" % (window_id, index),
                    "prefill_endpoint_id": route[0], "decode_endpoint_id": route[1],
                    "ttft_ms": 200.0, "tpot_ms": 55.0, "mean_itl_ms": 55.0,
                    "e2e_latency_ms": 7200.0,
                } for index in range(3)]
                return row, requests

            def _observe_window(self, window_id, window_dir, registry, telemetry):
                now = time.time()
                selected = {endpoint for pair in self.last_pairs for endpoint in pair}
                for endpoint in capabilities:
                    state = registry.get_state(endpoint)
                    state.queue_depth = 0
                    state.queue_depth_observed = True
                    state.kv_cache_usage_frac = 0.1
                    state.kv_cache_usage_observed = True
                    state.healthy = True
                    state.last_update_s = now
                    registry.update_state(state)
                    role = registry.get_spec(endpoint).role
                    telemetry.observe(EndpointTelemetrySample(
                        endpoint, now, energy_j=10.0,
                        completed_requests=1 if endpoint in selected else 0,
                        interval_s=1.0, queue_depth=0, kv_cache_usage_frac=0.1,
                        ttft_ms=200.0 if role == "prefill" and endpoint in selected else None,
                        tpot_ms=55.0 if role == "decode" and endpoint in selected else None,
                    ))

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            accepted_audit = temporary / "phase4b.json"
            accepted_audit.write_text(json.dumps({"valid": True, "smoke": False}))
            config["phase4c"]["accepted_phase4b_dynamic_audit"] = str(accepted_audit)
            config["phase4c"]["output_root"] = str(temporary / "out")
            config["phase4b"]["phase4a_summary"] = str(temporary / "unused.json")
            config["phase3d"]["minimum_dwell_s"] = 0.0
            config["routing_control_file"] = str(temporary / "routes.json")
            result = Harness(config, "fixture", backend=Backend()).run()
            for name in (
                "phase4c_results.csv", "phase4c_summary.csv", "phase4c_audit.json",
                "phase4c_summary.json", "phase4c_summary.md",
            ):
                self.assertTrue((result / name).is_file(), name)
            summary = json.loads((result / "phase4c_summary.json").read_text())
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["classification"], "A. PASSIVE ROBUSTLY SUFFICIENT")
            self.assertEqual(len(summary["policy_repeat_summaries"]), 18)
            self.assertEqual(len(summary["energy_comparisons"]), 3)


if __name__ == "__main__":
    unittest.main()
