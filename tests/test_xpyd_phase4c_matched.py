"""CPU-only tests for the Phase 4C.2 matched-routing experiment."""

import json
from pathlib import Path
import tempfile
import time
import unittest

from xpyd.phase3d_control import EndpointClockCapability, GPUReadback
from xpyd.phase4c_matched_routing import (
    POLICIES,
    Phase4CMatchedRoutingHarness,
    counterbalanced_blocks,
    load_config,
)


class Phase4CMatchedRoutingTests(unittest.TestCase):
    def test_checked_in_design_is_minimal_fixed_high_and_neptune_io(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json")
        settings = config["phase4c_matched"]
        self.assertEqual(tuple(settings["policies"]), POLICIES)
        self.assertEqual(settings["repeats"], 3)
        self.assertEqual(settings["requests_per_block"], 3)
        self.assertEqual(settings["priming_requests_per_block"], 3)
        self.assertEqual(settings["slo"], {"ttft_ms": 550.0, "tpot_ms": 65.0})
        self.assertEqual(
            {item["base_workload"] for item in settings["workloads"]},
            {"prefill_heavy", "both_heavy"},
        )
        self.assertEqual(settings["concentrated_pair"], ["P0", "D0"])
        self.assertEqual(config["client"]["max_concurrency"], 1)
        endpoint_nodes = {item["endpoint_id"]: item["node"] for item in config["endpoints"]}
        self.assertEqual(endpoint_nodes["P0"], "neptune")
        self.assertEqual(endpoint_nodes["P1"], "neptune")
        self.assertEqual(endpoint_nodes["D0"], "io")
        self.assertEqual(endpoint_nodes["D1"], "io")

    def test_block_order_pairs_policies_and_balances_first_position(self):
        workloads = [
            {"id": "prefill", "base_workload": "prefill_heavy"},
            {"id": "both", "base_workload": "both_heavy"},
        ]
        blocks = counterbalanced_blocks(workloads, 3, 71)
        self.assertEqual(len(blocks), 12)
        first_counts = {policy: 0 for policy in POLICIES}
        for index in range(0, len(blocks), 2):
            first, second = blocks[index:index + 2]
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[1], second[1])
            self.assertEqual(first[3]["base_workload"], second[3]["base_workload"])
            self.assertEqual({first[2], second[2]}, set(POLICIES))
            first_counts[first[2]] += 1
        self.assertEqual(first_counts, {"BALANCED_HIGH": 3, "P0D0_HIGH": 3})

    def test_launcher_selects_matched_module_without_new_gpu_mode(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn('XPYD_PHASE4C_MATCHED="${XPYD_PHASE4C_MATCHED:-0}"', source)
        self.assertIn("-m xpyd.phase4c_matched_routing", source)
        self.assertIn("export XPYD_PHASE4C_MATCHED_ACCEPTED_AUDIT", source)

    def test_full_mocked_plan_emits_12_windows_36_requests_and_hard_gates(self):
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
                pci_bus_id="0000:0%s:00.0" % endpoint[-1],
                supported_graphics_mhz=values,
                selected_low_mhz=values[0], selected_mid_mhz=values[1],
                selected_high_mhz=values[2],
            ) for endpoint, values in levels.items()
        }

        class Backend:
            def __init__(self):
                self.frequencies = {endpoint: values[-1] for endpoint, values in levels.items()}

            def discover(self, endpoint_id, node, gpu_id, configured_high_mhz):
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

        class Harness(Phase4CMatchedRoutingHarness):
            def _reference(self):
                return {
                    "path": "fixture", "sha256": "fixture", "near_signatures": {},
                    "oracles": {
                        workload: {"J_per_request": 100.0, "TTFT": 500.0, "TPOT": 55.0}
                        for workload in ("small_light", "prefill_heavy", "decode_heavy", "both_heavy")
                    },
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

            def _run_window(
                self, window_id, workload, count, requested, pairs, windows_root,
                *, require_all_pairs=False,
            ):
                directory = windows_root / window_id
                directory.mkdir(parents=True)
                (directory / "audit.json").write_text(
                    json.dumps({"valid": True}), encoding="utf-8"
                )
                fields = ["endpoint_id", "mean_queue_depth", "mean_kv_cache_usage_frac"]
                lines = [",".join(fields)] + [
                    "%s,0,0.01" % endpoint for endpoint in ("P0", "P1", "D0", "D1")
                ]
                (directory / "endpoint_summary.csv").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                return directory

            def _measurement_row(
                self, window_id, policy, workload, repeat, context,
                accepted, reference, window_dir,
            ):
                now = time.time()
                ttft = 500.0 if policy == "BALANCED_HIGH" else 600.0
                energy = 300.0 if policy == "BALANCED_HIGH" else 310.0
                gates = {
                    "logical_request_id": True, "requested_output_tokens": True,
                    "endpoint_assignment": True, "explicit_compatibility": True,
                    "nvml_energy_windows": True,
                    "no_invalidating_thermal_or_hw_slowdown": True,
                }
                row = {
                    "run_id": self.run_id, "window_id": window_id, "policy": policy,
                    "workload": workload["id"], "input_len": workload["input_len"],
                    "output_len": workload["output_len"], "rate_rps": workload["rate_rps"],
                    "repeat": repeat, "timing_start_unix_s": now - 1.0,
                    "timing_end_unix_s": now, "duration_s": 1.0,
                    "completed_requests": 3, "output_tokens": 384,
                    "total_gpu_gross_energy_j": energy,
                    "joules_per_request": energy / 3,
                    "joules_per_output_token": energy / 384,
                    "mean_ttft_ms": ttft, "p99_ttft_ms": ttft,
                    "mean_tpot_ms": 55.0, "p99_tpot_ms": 55.0,
                    "mean_itl_ms": 55.0, "mean_e2e_latency_ms": 7200.0,
                    "throughput_requests_s": 3.0,
                    "slo_pass": policy == "BALANCED_HIGH", "measurement_valid": True,
                    "route_distribution_json": "{}",
                    "selected_pairs_json": json.dumps(context.selected_pairs),
                    "phase4a_oracle_joules_per_request": 100.0,
                    "oracle_gap": 0.0, "oracle_gap_class": "<=5%",
                    "phase4a_near_optimal_match": False,
                    "hard_gates_json": json.dumps(gates),
                }
                for endpoint, frequencies in accepted.items():
                    target = context.target_frequencies[endpoint]
                    row[endpoint + "_requested_freq_mhz"] = target
                    row[endpoint + "_requested_level"] = "HIGH"
                    row[endpoint + "_observed_freq_mhz"] = target
                    row[endpoint + "_frequency_match_fraction"] = 1.0
                    row[endpoint + "_gross_energy_j"] = energy / 4
                routes = context.selected_pairs
                requests = []
                for index in range(3):
                    route = routes[index % len(routes)]
                    requests.append({
                        "window_id": window_id, "policy": policy,
                        "workload": workload["id"], "repeat": repeat,
                        "request_id": "%s-%d" % (window_id, index),
                        "prefill_endpoint_id": route[0],
                        "decode_endpoint_id": route[1], "ttft_ms": ttft,
                        "tpot_ms": 55.0, "mean_itl_ms": 55.0,
                        "e2e_latency_ms": 7200.0,
                    })
                return row, requests

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            accepted_audit = temporary / "phase4c1.json"
            accepted_audit.write_text(
                json.dumps({"valid": True, "phase": "4C1_aligned_slo_asymmetric_recovery"}),
                encoding="utf-8",
            )
            config["phase4c_matched"]["accepted_phase4c1_audit"] = str(accepted_audit)
            config["phase4c_matched"]["output_root"] = str(temporary / "out")
            config["phase4b"]["phase4a_summary"] = str(temporary / "unused.json")
            config["phase3d"]["minimum_dwell_s"] = 0.0
            config["routing_control_file"] = str(temporary / "routes.json")
            result = Harness(config, "fixture", backend=Backend()).run()
            for name in (
                "phase4c_matched_measurements.csv", "phase4c_matched_summary.csv",
                "phase4c_matched_audit.json", "phase4c_matched_summary.json",
                "phase4c_matched_summary.md",
            ):
                self.assertTrue((result / name).is_file(), name)
            summary = json.loads((result / "phase4c_matched_summary.json").read_text())
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["classification"], "ROUTING CONCENTRATION PENALTY OBSERVED")
            self.assertEqual(len(summary["policy_workload_summaries"]), 4)
            self.assertEqual(len(summary["matched_comparisons"]), 2)
            self.assertEqual(sum(summary["paired_request_categories"].values()), 18)
            audit = json.loads((result / "phase4c_matched_audit.json").read_text())
            self.assertTrue(all(audit["hard_gates"].values()))


if __name__ == "__main__":
    unittest.main()
