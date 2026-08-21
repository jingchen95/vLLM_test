"""CPU-only Phase 3C configuration and audit primitives."""

import csv
from pathlib import Path
import tempfile
import unittest

from xpyd.phase3c_substrate import (
    Phase3CError,
    _throttle_audit,
    _write_trace,
    build_registry_and_compatibility,
)


class Phase3CSubstrateTests(unittest.TestCase):
    def config(self):
        endpoints = []
        for endpoint_id, role, node, gpu in (
            ("P0", "prefill", "pnode", 0), ("P1", "prefill", "pnode", 1),
            ("D0", "decode", "dnode", 0), ("D1", "decode", "dnode", 1),
        ):
            endpoints.append({
                "endpoint_id": endpoint_id, "role": role, "node": node,
                "gpu_type": "fixture", "gpu_ids": [gpu], "tp_degree": 1,
                "http_uri": "http://%s:%d" % (node, 8000 + gpu),
                "kv_connector": "test-kv",
            })
        return {
            "endpoints": endpoints,
            "fixed_clocks": {item["endpoint_id"]: {"graphics_mhz": 1500} for item in endpoints},
            "compatible_pairs": [
                {"prefill_endpoint_id": p, "decode_endpoint_id": d,
                 "connector": "test-kv", "supported": True, "reason": "fixture"}
                for p in ("P0", "P1") for d in ("D0", "D1")
            ],
        }

    def test_registry_rejects_overlap_and_exact_pairs_are_enumerated(self):
        registry, table, pairs = build_registry_and_compatibility(self.config())
        self.assertEqual(len(registry.healthy_active()), 4)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(table.is_compatible(registry.get_spec("P1"), registry.get_spec("D0")))

    def test_throttle_audit_requires_available_noninvalidating_samples(self):
        records = [{
            "status": "success", "actual_start_wall_s": 2.0,
            "invalidating_thermal_or_hw_slowdown": False,
            "clock_throttle_reasons": ["sw_power_cap"],
        }]
        audit = _throttle_audit(records, 1.0, 3.0)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["observed_reasons"], ["sw_power_cap"])
        records[0]["invalidating_thermal_or_hw_slowdown"] = True
        self.assertFalse(_throttle_audit(records, 1.0, 3.0)["valid"])

    def test_default_phase3c_config_keeps_full_coverage_policy(self):
        config = self.config()
        self.assertNotIn("coverage_policy", config)
        _, _, pairs = build_registry_and_compatibility(config)
        self.assertEqual(set(pairs), {
            ("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"),
        })

    def test_dynamic_2p2d_launch_disables_endpoint_local_prefix_cache(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        phase3c_launch = source.split("run_phase3c_substrate()", 1)[1].split(
            "# ===========================================================================\n"
            "run_experiment_E()",
            1,
        )[0]
        self.assertIn('no_local_prefix_cache="--no-enable-prefix-caching"', phase3c_launch)
        for port in (8100, 8101, 8200, 8201):
            launch_line = next(
                line for line in phase3c_launch.splitlines()
                if "start_vllm_server" in line and " %d " % port in line
            )
            self.assertIn("${no_local_prefix_cache}", launch_line)

    def test_dynamic_2p2d_launch_disables_unvalidated_chunked_prefill(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        phase3c_launch = source.split("run_phase3c_substrate()", 1)[1].split(
            "# ===========================================================================\n"
            "run_experiment_E()",
            1,
        )[0]
        self.assertIn('no_chunked_prefill="--no-enable-chunked-prefill"', phase3c_launch)
        for port in (8100, 8101, 8200, 8201):
            launch_line = next(
                line for line in phase3c_launch.splitlines()
                if "start_vllm_server" in line and " %d " % port in line
            )
            self.assertIn("${no_chunked_prefill}", launch_line)

    def test_persistent_windows_use_distinct_request_id_namespaces(self):
        workloads = [{
            "id": "fixture", "input_len": 128, "output_len": 16,
            "count": 2, "rate_rps": 1.0,
        }]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            _write_trace(first, workloads, "phase3c-light-before")
            _write_trace(second, workloads, "phase3c-moderate")
            with first.open(newline="", encoding="utf-8") as stream:
                first_ids = {row["request_id"] for row in csv.DictReader(stream)}
            with second.open(newline="", encoding="utf-8") as stream:
                second_ids = {row["request_id"] for row in csv.DictReader(stream)}
        self.assertEqual(first_ids, {
            "phase3c-light-before-0000", "phase3c-light-before-0001",
        })
        self.assertEqual(second_ids, {
            "phase3c-moderate-0000", "phase3c-moderate-0001",
        })
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_request_id_namespace_rejects_unsafe_or_empty_values(self):
        workloads = [{
            "id": "fixture", "input_len": 1, "output_len": 1,
            "count": 1, "rate_rps": 1.0,
        }]
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate(("", "has spaces", "unsafe/slash")):
                with self.assertRaises(Phase3CError):
                    _write_trace(Path(directory) / ("%d.csv" % index), workloads, value)


if __name__ == "__main__":
    unittest.main()
