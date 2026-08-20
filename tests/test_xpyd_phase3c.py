"""CPU-only Phase 3C configuration and audit primitives."""

import unittest

from xpyd.phase3c_substrate import _throttle_audit, build_registry_and_compatibility


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


if __name__ == "__main__":
    unittest.main()
