"""CPU-only contracts for Stage C0 capacity/load calibration."""

import os
from pathlib import Path
import unittest

from xpyd.stagec0_capacity import (
    classify_points,
    linear_slope,
    load_configs,
    percentile,
    time_weighted_state,
)


class StageC0CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_checked_in_sweep_reuses_validated_topology_and_semantics(self):
        old = dict(os.environ)
        os.environ.update({
            "XPYD_PROXY_URI": "http://127.0.0.1:8090",
            "XPYD_PROXY_DIAGNOSTICS_LOG": "/tmp/proxy.jsonl",
            "XPYD_P0_HTTP_HOST": "127.0.0.1",
            "XPYD_P1_HTTP_HOST": "127.0.0.1",
            "XPYD_D0_HTTP_HOST": "127.0.0.1",
            "XPYD_D1_HTTP_HOST": "127.0.0.1",
            "XPYD_P_ADDR_HOST": "10.0.0.1",
            "XPYD_D_ADDR_HOST": "10.0.0.2",
        })
        try:
            runtime, workload = load_configs(
                self.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json",
                self.root / "paper/configs/xpyd_stagec0_capacity_neptune_europa.json",
            )
        finally:
            os.environ.clear()
            os.environ.update(old)
        rates = [
            item["offered_rate_rps"]
            for item in workload["capacity_calibration"]["load_points"]
        ]
        self.assertEqual(rates, [0.05, 0.10, 0.15, 0.20, 0.30])
        self.assertEqual(workload["capacity_calibration"]["max_concurrency"], 4)
        self.assertEqual(workload["capacity_calibration"]["control"], "disabled")
        self.assertEqual(
            {item["endpoint_id"]: item["node"] for item in runtime["endpoints"]},
            {"P0": "neptune", "P1": "neptune", "D0": "europa", "D1": "europa"},
        )

    def test_statistics_measure_trend_and_time_weighted_backlog(self):
        self.assertEqual(percentile([0.0, 10.0, 20.0], 0.5), 10.0)
        self.assertAlmostEqual(linear_slope([0, 1, 2], [0, 10, 20]), 10.0)
        mean, maximum = time_weighted_state(
            [(1.0, 2.0), (3.0, 0.0)], 0.0, 4.0
        )
        self.assertEqual(mean, 1.0)
        self.assertEqual(maximum, 2.0)

    def test_classification_selects_measured_safe_regimes_and_overload(self):
        rows = []
        for rate, queue, sustainable, slo in (
            (0.05, True, True, 0.0),
            (0.10, False, True, 0.0),
            (0.15, False, True, 0.02),
            (0.20, False, False, 0.20),
        ):
            rows.append({
                "offered_rate_rps": rate,
                "little_or_no_sustained_queue": queue,
                "sustainable": sustainable,
                "slo_violation_rate": slo,
                "classification": "UNCLASSIFIED",
            })
        chosen = classify_points(rows)
        self.assertEqual(chosen["LOW"]["offered_rate_rps"], 0.05)
        self.assertEqual(chosen["MEDIUM"]["offered_rate_rps"], 0.10)
        self.assertEqual(chosen["HIGH-SUSTAINABLE"]["offered_rate_rps"], 0.15)
        self.assertEqual(rows[-1]["classification"], "OVERLOAD")

    def test_launcher_has_dedicated_stagec0_mode(self):
        source = (self.root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("XPYD_STAGEC0_RUNTIME_CONFIG", source)
        self.assertIn("-m xpyd.stagec0_capacity", source)
        phase3c = (
            self.root / "paper/scripts/xpyd/phase3c_substrate.py"
        ).read_text(encoding="utf-8")
        self.assertIn('self.run_dir / "derived" / "endpoint_load.csv"', phase3c)


if __name__ == "__main__":
    unittest.main()
