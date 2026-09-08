"""CPU-only checks for Stage C0.5 TTFT aggregation."""

import unittest

from xpyd.stagec05_slo_calibration import percentile, sensitivity_rows


class StageC05SLOTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([0.0, 10.0, 20.0], 0.5), 10.0)
        self.assertEqual(percentile([0.0, 10.0], 0.9), 9.0)

    def test_sensitivity_separates_short_and_long_input(self):
        rows = [
            {"workload_class": "SS", "ttft_ms": 100.0},
            {"workload_class": "SL", "ttft_ms": 200.0},
            {"workload_class": "LS", "ttft_ms": 600.0},
            {"workload_class": "LL", "ttft_ms": 500.0},
        ]
        output = sensitivity_rows([("fixture", "0.10", rows)])
        selected = {
            row["request_group"]: row
            for row in output
            if row["load"] == "fixture" and row["ttft_slo_ms"] == 550
        }
        self.assertEqual(selected["SHORT_INPUT"]["violation_rate"], 0.0)
        self.assertEqual(selected["LONG_INPUT"]["violation_rate"], 0.5)
        self.assertEqual(selected["ALL"]["violation_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
