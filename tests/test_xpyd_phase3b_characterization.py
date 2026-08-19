"""CPU-only tests for controlled Phase 3B characterization."""

import json
from pathlib import Path
import tempfile
import unittest

from xpyd.phase3b_characterization import (
    _clock_audit,
    _metric_summary,
    _token_audit,
    balanced_schedule,
    load_config,
)


class CharacterizationTests(unittest.TestCase):
    def test_balanced_schedule_rotates_workload_order(self):
        workloads = [{"id": item} for item in ("a", "b", "c", "d")]
        schedule = balanced_schedule(workloads, 4)
        blocks = [schedule[index:index + 4] for index in range(0, 16, 4)]
        self.assertEqual(
            [[item["workload"]["id"] for item in block] for block in blocks],
            [
                ["a", "b", "c", "d"],
                ["b", "c", "d", "a"],
                ["c", "d", "a", "b"],
                ["d", "a", "b", "c"],
            ],
        )

    def test_clock_audit_requires_target_match_fraction(self):
        records = [
            {
                "status": "success",
                "actual_start_wall_s": float(index),
                "graphics_clock_mhz": 2520 if index < 9 else 2505,
                "memory_clock_mhz": 9001,
            }
            for index in range(10)
        ]
        failed = _clock_audit(records, 0, 9, {
            "graphics_mhz": 2520,
            "memory_mhz": 9001,
            "minimum_workload_match_fraction": 0.95,
        })
        self.assertFalse(failed["valid"])
        self.assertEqual(failed["graphics"]["target_match_fraction"], 0.9)
        passed = _clock_audit(records[:9], 0, 8, {
            "graphics_mhz": 2520,
            "memory_mhz": 9001,
            "minimum_workload_match_fraction": 0.95,
        })
        self.assertTrue(passed["valid"])

    def test_disaggregated_token_audit_does_not_double_count(self):
        load_record = {
            "endpoints": {
                "P0": {"window": {
                    "valid": True,
                    "delta_completed_requests": 10,
                    "delta_prompt_tokens": 1290,
                    "delta_generation_tokens": 10,
                }},
                "D0": {"window": {
                    "valid": True,
                    "delta_completed_requests": 10,
                    "delta_prompt_tokens": 1290,
                    "delta_generation_tokens": 1280,
                }},
            }
        }
        result = _token_audit(load_record, {
            "successful_requests": 10,
            "input_tokens_total": 1280,
            "output_tokens_total": 1280,
        })
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["checks"]), 6)

    def test_repeat_statistics_use_repeat_values(self):
        result = _metric_summary([1.0, 2.0, 3.0, 4.0, 100.0], "fixture")
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["median"], 3.0)
        self.assertIsNotNone(result["bootstrap_median_ci95"])

    def test_checked_in_config_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "paper/configs/xpyd_phase3b_characterization_l40s_l4.json"
        )
        self.assertEqual(config["experiment"]["repeats"], 5)
        self.assertEqual(len(config["workloads"]), 4)
        self.assertEqual(config["endpoints"][0]["node"], "uranus")
        self.assertEqual(config["fixed_clocks"]["D0"]["graphics_mhz"], 1500)

    def test_invalid_duplicate_workload_ids_fail(self):
        root = Path(__file__).resolve().parents[1]
        source = json.loads(
            (root / "paper/configs/xpyd_phase3b_characterization_l40s_l4.json")
            .read_text(encoding="utf-8")
        )
        source["workloads"][1]["id"] = source["workloads"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "unique"):
                load_config(path)


class IsolationGuardTests(unittest.TestCase):
    def test_characterization_orchestrator_contains_no_gpu_mutation(self):
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "paper/scripts/xpyd/phase3b_characterization.py"
        ).read_text(encoding="utf-8")
        for token in (
            "nvmlDevice" + "Set",
            "nvmlDevice" + "Reset",
            "nvidia-smi " + "-lgc",
            "nvidia-smi " + "-lmc",
            "nvidia-smi " + "-pl",
            "nvidia-smi " + "-pm",
        ):
            self.assertNotIn(token, source)

    def test_shell_clock_branch_excludes_persistence_and_power_limit(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        branch = source.split("set_characterization_clocks()", 1)[1].split(
            "reset_characterization_clocks()", 1
        )[0]
        self.assertIn("-lgc", branch)
        self.assertIn("-lmc", branch)
        self.assertNotIn("-pm", branch)
        self.assertNotIn("-pl", branch)


if __name__ == "__main__":
    unittest.main()
