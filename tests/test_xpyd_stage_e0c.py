"""CPU-only tests for Stage E0-c capacity and Poisson trace generation."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from xpyd.stage_e0c_capacity import (
    StageE0cError,
    StageE0cHarness,
    load_configs,
    saturated_throughput,
    step_trace,
)
from xpyd.workload_harness import WorkloadHarnessError, build_trace


class StageE0cTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def load(self):
        old = dict(os.environ)
        os.environ.update({
            "XPYD_PROXY_URI": "http://127.0.0.1:8090",
            "XPYD_PROXY_DIAGNOSTICS_LOG": "/tmp/proxy.jsonl",
            "XPYD_P0_HTTP_HOST": "127.0.0.1", "XPYD_P1_HTTP_HOST": "127.0.0.1",
            "XPYD_D0_HTTP_HOST": "127.0.0.1", "XPYD_D1_HTTP_HOST": "127.0.0.1",
            "XPYD_P_ADDR_HOST": "10.0.0.1", "XPYD_D_ADDR_HOST": "10.0.0.2",
        })
        try:
            return load_configs(
                self.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json",
                self.root / "paper/configs/xpyd_stage_e0c_capacity_neptune_europa.json",
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_config_contract(self):
        runtime, workload = self.load()
        settings = workload["stage_e0c"]
        self.assertEqual(settings["concurrency_ladder"][0], 4)
        self.assertEqual(settings["frequencies"], "all validated HIGH")
        self.assertEqual((settings["slo"]["ttft_ms"], settings["slo"]["tpot_ms"]), (600.0, 65.0))

    def test_uniform_trace_is_unchanged_by_poisson_support(self):
        _, workload = self.load()
        rows = step_trace(workload, "u", 0.5, 20, "uniform", 0)
        self.assertEqual([row["arrival_time_s"] for row in rows], [i / 0.5 for i in range(20)])

    def test_poisson_trace_is_seeded_and_keeps_exact_mix(self):
        _, workload = self.load()
        first = step_trace(workload, "p", 2.0, 400, "poisson", 7)
        again = step_trace(workload, "p", 2.0, 400, "poisson", 7)
        other = step_trace(workload, "p", 2.0, 400, "poisson", 8)
        times = [row["arrival_time_s"] for row in first]
        self.assertEqual(times, [row["arrival_time_s"] for row in again])
        self.assertNotEqual(times, [row["arrival_time_s"] for row in other])
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times, sorted(times))
        self.assertAlmostEqual(times[-1] / (len(times) - 1), 0.5, delta=0.08)
        counts = {key: sum(row["workload_class"] == key for row in first) for key in ("SS", "SL", "LS", "LL")}
        self.assertEqual(counts, {"SS": 140, "SL": 60, "LS": 140, "LL": 60})

    def test_poisson_requires_integer_seed(self):
        _, workload = self.load()
        workload = json.loads(json.dumps(workload))
        workload["scenarios"][0]["segments"][0]["arrival_process"] = "poisson"
        with self.assertRaises(WorkloadHarnessError):
            build_trace(workload, workload["scenarios"][0]["id"])

    def test_saturated_throughput_uses_interquartile_span(self):
        rows = [{"client_admitted_unix_s": 0.0, "e2e_latency_ms": 1000.0 * i,
                 "observed_output_tokens": 100} for i in range(1, 41)]
        result = saturated_throughput(rows)
        self.assertAlmostEqual(result["saturated_rps"], 1.0)
        self.assertAlmostEqual(result["saturated_output_tok_s"], 100.0)
        self.assertIsNone(saturated_throughput(rows[:4])["saturated_rps"])

    def harness(self, tmp, steps):
        runtime, workload = self.load()
        workload["output_root"] = str(tmp)
        harness = StageE0cHarness(runtime, workload, run_id="fixture")
        calls = []

        def fake_step(phase, step_id, concurrency, rate, count, process, seed):
            calls.append((phase, concurrency, rate, count, process))
            return dict(steps(phase, concurrency, rate), phase=phase, step_id=step_id,
                        max_concurrency=concurrency, offered_rate_rps=rate)

        harness._step = fake_step
        return harness, calls

    def test_ladder_stops_at_first_failure_and_sweeps_at_last_valid(self):
        def steps(phase, concurrency, rate):
            if phase == "ladder":
                return {"step_valid": concurrency < 32, "error": None if concurrency < 32 else "boom"}
            return {"step_valid": True, "sustainable": rate < 1.5}

        with tempfile.TemporaryDirectory() as tmp:
            harness, calls = self.harness(tmp, steps)
            run_dir = harness.run()
            ladder = [call for call in calls if call[0] == "ladder"]
            sweep = [call for call in calls if call[0] == "sweep"]
            self.assertEqual([call[1] for call in ladder], [4, 8, 16, 32])
            self.assertEqual({call[1] for call in sweep}, {16})
            self.assertTrue(all(call[4] == "poisson" for call in sweep))
            # Stops after two consecutive overloads: 1.5 and 2.0.
            self.assertEqual([call[2] for call in sweep], [0.3, 0.6, 1.0, 1.5, 2.0])
            summary = json.loads((run_dir / "e0c_summary.json").read_text())
            self.assertEqual(summary["max_validated_concurrency"], 16)
            self.assertEqual(summary["first_failed_concurrency"], 32)
            self.assertEqual(summary["max_sustainable_rate_rps"], 1.0)
            self.assertTrue((run_dir / "e0c_summary.md").is_file())

    def test_ladder_counts_hold_the_cap(self):
        def steps(phase, concurrency, rate):
            return {"step_valid": True, "sustainable": True}

        with tempfile.TemporaryDirectory() as tmp:
            harness, calls = self.harness(tmp, steps)
            harness.run()
            ladder = {call[1]: call for call in calls if call[0] == "ladder"}
            self.assertEqual(ladder[4][3], 24)
            self.assertEqual(ladder[64][3], 256)
            self.assertAlmostEqual(ladder[64][2], 64 / 5.0)

    def test_failed_validated_concurrency_raises_and_still_reports(self):
        def steps(phase, concurrency, rate):
            return {"step_valid": False, "error": "connector"}

        with tempfile.TemporaryDirectory() as tmp:
            harness, calls = self.harness(tmp, steps)
            with self.assertRaises(StageE0cError):
                harness.run()
            self.assertTrue((Path(tmp) / "fixture" / "e0c_summary.json").is_file())
            self.assertFalse([call for call in calls if call[0] == "sweep"])

    def test_window_exception_is_recorded_not_raised(self):
        runtime, workload = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            workload["output_root"] = tmp

            def runner(config, step_id, trace):
                self.assertEqual(config["client"]["max_concurrency"], 8)
                self.assertEqual(config["client"]["request_timeout_s"], 300.0)
                raise RuntimeError("client exited with status 1")

            harness = StageE0cHarness(runtime, workload, run_id="fixture", window_runner=runner)
            harness.run_dir.mkdir(parents=True)
            row = harness._step("ladder", "ladder-c008", 8, 1.6, 32, "uniform", 0)
            self.assertFalse(row["step_valid"])
            self.assertIn("client exited", row["error"])


if __name__ == "__main__":
    unittest.main()
