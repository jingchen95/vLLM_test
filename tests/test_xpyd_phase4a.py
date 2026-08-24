"""CPU-only Phase 4A action-space and empirical-oracle tests."""

from pathlib import Path
import unittest

from xpyd.phase4a_oracle import (
    aggregate_measurements,
    build_action_space,
    load_config,
    render_summary_markdown,
    select_oracles,
)


PAIRS = (("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"))
PROFILES = (
    {"id": "LL", "prefill_level": "LOW", "decode_level": "LOW"},
    {"id": "MM", "prefill_level": "MID", "decode_level": "MID"},
    {"id": "HH", "prefill_level": "HIGH", "decode_level": "HIGH"},
    {"id": "HL", "prefill_level": "HIGH", "decode_level": "LOW"},
    {"id": "LH", "prefill_level": "LOW", "decode_level": "HIGH"},
)


class Phase4AActionSpaceTests(unittest.TestCase):
    def test_checked_in_config_loads_with_four_workloads(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "paper/configs/xpyd_phase4a_oracle_neptune_io.json")
        self.assertEqual(len(config["workloads"]), 4)
        self.assertEqual(
            {(item["endpoint_id"], item["node"]) for item in config["endpoints"]},
            {("P0", "neptune"), ("P1", "neptune"), ("D0", "io"), ("D1", "io")},
        )

    def test_pruned_space_has_all_routes_profiles_and_one_static_baseline(self):
        actions = build_action_space(
            PAIRS, ("P0", "P1", "D0", "D1"), PROFILES, ("P0", "D0"),
            (
                {"id": "ML", "prefill_level": "MID", "decode_level": "LOW"},
                {"id": "MH", "prefill_level": "MID", "decode_level": "HIGH"},
            ),
        )
        self.assertEqual(len(actions), 23)
        self.assertEqual(sum(action.static_baseline for action in actions), 1)
        for pair in PAIRS:
            profiles = {
                action.profile_id for action in actions
                if (action.prefill_endpoint_id, action.decode_endpoint_id) == pair
            }
            self.assertTrue({"LL", "MM", "HH", "HL", "LH"}.issubset(profiles))
        baseline = next(action for action in actions if action.static_baseline)
        self.assertEqual(set(baseline.levels.values()), {"HIGH"})

    def test_unselected_endpoints_remain_active_at_low(self):
        actions = build_action_space(
            PAIRS, ("P0", "P1", "D0", "D1"), (PROFILES[2],),
            ("P0", "D0"), (),
        )
        selected = next(action for action in actions if action.config_id == "p0_d1_hh")
        self.assertEqual(selected.levels, {
            "P0": "HIGH", "P1": "LOW", "D0": "LOW", "D1": "HIGH",
        })

    def test_launcher_has_explicit_phase4a_entrypoint(self):
        source = (Path(__file__).resolve().parents[1] / "run_disagg_benchmark.sh").read_text()
        self.assertIn('XPYD_PHASE4A_CONFIG="${XPYD_PHASE4A_CONFIG:-}"', source)
        self.assertIn('args=(-m xpyd.phase4a_oracle --config "${XPYD_PHASE4A_CONFIG}")', source)
        self.assertIn('${XPYD_PHASE3C_CONFIG}${XPYD_PHASE3D_CONFIG}${XPYD_PHASE4A_CONFIG}', source)


def measurement(workload, config_id, repeat, energy, *, valid=True, slo=True, baseline=False):
    return {
        "workload": workload, "config_id": config_id, "repeat": repeat,
        "prefill_endpoint_id": "P0", "decode_endpoint_id": "D0",
        "profile_id": "ALL_HIGH" if baseline else "LL",
        "static_baseline": baseline, "measurement_valid": valid, "slo_pass": slo,
        "P0_requested_freq_mhz": 2520 if baseline else 1260,
        "P1_requested_freq_mhz": 2520 if baseline else 1260,
        "D0_requested_freq_mhz": 1500 if baseline else 750,
        "D1_requested_freq_mhz": 1500 if baseline else 750,
        "total_gpu_gross_energy_j": energy,
        "joules_per_request": energy / 2,
        "joules_per_output_token": energy / 256,
        "mean_ttft_ms": 500.0, "mean_tpot_ms": 60.0,
        "mean_itl_ms": 60.0, "mean_e2e_latency_ms": 8000.0,
        "throughput_requests_s": 0.25,
    }


class Phase4AOracleTests(unittest.TestCase):
    def test_markdown_report_formats_fractional_savings_as_percent(self):
        summary = {
            "valid": True,
            "action_space": {"measured_configurations": 1},
            "slo": {"ttft_ms": 1000.0, "tpot_ms": 80.0},
            "measurement_count": 3,
            "planned_measurement_count": 3,
            "oracles": [{
                "workload": "small", "best_config_id": "best",
                "second_best_config_id": "second", "J_per_request": 100.0,
                "energy_vs_static": 0.25, "TTFT": 500.0, "TPOT": 60.0,
                "near_optimal_5pct_count": 2,
            }],
            "answers": {
                "unique_best_configurations": ["best"],
                "best_configuration_differs_across_workloads": False,
                "largest_measured_main_effect": "P_side_DVFS",
                "oracle_savings_vs_static": {
                    "minimum_fraction": 0.25, "mean_fraction": 0.25,
                    "maximum_fraction": 0.25,
                },
                "optimum_shape": {"near_optimal_5pct_counts": [2]},
            },
            "hard_gates": {"complete_measurement_plan": True},
            "ready_for_phase4b": True,
            "error": None,
        }
        report = render_summary_markdown(summary)
        self.assertIn("| 25.00% |", report)
        self.assertIn("min `25.00%`, mean `25.00%`, max `25.00%`", report)

    def test_duplicate_repeat_does_not_satisfy_repeat_gate(self):
        rows = [
            measurement("small", "candidate", 1, 100.0),
            measurement("small", "candidate", 1, 101.0),
            measurement("small", "candidate", 2, 102.0),
        ]
        aggregate = aggregate_measurements(rows, 3)[0]
        self.assertFalse(aggregate["oracle_eligible"])

    def test_oracle_requires_every_repeat_valid_and_slo_safe(self):
        rows = []
        for repeat in range(1, 4):
            rows.append(measurement("small", "best", repeat, 100.0 + repeat))
            rows.append(measurement("small", "static", repeat, 150.0 + repeat, baseline=True))
            rows.append(measurement("small", "unsafe", repeat, 80.0, slo=repeat != 2))
        aggregates = aggregate_measurements(rows, 3)
        unsafe = next(item for item in aggregates if item["config_id"] == "unsafe")
        self.assertFalse(unsafe["oracle_eligible"])
        oracle = select_oracles(aggregates, [{"id": "small"}], 1000.0, 80.0)
        self.assertEqual(oracle[0]["best_config_id"], "best")
        self.assertGreater(oracle[0]["energy_vs_static"], 0.0)

    def test_missing_or_invalid_baseline_blocks_oracle(self):
        rows = [measurement("small", "best", repeat, 100.0) for repeat in range(1, 4)]
        rows += [
            measurement("small", "static", repeat, 150.0, baseline=True, valid=False)
            for repeat in range(1, 4)
        ]
        aggregates = aggregate_measurements(rows, 3)
        self.assertEqual(select_oracles(aggregates, [{"id": "small"}], 1000.0, 80.0), [])


if __name__ == "__main__":
    unittest.main()
