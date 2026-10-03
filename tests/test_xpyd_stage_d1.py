"""CPU-only semantic tests for the Stage D1 experiment contract."""

from pathlib import Path
from unittest.mock import patch
import unittest

from xpyd.adaptive_exploration import evidence_from_window
from xpyd.stage_d1 import POLICIES, load_configs, policy_order


class StageD1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.runtime = cls.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json"
        cls.experiment = cls.root / "paper/configs/xpyd_stage_d1_neptune_europa.json"

    def test_config_freezes_topology_slo_action_space_and_policies(self):
        environment = {
            "XPYD_ROUTING_POLICY": "d1_runtime_feedback",
            "XPYD_ROUTING_CONTROL_FILE": "/tmp/d1-control.json",
            "XPYD_PROXY_URI": "http://proxy",
            "XPYD_PROXY_DIAGNOSTICS_LOG": "/tmp/d1-diagnostics.jsonl",
            "XPYD_P0_HTTP_HOST": "p", "XPYD_P1_HTTP_HOST": "p",
            "XPYD_D0_HTTP_HOST": "d", "XPYD_D1_HTTP_HOST": "d",
            "XPYD_P_ADDR_HOST": "p", "XPYD_D_ADDR_HOST": "d",
            "XPYD_P0_SERVER_LOG": "p0", "XPYD_P1_SERVER_LOG": "p1",
            "XPYD_D0_SERVER_LOG": "d0", "XPYD_D1_SERVER_LOG": "d1",
        }
        with patch.dict("os.environ", environment, clear=False):
            runtime, experiment = load_configs(self.runtime, self.experiment)
        settings = experiment["stage_d1"]
        self.assertEqual(tuple(settings["policies"]), POLICIES)
        self.assertEqual(settings["slo"], {
            "ttft_ms": 600.0, "tpot_ms": 65.0, "attainment": 0.95,
        })
        self.assertEqual(settings["frequency_levels_mhz"]["P0"], {
            "LOW": 1260, "MID": 1890, "HIGH": 2520,
        })
        self.assertEqual(settings["frequency_levels_mhz"]["D1"], {
            "LOW": 750, "MID": 1125, "HIGH": 1500,
        })
        self.assertEqual(runtime["routing_policy"], "d1_runtime_feedback")
        self.assertFalse(any(settings[key] for key in (
            "predictor", "offline_profile", "rl", "bandit", "tp2"
        )))

    def test_policy_order_is_cyclic_across_load_repeat_pairs(self):
        starts = [policy_order(repeat, load)[0] for load in range(2) for repeat in range(1, 4)]
        self.assertEqual(starts, [
            POLICIES[0], POLICIES[1], POLICIES[2],
            POLICIES[3], POLICIES[0], POLICIES[1],
        ])

    def test_endpoint_evidence_uses_only_selected_endpoint_energy_and_pressure(self):
        metrics = {
            "observation_duration_s": 100,
            "client_waiting_at_offered_close": 0,
            "backlog_growing": False,
            "measurement_valid": True,
            "telemetry_fresh": True,
            "readback_valid": True,
            "P0_energy_j": 100,
            "P1_energy_j": 900,
            "endpoint_load": {
                "P0": {"waiting_mean": 0, "waiting_max": 0},
                "P1": {"waiting_mean": 99, "waiting_max": 99},
            },
        }
        requests = [{
            "ttft_ms": 500, "tpot_ms": 60, "input_len": 2048,
            "output_len": 128, "real_sse": True,
            "logical_request_id_propagated": True,
        }]
        evidence = evidence_from_window(
            "P", metrics, requests, observed_frequency_mhz=1890,
            slo_regime="TIGHT", ttft_slo_ms=600, tpot_slo_ms=65,
            endpoint_ids=("P0",),
        )
        self.assertEqual(evidence.gross_energy_j, 100)
        self.assertTrue(evidence.backlog_stable)

    def test_launcher_exposes_d1_preflight_and_full_modes(self):
        source = (self.root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("XPYD_STAGED1_STAGE", source)
        self.assertIn("-m xpyd.stage_d1", source)
        self.assertIn("XPYD_STAGED1_ACCEPTED_PREFLIGHT", source)


if __name__ == "__main__":
    unittest.main()
