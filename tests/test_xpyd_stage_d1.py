"""CPU-only semantic tests for the Stage D1 experiment contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from xpyd.adaptive_exploration import evidence_from_window
from xpyd.stage_d1 import (
    D1EvidenceLedger,
    ENDPOINTS,
    POLICIES,
    StageD1Harness,
    d1_preflight_acceptance,
    d1_context_signature,
    endpoint_evidence_summary,
    final_d1_safety_state_consistent,
    load_configs,
    normalize_d1_request,
    policy_order,
)
from xpyd.stagec2_dynamic import StageC2Error


class StageD1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.runtime = cls.root / "paper/configs/xpyd_stageb_runtime_neptune_europa.json"
        cls.experiment = cls.root / "paper/configs/xpyd_stage_d1_neptune_europa.json"

    @staticmethod
    def normalized_row(**overrides):
        row = {
            "request_id": "request-1",
            "workload_class": "LL",
            "input_len": "2048",
            "requested_output_len": "512",
            "observed_output_tokens": "17",
            "prefill_endpoint_id": "P0",
            "decode_endpoint_id": "D1",
            "ttft_ms": "500.0",
            "tpot_ms": "60.0",
            "send_unix_s": "10.0",
            "first_token_unix_s": "10.5",
            "complete_unix_s": "12.0",
            "client_admission_delay_ms": "2.0",
            "real_sse": "True",
            "logical_request_id_propagated": "True",
            "block_id": "b1",
            "policy": "JOINT-ROUTING-PER-ENDPOINT",
            "repeat": 1,
            "window_index": 1,
            "context": "HIGH-SUSTAINABLE",
            "load_point": "HIGH-SUSTAINABLE",
        }
        row.update(overrides)
        return row

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
        self.assertEqual(settings["preflight_warmup_windows"], 1)
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

    def test_real_normalized_schema_maps_to_canonical_record(self):
        record = normalize_d1_request(self.normalized_row())
        self.assertEqual(record["request_id"], "request-1")
        self.assertEqual(record["prefill_endpoint_id"], "P0")
        self.assertEqual(record["decode_endpoint_id"], "D1")
        self.assertEqual(record["requested_input_len"], 2048)
        self.assertEqual(record["requested_output_len"], 512)
        self.assertEqual(record["actual_output_len"], 17)
        self.assertEqual(record["request_class"], "LL")

    def test_missing_required_schema_field_fails_before_evidence(self):
        row = self.normalized_row()
        del row["prefill_endpoint_id"]
        with self.assertRaisesRegex(StageC2Error, "prefill_endpoint_id"):
            normalize_d1_request(row)

    def test_p0_d1_updates_only_selected_endpoint_evidence(self):
        request = normalize_d1_request(self.normalized_row())
        evidence = endpoint_evidence_summary([request], {
            "P0_energy_j": 100, "P1_energy_j": 200,
            "D0_energy_j": 300, "D1_energy_j": 400,
        })
        self.assertEqual(evidence["P0"]["request_count"], 1)
        self.assertEqual(evidence["D1"]["request_count"], 1)
        self.assertEqual(evidence["P1"]["request_count"], 0)
        self.assertEqual(evidence["D0"]["request_count"], 0)
        self.assertEqual(evidence["P0"]["requested_input_tokens"], 2048)
        self.assertEqual(evidence["D1"]["requested_output_tokens"], 512)
        self.assertEqual(evidence["P0"]["latency_count"], 1)
        self.assertEqual(evidence["D1"]["latency_count"], 1)
        self.assertEqual(evidence["P0"]["gross_energy_j"], 100)
        self.assertEqual(evidence["D1"]["gross_energy_j"], 400)

    def test_assigned_traffic_populates_all_endpoint_counts_tokens_and_latency(self):
        requests = [
            normalize_d1_request(self.normalized_row()),
            normalize_d1_request(self.normalized_row(
                request_id="request-2", workload_class="SS",
                input_len="128", requested_output_len="128",
                observed_output_tokens="128",
                prefill_endpoint_id="P1", decode_endpoint_id="D0",
            )),
        ]
        evidence = endpoint_evidence_summary(requests, {
            "P0_energy_j": 100, "P1_energy_j": 200,
            "D0_energy_j": 300, "D1_energy_j": 400,
        })
        for endpoint_id in ENDPOINTS:
            self.assertEqual(evidence[endpoint_id]["request_count"], 1)
            self.assertGreater(evidence[endpoint_id]["requested_input_tokens"], 0)
            self.assertGreater(evidence[endpoint_id]["requested_output_tokens"], 0)
            self.assertEqual(evidence[endpoint_id]["latency_count"], 1)

    def test_class_reaches_pair_feedback_and_energy_stays_endpoint_local(self):
        harness = StageD1Harness.__new__(StageD1Harness)
        harness.endpoint_energy = {item: None for item in ENDPOINTS}
        harness.endpoint_energy_frequency = {item: None for item in ENDPOINTS}
        harness.endpoint_latency = {item: None for item in ENDPOINTS}
        harness.pair_feedback = {}
        harness.pair_feedback_frequencies = {}
        harness.latest_endpoint_evidence = {}
        request = normalize_d1_request(self.normalized_row())
        harness._update_runtime_feedback({
            "P0_energy_j": 100, "P1_energy_j": 200,
            "D0_energy_j": 300, "D1_energy_j": 400,
        }, [request])
        self.assertEqual(harness.endpoint_energy["P0"], 100)
        self.assertIsNone(harness.endpoint_energy["P1"])
        self.assertEqual(harness.endpoint_energy["D1"], 400)
        self.assertEqual(
            harness.pair_feedback["LL"]["P0->D1"]["observations"], 1
        )

    def test_mixed_frequency_window_energy_is_not_routing_eligible(self):
        harness = StageD1Harness.__new__(StageD1Harness)
        harness.endpoint_energy = {item: None for item in ENDPOINTS}
        harness.endpoint_energy_frequency = {item: None for item in ENDPOINTS}
        harness.endpoint_latency = {item: None for item in ENDPOINTS}
        harness.pair_feedback = {}
        harness.pair_feedback_frequencies = {}
        harness.latest_endpoint_evidence = {}
        request = normalize_d1_request(self.normalized_row())
        harness._update_runtime_feedback({
            "P0_energy_j": 100, "P1_energy_j": 200,
            "D0_energy_j": 300, "D1_energy_j": 400,
            "D_mid_window_transition": True,
            "P_mid_window_transition": False,
        }, [request], {"P0": 2520, "P1": 2520, "D0": 1500, "D1": 1500})
        self.assertEqual(harness.endpoint_energy["P0"], 100)
        self.assertIsNone(harness.endpoint_energy["D1"])
        self.assertIsNone(harness.endpoint_energy_frequency["D1"])
        self.assertEqual(harness.pair_feedback_frequencies, {})

    def test_requested_output_budget_drives_context_not_actual_output(self):
        request = normalize_d1_request(self.normalized_row(
            requested_output_len="512", observed_output_tokens="1",
        ))
        settings = {
            "context_inference": {
                "long_input_tokens": 512,
                "long_output_tokens": 256,
                "prefill_heavy_fraction": 0.75,
                "decode_heavy_fraction": 0.50,
            }
        }
        self.assertEqual(
            d1_context_signature("D", [request], settings),
            "D:long-output-high",
        )

    def test_endpoint_controllers_have_independent_search_state(self):
        harness = StageD1Harness.__new__(StageD1Harness)
        harness.settings = json.loads(
            self.experiment.read_text(encoding="utf-8")
        )["stage_d1"]
        controllers = harness._controllers_endpoint()
        controllers["P0"].state.candidate_frequency = 1890
        self.assertIsNone(controllers["P1"].state.candidate_frequency)
        self.assertIsNot(controllers["P0"].state, controllers["P1"].state)

    def test_warmup_traffic_is_not_controller_evidence(self):
        request = normalize_d1_request(self.normalized_row())
        ledger = D1EvidenceLedger()
        ledger.record([request], evidence_eligible=False)
        self.assertEqual(ledger.warmup_request_count, 1)
        self.assertEqual(ledger.control_requests, [])
        ledger.record([request], evidence_eligible=True)
        self.assertEqual(len(ledger.control_requests), 1)

    def test_report_generation_accepts_canonical_schema(self):
        request = normalize_d1_request(self.normalized_row())
        policy = "JOINT-ROUTING-PER-ENDPOINT"
        row = {
            "block_id": "b1", "window_index": 1,
            "total_gross_energy_j": 1000, "request_count": 1,
            "load_point": "HIGH-SUSTAINABLE", "policy": policy,
            "repeat": 1, "trace_signature": "trace",
            "window_hard_gate_valid": True, "completed_rate_rps": 0.2,
            "P0_energy_j": 100, "P1_energy_j": 200,
            "D0_energy_j": 300, "D1_energy_j": 400,
        }
        diagnostic = {
            "logical_request_id": "request-1",
            "selected_prefill_endpoint_id": "P0",
            "selected_decode_endpoint_id": "D1",
            "routing_context": {
                "request_class": "LL", "input_tokens": 2048,
                "requested_output_tokens": 512,
            },
            "routing_decision": {
                "reason": "LOWER_PRESSURE",
                "candidate_states": [],
                "selected_state": {"known_safe": True},
            },
        }
        with TemporaryDirectory() as temporary:
            harness = StageD1Harness.__new__(StageD1Harness)
            harness.run_dir = Path(temporary)
            harness.stage = "full"
            harness.decision_rows = []
            harness.safety_rows = []
            harness.action_rows = []
            harness.segment_rows = []
            harness.search_rows = []
            harness.preflight_warmup_rows = []
            harness.evidence_ledger = D1EvidenceLedger()
            harness.routing_diagnostic_rows = [diagnostic]
            harness._write_d1_reports(
                [row], [request], 1, 1, True, None
            )
            summary = json.loads(
                (Path(temporary) / "stage_d1_summary.json").read_text()
            )
            self.assertTrue(summary["request_context_before_routing_audit"])

    @staticmethod
    def final_search_rows(*, contradictory=False):
        rows = []
        frequencies = {"P0": 2520, "P1": 2520, "D0": 1500, "D1": 1500}
        for endpoint_id, frequency in frequencies.items():
            rows.append({
                "endpoint_id": endpoint_id,
                "current_frequency_mhz": frequency,
                "final_validated_safe_frequency_mhz": frequency,
                "candidate_frequency_mhz": None,
                "unsafe_bound_mhz": (
                    frequency if contradictory and endpoint_id == "D0" else None
                ),
                "validated_safe_frequencies_json": json.dumps([frequency]),
                "revoked_frequencies_json": json.dumps(
                    [frequency] if contradictory and endpoint_id == "D0" else []
                ),
            })
        return rows

    def test_preflight_acceptance_fails_below_either_slo_attainment(self):
        common = {
            "stage": "preflight", "run_error": None,
            "structural_hard_gates": True, "routing_evidence_gates": True,
            "final_safety_state_consistent": True,
            "restoration_valid": True,
        }
        self.assertFalse(d1_preflight_acceptance(
            **common, ttft_attainment=0.94, tpot_attainment=1.0,
        ))
        self.assertFalse(d1_preflight_acceptance(
            **common, ttft_attainment=1.0, tpot_attainment=0.94,
        ))
        self.assertTrue(d1_preflight_acceptance(
            **common, ttft_attainment=0.95, tpot_attainment=0.95,
        ))

    def test_preflight_acceptance_rejects_contradictory_final_safety_state(self):
        self.assertTrue(final_d1_safety_state_consistent(
            self.final_search_rows()
        ))
        self.assertFalse(final_d1_safety_state_consistent(
            self.final_search_rows(contradictory=True)
        ))
        self.assertFalse(d1_preflight_acceptance(
            stage="preflight", run_error=None,
            structural_hard_gates=True, routing_evidence_gates=True,
            ttft_attainment=1.0, tpot_attainment=1.0,
            final_safety_state_consistent=False,
            restoration_valid=True,
        ))


if __name__ == "__main__":
    unittest.main()
