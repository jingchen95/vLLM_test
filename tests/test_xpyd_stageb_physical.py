"""CPU-only contract tests for the fail-closed Stage B physical sequence."""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

from xpyd.stageb_physical import audit_stage, load_configs
from xpyd.workload_harness import build_trace


class StageBPhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        old = dict(os.environ)
        os.environ.update({
            "XPYD_PROXY_URI": "http://127.0.0.1:8090",
            "XPYD_PROXY_DIAGNOSTICS_LOG": "/tmp/proxy.jsonl",
            "XPYD_ROUTING_CONTROL_FILE": "",
            "XPYD_P0_HTTP_HOST": "127.0.0.1",
            "XPYD_P1_HTTP_HOST": "127.0.0.1",
            "XPYD_D0_HTTP_HOST": "127.0.0.1",
            "XPYD_D1_HTTP_HOST": "127.0.0.1",
            "XPYD_P_ADDR_HOST": "10.0.0.1",
            "XPYD_D_ADDR_HOST": "10.0.0.2",
        })
        try:
            cls.runtime, cls.workload = load_configs(
                cls.root / "paper/configs/xpyd_phase4b_evaluation_neptune_io.json",
                cls.root / "paper/configs/xpyd_stageb_workload_harness.json",
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

    def _fixture(self, directory, stage, rows):
        stage_dir = Path(directory) / stage["id"]
        (stage_dir / "client").mkdir(parents=True)
        (stage_dir / "audit.json").write_text(
            json.dumps({"valid": True, "hard_gates": {"fixture": True}}),
            encoding="utf-8",
        )
        (stage_dir / "client" / "summary.json").write_text(json.dumps({
            "maximum_client_waiting_requests": 3,
            "maximum_client_in_flight_requests": stage["max_concurrency"],
            "mean_client_admission_delay_ms": 2.0,
            "p99_client_admission_delay_ms": 4.0,
            "workload_class_counts": {"SS": 8, "SL": 8, "LS": 8, "LL": 8},
        }), encoding="utf-8")
        fields = (
            "request_id", "workload_class", "segment",
            "client_admission_delay_ms",
        )
        with (stage_dir / "requests.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "request_id": row["request_id"],
                    "workload_class": row["workload_class"],
                    "segment": row["segment"],
                    "client_admission_delay_ms": 1.0,
                })
        with (stage_dir / "client" / "load_events.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for row in rows:
                for event in ("TRACE_ARRIVAL", "CLIENT_ADMIT", "REQUEST_COMPLETE"):
                    final = row is rows[-1] and event == "REQUEST_COMPLETE"
                    stream.write(json.dumps({
                        "event": event,
                        "request_id": row["request_id"],
                        "client_waiting_requests": 0 if final else 1,
                        "client_in_flight_requests": 0 if final else 1,
                    }) + "\n")
        return stage_dir

    def test_matrix_is_fixed_high_round_robin_and_gated_2_then_4(self):
        stages = [
            item for item in self.workload["proposed_validation_matrix"]
            if item.get("hardware")
        ]
        self.assertEqual([item["max_concurrency"] for item in stages], [2, 4, 4])
        self.assertTrue(all(item["control"] == "disabled" for item in stages))
        self.assertTrue(all(item["frequencies"] == "all validated HIGH" for item in stages))

    def test_valid_stage_audit_requires_both_client_and_proxy_load_evidence(self):
        stage = next(
            item for item in self.workload["proposed_validation_matrix"]
            if item.get("id") == "B1_concurrency_2_smoke"
        )
        rows = build_trace(self.workload, stage["scenario"], "fixture")
        with tempfile.TemporaryDirectory() as directory:
            stage_dir = self._fixture(directory, stage, rows)
            audit = audit_stage(
                stage, rows, stage_dir,
                {
                    "accepted_requests_total": 5,
                    "completed_requests_total": 5,
                    "failed_requests_total": 0,
                },
                {
                    "accepted_requests_total": 5 + len(rows),
                    "completed_requests_total": 5 + len(rows),
                    "failed_requests_total": 0,
                    "in_flight_requests": 0,
                    "maximum_in_flight_requests": 2,
                },
            )
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["hard_gates"]["exact_request_context_labels"])
        self.assertTrue(audit["hard_gates"]["proxy_request_accounting"])

    def test_label_mismatch_fails_closed(self):
        stage = next(
            item for item in self.workload["proposed_validation_matrix"]
            if item.get("id") == "B1_concurrency_2_smoke"
        )
        rows = build_trace(self.workload, stage["scenario"], "fixture")
        with tempfile.TemporaryDirectory() as directory:
            stage_dir = self._fixture(directory, stage, rows)
            with (stage_dir / "requests.csv").open(encoding="utf-8") as stream:
                request_rows = list(csv.DictReader(stream))
            request_rows[0]["workload_class"] = "INVALID"
            with (stage_dir / "requests.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(request_rows[0]))
                writer.writeheader()
                writer.writerows(request_rows)
            audit = audit_stage(
                stage, rows, stage_dir,
                {"accepted_requests_total": 0, "completed_requests_total": 0, "failed_requests_total": 0},
                {
                    "accepted_requests_total": len(rows),
                    "completed_requests_total": len(rows),
                    "failed_requests_total": 0,
                    "in_flight_requests": 0,
                    "maximum_in_flight_requests": 2,
                },
            )
        self.assertFalse(audit["valid"])
        self.assertFalse(audit["hard_gates"]["exact_request_context_labels"])

    def test_launcher_has_dedicated_stageb_mode(self):
        source = (self.root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("XPYD_STAGEB_RUNTIME_CONFIG", source)
        self.assertIn("-m xpyd.stageb_physical", source)


if __name__ == "__main__":
    unittest.main()
