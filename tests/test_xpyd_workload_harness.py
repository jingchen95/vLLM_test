"""CPU-only tests for realistic mixed-stream workload definitions."""

import json
from pathlib import Path
import tempfile
import unittest

from xpyd.workload_harness import (
    REQUEST_CLASSES,
    RequestBuckets,
    WorkloadHarnessError,
    build_trace,
    load_config,
    offered_window_summaries,
    write_trace,
)


class WorkloadHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_config(
            cls.root / "paper/configs/xpyd_stageb_workload_harness.json"
        )

    def test_explicit_bucket_boundaries_match_all_four_shapes(self):
        buckets = RequestBuckets(**self.config["bucket_definition"])
        observed = {
            key: buckets.classify(value["input_len"], value["output_len"])
            for key, value in self.config["shapes"].items()
        }
        self.assertEqual(observed, {key: key for key in REQUEST_CLASSES})

    def test_dynamic_trace_is_mixed_labeled_and_arrival_ordered(self):
        rows = build_trace(
            self.config, "dynamic_mixed_burst", "stageb-fixture"
        )
        self.assertEqual(len(rows), 80)
        self.assertEqual(len({row["request_id"] for row in rows}), 80)
        self.assertEqual(
            [row["arrival_time_s"] for row in rows],
            sorted(row["arrival_time_s"] for row in rows),
        )
        for segment in {row["segment"] for row in rows}:
            classes = {
                row["workload_class"] for row in rows if row["segment"] == segment
            }
            self.assertGreater(len(classes), 1)

    def test_offered_windows_retain_mix_and_token_totals(self):
        rows = build_trace(self.config, "stationary_mixed")
        windows = offered_window_summaries(
            rows, self.config["control_window_s"]
        )
        self.assertEqual(sum(row["logical_request_count"] for row in windows), 32)
        for window in windows:
            self.assertAlmostEqual(sum(window["class_fractions"].values()), 1.0)
            self.assertGreater(window["input_tokens"], 0)
            self.assertGreater(window["requested_output_tokens"], 0)

    def test_trace_round_trip_preserves_context_columns(self):
        rows = build_trace(self.config, "stationary_mixed", "roundtrip")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            write_trace(path, rows)
            text = path.read_text(encoding="utf-8")
        self.assertIn("workload_class,segment", text.splitlines()[0])
        self.assertIn("roundtrip-0000", text)

    def test_invalid_shape_classification_fails_closed(self):
        config = json.loads(json.dumps(self.config))
        config["shapes"]["SS"]["input_len"] = 2048
        with self.assertRaisesRegex(WorkloadHarnessError, "classifies as"):
            build_trace(config, "stationary_mixed")


if __name__ == "__main__":
    unittest.main()
