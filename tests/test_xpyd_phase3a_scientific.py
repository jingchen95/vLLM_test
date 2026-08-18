import tempfile
import unittest
from pathlib import Path

from xpyd.phase3a_scientific import (
    comparison_stats,
    histogram_bracket,
    rankdata,
    segment_proxy_records,
    source_digest,
    write_csv,
)


class Phase3AScientificAnalysisTests(unittest.TestCase):
    def test_rankdata_uses_average_rank_for_ties(self):
        self.assertEqual(rankdata([30.0, 10.0, 10.0, 20.0]), [4.0, 1.5, 1.5, 3.0])

    def test_comparison_stats_report_known_identity(self):
        result = comparison_stats(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            evidence_level="unit_test",
            alignment="exact",
        )
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["mean_absolute_difference_ms"], 0.0)
        self.assertEqual(result["pearson_correlation"], 1.0)
        self.assertEqual(result["spearman_correlation"], 1.0)

    def test_histogram_bracket_distinguishes_exact_and_bounded_targets(self):
        boundaries = [0.1, 0.25, 0.5, 1.0, "+Inf"]
        self.assertEqual(histogram_bracket(boundaries, 0.3), (0.25, 0.5, False))
        self.assertEqual(histogram_bracket(boundaries, 0.5), (0.25, 0.5, True))

    def test_proxy_segmentation_is_monotonic_and_cardinality_checked(self):
        diagnostics = [
            {"request_id": "second", "timestamps_monotonic_s": {"request_received": 2.0}},
            {"request_id": "first", "timestamps_monotonic_s": {"request_received": 1.0}},
            {"request_id": "third", "timestamps_monotonic_s": {"request_received": 3.0}},
        ]
        segmented = segment_proxy_records(diagnostics, [("warmup", 1), ("probe", 2)])
        self.assertEqual(segmented["warmup"][0]["request_id"], "first")
        self.assertEqual(
            [record["request_id"] for record in segmented["probe"]],
            ["second", "third"],
        )
        with self.assertRaisesRegex(RuntimeError, "cardinality"):
            segment_proxy_records(diagnostics, [("too_many", 4)])

    def test_source_digest_excludes_analysis_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.txt").write_text("raw", encoding="utf-8")
            (root / "analysis").mkdir()
            (root / "analysis" / "report.md").write_text("first", encoding="utf-8")
            before = source_digest(root)
            (root / "analysis" / "report.md").write_text("second", encoding="utf-8")
            self.assertEqual(source_digest(root), before)
            (root / "raw.txt").write_text("changed", encoding="utf-8")
            self.assertNotEqual(source_digest(root), before)

    def test_csv_output_uses_reproducible_lf_line_endings(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.csv"
            write_csv(output, [{"name": "phase3a", "status": "PASS"}])
            self.assertEqual(
                output.read_bytes(),
                b"name,status\nphase3a,PASS\n",
            )


if __name__ == "__main__":
    unittest.main()
