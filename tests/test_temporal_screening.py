import tempfile
import unittest
from pathlib import Path

import networkx as nx

from src.graph_analysis import (
    _benjamini_hochberg,
    build_graph,
    community_cut_edge_metrics,
    compute_temporal_decay,
    gml_export_copy,
)
from src.temporal_benchmark import run_temporal_signal_injection


class TemporalScreeningTests(unittest.TestCase):
    def test_shared_cut_edge_statistic_uses_internal_plus_cross_denominator(self):
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("a", "c")])
        metrics = community_cut_edge_metrics(graph, {"a": 0, "b": 0, "c": 1})
        left = next(item for item in metrics if item["community_id"] == 0)
        self.assertEqual(left["internal_edges"], 1)
        self.assertEqual(left["cross_edges"], 1)
        self.assertEqual(left["cut_edge_fraction"], 0.5)

    def test_benjamini_hochberg_returns_monotone_adjusted_values(self):
        adjusted = _benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
        self.assertEqual(adjusted, [0.02, 0.04, 0.04, 0.008])

    def test_zero_coverage_calendar_years_are_missing_not_zero_activity(self):
        graph = nx.DiGraph()
        graph.add_edge(
            "declining concept",
            "anchor concept",
            events=[
                {"year": 2018},
                {"year": 2018},
                {"year": 2019},
                {"year": 2022},
                {"year": 2023},
            ],
        )

        _, report = compute_temporal_decay(
            graph,
            cutoff_year=2023,
            return_report=True,
        )

        self.assertEqual(report["coverage"]["covered_years"], [2018, 2019, 2022, 2023])
        self.assertEqual(report["coverage"]["zero_coverage_years"], [2020, 2021])
        self.assertEqual(
            list(report["eligible_node_statistics"][0]["annual_share"]),
            ["2018", "2019", "2022", "2023"],
        )

    def test_temporal_signal_injection_is_deterministic_and_keeps_controls_separate(self):
        report = run_temporal_signal_injection(trials_per_condition=2, seed=7)
        controls = [row for row in report["summary"] if row["shape"] in {"stable", "increasing"}]
        self.assertEqual(len(controls), 2)
        self.assertTrue(all(row["interpretation"] == "false-positive rate control" for row in controls))
        self.assertTrue(all(row["detections"] == 0 for row in controls))

    def test_gml_export_preserves_nullable_event_provenance_as_text(self):
        graph = build_graph([{
            "subject": "method a",
            "subject_type": "METHOD",
            "relation": "USES",
            "object": "dataset b",
            "object_type": "DATASET",
            "year": 2024,
            "confidence": 0.9,
            "paper_id": None,
            "chunk_id": "paper-1:abstract:0",
        }])

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "knowledge_graph.gml"
            nx.write_gml(gml_export_copy(graph), target)
            self.assertTrue(target.is_file())
            exported = nx.read_gml(target)

        self.assertIsNone(graph["method a"]["dataset b"]["events"][0]["paper_id"])
        self.assertIn("events", exported["method a"]["dataset b"])


if __name__ == "__main__":
    unittest.main()
