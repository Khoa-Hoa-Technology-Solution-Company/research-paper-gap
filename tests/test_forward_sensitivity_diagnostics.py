import unittest

from src.forward_sensitivity_diagnostics import (
    paper_normalized_temporal_diagnostic,
    structural_gate_diagnostic,
)
from src.graph_analysis import build_graph


class ForwardSensitivityDiagnosticTests(unittest.TestCase):
    def test_structural_profiles_do_not_change_the_frozen_gate(self):
        graph = build_graph([
            {
                "subject": "a", "subject_type": "CONCEPT", "relation": "USES",
                "object": "b", "object_type": "CONCEPT", "year": 2020, "confidence": 0.9,
            },
            {
                "subject": "b", "subject_type": "CONCEPT", "relation": "USES",
                "object": "c", "object_type": "CONCEPT", "year": 2021, "confidence": 0.9,
            },
        ])
        report = structural_gate_diagnostic(graph)
        self.assertEqual(report["frozen_gate"]["minimum_global_size_ratio"], 0.05)
        self.assertIn("exploratory_absolute_10_nodes", report["profiles"])

    def test_paper_normalization_counts_unique_supporting_papers(self):
        graph = build_graph([
            {
                "subject": "topic", "subject_type": "CONCEPT", "relation": "USES",
                "object": "anchor", "object_type": "CONCEPT", "year": year,
                "confidence": 0.9, "paper_id": f"p-{year}",
            }
            for year in (2018, 2019, 2020, 2021, 2022)
        ])
        report = paper_normalized_temporal_diagnostic(graph, cutoff_year=2022)
        self.assertEqual(report["missing_paper_id_events"], 0)
        self.assertEqual(report["coverage"]["unique_papers_by_year"]["2018"], 1)


if __name__ == "__main__":
    unittest.main()
