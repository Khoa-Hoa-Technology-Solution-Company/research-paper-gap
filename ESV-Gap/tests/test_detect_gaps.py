import unittest

import networkx as nx

from src.detect_gaps import (
    detect_missing_links_common_neighbors,
    detect_evidence_map_empty_cells,
    detect_evidence_gaps,
    detect_temporal_decay,
    limitation_evidence_semantics,
)


def temporal_config():
    return {
        "gap_detection": {
            "temporal": {
                "decay_threshold": 0.3,
                "lookback_years": 2,
                "publication_counts": {
                    2020: 10,
                    2021: 10,
                    2022: 10,
                    2023: 10,
                    2024: 10,
                    2025: 10,
                },
                "snapshot_date": "2026-07-20",
                "exclude_incomplete_final_year": True,
            }
        },
        "gap_validation": {},
    }


class TemporalDecayTests(unittest.TestCase):
    def test_typed_evidence_map_requires_source_disjoint_paths(self):
        graph = nx.MultiDiGraph()
        for node, node_type, papers in (
            ("graph decoder", "METHOD", ["p1", "p2"]),
            ("rare-symbol dataset", "DATASET", ["p3", "p4"]),
            ("bridge a", "CONCEPT", ["p1", "p3"]),
            ("bridge b", "CONCEPT", ["p2", "p4"]),
        ):
            graph.add_node(node, type=node_type, papers=papers)
        graph.add_edge("graph decoder", "bridge a", source_paper="p1")
        graph.add_edge("bridge a", "rare-symbol dataset", source_paper="p3")
        graph.add_edge("graph decoder", "bridge b", source_paper="p2")
        graph.add_edge("bridge b", "rare-symbol dataset", source_paper="p4")
        config = {
            "gap_detection": {"evidence_map": {
                "min_marginal_papers": 2,
                "max_entities_per_type": 10,
                "top_k_candidates": 10,
            }},
            "gap_validation": {"max_path_length": 4, "min_independent_paths": 2},
        }
        gaps = detect_evidence_map_empty_cells(graph, config)
        target = next(
            gap for gap in gaps
            if {gap["head"], gap["tail"]} == {
                "graph decoder", "rare-symbol dataset"
            }
        )
        self.assertEqual(target["detector"], "typed_evidence_map_empty_cell")
        self.assertEqual(target["evidence_map"]["observed_joint_paper_count"], 0)
        self.assertGreaterEqual(
            target["evidence_map"]["source_disjoint_path_count"], 2
        )

    def test_missing_links_are_restricted_to_research_frame_type_pairs(self):
        graph = nx.MultiDiGraph()
        graph.add_node("method a", type="METHOD")
        graph.add_node("bridge", type="CONCEPT")
        graph.add_node("dataset a", type="DATASET")
        graph.add_node("method b", type="METHOD")
        graph.add_edge("method a", "bridge")
        graph.add_edge("bridge", "dataset a")
        graph.add_edge("method b", "bridge")
        gaps = detect_missing_links_common_neighbors(graph, 20)
        pairs = {frozenset((gap["head"], gap["tail"])) for gap in gaps}
        self.assertIn(frozenset(("method a", "dataset a")), pairs)
        self.assertNotIn(frozenset(("method a", "method b")), pairs)

    def test_zero_to_zero_windows_are_not_reported_as_decay(self):
        graph = nx.MultiDiGraph()
        concept = "legacy database control"
        for index in range(3):
            graph.add_edge(
                concept,
                f"old evidence {index}",
                source_paper=f"old-{index}",
                year=2020,
            )
        self.assertEqual(detect_temporal_decay(graph, temporal_config()), [])

    def test_positive_baseline_followed_by_zero_is_decay(self):
        graph = nx.MultiDiGraph()
        concept = "declining database control"
        for year in (2022, 2023):
            for index in range(2):
                graph.add_edge(
                    concept,
                    f"evidence {year}-{index}",
                    source_paper=f"paper-{year}-{index}",
                    year=year,
                )
        graph.add_edge(
            "unrelated current topic",
            "current evidence",
            source_paper="current-2025",
            year=2025,
        )
        gaps = detect_temporal_decay(graph, temporal_config())
        self.assertEqual([gap["concept"] for gap in gaps], [concept])
        self.assertEqual(gaps[0]["decay_rate"], 1.0)


class EvidenceGapTests(unittest.TestCase):
    def test_common_limitation_wording_is_recognised(self):
        result = limitation_evidence_semantics(
            "GPT-4V struggles with multilingual scenarios and complex tasks.",
            "GPT-4V",
        )
        self.assertTrue(result["valid"])

    def test_pronoun_limitation_requires_source_context(self):
        evidence = "Specifically, it showed limitations on non-Latin languages."
        without_context = limitation_evidence_semantics(evidence, "GPT-4V")
        with_context = limitation_evidence_semantics(
            evidence,
            "GPT-4V",
            "We evaluate GPT-4V. Specifically, it showed limitations.",
        )
        self.assertFalse(without_context["valid"])
        self.assertTrue(with_context["valid"])
        self.assertEqual(
            with_context["reason"],
            "explicit_unresolved_problem_context_supported",
        )

    def test_statement_that_no_limitation_exists_is_rejected(self):
        result = limitation_evidence_semantics(
            "No explicit limitation is stated for the proposed method.",
            "proposed method",
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "explicit_absence_of_limitation")

    def test_negated_requirement_is_an_advantage_not_a_limitation(self):
        result = limitation_evidence_semantics(
            "TAP does not require a predefined expression grammar.",
            "TAP",
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "negated_requirement_is_not_a_limitation")

    def test_limitation_of_existing_methods_is_not_assigned_to_new_model(self):
        result = limitation_evidence_semantics(
            "Existing image-to-markup methods could not segment symbols.",
            "Graph-to-Graph model",
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "limitation_is_attributed_to_other_methods")

    def test_lacks_relations_keep_paper_level_evidence(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(
            "visual encoder",
            "robust handwritten expression recognition",
            relation="LACKS",
            source_paper="paper-1",
            year=2024,
            confidence=0.9,
            evidence=(
                "The visual encoder has limitations, including robustness on "
                "handwritten expressions."
            ),
        )
        config = {
            "gap_detection": {
                "evidence": {
                    "min_confidence": 0.65,
                    "require_limitation_cue": True,
                }
            }
        }
        gaps = detect_evidence_gaps(graph, config)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["supporting_paper_ids"], ["paper-1"])
        self.assertEqual(gaps[0]["type"], "evidence_gap")

    def test_low_confidence_lacks_relation_is_not_promoted(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(
            "method",
            "capability",
            relation="LACKS",
            source_paper="paper-1",
            confidence=0.3,
            evidence="The method lacks this capability.",
        )
        self.assertEqual(detect_evidence_gaps(graph, {"gap_detection": {}}), [])


if __name__ == "__main__":
    unittest.main()
