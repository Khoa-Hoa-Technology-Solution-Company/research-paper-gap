import tempfile
import unittest
from pathlib import Path

import networkx as nx

from src.evidence_graph import (
    consolidate_evidence_candidates,
    limitation_tokens,
    split_limitation_dimensions,
)
from src.utils import save_jsonl


class EvidenceGraphTests(unittest.TestCase):
    def test_paraphrased_limitations_form_one_multi_source_cell(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            processed.mkdir()
            save_jsonl([
                {
                    "paperId": "p1", "title": "Paper one", "year": 2024,
                    "abstract": "The detector lacks inherent explainability.",
                },
                {
                    "paperId": "p2", "title": "Paper two", "year": 2025,
                    "abstract": "The IDS has limited interpretability.",
                },
            ], processed / "corpus_filtered.jsonl")
            graph = nx.MultiDiGraph()
            graph.add_node("detector", type="METHOD")
            graph.add_node("CICIDS", type="DATASET")
            graph.add_node("F1 score", type="METRIC")
            graph.add_edge(
                "detector", "CICIDS", relation="EVALUATES_ON", source_paper="p1"
            )
            graph.add_edge(
                "detector", "F1 score", relation="EVALUATES_ON", source_paper="p2"
            )
            candidates = [
                {
                    "subject": "detector",
                    "missing_capability": limitation,
                    "source_evidence": [{
                        "paper_id": paper_id, "year": year, "confidence": 0.9,
                        "evidence": evidence, "semantic_scope": "subject",
                    }],
                }
                for limitation, paper_id, year, evidence in (
                    (
                        "inherent explainability", "p1", 2024,
                        "The detector lacks inherent explainability.",
                    ),
                    (
                        "interpretability", "p2", 2025,
                        "The detector has limitations in interpretability.",
                    ),
                )
            ]
            config = {
                "project": {"domain": "intrusion detection"},
                "paths": {"processed_data": str(processed)},
                "gap_detection": {"evidence": {"cell_similarity_threshold": 0.72}},
                "gap_certification": {
                    "min_independent_limitation_sources": 2,
                    "min_full_text_characters": 1000,
                },
            }
            cells, evidence_graph = consolidate_evidence_candidates(
                graph, candidates, config
            )
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0]["supporting_paper_count"], 2)
            self.assertTrue(cells[0]["candidate_quality"][
                "certificate_readiness"
            ]["independent_sources_ready"])
            self.assertEqual(
                sum(
                    data.get("relation") == "REPORTS_LIMITATION"
                    for _, _, data in evidence_graph.edges(data=True)
                ),
                2,
            )

    def test_dependent_impact_clause_is_not_a_limitation_cell(self):
        parts = split_limitation_dimensions(
            "inherent explainability, limiting clinical adoption under regulations"
        )
        self.assertEqual(parts, ["inherent explainability"])

    def test_compound_zero_day_limitation_is_atomized_for_recurrence(self):
        parts = split_limitation_dimensions(
            "insufficient evaluation against encrypted and zero-day attacks"
        )
        self.assertIn("zero-day attacks", parts)

    def test_aliases_normalize_interpretability_to_explainability(self):
        self.assertEqual(
            limitation_tokens("model interpretability"), {"explainability"}
        )


if __name__ == "__main__":
    unittest.main()
