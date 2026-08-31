import unittest

import networkx as nx

from src.synthesise_research_gap import evaluate_synthesis_candidate


def synthesis_config():
    return {
        "project": {"domain": "handwritten mathematical expression recognition"},
        "gap_validation": {"snapshot_date": "2026-08-23"},
        "gap_synthesis": {
            "min_screened_corpus_size": 30,
            "min_signal_families": 2,
            "min_graph_signal_families": 1,
            "signal_similarity_threshold": 0.60,
            "max_direct_documents": 2,
            "min_external_queries": 2,
        },
    }


def certified_synthesis_config():
    config = synthesis_config()
    config["gap_certification"] = {
        "enabled": True,
        "min_independent_limitation_sources": 2,
        "min_source_text_level": "full_text",
        "min_full_text_characters": 1000,
        "min_external_queries": 3,
        "require_all_queries_completed": True,
    }
    return config


def completed_closure_record():
    return {
        "performed": True,
        "completed_query_count": 3,
        "closure_hits": [],
        "queries": [
            {"query": f"q{index}", "performed": True}
            for index in range(1, 4)
        ],
    }


def documents(count=30):
    items = [{
        "paperId": "p1",
        "title": "Baseline recognizer limitations",
        "abstract": "The baseline recognizer remains unable to recognize rare symbols.",
    }]
    items.extend({
        "paperId": f"p{index}",
        "title": f"Unrelated study {index}",
        "abstract": "A different handwriting topic and experiment.",
    } for index in range(2, count + 1))
    return items


class ResearchGapSynthesisTests(unittest.TestCase):
    def _certificate_candidate_and_graph(self):
        graph = nx.MultiDiGraph()
        graph.add_node("baseline recognizer", type="METHOD")
        graph.add_node("robustness to rare symbols", type="CONCEPT")
        graph.add_node("CROHME 2023", type="DATASET")
        graph.add_node("expression recognition accuracy", type="METRIC")
        graph.add_edge(
            "baseline recognizer", "robustness to rare symbols",
            relation="LACKS", source_paper="p1",
        )
        graph.add_edge(
            "baseline recognizer", "CROHME 2023",
            relation="EVALUATES_ON", source_paper="p1",
        )
        graph.add_edge(
            "rare symbol robustness", "low-frequency notation",
            relation="RELATED_TO", source_paper="p3",
        )
        candidate = {
            "type": "evidence_gap",
            "subject": "baseline recognizer",
            "missing_capability": "robustness to rare symbols",
            "supporting_paper_ids": ["p1", "p2"],
            "mean_evidence_confidence": 0.9,
            "source_evidence": [
                {
                    "paper_id": paper_id,
                    "confidence": 0.9,
                    "evidence": (
                        "The baseline recognizer remains unable to recognize rare symbols."
                    ),
                }
                for paper_id in ("p1", "p2")
            ],
            "validation": {
                "closure_corpus_available": True,
                "closure_hits": [],
                "metrics": {"stability": 0.9, "specificity": 0.9},
                "external_closure_search": completed_closure_record(),
            },
        }
        return candidate, graph

    def _full_text_documents(self):
        records = documents()
        evidence = (
            "The baseline recognizer remains unable to recognize rare symbols. "
        )
        for record in records[:2]:
            record["full_text"] = evidence + ("supporting experimental detail " * 80)
        return records

    def test_full_text_two_source_candidate_receives_corpus_bounded_certificate(self):
        candidate, graph = self._certificate_candidate_and_graph()
        result = evaluate_synthesis_candidate(
            candidate,
            {
                "orphan_clusters": [{
                    "members": ["rare symbol robustness", "low-frequency notation"],
                    "inter_edge_ratio": 0.0,
                }],
                "missing_links": [], "temporal_decay": [],
            },
            graph,
            self._full_text_documents(),
            certified_synthesis_config(),
        )
        self.assertTrue(result["decision"]["passed"])
        self.assertTrue(result["gap_certificate"]["passed"])
        self.assertEqual(result["status"], "certified_corpus_bounded_research_gap")

    def test_abstract_only_evidence_is_not_certified(self):
        candidate, graph = self._certificate_candidate_and_graph()
        result = evaluate_synthesis_candidate(
            candidate,
            {"orphan_clusters": [{
                "members": ["rare symbol robustness", "low-frequency notation"],
                "inter_edge_ratio": 0.0,
            }], "missing_links": [], "temporal_decay": []},
            graph,
            documents(),
            certified_synthesis_config(),
        )
        self.assertTrue(result["decision"]["passed"])
        self.assertFalse(result["gap_certificate"]["passed"])
        self.assertIn(
            "source_text_evidence_sufficient",
            result["gap_certificate"]["failed_gates"],
        )
        self.assertEqual(result["status"], "evidence_cleared_hypothesis_not_certified")

    def test_one_limitation_source_is_not_certified(self):
        candidate, graph = self._certificate_candidate_and_graph()
        candidate["supporting_paper_ids"] = ["p1"]
        candidate["source_evidence"] = candidate["source_evidence"][:1]
        result = evaluate_synthesis_candidate(
            candidate,
            {"orphan_clusters": [{
                "members": ["rare symbol robustness", "low-frequency notation"],
                "inter_edge_ratio": 0.0,
            }], "missing_links": [], "temporal_decay": []},
            graph,
            self._full_text_documents(),
            certified_synthesis_config(),
        )
        self.assertTrue(result["decision"]["passed"])
        self.assertFalse(result["gap_certificate"]["passed"])
        self.assertIn(
            "independent_limitation_sources",
            result["gap_certificate"]["failed_gates"],
        )

    def test_convergent_answerable_candidate_is_accepted_without_human_input(self):
        graph = nx.MultiDiGraph()
        graph.add_node("baseline recognizer", type="METHOD")
        graph.add_node("robustness to rare symbols", type="CONCEPT")
        graph.add_node("CROHME 2023", type="DATASET")
        graph.add_node("expression recognition accuracy", type="METRIC")
        graph.add_edge(
            "baseline recognizer",
            "robustness to rare symbols",
            relation="LACKS",
            source_paper="p1",
        )
        graph.add_edge(
            "baseline recognizer",
            "CROHME 2023",
            relation="EVALUATES_ON",
            source_paper="p1",
        )
        graph.add_edge(
            "baseline recognizer",
            "expression recognition accuracy",
            relation="EVALUATES_ON",
            source_paper="p1",
        )
        graph.add_edge(
            "rare symbol robustness",
            "low-frequency notation",
            relation="RELATED_TO",
            source_paper="p2",
        )
        candidate = {
            "type": "evidence_gap",
            "subject": "baseline recognizer",
            "missing_capability": "robustness to rare symbols",
            "supporting_paper_ids": ["p1"],
            "mean_evidence_confidence": 0.9,
            "source_evidence": [{
                "paper_id": "p1",
                "confidence": 0.9,
                "evidence": (
                    "The baseline recognizer remains unable to recognize rare symbols."
                ),
            }],
            "validation": {
                "closure_corpus_available": True,
                "closure_hits": [],
                "metrics": {"stability": 0.9, "specificity": 0.9},
                "external_closure_search": {
                    "performed": True,
                    "completed_query_count": 2,
                    "closure_hits": [],
                },
            },
        }
        raw = {
            "orphan_clusters": [{
                "members": ["rare symbol robustness", "low-frequency notation"],
                "inter_edge_ratio": 0.0,
            }],
            "missing_links": [],
            "temporal_decay": [],
        }
        result = evaluate_synthesis_candidate(
            candidate, raw, graph, documents(), synthesis_config()
        )
        self.assertTrue(result["decision"]["passed"])
        self.assertEqual(result["status"], "automatically_supported_research_gap")
        self.assertTrue(result["answerable"])
        self.assertIn("CROHME 2023", result["research_question"])

    def test_same_source_orphan_and_corpus_scarcity_do_not_fake_convergence(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(
            "baseline recognizer", "rare symbol robustness",
            relation="LACKS", source_paper="p1",
        )
        graph.add_edge(
            "rare symbol robustness", "low-frequency notation",
            relation="RELATED_TO", source_paper="p1",
        )
        candidate = {
            "type": "evidence_gap",
            "subject": "baseline recognizer",
            "missing_capability": "rare symbol robustness",
            "supporting_paper_ids": ["p1"],
            "source_evidence": [{
                "paper_id": "p1",
                "evidence": "The baseline recognizer remains unable to recognize rare symbols.",
            }],
            "validation": {
                "closure_corpus_available": True,
                "closure_hits": [],
                "metrics": {"stability": 0.9, "specificity": 0.9},
                "external_closure_search": {
                    "performed": True, "completed_query_count": 2,
                    "closure_hits": [],
                },
            },
        }
        raw = {
            "orphan_clusters": [{
                "members": ["rare symbol robustness", "low-frequency notation"]
            }],
            "missing_links": [], "temporal_decay": [],
        }
        result = evaluate_synthesis_candidate(
            candidate, raw, graph, documents(), synthesis_config()
        )
        self.assertFalse(result["decision"]["passed"])
        self.assertFalse(result["decision"]["hard_gates"]["multi_signal_convergence"])
        self.assertIn("same_source_structural_context", result["convergent_signals"])
        self.assertIn("corpus_scarcity", result["convergent_signals"])

    def test_generic_member_and_unrelated_cluster_paper_do_not_corroborate(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(
            "existing solutions", "high-performance encryption",
            relation="LACKS", source_paper="p1",
        )
        graph.add_edge(
            "storage engine", "performance",
            relation="IMPROVES", source_paper="p2",
        )
        graph.add_edge(
            "unrelated node", "another node",
            relation="RELATED_TO", source_paper="p3",
        )
        candidate = {
            "type": "evidence_gap", "subject": "existing solutions",
            "missing_capability": "high-performance encryption",
            "supporting_paper_ids": ["p1"],
            "source_evidence": [{
                "paper_id": "p1",
                "evidence": "Existing solutions lack high-performance encryption.",
            }],
            "validation": {
                "closure_corpus_available": True, "closure_hits": [],
                "metrics": {"stability": 0.9, "specificity": 0.9},
                "external_closure_search": {
                    "performed": True, "completed_query_count": 2,
                    "closure_hits": [],
                },
            },
        }
        raw = {
            "orphan_clusters": [{
                "members": ["performance", "unrelated node", "another node"]
            }],
            "missing_links": [], "temporal_decay": [],
        }
        result = evaluate_synthesis_candidate(
            candidate, raw, graph, documents(), synthesis_config()
        )
        self.assertFalse(result["decision"]["hard_gates"]["multi_signal_convergence"])
        self.assertNotIn("structural_undercoverage", result["convergent_signals"])

    def test_problem_already_addressed_by_source_is_rejected(self):
        graph = nx.MultiDiGraph()
        graph.add_node("pattern generation strategies", type="METHOD")
        graph.add_node("large training data", type="DATASET")
        graph.add_node("lack of large training data", type="CONCEPT")
        graph.add_edge(
            "pattern generation strategies",
            "large training data",
            relation="LACKS",
            source_paper="p1",
        )
        graph.add_edge(
            "pattern generation strategies",
            "lack of large training data",
            relation="ADDRESSES",
            source_paper="p1",
        )
        candidate = {
            "type": "evidence_gap",
            "subject": "pattern generation strategies",
            "missing_capability": "large training data",
            "supporting_paper_ids": ["p1"],
            "source_evidence": [{
                "paper_id": "p1",
                "evidence": "The lack of large training data is a serious issue.",
            }],
            "validation": {
                "closure_corpus_available": True,
                "closure_hits": [],
                "metrics": {"stability": 0.9, "specificity": 0.9},
                "external_closure_search": {
                    "performed": True,
                    "completed_query_count": 2,
                    "closure_hits": [],
                },
            },
        }
        raw = {
            "orphan_clusters": [{"members": ["large-scale dataset"]}],
            "missing_links": [],
            "temporal_decay": [],
        }
        result = evaluate_synthesis_candidate(
            candidate, raw, graph, documents(), synthesis_config()
        )
        self.assertFalse(result["decision"]["passed"])
        self.assertFalse(
            result["decision"]["hard_gates"]["not_already_resolved"]
        )
        self.assertTrue(result["source_resolution_edges"])

    def test_source_abstract_solution_clause_blocks_missed_triple_resolution(self):
        """A source can solve its own limitation via a different method node."""
        graph = nx.MultiDiGraph()
        graph.add_node("existing healthcare IoT IDS", type="METHOD")
        graph.add_node("inherent explainability", type="CONCEPT")
        graph.add_edge(
            "existing healthcare IoT IDS", "inherent explainability",
            relation="LACKS", source_paper="p1",
        )
        candidate = {
            "type": "evidence_gap",
            "subject": "existing healthcare IoT IDS",
            "missing_capability": "inherent explainability",
            "supporting_paper_ids": ["p1"],
            "source_evidence": [{
                "paper_id": "p1",
                "evidence": "Existing IDS lack inherent explainability.",
            }],
            "validation": {
                "closure_corpus_available": True, "closure_hits": [],
                "domain_relevant": True, "problem_relevant": True,
                "metrics": {
                    "stability": 0.9, "specificity": 0.9,
                    "candidate_problem_relevance": 0.9,
                },
                "external_closure_search": {
                    "performed": True, "completed_query_count": 2,
                    "closure_hits": [],
                },
            },
        }
        source_documents = [{
            "paperId": "p1",
            "title": "Explainable healthcare IoT IDS",
            "abstract": (
                "Existing IDS lack inherent explainability. This paper presents "
                "an explainable attention network that embeds interpretability "
                "directly into the detection mechanism."
            ),
        }]
        result = evaluate_synthesis_candidate(
            candidate,
            {"orphan_clusters": [], "missing_links": [], "temporal_decay": []},
            graph,
            source_documents + documents(29),
            synthesis_config(),
        )
        self.assertFalse(result["decision"]["passed"])
        self.assertFalse(result["decision"]["hard_gates"]["not_already_resolved"])
        self.assertTrue(result["source_resolution_documents"])

    def test_low_recorded_problem_relevance_cannot_pass_synthesis_gate(self):
        graph = nx.MultiDiGraph()
        candidate = {
            "type": "evidence_gap",
            "subject": "traditional IDS methods",
            "missing_capability": "zero-day attack detection",
            "supporting_paper_ids": ["p1"],
            "source_evidence": [{
                "paper_id": "p1",
                "evidence": "Traditional IDS methods struggle with zero-day attacks.",
            }],
            "validation": {
                "closure_corpus_available": True, "closure_hits": [],
                "domain_relevant": True, "problem_relevant": True,
                "metrics": {
                    "stability": 0.9, "specificity": 0.9,
                    "candidate_problem_relevance": 0.3158,
                },
                "external_closure_search": {
                    "performed": True, "completed_query_count": 2,
                    "closure_hits": [],
                },
            },
        }
        result = evaluate_synthesis_candidate(
            candidate,
            {"orphan_clusters": [], "missing_links": [], "temporal_decay": []},
            graph,
            documents(),
            synthesis_config(),
        )
        self.assertFalse(result["decision"]["passed"])
        self.assertFalse(
            result["decision"]["hard_gates"]["candidate_problem_aligned_with_domain"]
        )

    def test_source_disjoint_empty_evidence_cell_can_form_gap(self):
        graph = nx.MultiDiGraph()
        graph.add_node("graph neural decoder", type="METHOD")
        graph.add_node("rare notation robustness", type="CONCEPT")
        graph.add_node("CROHME 2023", type="DATASET")
        graph.add_node("expression recognition accuracy", type="METRIC")
        graph.add_edge(
            "graph neural decoder", "CROHME 2023",
            relation="EVALUATES_ON", source_paper="p1",
        )
        graph.add_edge(
            "rare notation robustness", "expression recognition accuracy",
            relation="EVALUATES_ON", source_paper="p2",
        )
        candidate = {
            "type": "missing_link",
            "head": "graph neural decoder",
            "tail": "rare notation robustness",
            "prediction_score": 8.0,
            "validation": {
                "existing_direct_edge": False,
                "supporting_paper_ids": ["p1", "p2"],
                "independent_evidence_path_count": 2,
                "independent_evidence_paths": [
                    {"nodes": ["graph neural decoder", "bridge a", "rare notation robustness"], "papers": ["p1"]},
                    {"nodes": ["graph neural decoder", "bridge b", "rare notation robustness"], "papers": ["p2"]},
                ],
                "closure_corpus_available": True,
                "closure_hits": [],
                "metrics": {"stability": 0.9, "specificity": 0.9},
                "external_closure_search": {
                    "performed": True,
                    "completed_query_count": 2,
                    "closure_hits": [],
                },
            },
        }
        raw = {"orphan_clusters": [], "missing_links": [candidate], "temporal_decay": []}
        result = evaluate_synthesis_candidate(
            candidate, raw, graph, documents(), synthesis_config()
        )
        self.assertTrue(result["decision"]["passed"])
        self.assertEqual(
            result["semantic_audit"]["reason"],
            "source_disjoint_empty_evidence_cell",
        )
        self.assertIn("direct relationship", result["claim"])


if __name__ == "__main__":
    unittest.main()
