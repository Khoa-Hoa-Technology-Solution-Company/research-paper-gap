import unittest

import networkx as nx

from src.validate_gaps import (
    closure_hits,
    independent_evidence_paths,
    specificity_score,
    validate_candidate,
)
from experiments.run_corpus_validation import build_graph


def validation_config():
    return {
        "gap_validation": {
            "min_supporting_papers": 2,
            "min_independent_paths": 2,
            "min_surviving_paths": 1,
            "max_path_length": 4,
            "min_specificity": 0.55,
            "bootstrap_repeats": 60,
            "edge_keep_probability": 0.8,
            "paper_keep_probability": 0.8,
            "plausible_edge_add_probability": 0.5,
            "min_stability": 0.7,
            "orphan_isolation_threshold": 0.8,
            "temporal_decay_threshold": 0.3,
            "temporal_lookback_years": 2,
            "closure_token_coverage": 0.6,
            "random_seed": 42,
            "max_closure_hits_to_record": 10,
            "weights": {
                "provenance": 0.25,
                "specificity": 0.2,
                "path_diversity": 0.2,
                "stability": 0.3,
                "closure_clearance": 0.1,
            },
        }
    }


def supported_bridge_graph(include_direct_edge=False):
    graph = nx.MultiDiGraph()
    graph.add_node("runtime anomaly detection", papers=["p1", "p2"])
    graph.add_node("service mesh telemetry", papers=["p3", "p4"])
    graph.add_node("distributed tracing", papers=["p1", "p3"])
    graph.add_node("policy enforcement", papers=["p2", "p4"])
    graph.add_edge("runtime anomaly detection", "distributed tracing", source_paper="p1", year=2022)
    graph.add_edge("distributed tracing", "service mesh telemetry", source_paper="p3", year=2023)
    graph.add_edge("runtime anomaly detection", "policy enforcement", source_paper="p2", year=2022)
    graph.add_edge("policy enforcement", "service mesh telemetry", source_paper="p4", year=2024)
    if include_direct_edge:
        graph.add_edge("runtime anomaly detection", "service mesh telemetry", source_paper="p5", year=2025)
    return graph


class GapValidationTests(unittest.TestCase):
    def test_specificity_penalises_placeholders(self):
        self.assertEqual(specificity_score(["proposed framework"]), 0.0)
        self.assertGreater(specificity_score(["service mesh telemetry"]), 0.8)

    def test_missing_plausible_pool_fails_closed_to_review(self):
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
            "prediction_score": 4.0,
        }
        result = validate_candidate(supported_bridge_graph(), candidate, validation_config(), [])
        self.assertEqual(result["status"], "review_required")
        self.assertIn("plausible_edge_stress_unavailable", result["reasons"])
        self.assertIsNone(result["bootstrap"]["mode_survival"]["plausible_edge_addition"])
        self.assertFalse(result["bootstrap"]["mode_available"]["plausible_edge_addition"])
        self.assertGreaterEqual(result["metrics"]["stability"], 0.7)
        self.assertGreaterEqual(result["supporting_paper_count"], 2)

    def test_irrelevant_plausible_edge_does_not_count_as_stress_test(self):
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
            "plausible_edges": [{"head": "unrelated", "tail": "other"}],
        }
        result = validate_candidate(supported_bridge_graph(), candidate, validation_config(), [])
        self.assertEqual(result["status"], "review_required")
        self.assertEqual(result["bootstrap"]["plausible_stress_edge_count"], 0)

    def test_relevant_plausible_edge_is_evaluated_not_assumed_survived(self):
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
            "plausible_edges": [{
                "head": "runtime anomaly detection",
                "tail": "service mesh telemetry",
            }],
        }
        result = validate_candidate(supported_bridge_graph(), candidate, validation_config(), [])
        addition = result["bootstrap"]["mode_survival"]["plausible_edge_addition"]
        self.assertIsNotNone(addition)
        self.assertLess(addition, 1.0)
        self.assertTrue(result["bootstrap"]["mode_available"]["plausible_edge_addition"])

    def test_explicitly_completed_empty_search_is_evidence_of_no_closing_edge(self):
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
            "plausible_edges": [],
            "plausible_edge_search_performed": True,
        }
        result = validate_candidate(supported_bridge_graph(), candidate, validation_config(), [])
        self.assertEqual(result["status"], "automatically_eligible")
        self.assertEqual(result["bootstrap"]["mode_survival"]["plausible_edge_addition"], 1.0)
        self.assertNotIn("plausible_edge_stress_unavailable", result["reasons"])

    def test_existing_link_is_routed_to_review(self):
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
        }
        result = validate_candidate(supported_bridge_graph(True), candidate, validation_config(), [])
        self.assertEqual(result["status"], "review_required")
        self.assertIn("observed_relation_requires_qualified_review", result["reasons"])

    def test_single_path_is_rejected_even_when_edge_survival_is_high(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("runtime anomaly detection", "bridge", source_paper="p1", year=2022)
        graph.add_edge("bridge", "service mesh telemetry", source_paper="p2", year=2023)
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
        }
        result = validate_candidate(graph, candidate, validation_config(), [])
        self.assertEqual(len(independent_evidence_paths(graph, candidate["head"], candidate["tail"])), 1)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("insufficient_source_disjoint_evidence_paths", result["reasons"])

    def test_path_provenance_does_not_count_unrelated_incident_papers(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("head", "bridge", source_paper="shared")
        graph.add_edge("bridge", "tail", source_paper="shared")
        graph.add_edge("head", "unrelated-a", source_paper="p2")
        graph.add_edge("tail", "unrelated-b", source_paper="p3")
        candidate = {"type": "missing_link", "head": "head", "tail": "tail"}
        result = validate_candidate(graph, candidate, validation_config(), [])
        self.assertEqual(result["supporting_paper_ids"], ["shared"])
        self.assertEqual(result["status"], "rejected")

    def test_local_closure_routes_candidate_to_review(self):
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
        }
        documents = [{
            "paper_id": "known-work",
            "title": "Runtime anomaly detection using service mesh telemetry",
            "abstract": "We connect both techniques in production microservices.",
        }]
        self.assertEqual(len(closure_hits(candidate, documents)), 1)
        result = validate_candidate(supported_bridge_graph(), candidate, validation_config(), documents)
        self.assertEqual(result["status"], "review_required")

    def test_type_suffix_does_not_hide_local_coverage(self):
        candidate = {
            "type": "missing_link",
            "head": "JWT [METHOD]",
            "tail": "OAuth",
        }
        documents = [{
            "paperId": "known-work",
            "title": "Applying OAuth2 and JWT protocols",
            "abstract": "OAuth2 and JWT secure distributed API gateways.",
        }]
        hits = closure_hits(candidate, documents)
        self.assertEqual([hit["paper_id"] for hit in hits], ["known-work"])

    def test_type_and_case_variants_cannot_form_self_link(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("Scalability [CONCEPT]", "bridge-a", source_paper="p1")
        graph.add_edge("bridge-a", "scalability [METRIC]", source_paper="p2")
        graph.add_edge("Scalability [CONCEPT]", "bridge-b", source_paper="p3")
        graph.add_edge("bridge-b", "scalability [METRIC]", source_paper="p4")
        candidate = {
            "type": "missing_link",
            "head": "Scalability [CONCEPT]",
            "tail": "scalability [METRIC]",
        }
        result = validate_candidate(graph, candidate, validation_config(), [])
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["canonical_self_link"])
        self.assertIn("canonical_self_link", result["reasons"])

    def test_corpus_adapter_merges_type_and_case_variants(self):
        triples = [
            {
                "subject": "Scalability [CONCEPT]",
                "object": "throughput",
                "paper_id": "p1",
            },
            {
                "subject": "scalability [METRIC]",
                "object": "latency",
                "paper_id": "p2",
            },
        ]
        graph = build_graph(triples)
        self.assertIn("Scalability", graph)
        self.assertNotIn("scalability [METRIC]", graph)
        self.assertEqual(sum(node.casefold() == "scalability" for node in graph), 1)

    def test_temporal_closure_checks_post_peak_mentions(self):
        candidate = {
            "type": "temporal_decay",
            "concept": "service mesh telemetry",
            "peak_year": 2022,
        }
        documents = [
            {"paper_id": "old", "year": 2021, "title": "Service mesh telemetry"},
            {"paper_id": "new", "year": 2024, "title": "New service mesh telemetry study"},
        ]
        hits = closure_hits(candidate, documents)
        self.assertEqual([hit["paper_id"] for hit in hits], ["new"])

    def test_missing_closure_corpus_fails_to_manual_review(self):
        candidate = {
            "type": "missing_link",
            "head": "runtime anomaly detection",
            "tail": "service mesh telemetry",
        }
        result = validate_candidate(supported_bridge_graph(), candidate, validation_config())
        self.assertEqual(result["status"], "review_required")
        self.assertIn("source_closure_corpus_unavailable", result["reasons"])


if __name__ == "__main__":
    unittest.main()
