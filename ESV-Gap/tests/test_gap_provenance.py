import unittest

import networkx as nx

from src.gap_provenance import build_paper_index, resolve_gap_provenance


def add_relation(graph, source, target, paper_id, relation="USES", year=2024):
    graph.add_edge(
        source,
        target,
        relation=relation,
        source_paper=paper_id,
        year=year,
        evidence=f"{source} {relation} {target}",
    )


class GapProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.MultiDiGraph()
        add_relation(self.graph, "A", "Bridge", "p1")
        add_relation(self.graph, "Bridge", "B", "p2", relation="ADDRESSES")
        add_relation(self.graph, "Cluster A", "Cluster B", "p3")
        add_relation(self.graph, "Time concept", "Method", "p4", year=2022)
        self.papers = build_paper_index([
            {"paperId": "p1", "title": "Paper One", "year": 2021},
            {"paperId": "p2", "title": "Paper Two", "year": 2022},
            {"paperId": "p3", "title": "Paper Three", "year": 2023},
            {"paperId": "p4", "title": "Paper Four", "year": 2024},
        ])

    def test_missing_link_uses_motivating_path_papers(self):
        result = resolve_gap_provenance(
            self.graph,
            {"type": "missing_link", "head": "A", "tail": "B"},
            self.papers,
        )
        self.assertEqual(set(result["paper_ids"]), {"p1", "p2"})
        self.assertEqual(result["evidence_paths"][0]["nodes"], ["A", "Bridge", "B"])

    def test_orphan_cluster_uses_only_internal_edges(self):
        result = resolve_gap_provenance(
            self.graph,
            {
                "type": "orphan_cluster",
                "members": ["Cluster A", "Cluster B"],
            },
            self.papers,
        )
        self.assertEqual(result["paper_ids"], ["p3"])

    def test_temporal_decay_uses_incident_relation_papers(self):
        result = resolve_gap_provenance(
            self.graph,
            {"type": "temporal_decay", "concept": "Time concept"},
            self.papers,
        )
        self.assertEqual(result["paper_ids"], ["p4"])
        self.assertEqual(result["papers"][0]["title"], "Paper Four")


if __name__ == "__main__":
    unittest.main()
