import json
import tempfile
import unittest
from pathlib import Path

from src.build_graph import build_knowledge_graph


class EmptyGraphTests(unittest.TestCase):
    def test_empty_triple_set_builds_a_safe_null_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triples_dir = root / "triples"
            graph_dir = root / "graph"
            triples_dir.mkdir()
            (triples_dir / "all_triples.json").write_text("[]", encoding="utf-8")

            config = {
                "paths": {
                    "triples": str(triples_dir),
                    "graph": str(graph_dir),
                },
                "graph": {
                    "fuzzy_match_threshold": 85,
                    "semantic_similarity_threshold": 0.85,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "min_edge_confidence": 0.3,
                },
            }

            graph = build_knowledge_graph(config)

            self.assertEqual(graph.number_of_nodes(), 0)
            self.assertEqual(graph.number_of_edges(), 0)
            self.assertTrue((graph_dir / "knowledge_graph.pkl").exists())
            self.assertTrue((graph_dir / "knowledge_graph.graphml").exists())


if __name__ == "__main__":
    unittest.main()
