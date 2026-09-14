import json
import tempfile
import unittest

from src.build_graph import deduplicate_entities
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


class EntityDeduplicationTests(unittest.TestCase):
    def test_fuzzy_merge_does_not_merge_different_versions_or_types(self):
        entities = {
            "ComPRePS 1.0": {"type": "TOOL", "occurrences": 2},
            "ComPRePS 2.0": {"type": "TOOL", "occurrences": 1},
            "cubicles": {"type": "CONCEPT", "occurrences": 1},
            "CubicleOS": {"type": "METHOD", "occurrences": 2},
        }
        mapping, _ = deduplicate_entities(entities)
        self.assertNotEqual(mapping["ComPRePS 1.0"], mapping["ComPRePS 2.0"])
        self.assertNotEqual(mapping["cubicles"], mapping["CubicleOS"])

    def test_acronym_merges_with_same_typed_expansion(self):
        entities = {
            "HME": {"type": "CONCEPT", "occurrences": 2},
            "handwritten mathematical expression": {
                "type": "CONCEPT",
                "occurrences": 1,
            },
        }
        mapping, _ = deduplicate_entities(entities)
        self.assertEqual(
            mapping["HME"], mapping["handwritten mathematical expression"]
        )


if __name__ == "__main__":
    unittest.main()
