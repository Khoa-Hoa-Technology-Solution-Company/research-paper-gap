import json
import tempfile
import unittest
from pathlib import Path

from src.create_lexical_reproduction_corpus import build_lexical_corpus


class LexicalReproductionCorpusTests(unittest.TestCase):
    def test_snapshot_is_deterministic_and_records_decisions(self):
        records = [
            {
                "paperId": "include",
                "title": "Microservice authentication",
                "abstract": "Security policy for cloud-native architecture.",
                "year": 2024,
                "citationCount": 2,
            },
            {
                "paperId": "exclude",
                "title": "A compiler survey",
                "abstract": "No relevant architecture or security terms.",
                "year": 2024,
                "citationCount": 3,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "retrieved.json"
            source.write_text(json.dumps(records), encoding="utf-8")
            output = root / "out"

            manifest = build_lexical_corpus(source, output)

            self.assertEqual(manifest["source_snapshot"]["retrieved_records"], 2)
            self.assertEqual(manifest["screening"]["retained_records"], 1)
            screened = json.loads((output / "screened_papers.json").read_text(encoding="utf-8"))
            self.assertEqual([paper["paperId"] for paper in screened], ["include"])
            self.assertTrue((output / "screening_decisions.csv").exists())
            self.assertTrue((output / "corpus_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
