import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.full_text import enrich_candidate_source_full_text
from src.utils import load_jsonl, save_json, save_jsonl


class _Response:
    content = b"%PDF fake"

    def raise_for_status(self):
        return None


class FullTextTests(unittest.TestCase):
    def test_only_candidate_source_with_direct_oa_pdf_is_enriched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            outputs = root / "outputs"
            processed.mkdir()
            outputs.mkdir()
            corpus_path = processed / "corpus_filtered.jsonl"
            save_jsonl([{
                "paperId": "p1",
                "title": "Open paper",
                "abstract": "A limitation is reported.",
                "openAccessPdf": {"url": "https://example.test/paper.pdf", "status": "GOLD"},
            }], corpus_path)
            save_json({
                "evidence_gaps": [{
                    "supporting_paper_ids": ["p1"],
                    "candidate_quality": {
                        "score": 0.9,
                        "certificate_readiness": {
                            "independent_sources_ready": True
                        },
                    },
                }]
            }, outputs / "detected_gaps_raw.json")
            config = {
                "paths": {
                    "processed_data": str(processed), "outputs": str(outputs)
                },
                "full_text": {"enabled": True},
                "gap_certification": {"min_full_text_characters": 1000},
            }
            with patch(
                "src.full_text.extract_pdf_text", return_value="evidence " * 200
            ):
                report = enrich_candidate_source_full_text(
                    config, request_get=lambda *args, **kwargs: _Response()
                )
            self.assertEqual(report["enriched"], 1)
            paper = list(load_jsonl(corpus_path))[0]
            self.assertGreater(len(paper["full_text"]), 1000)
            self.assertEqual(
                paper["full_text_provenance"]["retrieval_kind"],
                "direct_open_access_pdf",
            )


if __name__ == "__main__":
    unittest.main()
