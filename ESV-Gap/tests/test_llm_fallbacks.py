import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.extract_triples import (
    ExtractionRateLimitError,
    _retry_after_seconds,
    extract_all_triples,
    extract_triples_from_text,
)
from src.rag_baseline import run_mulla_rag


class LlmFallbackTests(unittest.TestCase):
    @patch("src.extract_triples.time.sleep")
    def test_extraction_retries_a_rate_limit(self, mock_sleep):
        client = Mock()
        success = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=(
                '{"triples":[{"subject":{"name":"KG","type":"METHOD"},'
                '"relation":"ADDRESSES","object":{"name":"gap","type":"CONCEPT"}}]}'
            )))]
        )
        client.chat.completions.create.side_effect = [
            Exception("429 rate_limit: Please try again in 4.9s"),
            success,
        ]

        triples = extract_triples_from_text(
            client, "model", "{text_chunk}", "domain", "title", 2025, "text"
        )

        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0]["confidence"], 0.5)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        mock_sleep.assert_called_once_with(5.9)

    def test_retry_after_parser_preserves_minutes(self):
        self.assertAlmostEqual(
            _retry_after_seconds("Please try again in 6m12.038s"),
            372.038,
        )

    @patch("src.extract_triples.time.sleep")
    def test_daily_quota_stops_without_false_empty_result(self, mock_sleep):
        client = Mock()
        client.chat.completions.create.side_effect = Exception(
            "429 rate_limit_exceeded: tokens per day (TPD); "
            "Please try again in 6m12.038s"
        )

        with self.assertRaises(ExtractionRateLimitError):
            extract_triples_from_text(
                client, "model", "{text_chunk}", "domain", "title", 2025, "text"
            )

        self.assertEqual(client.chat.completions.create.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("src.extract_triples.OpenAI")
    @patch("src.extract_triples.extract_paper_triples")
    def test_resume_preserves_checkpointed_triple_count(
        self, mock_extract_paper, _mock_openai
    ):
        mock_extract_paper.side_effect = ExtractionRateLimitError("quota")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            triples = root / "triples"
            prompts = root / "prompts"
            processed.mkdir()
            triples.mkdir()
            prompts.mkdir()

            papers = [
                {"paperId": "p1", "title": "Done", "abstract": "done"},
                {"paperId": "p2", "title": "Pending", "abstract": "pending"},
            ]
            (processed / "corpus_filtered.jsonl").write_text(
                "".join(json.dumps(paper) + "\n" for paper in papers),
                encoding="utf-8",
            )
            (prompts / "triple_extraction.txt").write_text(
                "{domain} {title} {year} {text_chunk}", encoding="utf-8"
            )
            (triples / "paper_p1.json").write_text(json.dumps({
                "paperId": "p1",
                "num_triples": 2,
                "triples": [{"relation": "USES"}, {"relation": "EXTENDS"}],
            }), encoding="utf-8")
            (triples / "extraction_progress.json").write_text(json.dumps({
                "completed_ids": ["p1"],
                "total_triples": 2,
            }), encoding="utf-8")

            config = {
                "paths": {
                    "processed_data": str(processed),
                    "triples": str(triples),
                    "prompts": str(prompts),
                },
                "extraction": {
                    "model": "test-model",
                    "chunk_size": 1500,
                    "chunk_overlap": 200,
                },
                "project": {"domain": "test domain"},
                "api_keys": {"groq": "gsk_test"},
            }

            with self.assertRaises(ExtractionRateLimitError):
                extract_all_triples(config)

            progress = json.loads(
                (triples / "extraction_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["completed_ids"], ["p1"])
            self.assertEqual(progress["total_triples"], 2)
            self.assertEqual(mock_extract_paper.call_count, 1)

    @patch("src.rag_baseline.time.sleep")
    @patch("src.rag_baseline.call_llm")
    @patch("src.rag_baseline.SentenceTransformer", None)
    def test_rag_uses_tfidf_without_sentence_transformers(
        self, mock_call_llm, _mock_sleep
    ):
        mock_call_llm.return_value = (
            "RESEARCH_GAPS: gap\n"
            "RESEARCH_DIRECTION: direction\n"
            "SOLUTION_APPROACH: solution\n"
            "REMAINING_GAPS: remaining"
        )
        papers = [{
            "paperId": "p1",
            "title": "Paper one",
            "year": 2025,
            "abstract": "knowledge graph evidence retrieval and research gaps",
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rag.json"
            results = run_mulla_rag(
                papers, "research gaps", Mock(), "model", output_path
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["retrieval_backend"], "tfidf_fallback")
        self.assertEqual(results[0]["retrieved_context"], [])


if __name__ == "__main__":
    unittest.main()
