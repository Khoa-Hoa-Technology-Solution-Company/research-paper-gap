import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import config
from src.closure_search import (
    _deduplicate_query_variants,
    build_closure_run_manifest,
    deterministic_query_variants,
)
from src.extract_triples import _locate_evidence, chunk_provenance_from_record
from src.fetch_papers import build_chunk_records, chunk_text_with_offsets
from src.llm_client import call_llm_with_provenance
from src.provenance import PROVENANCE_SCHEMA_VERSION, stable_chunk_id


class _FakeCompletions:
    def create(self, **_kwargs):
        message = SimpleNamespace(content='[{"subject": "A"}]')
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(
            choices=[choice],
            id="provider-response-1",
            model="test-model-revision",
            created=1234567890,
            system_fingerprint="test-fingerprint",
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class ProvenanceTests(unittest.TestCase):
    def test_chunk_records_preserve_exact_source_offsets(self):
        abstract = "First sentence. Second sentence! Third sentence?"
        chunks = chunk_text_with_offsets(abstract, max_words=3)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(
                chunk["text"], abstract[chunk["source_char_start"]:chunk["source_char_end"]]
            )

        records = build_chunk_records(
            {"paperId": "paper-1", "title": "Example", "abstract": abstract, "year": 2025},
            max_words=3,
        )
        self.assertEqual(records[0]["paper_id"], "paper-1")
        self.assertEqual(records[0]["source_char_start"], 0)
        self.assertEqual(records[0]["chunking_algorithm_version"], "deterministic-sentence-span-chunking-v1")
        self.assertNotEqual(records[0]["chunk_id"], records[1]["chunk_id"])

    def test_stable_chunk_id_changes_when_content_changes(self):
        left = stable_chunk_id("paper", "abstract", 0, "same")
        right = stable_chunk_id("paper", "abstract", 0, "changed")
        self.assertNotEqual(left, right)

    def test_evidence_offsets_are_only_emitted_for_a_unique_exact_match(self):
        unique = _locate_evidence("One source sentence.", "source")
        self.assertEqual(unique["evidence_location_status"], "unique_verbatim_match")
        self.assertEqual(unique["evidence_char_start"], 4)

        ambiguous = _locate_evidence("repeat repeat", "repeat")
        self.assertEqual(ambiguous["evidence_location_status"], "ambiguous_verbatim_match")
        self.assertIsNone(ambiguous["evidence_char_start"])
        self.assertEqual(ambiguous["evidence_match_count"], 2)

    def test_legacy_chunk_without_offsets_is_explicitly_marked(self):
        provenance = chunk_provenance_from_record(
            {"paperId": "old-paper", "title": "Old", "chunk_index": 0, "text": "Legacy chunk."},
            fallback_index=0,
        )
        self.assertEqual(provenance["chunk_offset_status"], "not_recorded_in_input_chunk")
        self.assertIsNone(provenance["source_char_start"])
        self.assertEqual(provenance["chunk_id_status"], "generated_for_this_rerun")

    def test_deterministic_closure_variants_and_deduplication_are_repeatable(self):
        claim = "Evaluate zero trust authentication for microservice service meshes."
        first = deterministic_query_variants(claim)
        second = deterministic_query_variants(claim)
        self.assertEqual(first, second)
        self.assertTrue(all(item["origin"] == "deterministic" for item in first))
        unique, manifest = _deduplicate_query_variants(first + [dict(first[0], query_id="duplicate")])
        self.assertLess(len(unique), len(first) + 1)
        self.assertEqual(manifest["dropped_variants"][0]["reason"], "normalized_duplicate")

    def test_closure_run_manifest_is_constructible_without_network(self):
        audit = [{
            "claim_id": "c-1",
            "claim_sha256": "abc",
            "closure_status": "retrieved--human-review-required",
            "candidate_count": 2,
            "retrieval_manifest": {"query_executions": [{"query_id": "deterministic-verbatim"}]},
        }]
        manifest = build_closure_run_manifest(
            audit,
            input_path=None,
            limit=20,
            include_llm_variants=False,
            citation_limit=10,
            citation_candidate_limit=5,
        )
        self.assertEqual(manifest["schema_version"], PROVENANCE_SCHEMA_VERSION)
        self.assertFalse(manifest["retrieval_configuration"]["include_llm_variants"])
        self.assertEqual(manifest["claims"][0]["query_ids"], ["deterministic-verbatim"])

    def test_llm_call_logs_non_secret_reproducibility_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            call_log = os.path.join(temp_dir, "llm_calls.jsonl")
            response_dir = os.path.join(temp_dir, "responses")
            with (
                patch.object(config, "PROVENANCE_DIR", temp_dir),
                patch.object(config, "LLM_CALL_LOG_PATH", call_log),
                patch.object(config, "LLM_RESPONSE_DIR", response_dir),
                patch.object(config, "LLM_DELAY", 0.0),
                patch.object(config, "LLM_RAW_RESPONSE_POLICY", "hash-only"),
                patch("src.llm_client.get_llm_client", return_value=_FakeClient()),
            ):
                result = call_llm_with_provenance(
                    "A prompt that should be hashed, not stored.",
                    temperature=0.0,
                    prompt_version="unit-test-v1",
                    purpose="unit-test",
                )
            self.assertEqual(result.provenance["status"], "succeeded")
            self.assertEqual(result.provenance["generation_parameters"]["temperature"], 0.0)
            self.assertIsNotNone(result.provenance["response"]["assistant_message_content_sha256"])
            self.assertIsNone(result.provenance["response"]["storage_path"])
            with open(call_log, encoding="utf-8") as handle:
                logged = json.loads(handle.read())
            self.assertNotIn("A prompt that should be hashed", json.dumps(logged))
            self.assertEqual(logged["prompt_version"], "unit-test-v1")


if __name__ == "__main__":
    unittest.main()
