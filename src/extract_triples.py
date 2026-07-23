import json
import os
import argparse
import re
from typing import Any
from src import config
from src.llm_client import LLMCallError, call_llm_with_provenance
from src.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    file_sha256,
    sha256_json,
    sha256_text,
    stable_chunk_id,
    utc_now_iso,
)

def _locate_evidence(chunk_text: str, evidence_quote: str) -> dict:
    """Return auditable character/sentence offsets when a quote is verbatim.

    LLM-provided quotations are not assumed to be verbatim.  Missing offsets
    are retained as null instead of inventing a location.
    """
    location = {
        "evidence_char_start": None,
        "evidence_char_end": None,
        "sentence_start": None,
        "sentence_end": None,
        "evidence_match_count": 0,
        "evidence_quote_sha256": sha256_text(evidence_quote or ""),
        "evidence_location_status": "unmatched",
    }
    if not evidence_quote:
        return location
    folded_chunk = chunk_text.casefold()
    folded_quote = evidence_quote.casefold()
    starts: list[int] = []
    cursor = 0
    while True:
        start = folded_chunk.find(folded_quote, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + max(1, len(folded_quote))
    location["evidence_match_count"] = len(starts)
    if not starts:
        return location
    # A repeated quote has no unique source offset.  Avoid pretending that the
    # first occurrence was the evidence intended by the extractor.
    if len(starts) > 1:
        location["evidence_location_status"] = "ambiguous_verbatim_match"
        return location
    start = starts[0]
    end = start + len(evidence_quote)
    sentence_bounds = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", chunk_text, flags=re.DOTALL):
        if match.group(0).strip():
            sentence_bounds.append((match.start(), match.end()))
    covered = [index for index, (left, right) in enumerate(sentence_bounds) if left < end and right > start]
    location.update({
        "evidence_char_start": start,
        "evidence_char_end": end,
        "sentence_start": covered[0] if covered else None,
        "sentence_end": covered[-1] if covered else None,
        "evidence_location_status": "unique_verbatim_match",
    })
    return location


def chunk_provenance_from_record(chunk: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    """Normalize a forward chunk record without pretending legacy offsets exist."""
    text = str(chunk.get("text") or "")
    paper_id = chunk.get("paper_id") or chunk.get("paperId")
    section_label = chunk.get("section_label", "unknown")
    chunk_index = chunk.get("chunk_index", fallback_index)
    supplied_chunk_id = chunk.get("chunk_id")
    start, end = chunk.get("source_char_start"), chunk.get("source_char_end")
    offsets_are_integers = (
        isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and 0 <= start <= end
    )
    if offsets_are_integers and end - start != len(text):
        offset_status = "recorded_offset_length_mismatch"
    elif offsets_are_integers:
        offset_status = "recorded_exact_chunk_span"
    else:
        start, end, offset_status = None, None, "not_recorded_in_input_chunk"
    return {
        "paper_id": paper_id,
        "paper_title": chunk.get("paper_title") or chunk.get("title"),
        "chunk_id": supplied_chunk_id or stable_chunk_id(paper_id, section_label, chunk_index, text),
        "chunk_id_status": "recorded" if supplied_chunk_id else "generated_for_this_rerun",
        "chunk_index": chunk_index,
        "section_label": section_label,
        "source_text_kind": chunk.get("source_text_kind", section_label),
        "source_text_sha256": chunk.get("source_text_sha256"),
        "chunk_text_sha256": chunk.get("chunk_text_sha256") or sha256_text(text),
        "source_char_start": start,
        "source_char_end": end,
        "chunk_offset_status": offset_status,
        "chunking_algorithm_version": chunk.get("chunking_algorithm_version"),
        "chunking_max_words": chunk.get("chunking_max_words"),
        "external_ids": chunk.get("external_ids") or chunk.get("externalIds"),
        "doi": chunk.get("doi"),
    }


def extract_triples_from_chunk(
    chunk_text: str,
    year: int,
    provenance: dict[str, Any] | None = None,
    *,
    return_call_provenance: bool = False,
) -> list[dict] | tuple[list[dict], dict[str, Any] | None]:
    """Extract triples from one chunk and attach future-rerun provenance.

    This does not retrofit the historical artifact.  It records enough for a
    future rerun to identify the source chunk, model request, and evidence
    offset policy without claiming model-reported scores are calibrated.
    """
    prompt = f"""
    You are an expert in Software Engineering research. Read the following text chunk and extract typed triples of the form <Subject, Relation, Object>.
    
    CONSTRAINTS:
    - Subject and Object entity types MUST belong to: {config.ENTITY_TYPES}
    - Relation types MUST belong to: {config.RELATION_TYPES}
    - Assign a Confidence Score (0.0 to 1.0) for each triple.
    - Provide a short "evidence_quote" from the text showing where the relation is stated.
    
    Text Chunk:
    "{chunk_text}"
    
    Output strictly as a JSON array of objects, with no markdown formatting (like ```json), backticks, or explanation.
    JSON Schema:
    [
      {{
        "subject": "Name of subject entity (canonical, noun phrase)",
        "subject_type": "One of: {config.ENTITY_TYPES}",
        "relation": "One of: {config.RELATION_TYPES}",
        "object": "Name of object entity (canonical, noun phrase)",
        "object_type": "One of: {config.ENTITY_TYPES}",
        "confidence": 0.95,
        "evidence_quote": "Exact quote from text"
      }}
    ]
    """
    
    call_provenance: dict[str, Any] | None = None
    try:
        result = call_llm_with_provenance(
            prompt,
            temperature=0.1,
            prompt_version=config.TRIPLE_EXTRACTION_PROMPT_VERSION,
            purpose="typed-triple-extraction",
        )
        content = result.text
        call_provenance = result.provenance
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
                
        triples = json.loads(content)
        valid_triples = []
        
        for t in triples:
            sub = t.get("subject", "").strip()
            sub_type = t.get("subject_type", "").strip().upper()
            rel = t.get("relation", "").strip().upper()
            obj = t.get("object", "").strip()
            obj_type = t.get("object_type", "").strip().upper()
            conf = float(t.get("confidence", 0.0))
            quote = t.get("evidence_quote", "").strip()
            
            # Validation
            if not sub or not obj or not rel:
                continue
            if sub_type not in config.ENTITY_TYPES:
                continue
            if obj_type not in config.ENTITY_TYPES:
                continue
            if rel not in config.RELATION_TYPES:
                continue
            if conf < config.TRIPLE_CONFIDENCE_THRESHOLD:
                continue
                
            call_identity = call_provenance["llm"]
            response_metadata = call_provenance.get("response", {})
            record: dict[str, Any] = {
                "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
                "subject": sub,
                "subject_type": sub_type,
                "relation": rel,
                "object": obj,
                "object_type": obj_type,
                # Retain the legacy key for compatibility with existing
                # filtering/graph code.  The explicit name prevents the
                # paper from treating this self-reported value as calibrated
                # probability.
                "confidence": conf,
                "model_reported_extraction_score": conf,
                "evidence_quote": quote,
                "year": year,
                # Keep model_name for compatibility with older readers, while
                # the explicit fields preserve provider/model/revision details.
                "model_name": call_identity["model_identifier"],
                "model_provider": call_identity["provider"],
                "model_identifier": call_identity["model_identifier"],
                "model_revision": call_identity["configured_model_revision"],
                "model_release_date": call_identity["configured_model_release_date"],
                "api_version": call_identity["configured_api_version"],
                "api_date": call_identity["configured_api_date"],
                "prompt_version": config.TRIPLE_EXTRACTION_PROMPT_VERSION,
                "prompt_sha256": call_provenance["prompt_sha256"],
                "extraction_call_id": call_provenance["call_id"],
                "extraction_requested_at_utc": call_provenance["request_started_at_utc"],
                "extraction_completed_at_utc": call_provenance.get("completed_at_utc"),
                "generation_parameters": call_provenance["generation_parameters"],
                "retry_policy": call_provenance["retry_policy"],
                "raw_response_storage_policy": call_provenance["raw_response_storage_policy"],
                "raw_response_sha256": response_metadata.get("raw_provider_response_sha256"),
                "assistant_response_content_sha256": response_metadata.get("assistant_message_content_sha256"),
                "raw_response_storage_path": response_metadata.get("storage_path"),
                "provider_response_id": response_metadata.get("provider_response_id"),
                "provider_system_fingerprint": response_metadata.get("provider_system_fingerprint"),
            }
            record.update(_locate_evidence(chunk_text, quote))
            if provenance:
                record.update({
                    "paper_id": provenance.get("paper_id"),
                    "paper_title": provenance.get("paper_title"),
                    "chunk_id": provenance.get("chunk_id"),
                    "chunk_id_status": provenance.get("chunk_id_status", "not_recorded"),
                    "chunk_index": provenance.get("chunk_index"),
                    "section_label": provenance.get("section_label", "unknown"),
                    "source_text_kind": provenance.get("source_text_kind", provenance.get("section_label", "unknown")),
                    "source_text_sha256": provenance.get("source_text_sha256"),
                    "chunk_text_sha256": provenance.get("chunk_text_sha256", sha256_text(chunk_text)),
                    "chunk_source_char_start": provenance.get("source_char_start"),
                    "chunk_source_char_end": provenance.get("source_char_end"),
                    "chunk_offset_status": provenance.get("chunk_offset_status", "not_recorded"),
                    "chunking_algorithm_version": provenance.get("chunking_algorithm_version"),
                    "chunking_max_words": provenance.get("chunking_max_words"),
                    "paper_external_ids": provenance.get("external_ids"),
                    "paper_doi": provenance.get("doi"),
                })
                if record["evidence_location_status"] == "unique_verbatim_match":
                    source_start = provenance.get("source_char_start")
                    if isinstance(source_start, int) and not isinstance(source_start, bool):
                        record["evidence_source_char_start"] = source_start + int(record["evidence_char_start"])
                        record["evidence_source_char_end"] = source_start + int(record["evidence_char_end"])
                    else:
                        record["evidence_source_char_start"] = None
                        record["evidence_source_char_end"] = None
                else:
                    record["evidence_source_char_start"] = None
                    record["evidence_source_char_end"] = None
            valid_triples.append(record)
        return (valid_triples, call_provenance) if return_call_provenance else valid_triples
    except LLMCallError as exc:
        print(f"[!] Error extracting triples: {exc}")
        return ([], exc.provenance) if return_call_provenance else []
    except Exception as e:
        print(f"[!] Error extracting triples: {e}")
        return ([], call_provenance) if return_call_provenance else []

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(raw_path: str, progress_path: str, triples: list[dict], processed: set[int]) -> None:
    with open(raw_path, "w", encoding="utf-8") as handle:
        json.dump(triples, handle, ensure_ascii=False, indent=2)
    with open(progress_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "processed_chunk_indices": sorted(processed),
            "updated_at_utc": utc_now_iso(),
        }, handle, indent=2)


def _write_extraction_manifest(
    *,
    chunks_path: str,
    raw_path: str,
    start: int,
    end: int,
    processed: set[int],
    execution_records: list[dict[str, Any]],
    skipped_chunk_indices: list[int],
) -> str:
    """Persist a run-level index of calls without embedding source text."""
    output_path = os.path.join(config.TRIPLES_DIR, "extraction_provenance_manifest.json")
    manifest = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "record_type": "triple_extraction_run_manifest",
        "generated_at_utc": utc_now_iso(),
        "run_id": config.RUN_ID or None,
        "input_chunks": {
            "path": os.path.relpath(chunks_path, config.DATA_DIR).replace(os.sep, "/"),
            "sha256": file_sha256(chunks_path),
        },
        "requested_chunk_range": {"start_inclusive": start, "end_exclusive": end},
        "processed_chunk_indices_after_run": sorted(processed),
        "skipped_checkpointed_chunk_indices": skipped_chunk_indices,
        "new_execution_count": len(execution_records),
        "executions": execution_records,
        "output_raw_triples": {
            "path": os.path.relpath(raw_path, config.DATA_DIR).replace(os.sep, "/"),
            "sha256": file_sha256(raw_path) if os.path.exists(raw_path) else None,
        },
        "raw_response_policy": config.LLM_RAW_RESPONSE_POLICY,
        "limitations": [
            "This manifest describes only calls executed by the current forward schema.",
            "A chunk from a legacy input with missing source offsets remains marked as such; no offsets are inferred.",
        ],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_path


def run_extraction(start: int = 0, end: int | None = None) -> list[dict]:
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        print(f"[!] File {chunks_path} not found. Please run fetch_papers first.")
        return []
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    end = min(end if end is not None else len(chunks), len(chunks))
    if start < 0 or start >= end:
        raise ValueError(f"Invalid chunk range [{start}, {end}) for {len(chunks)} chunks")
    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    progress_path = os.path.join(config.TRIPLES_DIR, "extraction_progress.json")
    all_triples = _load_json(raw_path, [])
    processed = set(_load_json(progress_path, {"processed_chunk_indices": []}).get("processed_chunk_indices", []))
    print(f"[*] Processing chunks [{start}, {end}) of {len(chunks)}; {len(processed)} already checkpointed.")
    execution_records: list[dict[str, Any]] = []
    skipped_chunk_indices: list[int] = []
    
    for i in range(start, end):
        if i in processed:
            print(f"[*] Skipping checkpointed chunk {i + 1}/{len(chunks)}")
            skipped_chunk_indices.append(i)
            continue
        c = chunks[i]
        print(f"[*] Extracting triples from chunk {i+1}/{len(chunks)} of '{c['title'][:40]}...'")
        chunk_provenance = chunk_provenance_from_record(c, i)
        triples, call_provenance = extract_triples_from_chunk(
            c["text"],
            c["year"],
            provenance=chunk_provenance,
            return_call_provenance=True,
        )
        print(f"[+] Extracted {len(triples)} valid triples from chunk.")
        execution_records.append({
            "chunk_list_index": i,
            "paper_id": chunk_provenance["paper_id"],
            "chunk_id": chunk_provenance["chunk_id"],
            "chunk_id_status": chunk_provenance["chunk_id_status"],
            "chunk_text_sha256": chunk_provenance["chunk_text_sha256"],
            "chunk_offset_status": chunk_provenance["chunk_offset_status"],
            "call_id": call_provenance.get("call_id") if call_provenance else None,
            "call_status": call_provenance.get("status") if call_provenance else "not_started",
            "raw_response_sha256": (
                call_provenance.get("response", {}).get("raw_provider_response_sha256")
                if call_provenance else None
            ),
            "valid_triple_count": len(triples),
            "valid_triples_sha256": sha256_json(triples),
        })
        call_succeeded = bool(call_provenance and call_provenance.get("status") == "succeeded")
        all_triples.extend(triples)
        # A provider failure can yield an empty list. Do not checkpoint that
        # chunk as completed: a resumed forward run must retry it rather than
        # silently treating an unavailable extraction as a valid no-triple
        # result. A successful response with zero valid triples is completed.
        if call_succeeded:
            processed.add(i)
        else:
            print(f"[!] Chunk {i + 1} was not checkpointed because its LLM call did not succeed.")
        _save_checkpoint(raw_path, progress_path, all_triples, processed)
    print(f"[+] Checkpointed {len(all_triples)} raw triples to {raw_path}")
    manifest_path = _write_extraction_manifest(
        chunks_path=chunks_path,
        raw_path=raw_path,
        start=start,
        end=end,
        processed=processed,
        execution_records=execution_records,
        skipped_chunk_indices=skipped_chunk_indices,
    )
    print(f"[+] Saved extraction provenance manifest to {manifest_path}")
    return all_triples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract triples with resumable chunk checkpoints.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive chunk index.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive chunk index.")
    args = parser.parse_args()
    run_extraction(args.start, args.end)
