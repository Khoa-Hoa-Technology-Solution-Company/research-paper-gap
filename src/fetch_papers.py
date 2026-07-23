import json
import os
import time
import requests
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


CHUNKING_ALGORITHM_VERSION = "deterministic-sentence-span-chunking-v1"
PAPER_RETRIEVAL_FIELDS = (
    "paperId,title,abstract,year,citationCount,externalIds,publicationVenue"
)

def expand_query_with_provenance(topic: str) -> tuple[list[str], dict[str, Any]]:
    """Expand a topic while retaining the model-call provenance for a rerun."""
    print(f"[*] Expanding query for topic: '{topic}' using LLM...")
    prompt = f"""
    You are an expert academic researcher. Generate 5 diverse, highly specific search terms or phrases 
    to retrieve scientific articles about the following topic from Semantic Scholar:
    Topic: "{topic}"
    
    Output strictly as a JSON array of strings, with no additional text or markdown formatting (like ```json).
    Example: ["term 1", "term 2", "term 3", "term 4", "term 5"]
    """
    
    manifest: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "operation": "query_expansion",
        "topic": topic,
        "topic_sha256": sha256_text(topic),
        "prompt_version": config.QUERY_EXPANSION_PROMPT_VERSION,
    }
    try:
        result = call_llm_with_provenance(
            prompt,
            temperature=0.2,
            prompt_version=config.QUERY_EXPANSION_PROMPT_VERSION,
            purpose="literature-query-expansion",
        )
        content = result.text
        manifest["llm_call"] = result.provenance
        # Clean up any potential markdown backticks
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
        
        queries = json.loads(content)
        if isinstance(queries, list):
            queries = [str(query).strip() for query in queries if str(query).strip()]
            # Include original topic
            if topic not in queries:
                queries.insert(0, topic)
            manifest.update({"status": "succeeded", "queries": queries})
            print(f"[+] Expanded queries: {queries}")
            return queries, manifest
    except LLMCallError as exc:
        manifest.update({"status": "failed", "llm_call": exc.provenance})
        print(f"[!] Error expanding query: {exc}. Falling back to original topic.")
    except Exception as e:
        manifest.update({
            "status": "failed",
            "error_type": type(e).__name__,
            "error_sha256": sha256_text(str(e)),
        })
        print(f"[!] Error expanding query: {e}. Falling back to original topic.")

    manifest.setdefault("status", "fallback")
    manifest["queries"] = [topic]
    return [topic], manifest


def expand_query(topic: str) -> list[str]:
    """Compatibility wrapper that returns expanded queries without a manifest."""
    queries, _ = expand_query_with_provenance(topic)
    return queries

def _semantic_scholar_headers() -> dict[str, str]:
    return {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY} if config.SEMANTIC_SCHOLAR_API_KEY else {}


def search_papers(
    queries: list[str], limit_per_query: int = 20, *, return_manifest: bool = False
) -> list[dict] | tuple[list[dict], dict[str, Any]]:
    """Retrieve paper metadata and retain a replay/audit manifest.

    The API is mutable, so the manifest records every request parameter, HTTP
    response hash, and returned paper ID.  It never records the API key.
    """
    print(f"[*] Querying Semantic Scholar API for {len(queries)} search terms...")
    endpoint = f"{config.SEMANTIC_SCHOLAR_API_BASE_URL}/paper/search"
    headers = _semantic_scholar_headers()

    seen_paper_ids = set()
    papers_by_id: dict[str, dict[str, Any]] = {}
    papers = []
    manifest: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "operation": "semantic_scholar_paper_search",
        "endpoint": endpoint,
        "fields": PAPER_RETRIEVAL_FIELDS,
        "queries_requested": list(queries),
        "limit_per_query": limit_per_query,
        "timeout_seconds": config.SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS,
        "inter_request_delay_seconds": config.SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS,
        "api_key_configured": bool(config.SEMANTIC_SCHOLAR_API_KEY),
        "deduplication_rule": "keep first non-empty-abstract record by Semantic Scholar paperId; retain all retrieval events on that record",
        "request_events": [],
    }

    for q in queries:
        print(f"[*] Querying: '{q}'...")
        retrieved_for_query = 0
        for offset in range(0, limit_per_query, 100):
            params = {
                "query": q,
                "limit": min(100, limit_per_query - offset),
                "offset": offset,
                "fields": PAPER_RETRIEVAL_FIELDS,
            }
            event: dict[str, Any] = {
                "query": q,
                "query_sha256": sha256_text(q),
                "request_parameters": dict(params),
                "requested_at_utc": utc_now_iso(),
            }
            try:
                if config.SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS:
                    time.sleep(config.SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS)
                response = requests.get(
                    endpoint,
                    params=params,
                    headers=headers,
                    timeout=config.SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS,
                )
                event["completed_at_utc"] = utc_now_iso()
                event["http_status"] = response.status_code
                event["response_sha256"] = sha256_text(response.text)
                if response.status_code != 200:
                    print(f"[!] API request failed with status code {response.status_code}: {response.text}")
                    event["status"] = "http_error"
                    manifest["request_events"].append(event)
                    break
                results = response.json().get("data", [])
                event.update({
                    "status": "succeeded",
                    "result_count": len(results),
                    "result_paper_ids": [item.get("paperId") for item in results],
                })
                manifest["request_events"].append(event)
                retrieved_for_query += len(results)
                for r in results:
                    paper_id = r.get("paperId")
                    retrieval_event = {
                        "query": q,
                        "query_sha256": sha256_text(q),
                        "offset": offset,
                        "request_event_index": len(manifest["request_events"]) - 1,
                        "response_record_sha256": sha256_json(r),
                    }
                    if paper_id and paper_id not in seen_paper_ids and r.get("abstract"):
                        seen_paper_ids.add(paper_id)
                        paper = {
                            "paperId": paper_id,
                            "paper_id": paper_id,
                            "title": r.get("title"),
                            "abstract": r.get("abstract"),
                            "year": r.get("year"),
                            "citationCount": r.get("citationCount", 0),
                            "externalIds": r.get("externalIds", {}),
                            "publicationVenue": r.get("publicationVenue"),
                            "retrieval_provenance": {
                                "schema_version": PROVENANCE_SCHEMA_VERSION,
                                "source": "Semantic Scholar Graph API",
                                "endpoint": endpoint,
                                "retrieval_events": [retrieval_event],
                            },
                        }
                        papers.append(paper)
                        papers_by_id[str(paper_id)] = paper
                    elif paper_id and r.get("abstract") and str(paper_id) in papers_by_id:
                        papers_by_id[str(paper_id)]["retrieval_provenance"]["retrieval_events"].append(retrieval_event)
                if len(results) < params["limit"]:
                    break
            except Exception as e:
                event.update({
                    "status": "request_exception",
                    "completed_at_utc": utc_now_iso(),
                    "error_type": type(e).__name__,
                    "error_sha256": sha256_text(str(e)),
                })
                manifest["request_events"].append(event)
                print(f"[!] Error querying Semantic Scholar for query '{q}': {e}")
                break
        print(f"[+] Retrieved {retrieved_for_query} records for query '{q}'")
            
    manifest.update({
        "completed_at_utc": utc_now_iso(),
        "unique_nonempty_abstract_papers": len(papers),
        "stored_paper_ids": [paper.get("paperId") for paper in papers],
        "stored_papers_sha256": sha256_json(papers),
    })
    print(f"[+] Total unique papers with abstracts retrieved: {len(papers)}")
    return (papers, manifest) if return_manifest else papers

def sent_tokenize_fallback(text: str) -> list[str]:
    """
    Split text into sentences. Uses NLTK if available, falls back to regex.
    """
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
        return nltk.sent_tokenize(text)
    except Exception as e:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def screen_relevance(
    papers: list[dict],
    topic: str,
    max_papers: int = 100,
    batch_size: int = 10,
    *,
    return_manifest: bool = False,
) -> list[dict] | tuple[list[dict], dict[str, Any]]:
    """Use an LLM relevance screen and emit batch-level provenance on request."""
    print(f"[*] Screening {len(papers)} papers for relevance to topic '{topic}'...")
    screened = []
    manifest: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "operation": "llm_relevance_screening",
        "topic": topic,
        "topic_sha256": sha256_text(topic),
        "prompt_version": config.RELEVANCE_SCREENING_PROMPT_VERSION,
        "threshold": config.RELEVANCE_THRESHOLD,
        "batch_size": batch_size,
        "max_papers": max_papers,
        "fallback_policy": "retain an entire batch with a clearly marked fallback reason if its LLM response cannot be parsed",
        "batches": [],
    }
    # Screening one paper per request makes a 300-paper study unnecessarily
    # slow and increases rate-limit exposure.  Each item carries a stable index
    # so a malformed response can safely fall back to keeping that item.
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        records = [
            {"id": start + index, "title": paper.get("title", ""),
             "abstract": paper.get("abstract", "")}
            for index, paper in enumerate(batch)
        ]
        prompt = f"""
        Evaluate the relevance of each scientific paper below to the research topic: "{topic}".
        Return strictly a JSON array with one object per input record and no
        markdown. Each object must be {{"id": integer, "score": float from
        0.0 to 1.0, "reason": short string}}. Do not omit any id.

        Records:
        {json.dumps(records, ensure_ascii=False)}
        """
        batch_manifest: dict[str, Any] = {
            "batch_start_index": start,
            "input_paper_ids": [paper.get("paperId") or paper.get("paper_id") for paper in batch],
            "input_sha256": sha256_json(records),
        }
        try:
            call_result = call_llm_with_provenance(
                prompt,
                temperature=0.0,
                prompt_version=config.RELEVANCE_SCREENING_PROMPT_VERSION,
                purpose="literature-relevance-screening",
            )
            content = call_result.text
            batch_manifest["llm_call"] = call_result.provenance
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
            
            results = json.loads(content)
            if not isinstance(results, list):
                raise ValueError("Batch relevance response is not a JSON array")
            scores = {int(item["id"]): item for item in results if "id" in item}
            for index, paper in enumerate(batch):
                result = scores.get(start + index)
                if result is None:
                    raise ValueError(f"Missing relevance result for record {start + index}")
                score = float(result.get("score", 0.0))
                print(f"[*] Paper: '{paper.get('title', '')[:50]}...' -> Score: {score}")
                if score >= config.RELEVANCE_THRESHOLD:
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = result.get("reason", "")
                    paper["screening_batch_call_id"] = call_result.provenance["call_id"]
                    paper["screening_method"] = "llm-batched"
                    screened.append(paper)
            batch_manifest.update({
                "status": "succeeded",
                "retained_paper_ids": [paper.get("paperId") or paper.get("paper_id") for paper in screened if paper in batch],
            })
        except LLMCallError as exc:
            batch_manifest.update({"status": "failed_fallback_retained", "llm_call": exc.provenance})
            print(f"[!] Error screening batch {start}-{start + len(batch) - 1}: {exc}. Keeping batch by default.")
            for paper in batch:
                paper["relevance_score"] = 0.8
                paper["relevance_reason"] = "Batch screening error; retained for audit"
                paper["screening_method"] = "llm-batch-fallback-retained"
                paper["screening_batch_call_id"] = exc.provenance.get("call_id")
                screened.append(paper)
        except Exception as e:
            batch_manifest.update({
                "status": "failed_fallback_retained",
                "error_type": type(e).__name__,
                "error_sha256": sha256_text(str(e)),
            })
            print(f"[!] Error screening batch {start}-{start + len(batch) - 1}: {e}. Keeping batch by default.")
            for paper in batch:
                paper["relevance_score"] = 0.8
                paper["relevance_reason"] = "Batch screening error; retained for audit"
                paper["screening_method"] = "llm-batch-fallback-retained"
                screened.append(paper)
        manifest["batches"].append(batch_manifest)
            
    print(f"[+] {len(screened)} papers survived relevance screening (threshold >= {config.RELEVANCE_THRESHOLD})")
    
    # Sort by relevance_score descending, then citationCount descending
    screened.sort(key=lambda x: (x.get("relevance_score", 0.0), x.get("citationCount", 0)), reverse=True)
    
    retained_before_cap = len(screened)
    if len(screened) > max_papers:
        print(f"[*] Truncating screened papers from {len(screened)} to {max_papers} based on relevance and citations.")
        screened = screened[:max_papers]
        
    manifest.update({
        "completed_at_utc": utc_now_iso(),
        "retained_before_cap": retained_before_cap,
        "retained_after_cap": len(screened),
        "retained_paper_ids": [paper.get("paperId") or paper.get("paper_id") for paper in screened],
    })
    return (screened, manifest) if return_manifest else screened


def screen_relevance_lexical(
    papers: list[dict], max_papers: int = 300, *, return_manifest: bool = False
) -> list[dict] | tuple[list[dict], dict[str, Any]]:
    """Apply a transparent first-pass screen for the microservices-security corpus.

    This is deliberately a *retrieval screen*, not a claim of expert relevance
    judgment. The resulting score and matched terms are retained for auditing;
    a later blinded/manual or LLM calibration may supersede this decision.
    """
    architecture_terms = ("microservice", "micro-service", "service mesh", "cloud-native")
    security_terms = (
        "security", "secure", "authentication", "authorization", "access control",
        "zero trust", "vulnerability", "attack", "threat", "privacy", "encryption",
        "identity", "policy", "oauth", "mTLS", "mtls",
    )
    screened = []
    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        architecture_hits = [term for term in architecture_terms if term in text]
        security_hits = [term for term in security_terms if term.lower() in text]
        if not architecture_hits or not security_hits:
            continue
        score = min(0.95, 0.50 + 0.15 * len(architecture_hits) + 0.05 * len(security_hits))
        paper["relevance_score"] = round(score, 2)
        paper["relevance_reason"] = (
            "Transparent lexical retrieval screen; architecture terms="
            f"{architecture_hits}; security terms={security_hits[:5]}"
        )
        paper["screening_method"] = "lexical-first-pass"
        screened.append(paper)
    screened.sort(key=lambda x: (x["relevance_score"], x.get("citationCount", 0)), reverse=True)
    screened = screened[:max_papers]
    manifest = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "operation": "lexical_relevance_screening",
        "rule_version": "microservices-security-lexical-v1",
        "input_count": len(papers),
        "max_papers": max_papers,
        "include_rule": "at least one architecture term AND at least one security term",
        "architecture_terms": list(architecture_terms),
        "security_terms": list(security_terms),
        "ranking_rule": "relevance score descending, then citation count descending",
        "retained_after_cap": len(screened),
        "retained_paper_ids": [paper.get("paperId") or paper.get("paper_id") for paper in screened],
        "completed_at_utc": utc_now_iso(),
    }
    return (screened, manifest) if return_manifest else screened

def chunk_text_with_offsets(text: str, max_words: int = 1000) -> list[dict[str, Any]]:
    """Chunk text using deterministic spans, preserving source offsets.

    ``source_char_end`` is exclusive.  The chunk text is always the exact
    substring at ``text[start:end]``; if a source cannot be represented that
    way, callers should retain null offsets rather than inventing them.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if not text or not text.strip():
        return []

    # This intentionally does not depend on a downloadable sentence model.
    # It is not a linguistic claim; it is a stable boundary rule for a future
    # rerun manifest.  Abbreviations may make imperfect boundaries, but those
    # boundaries and the source offsets are fully recoverable.
    spans = [
        (match.start(), match.end())
        for match in re.finditer(r"\S.*?(?:[.!?]+(?=\s|$)|$)", text, flags=re.DOTALL)
    ]
    if not spans:
        spans = [(0, len(text))]

    chunks: list[dict[str, Any]] = []
    current_start: int | None = None
    current_end: int | None = None
    current_word_count = 0
    for start, end in spans:
        sentence = text[start:end]
        word_count = len(sentence.split())
        if current_start is not None and current_word_count + word_count > max_words:
            chunks.append({
                "text": text[current_start:current_end],
                "source_char_start": current_start,
                "source_char_end": current_end,
                "word_count": current_word_count,
            })
            current_start, current_end, current_word_count = start, end, word_count
        elif current_start is None:
            current_start, current_end, current_word_count = start, end, word_count
        else:
            current_end = end
            current_word_count += word_count

    if current_start is not None and current_end is not None:
        chunks.append({
            "text": text[current_start:current_end],
            "source_char_start": current_start,
            "source_char_end": current_end,
            "word_count": current_word_count,
        })
    return chunks


def chunk_text(text: str, max_words: int = 1000) -> list[str]:
    """Compatibility wrapper that returns only chunk text."""
    return [chunk["text"] for chunk in chunk_text_with_offsets(text, max_words)]


def build_chunk_records(
    paper: dict[str, Any], *, max_words: int = 1000, section_label: str = "abstract"
) -> list[dict[str, Any]]:
    """Attach immutable paper/chunk identifiers and source offsets to chunks."""
    source_value = paper.get(section_label, "") if section_label != "abstract" else paper.get("abstract", "")
    source_text = str(source_value or "")
    paper_id = paper.get("paperId") or paper.get("paper_id")
    source_hash = sha256_text(source_text)
    records = []
    for index, chunk in enumerate(chunk_text_with_offsets(source_text, max_words=max_words)):
        chunk_id = stable_chunk_id(paper_id, section_label, index, chunk["text"])
        external_ids = paper.get("externalIds") or paper.get("external_ids") or {}
        records.append({
            # ``paperId`` remains for legacy pipeline compatibility; snake case
            # is the canonical forward-schema spelling.
            "paperId": paper_id,
            "paper_id": paper_id,
            "paper_title": paper.get("title"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "external_ids": external_ids,
            "doi": external_ids.get("DOI") or external_ids.get("doi"),
            "chunk_id": chunk_id,
            "chunk_index": index,
            "section_label": section_label,
            "source_text_kind": section_label,
            "source_text_sha256": source_hash,
            "source_char_start": chunk["source_char_start"],
            "source_char_end": chunk["source_char_end"],
            "chunk_text_sha256": sha256_text(chunk["text"]),
            "chunk_word_count": chunk["word_count"],
            "chunking_algorithm_version": CHUNKING_ALGORITHM_VERSION,
            "chunking_max_words": max_words,
            "text": chunk["text"],
        })
    return records


def write_stage1_provenance_manifest(
    *,
    topic: str,
    query_expansion: dict[str, Any],
    retrieval: dict[str, Any],
    screening: dict[str, Any],
    artifact_paths: dict[str, str],
    output_path: str | None = None,
) -> str:
    """Write a manifest tying Stage 1's immutable artifacts to its settings."""
    output_path = output_path or os.path.join(config.RAW_PAPERS_DIR, "stage1_provenance_manifest.json")
    manifest = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "record_type": "stage1_literature_provenance_manifest",
        "generated_at_utc": utc_now_iso(),
        "run_id": config.RUN_ID or None,
        "topic": topic,
        "topic_sha256": sha256_text(topic),
        "query_expansion": query_expansion,
        "retrieval": retrieval,
        "screening": screening,
        "artifact_sha256": {
            label: file_sha256(path)
            for label, path in artifact_paths.items()
            if os.path.exists(path)
        },
        "limitations": [
            "A later Semantic Scholar query can return different records; this manifest records the observed response IDs and hashes rather than claiming the live API is immutable.",
            "This forward schema does not reconstruct historical metadata missing from legacy artifacts.",
        ],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and screen papers from Semantic Scholar.")
    parser.add_argument("--topic", type=str, default="Microservices security", help="The topic to search for.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum records per expanded query.")
    parser.add_argument("--max-papers", type=int, default=300, help="Maximum screened papers to retain.")
    parser.add_argument("--screening-mode", choices=["llm", "lexical"], default="llm")
    args = parser.parse_args()
    
    # 1. Expand Query
    expanded, expansion_manifest = expand_query_with_provenance(args.topic)
    
    # 2. Search Papers
    fetched, retrieval_manifest = search_papers(
        expanded, limit_per_query=args.limit, return_manifest=True
    )
    
    # Save raw papers metadata
    raw_path = os.path.join(config.RAW_PAPERS_DIR, "papers_metadata.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(fetched, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved raw papers metadata to {raw_path}")
    
    # 3. Screen Relevance
    if args.screening_mode == "lexical":
        screened, screening_manifest = screen_relevance_lexical(
            fetched, args.max_papers, return_manifest=True
        )
    else:
        screened, screening_manifest = screen_relevance(
            fetched, args.topic, args.max_papers, return_manifest=True
        )
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    with open(screened_path, "w", encoding="utf-8") as f:
        json.dump(screened, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved screened papers to {screened_path}")
    
    # 4. Chunk Screened Papers
    all_chunks = []
    for p in screened:
        all_chunks.extend(build_chunk_records(p, max_words=1000, section_label="abstract"))
            
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[+] Created {len(all_chunks)} chunks from screened papers. Saved to {chunks_path}")
    manifest_path = write_stage1_provenance_manifest(
        topic=args.topic,
        query_expansion=expansion_manifest,
        retrieval=retrieval_manifest,
        screening=screening_manifest,
        artifact_paths={
            "papers_metadata": raw_path,
            "screened_papers": screened_path,
            "chunks": chunks_path,
        },
    )
    print(f"[+] Saved Stage 1 provenance manifest to {manifest_path}")

