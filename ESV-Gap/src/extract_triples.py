"""
Stage 2: Triple Extraction using LLM

Extracts structured knowledge triples (entity, relation, entity)
from each paper's abstract using GPT-4 with a controlled output schema.

Usage:
    python run_pipeline.py --stage extract
"""

import json
import re
import time
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm 
from openai import OpenAI  # Backward-compatible extension/test patch point.
from src.utils import (
    get_logger, save_json, load_json, save_jsonl, load_jsonl,
    ensure_dir, chunk_text, clean_text
)
from src.groq_key_pool import create_groq_client
from src.llm_errors import (
    LLMAuthenticationError,
    authentication_error_message,
    is_authentication_error,
)

logger = get_logger("extract")
load_dotenv()  # Load environment variables from .env file


_LIMITATION_PATTERNS = (
    (
        re.compile(
            r"^(?:however[,;]?\s+|yet\s+|in summary[,;]?\s+)?"
            r"(?P<subject>.{2,100}?)\s+(?:also\s+)?lacks?\s+"
            r"(?P<object>[^.;:]{3,220})",
            re.IGNORECASE,
        ),
        0.92,
    ),
    (
        re.compile(
            r"^(?:however[,;]?\s+)?(?P<subject>.{2,100}?)\s+"
            r"(?:faces?|faced|suffers?|suffered)\s+(?:from\s+)?"
            r"(?:significant\s+)?limitations?\s+(?:due\s+to|in|of)\s+"
            r"(?P<object>[^.;:]{3,220})",
            re.IGNORECASE,
        ),
        0.88,
    ),
)

_OPEN_CHALLENGE_PATTERN = re.compile(
    r"(?P<challenge>[^.;:]{5,180}?)\s+remains?\s+an?\s+open\s+challenge\b",
    re.IGNORECASE,
)


def _clean_limitation_phrase(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip(" ,:;-\"")
    value = re.sub(r"^(?:and|but|while)\s+", "", value, flags=re.IGNORECASE)
    return value


def deterministic_limitation_triples(domain, title, text):
    """Recover only explicit limitation statements missed by the LLM.

    These patterns intentionally exclude generic future-work prose. Downstream
    semantic, provenance, convergence, and closure gates still decide whether
    the extracted statement can participate in a research-gap claim.
    """
    triples = []
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        matched = False
        for pattern, confidence in _LIMITATION_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            subject = _clean_limitation_phrase(match.group("subject"))
            missing = _clean_limitation_phrase(match.group("object"))
            if re.match(r"^(?:it|they|their|this|these|those|we|our)\b", subject, re.I):
                subject = str(title).strip()
            if not subject or not missing:
                continue
            triples.append({
                "subject": {"name": subject, "type": "CONCEPT"},
                "relation": "LACKS",
                "object": {"name": missing, "type": "LIMITATION"},
                "confidence": confidence,
                "evidence": sentence,
                "extraction_method": "deterministic_limitation_pattern",
            })
            matched = True
            break
        if matched:
            continue
        open_match = _OPEN_CHALLENGE_PATTERN.search(sentence)
        if open_match:
            challenge = _clean_limitation_phrase(open_match.group("challenge"))
            challenge = re.sub(
                r"^(?:however[,;]?\s+|yet\s+)", "", challenge,
                flags=re.IGNORECASE,
            )
            if challenge:
                triples.append({
                    "subject": {"name": str(domain).strip(), "type": "CONCEPT"},
                    "relation": "LACKS",
                    "object": {"name": challenge, "type": "LIMITATION"},
                    "confidence": 0.85,
                    "evidence": sentence,
                    "extraction_method": "deterministic_open_challenge_pattern",
                })
    return triples


def augment_with_deterministic_limitations(result, paper, domain):
    """Add non-duplicate deterministic limitations to one paper result."""
    triples = list(result.get("triples", []))
    existing_lacks_evidence = {
        re.sub(r"\s+", " ", str(item.get("evidence", ""))).strip().casefold()
        for item in triples
        if str(item.get("relation", "")).upper() == "LACKS"
    }
    additions = deterministic_limitation_triples(
        domain, paper.get("title", ""), paper.get("abstract", "")
    )
    paper_id = paper.get("paperId", "")
    year = paper.get("year")
    for triple in additions:
        evidence_key = re.sub(
            r"\s+", " ", str(triple.get("evidence", ""))
        ).strip().casefold()
        if evidence_key in existing_lacks_evidence:
            continue
        triple["source_paper_id"] = paper_id
        triple["source_year"] = year
        triples.append(triple)
        existing_lacks_evidence.add(evidence_key)
    return {
        **result,
        "triples": triples,
        "num_triples": len(triples),
        "deterministic_limitations_added": len(triples) - len(result.get("triples", [])),
    }


class ExtractionRateLimitError(RuntimeError):
    """The provider asked the resumable extraction job to pause."""


def _normalise_triple_payload(result):
    """Return only structurally valid triples from a decoded response."""
    if not isinstance(result, dict):
        return []
    valid_triples = []
    raw = result.get("triples", [])
    triples = [triple for triple in raw if isinstance(triple, dict)]
    for triple in triples:
        if (
            "subject" in triple
            and "relation" in triple
            and "object" in triple
            and isinstance(triple["subject"], dict)
            and isinstance(triple["object"], dict)
            and "name" in triple["subject"]
            and "name" in triple["object"]
        ):
            triple["subject"]["name"] = str(triple["subject"]["name"]).strip()
            triple["object"]["name"] = str(triple["object"]["name"]).strip()
            triple.setdefault("confidence", 0.5)
            valid_triples.append(triple)
    return valid_triples


def _decode_generated_payload(content):
    """Decode JSON, including one known Groq missing-item-brace defect.

    The repair is intentionally narrow: it only inserts an opening object brace
    between array items when a new ``subject`` key follows a closed object.
    Anything else remains a failed extraction instead of being guessed.
    """
    text = str(content or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        repaired = re.sub(
            r"}\s*,\s*\"subject\"\s*:",
            '},{"subject":',
            text,
        )
        if repaired == text:
            return None
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            return None


def _failed_generation(exc):
    """Extract provider-supplied invalid JSON for conservative local repair."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("failed_generation"):
            return str(error["failed_generation"])
    return ""


def _retry_after_seconds(error_text):
    """Parse Groq durations such as ``6m12.03s`` without dropping minutes."""
    match = re.search(
        r"try again in\s*(?:(?P<hours>[0-9.]+)h)?"
        r"(?:(?P<minutes>[0-9.]+)m)?(?:(?P<seconds>[0-9.]+)s)?",
        error_text,
        re.I,
    )
    if not match or not any(match.groupdict().values()):
        return None
    return (
        float(match.group("hours") or 0) * 3600
        + float(match.group("minutes") or 0) * 60
        + float(match.group("seconds") or 0)
    )

def load_extraction_prompt(prompts_dir):
    """Load the triple extraction prompt template."""
    path = Path(prompts_dir) / "triple_extraction.txt"
    with open(path, "r") as f:
        return f.read()


def extract_triples_from_text(
    client,
    model,
    prompt_template,
    domain,
    title,
    year,
    text,
    temperature=0.1,
    max_retries=3,
):
    """
    Extract triples from a single text chunk.
    Returns list of triple dicts.
    """
    prompt = prompt_template.replace("{domain}", str(domain))\
                        .replace("{title}", str(title))\
                        .replace("{year}", str(year))\
                        .replace("{text_chunk}", str(text))
    
    for attempt in range(max(1, int(max_retries))):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research knowledge extraction agent. Output ONLY valid JSON with no additional text or markdown formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = _decode_generated_payload(content)
            if result is None:
                raise json.JSONDecodeError("invalid generated JSON", str(content), 0)
            return _normalise_triple_payload(result)

        except json.JSONDecodeError as exc:
            if attempt + 1 < max_retries:
                logger.warning(
                    "JSON parse error for '%s...'; retry %d/%d",
                    title[:40],
                    attempt + 1,
                    max_retries,
                )
                continue
            logger.warning(f"JSON parse error for '{title[:40]}...': {exc}")
            return []
        except Exception as exc:
            if is_authentication_error(exc):
                raise LLMAuthenticationError(authentication_error_message()) from exc
            failed_payload = _failed_generation(exc)
            if failed_payload:
                repaired = _decode_generated_payload(failed_payload)
                triples = _normalise_triple_payload(repaired)
                if triples:
                    logger.warning(
                        "Recovered %d triples from provider-rejected JSON for '%s...'.",
                        len(triples),
                        title[:40],
                    )
                    return triples
                if attempt + 1 < max_retries:
                    logger.warning(
                        "Provider rejected JSON for '%s...'; retry %d/%d",
                        title[:40],
                        attempt + 1,
                        max_retries,
                    )
                    continue
            error_text = str(exc)
            rate_limited = "429" in error_text or "rate_limit" in error_text
            daily_quota = "tokens per day" in error_text.lower() or "tpd" in error_text.lower()
            suggested = _retry_after_seconds(error_text)

            # Do not burn more requests or write false zero-triple results when
            # the daily budget is exhausted or the requested pause is long.
            if rate_limited and (daily_quota or (suggested is not None and suggested > 30)):
                wait_note = (
                    f" Provider suggests retrying in about {suggested:.0f} seconds."
                    if suggested is not None else ""
                )
                raise ExtractionRateLimitError(
                    "Groq extraction quota is temporarily exhausted. Progress was "
                    f"checkpointed and can be resumed later.{wait_note}"
                ) from exc

            if rate_limited and attempt + 1 < max_retries:
                wait = min(max((suggested or 5.0) + 1.0, 2.0), 30.0)
                logger.warning(
                    f"Extraction rate limited for '{title[:40]}...'; "
                    f"retry {attempt + 1}/{max_retries} in {wait:.1f}s."
                )
                time.sleep(wait)
                continue
            if rate_limited:
                raise ExtractionRateLimitError(
                    "Groq remained rate-limited after bounded retries. Progress "
                    "was checkpointed and can be resumed later."
                ) from exc
            logger.warning(f"Extraction failed for '{title[:40]}...': {exc}")
            return []

    return []


def extract_paper_triples(client, model, prompt_template, domain, paper, chunk_size=1500, chunk_overlap=200):
    """
    Extract triples from a single paper.
    Uses the abstract (and full text if available).
    Returns a dict with paper info and extracted triples.
    """
    title = paper.get("title", "Unknown")
    year = paper.get("year", "Unknown")
    paper_id = paper.get("paperId", "unknown")
    
    # Use abstract as primary text source
    text = clean_text(paper.get("abstract", ""))
    
    if not text:
        logger.warning(f"No text for paper: {title[:50]}")
        return {
            "paperId": paper_id,
            "title": title,
            "year": year,
            "num_triples": 0,
            "triples": [],
        }
    
    # For abstracts, usually no chunking needed (they're short)
    # But if we have full text later, chunking kicks in
    all_triples = []
    
    if len(text.split()) > chunk_size:
        chunks = chunk_text(text, chunk_size, chunk_overlap)
    else:
        chunks = [text]
    
    for chunk in chunks:
        triples = extract_triples_from_text(
            client, model, prompt_template, domain, title, year, chunk
        )
        
        # Tag each triple with source info
        for t in triples:
            t["source_paper_id"] = paper_id
            t["source_year"] = year
        
        all_triples.extend(triples)
    
    result = {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "num_triples": len(all_triples),
        "triples": all_triples,
    }
    return augment_with_deterministic_limitations(result, paper, domain)


def backfill_deterministic_limitations(config):
    """Augment checkpointed paper files without making any API requests."""
    proc_dir = Path(config["paths"]["processed_data"])
    triples_dir = ensure_dir(config["paths"]["triples"])
    corpus_path = proc_dir / "corpus_filtered.jsonl"
    if not corpus_path.exists():
        return {"papers_updated": 0, "limitations_added": 0}
    papers = {
        str(paper.get("paperId", "")): paper
        for paper in load_jsonl(corpus_path)
        if paper.get("paperId")
    }
    papers_updated = 0
    limitations_added = 0
    for paper_file in triples_dir.glob("paper_*.json"):
        result = load_json(paper_file)
        paper = papers.get(str(result.get("paperId", "")))
        if not paper:
            continue
        augmented = augment_with_deterministic_limitations(
            result, paper, config.get("project", {}).get("domain", "")
        )
        added = int(augmented.get("deterministic_limitations_added", 0))
        if added:
            save_json(augmented, paper_file)
            papers_updated += 1
            limitations_added += added

    all_results = [
        load_json(path) for path in triples_dir.glob("paper_*.json")
    ]
    all_triples = [
        triple
        for result in all_results
        for triple in result.get("triples", [])
    ]
    if all_results:
        save_json(all_triples, triples_dir / "all_triples.json")
        progress_path = triples_dir / "extraction_progress.json"
        progress = load_json(progress_path) if progress_path.exists() else {}
        progress["total_triples"] = len(all_triples)
        save_json(progress, progress_path)
    logger.info(
        "Deterministic limitation backfill: %d paper(s), %d statement(s) added",
        papers_updated,
        limitations_added,
    )
    return {
        "papers_updated": papers_updated,
        "limitations_added": limitations_added,
    }


def extract_all_triples(config, progress_callback=None):
    """
    Main extraction function.
    Processes all papers in the filtered corpus and extracts triples.
    """
    proc_dir = Path(config["paths"]["processed_data"])
    triples_dir = ensure_dir(config["paths"]["triples"])
    
    ext_config = config["extraction"]
    model = ext_config["model"]
    domain = config["project"]["domain"]
    chunk_size = ext_config["chunk_size"]
    chunk_overlap = ext_config["chunk_overlap"]
    
    # Load filtered corpus
    corpus_path = proc_dir / "corpus_filtered.jsonl"
    if not corpus_path.exists():
        logger.error(f"Filtered corpus not found: {corpus_path}")
        logger.error("Run 'python run_pipeline.py --stage filter' first.")
        return
    
    papers = load_jsonl(corpus_path)
    logger.info(f"Loaded {len(papers)} papers from filtered corpus")
    
    # Load prompt template
    prompt_template = load_extraction_prompt(config["paths"]["prompts"])
    
    # Initialise Groq client through the OpenAI-compatible SDK.
    # Prefer the key supplied in config (used by the Streamlit app), with
    # environment variables retained for CLI runs.
    client = create_groq_client(config, timeout=30.0, max_retries=1)
    logger.info("Groq key pool: %d key(s) available for extraction", client.pool_size)
    
    # --- Check for existing progress ---
    progress_path = triples_dir / "extraction_progress.json"
    completed_ids = set()
    all_results = []
    
    if progress_path.exists():
        existing = load_json(progress_path)
        completed_ids = set(existing.get("completed_ids", []))
        logger.info(f"Resuming from checkpoint: {len(completed_ids)} papers already done")
    
    # Load any existing per-paper results
    for paper_file in triples_dir.glob("paper_*.json"):
        try:
            data = load_json(paper_file)
            all_results.append(data)
            if data.get("paperId"):
                completed_ids.add(data["paperId"])
        except:
            pass
    
    # --- Extract triples ---
    logger.info(f"Extracting triples with {model}")
    logger.info(f"Domain: {domain}")
    
    # Include checkpointed papers in the displayed count. Without this,
    # another quota pause during resume would overwrite progress with zero.
    total_triples = sum(
        int(result.get("num_triples", len(result.get("triples", []))) or 0)
        for result in all_results
    )
    
    for i, paper in enumerate(tqdm(papers, desc="Extracting triples")):
        paper_id = paper.get("paperId", "")
        
        # Skip already processed
        if paper_id in completed_ids:
            if progress_callback:
                progress_callback(i + 1, len(papers))
            continue
        
        try:
            result = extract_paper_triples(
                client, model, prompt_template, domain, paper,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        except ExtractionRateLimitError:
            save_json(
                {"completed_ids": sorted(completed_ids), "total_triples": total_triples},
                progress_path,
            )
            raise
        
        total_triples += result["num_triples"]
        all_results.append(result)
        completed_ids.add(paper_id)
        
        # Save per-paper result
        save_json(result, triples_dir / f"paper_{paper_id[:12]}.json")
        
        # Save progress checkpoint every 10 papers
        if (i + 1) % 10 == 0:
            save_json(
                {"completed_ids": list(completed_ids), "total_triples": total_triples},
                progress_path
            )
            logger.info(f"  Checkpoint: {len(completed_ids)}/{len(papers)} papers, {total_triples} triples")

        if progress_callback:
            progress_callback(i + 1, len(papers))
        
        # Rate limiting
        time.sleep(1.0)

    # Old checkpoint files may predate the deterministic high-precision
    # limitation extractor. Backfill them locally before graph construction.
    backfill_deterministic_limitations(config)
    
    # --- Aggregate all triples ---
    all_triples = []
    for paper_file in triples_dir.glob("paper_*.json"):
        all_triples.extend(load_json(paper_file).get("triples", []))
    
    # Save aggregated triples
    save_json(all_triples, triples_dir / "all_triples.json")
    save_json(
        {"completed_ids": list(completed_ids), "total_triples": len(all_triples)},
        progress_path
    )
    
    # --- Print stats ---
    papers_with_triples = sum(1 for r in all_results if r.get("num_triples", 0) > 0)
    avg_triples = len(all_triples) / max(len(all_results), 1)
    
    # Count relation types
    relation_counts = {}
    entity_types = {}
    for t in all_triples:
        rel = t.get("relation", "UNKNOWN")
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        for role in ["subject", "object"]:
            etype = t.get(role, {}).get("type", "UNKNOWN")
            entity_types[etype] = entity_types.get(etype, 0) + 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"  EXTRACTION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Papers processed: {len(all_results)}")
    logger.info(f"  Papers with triples: {papers_with_triples}")
    logger.info(f"  Total triples: {len(all_triples)}")
    logger.info(f"  Avg triples/paper: {avg_triples:.1f}")
    logger.info(f"\n  Relation distribution:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {rel}: {count}")
    logger.info(f"\n  Entity type distribution:")
    for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        logger.info(f"    {etype}: {count}")
    logger.info(f"\n  Saved to: {triples_dir / 'all_triples.json'}")

    return all_triples


if __name__ == "__main__":
    import yaml # type: ignore
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    extract_all_triples(config)
