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
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm 
from openai import OpenAI 
from src.utils import (
    get_logger, save_json, load_json, save_jsonl, load_jsonl,
    ensure_dir, chunk_text, clean_text
)

logger = get_logger("extract")
load_dotenv()  # Load environment variables from .env file


class ExtractionRateLimitError(RuntimeError):
    """The provider asked the resumable extraction job to pause."""


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
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            valid_triples = []
            raw = result.get("triples", [])
            triples = [t for t in raw if isinstance(t, dict)]
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
                    triple["subject"]["name"] = triple["subject"]["name"].strip()
                    triple["object"]["name"] = triple["object"]["name"].strip()
                    triple.setdefault("confidence", 0.5)
                    valid_triples.append(triple)

            return valid_triples

        except json.JSONDecodeError as exc:
            logger.warning(f"JSON parse error for '{title[:40]}...': {exc}")
            return []
        except Exception as exc:
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
    
    return {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "num_triples": len(all_triples),
        "triples": all_triples,
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
    groq_api_key = (
        config.get("api_keys", {}).get("groq")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not groq_api_key:
        raise ValueError(
            "Missing Groq API key. Enter it in the app or set GROQ_API_KEY."
        )
    client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=30.0,
        max_retries=1,
        )
    
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
    
    # --- Aggregate all triples ---
    all_triples = []
    for result in all_results:
        all_triples.extend(result.get("triples", []))
    
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
