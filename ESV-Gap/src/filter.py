"""
Stage 1b: Corpus Filtering

Filters the raw collected papers down to the final corpus
using LLM-based relevance screening (mimicking PRISMA-style
title/abstract screening).

Usage:
    python run_pipeline.py --stage filter
"""

import json
import math
import re
import time
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from src.utils import get_logger, save_json, save_jsonl, load_jsonl, ensure_dir
from src.groq_key_pool import create_groq_client, is_rate_limit_error
from src.llm_errors import (
    LLMAuthenticationError,
    LLMRateLimitError,
    authentication_error_message,
    is_authentication_error,
    rate_limit_error_message,
)

logger = get_logger("filter")
load_dotenv()  # Load environment variables from .env file

DOMAIN_STOPWORDS = {
    "a", "an", "and", "for", "in", "of", "on", "the", "to", "with",
}


def _semantic_anchor_text(value):
    """Collapse common scholarly aliases into comparable topic anchors."""
    text = str(value or "").casefold()
    replacements = (
        (r"\b(?:internet\s+of\s+things|iot)(?:\s+networks?)?\b", " iot "),
        (r"\b(?:intrusion\s+detection(?:\s+systems?)?|ids)\b", " intrusiondetection "),
        (
            r"\b(?:adversarial\s+(?:robustness|attacks?|examples?)|"
            r"evasion\s+attacks?|poisoning\s+attacks?)\b",
            " adversarialrobustness ",
        ),
        (
            r"\b(?:deep[-\s]+learning|deep\s+neural\s+networks?|"
            r"neural\s+networks?)\b",
            " deeplearning ",
        ),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def _domain_root(token):
    token = str(token).casefold()
    if token.startswith((
        "secur", "vulnerab", "attack", "exploit", "threat", "encrypt",
        "cryptograph", "isolat", "malware", "privacy", "protect",
    )) or token in {"xss", "sqli", "csrf"}:
        return "secur"
    if token.startswith("monolith"):
        return "monolith"
    if token.startswith("microservic"):
        return "microservice"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def domain_anchor_coverage(domain, title, abstract):
    """Measure substantive lexical anchoring to the requested topic."""
    domain_roots = {
        _domain_root(token)
        for token in re.findall(r"[a-z0-9]+", _semantic_anchor_text(domain))
        if token not in DOMAIN_STOPWORDS
    }
    if not domain_roots:
        return 1.0
    text_roots = {
        _domain_root(token)
        for token in re.findall(
            r"[a-z0-9]+", _semantic_anchor_text(f"{title} {abstract}")
        )
    }
    return len(domain_roots.intersection(text_roots)) / len(domain_roots)


def select_query_year_balanced(papers, target_size):
    """Cap a relevant corpus without collapsing it to one query or year.

    Input order remains the within-stratum quality ranking. Selection then
    round-robins over (retrieval query, publication year) strata.
    """
    if len(papers) <= target_size:
        return list(papers)
    strata = {}
    for paper in papers:
        queries = paper.get("_matched_queries", []) or ["untracked-query"]
        year = str(paper.get("year") or "unknown-year")
        for query in queries:
            strata.setdefault((str(query), year), []).append(paper)
    selected = []
    selected_ids = set()
    positions = {key: 0 for key in strata}
    def stratum_order(key):
        try:
            year_order = -int(key[1])
        except (TypeError, ValueError):
            year_order = 1
        return year_order, key[0].casefold()

    ordered_strata = sorted(strata, key=stratum_order)
    while len(selected) < target_size:
        progressed = False
        for key in ordered_strata:
            bucket = strata[key]
            while positions[key] < len(bucket):
                paper = bucket[positions[key]]
                positions[key] += 1
                identity = str(
                    paper.get("paperId") or paper.get("paper_id") or id(paper)
                )
                if identity in selected_ids:
                    continue
                selected.append(paper)
                selected_ids.add(identity)
                progressed = True
                break
            if len(selected) >= target_size:
                break
        if not progressed:
            break
    return selected

def load_prompt_template(prompts_dir):
    """Load the abstract filtering prompt template."""
    path = Path(prompts_dir) / "filter_abstract.txt"
    with open(path, "r") as f:
        return f.read()


def screen_paper(client, model, prompt_template, domain, title, abstract):
    """
    Use LLM to screen a single paper for relevance.
    Returns dict with relevant (bool), confidence (float), reason (str).
    Includes automatic retry with backoff for rate limiting.
    """
    prompt = prompt_template.replace("{domain}", str(domain))\
                            .replace("{title}", str(title))\
                            .replace("{abstract}", str(abstract))

    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research paper screening assistant. Respond only with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return {
                "relevant": bool(result.get("relevant", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "reason": str(result.get("reason", "")),
            }

        except Exception as e:
            if is_authentication_error(e):
                raise LLMAuthenticationError(authentication_error_message()) from e
            err_str = str(e)
            if is_rate_limit_error(e):
                daily_quota = any(marker in err_str.casefold() for marker in (
                    "tokens per day", "requests per day", "daily quota", " tpd", " rpd"
                ))
                if daily_quota or attempt + 1 >= max_retries:
                    raise LLMRateLimitError(rate_limit_error_message()) from e
                wait = 10 * (attempt + 1)   # 10s, 20s, 30s, 40s, 50s
                logger.warning(
                    f"All configured keys were rate limited. Waiting {wait}s before retry "
                    f"{attempt + 1}/{max_retries}..."
                )
                time.sleep(wait)
                continue
            logger.warning(f"Screening failed for '{title[:50]}...': {e}")
            return {"relevant": False, "confidence": 0.0, "reason": f"Error: {e}"}

    raise LLMRateLimitError(rate_limit_error_message())


def filter_corpus(config, progress_callback=None):
    """
    Main filtering function.
    Loads raw papers, screens with LLM, filters to target corpus size.
    """
    raw_dir  = Path(config["paths"]["raw_data"])
    proc_dir = ensure_dir(config["paths"]["processed_data"])

    filter_config = config["filtering"]
    target_size   = filter_config["target_corpus_size"]
    model         = filter_config["relevance_model"]
    threshold     = filter_config["relevance_threshold"]
    lexical_guard = bool(filter_config.get("lexical_domain_guard", True))
    anchor_threshold = float(filter_config.get("min_domain_anchor_coverage", 0.60))
    screening_multiplier = float(
        filter_config.get("screening_candidate_multiplier", 1.25)
    )
    screening_target = max(target_size, math.ceil(target_size * screening_multiplier))
    domain        = config["project"]["domain"]

    # Load raw papers
    raw_path = raw_dir / "all_papers_raw.jsonl"
    if not raw_path.exists():
        logger.error(f"Raw data not found: {raw_path}")
        logger.error("Run 'python run_pipeline.py --stage collect' first.")
        return []

    papers = load_jsonl(raw_path)
    logger.info(f"Loaded {len(papers)} raw papers")

    # Load prompt template
    prompt_template = load_prompt_template(config["paths"]["prompts"])

    # Initialise client — Groq with OpenAI-compatible SDK.
    # The Streamlit app passes the key through config; keep the environment
    # variable as a fallback for CLI runs.
    client = create_groq_client(config, timeout=30.0, max_retries=0)
    logger.info("Groq key pool: %d key(s) available for screening", client.pool_size)

    # --- Screen papers ---
    logger.info(f"Screening papers with {model} (threshold: {threshold})")
    logger.info(f"This may take a while for {len(papers)} papers...")

    screened        = []
    relevant_count  = 0

    for i, paper in enumerate(tqdm(papers, desc="Screening")):
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not abstract:
            if progress_callback:
                progress_callback(i + 1, len(papers))
            continue

        anchor_coverage = domain_anchor_coverage(domain, title, abstract)
        used_llm = False
        if lexical_guard and anchor_coverage < anchor_threshold:
            result = {
                "relevant": False,
                "confidence": 0.0,
                "reason": (
                    "Rejected before LLM screening by deterministic domain-anchor "
                    f"guard: coverage {anchor_coverage:.2f} is below "
                    f"{anchor_threshold:.2f}."
                ),
            }
        else:
            used_llm = True
            result = screen_paper(
                client, model, prompt_template, domain, title, abstract
            )

        result["domain_anchor_coverage"] = round(anchor_coverage, 4)

        paper["screening"] = result
        screened.append(paper)

        if result["relevant"] and result["confidence"] >= threshold:
            relevant_count += 1

        # Save progress every 50 papers
        if (i + 1) % 50 == 0:
            save_jsonl(screened, proc_dir / "screening_progress.jsonl")
            logger.info(
                f"  Progress: {i + 1}/{len(papers)}, "
                f"relevant so far: {relevant_count}"
            )

        if progress_callback:
            progress_callback(i + 1, len(papers))

        # Rate limiting — 3 seconds keeps us safely under 6k TPM on free tier
        # The collection pool is deliberately larger than the requested final
        # corpus. Stop as soon as enough relevant papers have been retained.
        if relevant_count >= screening_target:
            logger.info(
                "  Reached balancing pool of %d relevant papers for target %d "
                "after screening %d/%d",
                screening_target, target_size,
                i + 1,
                len(papers),
            )
            break

        if used_llm:
            time.sleep(3)

    # --- Filter to relevant papers ---
    relevant_papers = [
        p for p in screened
        if p["screening"]["relevant"] and p["screening"]["confidence"] >= threshold
    ]

    logger.info(f"\nScreening complete:")
    logger.info(f"  Total screened:            {len(screened)}")
    logger.info(f"  Relevant (above threshold): {len(relevant_papers)}")

    # --- Sort by confidence * citations to get best papers ---
    relevant_papers.sort(
        key=lambda p: (
            p["screening"]["confidence"]
            * (1 + p.get("citationCount", 0) ** 0.5)
        ),
        reverse=True,
    )

    # --- Cap to target corpus size ---
    if len(relevant_papers) > target_size:
        final_corpus = select_query_year_balanced(relevant_papers, target_size)
        logger.info(
            "  Query/year-balanced cap to target size: %d", target_size
        )
    else:
        final_corpus = relevant_papers
        logger.info(
            f"  Final corpus size: {len(final_corpus)} "
            f"(target was {target_size})"
        )

    # --- Save results ---
    save_jsonl(final_corpus, proc_dir / "corpus_filtered.jsonl")
    save_jsonl(screened,     proc_dir / "all_screened.jsonl")

    # Save human-readable summary
    summary = []
    for p in final_corpus:
        summary.append({
            "paperId":        p.get("paperId"),
            "title":          p.get("title"),
            "year":           p.get("year"),
            "citations":      p.get("citationCount", 0),
            "relevance_score": p["screening"]["confidence"],
            "reason":         p["screening"]["reason"],
        })
    save_json(summary, proc_dir / "corpus_summary.json")
    save_json({
        "requested_target": target_size,
        "raw_paper_count": len(papers),
        "screened_paper_count": len(screened),
        "retained_paper_count": len(final_corpus),
        "target_attainment": round(len(final_corpus) / max(target_size, 1), 4),
        "retention_rate": round(len(final_corpus) / max(len(screened), 1), 4),
        "mean_domain_anchor_coverage": round(
            sum(
                float(p.get("screening", {}).get("domain_anchor_coverage", 0.0))
                for p in final_corpus
            ) / max(len(final_corpus), 1),
            4,
        ),
        "adequate_for_requested_target": len(final_corpus) >= target_size,
        "selection_strategy": "query_year_balanced_round_robin",
        "screening_candidate_multiplier": screening_multiplier,
        "warning": (
            None if len(final_corpus) >= target_size else
            "The requested number is a target, not the retained corpus size. "
            "Gap claims remain bounded to the smaller screened corpus."
        ),
    }, proc_dir / "screening_diagnostics.json")

    # --- Print stats ---
    years    = [p["year"] for p in final_corpus if p.get("year")]
    year_dist = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1

    logger.info(f"\n{'='*50}")
    logger.info(f"  FILTERING COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Final corpus: {len(final_corpus)} papers")
    logger.info(f"  Year distribution:")
    for y in sorted(year_dist.keys()):
        logger.info(f"    {y}: {year_dist[y]} papers")
    logger.info(f"  Saved to: {proc_dir / 'corpus_filtered.jsonl'}")

    return final_corpus


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    filter_corpus(config)
