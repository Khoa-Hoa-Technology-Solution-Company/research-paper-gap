"""
B5: LightRAG Baseline - Lightweight KG-enhanced RAG.
Extracts key entities from abstracts, builds entity co-occurrence graph,
retrieves connected entities as context for LLM gap generation.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.llm_client import call_llm


def build_entity_cooccurrence(papers, triples):
    """Build lightweight entity co-occurrence from triples, keyed by paper."""
    # Map entities to papers via year (approximate)
    year_entities = defaultdict(set)
    for t in triples:
        year_entities[t["year"]].add(t["subject"])
        year_entities[t["year"]].add(t["object"])

    # Build co-occurrence: entities that appear in the same year's triples
    cooccurrence = defaultdict(set)
    for year, entities in year_entities.items():
        for e in entities:
            cooccurrence[e].update(entities - {e})

    return cooccurrence


def run_lightrag_baseline(papers, triples):
    """LightRAG: entity-based retrieval + LLM gap generation."""
    all_gaps = []
    print(f"[*] Running LightRAG Baseline (B5) on {len(papers)} papers...")

    cooccurrence = build_entity_cooccurrence(papers, triples)
    all_entities = set(cooccurrence.keys())

    for idx, p in enumerate(papers):
        title = p["title"]
        abstract = p["abstract"]
        print(f"[*] Processing paper {idx+1}/{len(papers)}: '{title[:40]}...'")

        # Find entities mentioned in abstract
        abstract_lower = abstract.lower()
        matched = [e for e in all_entities if e.lower() in abstract_lower][:5]

        # Get co-occurring entities as context
        related = set()
        for e in matched:
            related.update(list(cooccurrence.get(e, set()))[:5])

        entity_context = ", ".join(list(related)[:15])

        prompt = f"""Act as a Software Engineering researcher. You are given a paper and related entities from a lightweight knowledge graph.

Target Paper:
Title: {title}
Abstract: {abstract}

Related KG Entities: {entity_context}

Based on the paper and related entities, identify a research gap.
Output strictly as a JSON object (no markdown):
{{
  "Grounds": "Evidence from the paper and entity relationships.",
  "Claim": "The research gap statement.",
  "Warrant": "Technical justification.",
  "Bucket": "near_term_feasible or long_term_or_speculative"
}}"""

        try:
            content = call_llm(prompt, temperature=0.2)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            gap = json.loads(content)
            gap["source_paper"] = title
            gap["type"] = "lightrag"
            all_gaps.append(gap)
        except Exception as e:
            print(f"[!] Error: {e}")

    return all_gaps


def main():
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")

    with open(resolved_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    with open(screened_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    gaps = run_lightrag_baseline(papers, triples)

    out_path = os.path.join(config.GAPS_DIR, "baseline_lightrag.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved B5 LightRAG gaps ({len(gaps)}) to {out_path}")


if __name__ == "__main__":
    main()
