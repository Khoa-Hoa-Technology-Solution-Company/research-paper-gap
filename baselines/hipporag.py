"""
B6: HippoRAG Baseline - Passage-level retrieval augmented by KG entity associations.
Uses the KG to re-rank retrieved passages based on entity overlap, then prompts LLM.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.llm_client import call_llm


def compute_entity_overlap(text, kg_entities):
    """Score text by how many KG entities it mentions."""
    text_lower = text.lower()
    return sum(1 for e in kg_entities if e.lower() in text_lower)


def run_hipporag_baseline(papers, chunks, triples):
    """
    HippoRAG: For each paper, retrieve chunks re-ranked by KG entity density,
    then prompt LLM with top-k enriched passages.
    """
    all_gaps = []
    print(f"[*] Running HippoRAG Baseline (B6) on {len(papers)} papers...")

    # Collect KG entities
    kg_entities = set()
    for t in triples:
        kg_entities.add(t["subject"])
        kg_entities.add(t["object"])

    for idx, p in enumerate(papers):
        title = p["title"]
        abstract = p["abstract"]
        print(f"[*] Processing paper {idx+1}/{len(papers)}: '{title[:40]}...'")

        # Get chunks not from this paper
        other_chunks = [c for c in chunks if c["title"] != title]

        # Score and rank chunks by KG entity overlap
        scored = []
        for c in other_chunks:
            score = compute_entity_overlap(c["text"], kg_entities)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored[:3]

        context = "\n\n".join([
            f"Passage {i+1} (KG entity overlap: {s}):\n{c['text'][:500]}"
            for i, (s, c) in enumerate(top_chunks)
        ])

        prompt = f"""Act as a Software Engineering researcher. You are given a target paper and passages from related papers, ranked by their connection to a knowledge graph.

Target Paper:
Title: {title}
Abstract: {abstract}

KG-Ranked Passages:
{context}

Based on the paper and KG-ranked passages, identify a research gap.
Output strictly as a JSON object (no markdown):
{{
  "Grounds": "Evidence from the paper and passages.",
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
            gap["type"] = "hipporag"
            all_gaps.append(gap)
        except Exception as e:
            print(f"[!] Error: {e}")

    return all_gaps


def main():
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")

    with open(resolved_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    with open(screened_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    gaps = run_hipporag_baseline(papers, chunks, triples)

    out_path = os.path.join(config.GAPS_DIR, "baseline_hipporag.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved B6 HippoRAG gaps ({len(gaps)}) to {out_path}")


if __name__ == "__main__":
    main()
