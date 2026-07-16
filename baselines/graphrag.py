"""
B4: GraphRAG Baseline - Multi-hop subgraph retrieval + LLM gap generation.
Builds a local KG from triples, retrieves multi-hop subgraphs around key entities,
and uses subgraph context for LLM gap generation.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.llm_client import call_llm
from src.graph_analysis import build_graph


def extract_subgraph_context(G, seed_nodes, max_hops=2):
    """Extract multi-hop neighborhood around seed nodes as text context."""
    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(max_hops):
        new_frontier = set()
        for node in frontier:
            for neighbor in list(G.successors(node)) + list(G.predecessors(node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_frontier.add(neighbor)
        frontier = new_frontier

    # Build text description of subgraph
    lines = []
    for u, v, data in G.edges(data=True):
        if u in visited and v in visited:
            rel = data.get("relation", "RELATED_TO")
            lines.append(f"({u}) --[{rel}]--> ({v})")

    return "\n".join(lines[:30])  # Limit to 30 edges to avoid token bloat


def run_graphrag_baseline(G, papers):
    """
    GraphRAG: For each paper, identify key entities in the KG,
    retrieve multi-hop subgraph, use as context for LLM gap generation.
    """
    all_gaps = []
    print(f"[*] Running GraphRAG Baseline (B4) on {len(papers)} papers...")

    graph_nodes = set(G.nodes())

    for idx, p in enumerate(papers):
        title = p["title"]
        abstract = p["abstract"]
        print(f"[*] Processing paper {idx+1}/{len(papers)}: '{title[:40]}...'")

        # Find entities from this paper's abstract that exist in KG
        abstract_lower = abstract.lower()
        seed_nodes = [n for n in graph_nodes if n.lower() in abstract_lower][:5]

        if not seed_nodes:
            # Fallback: use first 3 graph nodes
            seed_nodes = list(graph_nodes)[:3]

        subgraph_text = extract_subgraph_context(G, seed_nodes, max_hops=2)

        prompt = f"""Act as a Software Engineering researcher. You are given:
1. A target paper's abstract
2. A knowledge graph subgraph showing relationships between concepts in the literature

Target Paper:
Title: {title}
Abstract: {abstract}

Knowledge Graph Context (multi-hop subgraph):
{subgraph_text}

Based on the graph structure and the paper, identify an implicit research gap.
Output strictly as a JSON object (no markdown):
{{
  "Grounds": "Evidence from the graph and paper that suggests a gap.",
  "Claim": "The research gap statement.",
  "Warrant": "Technical justification for why this gap matters.",
  "Bucket": "more_probable or least_probable"
}}"""

        try:
            content = call_llm(prompt, temperature=0.2)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            gap = json.loads(content)
            gap["source_paper"] = title
            gap["type"] = "graphrag"
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

    G = build_graph(triples)
    gaps = run_graphrag_baseline(G, papers)

    out_path = os.path.join(config.GAPS_DIR, "baseline_graphrag.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved B4 GraphRAG gaps ({len(gaps)}) to {out_path}")


if __name__ == "__main__":
    main()
