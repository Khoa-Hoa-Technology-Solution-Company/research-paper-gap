"""
Scalability Experiment - Profiles performance for 20, 50, 100, and 200 papers
by scaling the graph size to distinct nodes and profiling ER, Louvain, and Decay.
"""
import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.entity_resolution import resolve_entities
from src.graph_analysis import build_graph, detect_orphan_clusters, compute_temporal_decay


def generate_distinct_triples(triples, scale_factor, suffix):
    """
    Generate distinct triples by renaming entities uniquely so they do NOT merge,
    thus scaling the graph nodes and edges size.
    """
    scaled = []
    # Add original
    scaled.extend(triples)
    
    # Add uniquely renamed copies
    for f in range(1, scale_factor):
        for t in triples:
            # We rename entities to simulate new papers with new distinct concepts
            new_sub = f"{t['subject']}_{suffix}_{f}"
            new_obj = f"{t['object']}_{suffix}_{f}"
            scaled.append({
                "subject": new_sub,
                "subject_type": t["subject_type"],
                "relation": t["relation"],
                "object": new_obj,
                "object_type": t["object_type"],
                "confidence": t["confidence"],
                "evidence_quote": t["evidence_quote"],
                "year": t["year"]
            })
    return scaled


def run_scalability_experiment():
    print("=" * 60)
    print("SCALABILITY EXPERIMENT PROFILING (DISTINCT NODES)")
    print("=" * 60)

    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    if not os.path.exists(raw_path):
        print("[!] raw_triples.json not found.")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_triples = json.load(f)

    # This is a synthetic triple-replication benchmark, not a paper-count
    # experiment. It starts from raw triples and temporarily disables normal
    # entity merging so that graph size can be controlled.
    scales = [
        {"name": "Raw-triple seed (1x)", "factor": 1, "suffix": ""},
        {"name": "Synthetic replication (3x)", "factor": 3, "suffix": "x3"},
        {"name": "Synthetic replication (5x)", "factor": 5, "suffix": "x5"},
        {"name": "Synthetic replication (10x)", "factor": 10, "suffix": "x10"},
    ]

    results = []

    for s in scales:
        print(f"\n[*] Profiling scale: {s['name']}...")
        scaled_raw = generate_distinct_triples(raw_triples, s["factor"], s["suffix"])
        
        # 1. Entity Resolution Time (simulating ER matching, S-BERT encoding)
        er_start = time.time()
        # We temporarily set config threshold very high (98%) to prevent merging,
        # ensuring nodes and edges scale up properly for graph profiling.
        orig_fuzzy = config.FUZZY_MATCH_THRESHOLD
        orig_cosine = config.COSINE_SIMILARITY_THRESHOLD
        config.FUZZY_MATCH_THRESHOLD = 99
        config.COSINE_SIMILARITY_THRESHOLD = 0.99
        
        resolved, mapping = resolve_entities(scaled_raw)
        
        config.FUZZY_MATCH_THRESHOLD = orig_fuzzy
        config.COSINE_SIMILARITY_THRESHOLD = orig_cosine
        
        er_time = time.time() - er_start
        
        # 2. Graph building time
        graph_start = time.time()
        G = build_graph(resolved)
        graph_build_time = time.time() - graph_start
        
        # 3. Louvain Modularity clustering time
        louvain_start = time.time()
        orphans = detect_orphan_clusters(G)
        louvain_time = time.time() - louvain_start
        
        # 4. Temporal Decay analysis time
        decay_start = time.time()
        stagnant = compute_temporal_decay(G)
        decay_time = time.time() - decay_start
        
        graph_total_time = time.time() - graph_start
        total_time = er_time + graph_total_time

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        print(f"  Nodes: {n_nodes}, Edges: {n_edges}")
        print(f"  ER Time: {er_time:.4f}s, Modularity Time: {louvain_time:.4f}s, Decay Time: {decay_time:.4f}s")
        print(f"  Total Time: {total_time:.4f}s")

        results.append({
            "scale": s["name"],
            "nodes": n_nodes,
            "edges": n_edges,
            "er_seconds": round(er_time, 4),
            "graph_build_seconds": round(graph_build_time, 4),
            "louvain_seconds": round(louvain_time, 4),
            "decay_seconds": round(decay_time, 4),
            "total_seconds": round(total_time, 4),
            "gaps_count": len(orphans) + len(stagnant)
        })

    out_path = os.path.join(config.DATA_DIR, "scalability_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Scalability results saved to {out_path}")

    # Output Markdown Table
    print("\n| Corpus Scale | Nodes | Edges | ER Time (s) | Modularity Time (s) | Decay Time (s) | Total Time (s) | Gaps Candidates |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['scale']} | {r['nodes']} | {r['edges']} | {r['er_seconds']:.4f} | {r['louvain_seconds']:.4f} | {r['decay_seconds']:.4f} | {r['total_seconds']:.4f} | {r['gaps_count']} |")


if __name__ == "__main__":
    run_scalability_experiment()
