"""
Runtime Analysis - Measures execution times of various pipeline components.
Runs offline to profile performance.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.entity_resolution import resolve_entities
from src.graph_analysis import build_graph, detect_orphan_clusters, compute_temporal_decay


def profile_runtime():
    print("=" * 60)
    print("RUNTIME PERFORMANCE PROFILING")
    print("=" * 60)

    # 1. Load data files
    start_time = time.time()
    
    raw_triples_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    if not os.path.exists(raw_triples_path):
        print("[!] raw_triples.json not found. Run Stage 2 first.")
        return

    with open(raw_triples_path, "r", encoding="utf-8") as f:
        raw_triples = json.load(f)
    
    load_time = time.time() - start_time
    print(f"[*] Data loading time: {load_time:.4f} seconds ({len(raw_triples)} raw triples)")

    # 2. Profile Entity Resolution (Fuzzy + S-BERT)
    print("\n[*] Profiling Entity Resolution Stage...")
    er_start = time.time()
    resolved_triples, entity_mapping = resolve_entities(raw_triples)
    er_time = time.time() - er_start
    print(f"[+] Entity Resolution took: {er_time:.4f} seconds")
    
    # 3. Profile Graph Analysis (Build Graph, Louvain, Decay)
    print("\n[*] Profiling Graph Analysis Stage...")
    graph_start = time.time()
    G = build_graph(resolved_triples)
    graph_build_time = time.time() - graph_start
    
    louvain_start = time.time()
    orphans = detect_orphan_clusters(G)
    louvain_time = time.time() - louvain_start
    
    decay_start = time.time()
    stagnant = compute_temporal_decay(G)
    decay_time = time.time() - decay_start
    
    graph_total_time = time.time() - graph_start
    print(f"[+] Graph Build took:          {graph_build_time:.4f} seconds")
    print(f"[+] Louvain community clustering: {louvain_time:.4f} seconds")
    print(f"[+] Temporal decay analysis:     {decay_time:.4f} seconds")
    print(f"[+] Total Stage 3 took:         {graph_total_time:.4f} seconds")

    # 4. Profile Evaluation (Offline components)
    print("\n[*] Profiling Evaluation Stage...")
    eval_start = time.time()
    from src.evaluate import count_unique_gaps
    
    # Load gaps to evaluate unique counting time
    gaps_path = os.path.join(config.GAPS_DIR, "kgtabi_gaps.json")
    if os.path.exists(gaps_path):
        with open(gaps_path, "r", encoding="utf-8") as f:
            kg_gaps = json.load(f)
        unique_count_start = time.time()
        unique_count = count_unique_gaps(kg_gaps)
        unique_count_time = time.time() - unique_count_start
        print(f"[+] Unique gap calculation (S-BERT): {unique_count_time:.4f} seconds")
    else:
        print("[!] kgtabi_gaps.json not found, skipping unique gap timing.")
        unique_count_time = 0.0

    eval_time = time.time() - eval_start
    
    # Summarize runtimes
    runtime_summary = {
        "data_loading_seconds": round(load_time, 4),
        "entity_resolution_seconds": round(er_time, 4),
        "graph_build_seconds": round(graph_build_time, 4),
        "louvain_clustering_seconds": round(louvain_time, 4),
        "temporal_decay_seconds": round(decay_time, 4),
        "unique_gap_clustering_seconds": round(unique_count_time, 4),
        "triples_count": len(raw_triples),
        "entities_count": len(G.nodes())
    }

    out_path = os.path.join(config.DATA_DIR, "runtime_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(runtime_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Runtime analysis results saved to {out_path}")
    
    # Print Markdown Table
    print("\n| Pipeline Component | Runtime (seconds) | Throughput / Info |")
    print("| :--- | :---: | :--- |")
    print(f"| Data Loading | {load_time:.4f} | loaded {len(raw_triples)} raw triples |")
    print(f"| Entity Resolution (Fuzzy + S-BERT) | {er_time:.4f} | {len(raw_triples)/er_time:.1f} triples/sec |")
    print(f"| Graph Build | {graph_build_time:.4f} | built networkx DiGraph |")
    print(f"| Louvain Community Clustering | {louvain_time:.4f} | detected {len(orphans)} orphan clusters |")
    print(f"| Temporal Decay Analysis | {decay_time:.4f} | checked {len(G.nodes())} concepts |")
    print(f"| Unique Gap Deduplication (S-BERT) | {unique_count_time:.4f} | clustered generated gaps |")
    
    return runtime_summary


if __name__ == "__main__":
    profile_runtime()
