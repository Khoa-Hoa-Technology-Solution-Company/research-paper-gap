"""
Sensitivity Analysis - Sweep thresholds and measure impact on orphan clusters and stagnant concepts.
Runs entirely offline on existing resolved_triples.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.graph_analysis import build_graph, compute_temporal_decay, louvain_partition
from src.entity_resolution import resolve_entities


def sweep_louvain_thresholds(G):
    """Sweep Louvain size_ratio and bridge_ratio thresholds."""
    partition = louvain_partition(G)

    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    total_nodes = G.number_of_nodes()
    G_undirected = G.to_undirected()
    results = []

    size_ratios = [0.03, 0.05, 0.07, 0.10]
    bridge_ratios = [0.05, 0.10, 0.15, 0.20]

    for sr in size_ratios:
        for br in bridge_ratios:
            orphan_count = 0
            for cid, nodes in communities.items():
                r_size = len(nodes) / total_nodes
                internal = sum(1 for u, v in G_undirected.edges() if u in nodes and v in nodes)
                cross = sum(1 for u, v in G_undirected.edges() if (u in nodes) != (v in nodes))
                r_bridge = cross / (internal + cross) if (internal + cross) else 0.0

                if r_size >= sr and r_bridge <= br:
                    orphan_count += 1

            results.append({
                "min_size_ratio": sr,
                "max_bridge_ratio": br,
                "orphan_clusters": orphan_count,
            })

    return results


def sweep_decay_thresholds(G):
    """Sweep the recent-activity threshold using the full trend-test logic."""
    results = []
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    for thresh in thresholds:
        original_threshold = config.TEMPORAL_DECAY_THRESHOLD
        config.TEMPORAL_DECAY_THRESHOLD = thresh
        try:
            stagnant_count = len(compute_temporal_decay(G))
        finally:
            config.TEMPORAL_DECAY_THRESHOLD = original_threshold
        results.append({
            "decay_threshold": thresh,
            "stagnant_concepts": stagnant_count,
        })

    return results


def sweep_entity_resolution_thresholds(raw_triples):
    """Sweep fuzzy match and cosine similarity thresholds."""
    results = []
    fuzzy_vals = [75, 80, 85, 90, 95]
    cosine_vals = [0.75, 0.80, 0.85, 0.90, 0.95]

    # Collect raw entities
    raw_entity_set = set()
    for t in raw_triples:
        raw_entity_set.add(t["subject"])
        raw_entity_set.add(t["object"])
    raw_count = len(raw_entity_set)

    for fz in fuzzy_vals:
        for cs in cosine_vals:
            # Temporarily override thresholds
            orig_fz = config.FUZZY_MATCH_THRESHOLD
            orig_cs = config.COSINE_SIMILARITY_THRESHOLD
            config.FUZZY_MATCH_THRESHOLD = fz
            config.COSINE_SIMILARITY_THRESHOLD = cs

            try:
                resolved, mapping = resolve_entities(raw_triples)
                resolved_entities = set()
                for t in resolved:
                    resolved_entities.add(t["subject"])
                    resolved_entities.add(t["object"])
                merged = raw_count - len(resolved_entities)
            except Exception:
                merged = 0
                resolved_entities = raw_entity_set

            config.FUZZY_MATCH_THRESHOLD = orig_fz
            config.COSINE_SIMILARITY_THRESHOLD = orig_cs

            results.append({
                "fuzzy_threshold": fz,
                "cosine_threshold": cs,
                "entities_after": len(resolved_entities),
                "entities_merged": merged,
            })

    return results


def run_sensitivity_analysis(skip_entity_resolution: bool = False):
    print("=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load data
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")

    with open(resolved_path, "r", encoding="utf-8") as f:
        resolved_triples = json.load(f)
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_triples = json.load(f)

    G = build_graph(resolved_triples)

    # 1. Louvain threshold sweep
    print("\n--- Louvain Threshold Sweep ---")
    louvain_results = sweep_louvain_thresholds(G)
    print(f"  {'Size Ratio':>12} {'Bridge Ratio':>14} {'Orphan Clusters':>16}")
    for r in louvain_results:
        print(f"  {r['min_size_ratio']:>12.2f} {r['max_bridge_ratio']:>14.2f} {r['orphan_clusters']:>16}")

    # 2. Decay threshold sweep
    print("\n--- Temporal Decay Threshold Sweep ---")
    decay_results = sweep_decay_thresholds(G)
    print(f"  {'Decay Threshold':>16} {'Stagnant Concepts':>18}")
    for r in decay_results:
        print(f"  {r['decay_threshold']:>16.2f} {r['stagnant_concepts']:>18}")

    # Entity resolution is unaffected by the Louvain random state. Reuse the
    # previously recorded sweep when refreshing topology-only sensitivity.
    er_results = []
    if skip_entity_resolution:
        existing_path = os.path.join(config.DATA_DIR, "sensitivity_results.json")
        if os.path.exists(existing_path):
            with open(existing_path, "r", encoding="utf-8") as f:
                er_results = json.load(f).get("entity_resolution_sweep", [])
        print("\n--- Entity Resolution Threshold Sweep ---")
        print("[*] Reused prior deterministic ER sweep; not recomputed for this topology refresh.")
    else:
        print("\n--- Entity Resolution Threshold Sweep ---")
        er_results = sweep_entity_resolution_thresholds(raw_triples)
        print(f"  {'Fuzzy':>6} {'Cosine':>8} {'Entities After':>15} {'Merged':>8}")
        for r in er_results:
            print(f"  {r['fuzzy_threshold']:>6} {r['cosine_threshold']:>8.2f} {r['entities_after']:>15} {r['entities_merged']:>8}")

    # Save all results
    all_results = {
        "louvain_sweep": louvain_results,
        "decay_sweep": decay_results,
        "entity_resolution_sweep": er_results,
    }
    out_path = os.path.join(config.DATA_DIR, "sensitivity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Sensitivity analysis results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run KG-TABI sensitivity analysis.")
    parser.add_argument(
        "--skip-entity-resolution",
        action="store_true",
        help="Reuse the existing ER sweep while refreshing topology-only results.",
    )
    args = parser.parse_args()
    run_sensitivity_analysis(skip_entity_resolution=args.skip_entity_resolution)
