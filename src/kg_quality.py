"""
KG Quality Evaluation - Measures triple extraction and entity resolution quality.
Runs entirely offline on existing data files.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config


def evaluate_kg_quality():
    # Load data
    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    mapping_path = os.path.join(config.TRIPLES_DIR, "entity_mapping.json")
    gml_path = os.path.join(config.GRAPH_DIR, "knowledge_graph.gml")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_triples = json.load(f)
    with open(resolved_path, "r", encoding="utf-8") as f:
        resolved_triples = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        entity_mapping = json.load(f)

    # --- 1. Triple Extraction Statistics ---
    print("=" * 60)
    print("1. TRIPLE EXTRACTION STATISTICS")
    print("=" * 60)

    total_raw = len(raw_triples)
    total_resolved = len(resolved_triples)
    print(f"  Raw triples extracted:      {total_raw}")
    print(f"  Resolved triples (post-ER): {total_resolved}")
    print(f"  Triples removed by ER:      {total_raw - total_resolved}")

    # Confidence distribution
    confidences = [t["confidence"] for t in raw_triples]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    high_conf = sum(1 for c in confidences if c >= 0.7)
    med_conf = sum(1 for c in confidences if 0.3 <= c < 0.7)
    print(f"\n  Confidence distribution:")
    print(f"    Average confidence:       {avg_conf:.3f}")
    print(f"    High (>= 0.7):            {high_conf} ({100*high_conf/total_raw:.1f}%)")
    print(f"    Medium (0.3 - 0.7):       {med_conf} ({100*med_conf/total_raw:.1f}%)")

    # Per-paper triple count
    paper_years = {}
    for t in raw_triples:
        y = t["year"]
        paper_years[y] = paper_years.get(y, 0) + 1
    print(f"\n  Triples per publication year:")
    for y in sorted(paper_years.keys()):
        print(f"    {y}: {paper_years[y]} triples")

    # --- 2. Entity Type Distribution ---
    print("\n" + "=" * 60)
    print("2. ENTITY TYPE DISTRIBUTION")
    print("=" * 60)

    entity_types_raw = {}
    raw_entities = {}
    for t in raw_triples:
        for role, type_key in [("subject", "subject_type"), ("object", "object_type")]:
            name = t[role]
            etype = t[type_key]
            raw_entities[name] = etype
            entity_types_raw[etype] = entity_types_raw.get(etype, 0) + 1

    resolved_entities = {}
    entity_types_resolved = {}
    for t in resolved_triples:
        for role, type_key in [("subject", "subject_type"), ("object", "object_type")]:
            name = t[role]
            etype = t[type_key]
            resolved_entities[name] = etype
            entity_types_resolved[etype] = entity_types_resolved.get(etype, 0) + 1

    print(f"  {'Type':<12} {'Raw Count':>10} {'Resolved Count':>15}")
    print(f"  {'-'*12} {'-'*10} {'-'*15}")
    all_types = sorted(set(list(entity_types_raw.keys()) + list(entity_types_resolved.keys())))
    for etype in all_types:
        raw_c = entity_types_raw.get(etype, 0)
        res_c = entity_types_resolved.get(etype, 0)
        print(f"  {etype:<12} {raw_c:>10} {res_c:>15}")

    # --- 3. Relation Type Distribution ---
    print("\n" + "=" * 60)
    print("3. RELATION TYPE DISTRIBUTION")
    print("=" * 60)

    rel_types = {}
    for t in resolved_triples:
        r = t["relation"]
        rel_types[r] = rel_types.get(r, 0) + 1

    for r in sorted(rel_types.keys(), key=lambda x: rel_types[x], reverse=True):
        print(f"  {r:<16} {rel_types[r]:>5}")

    # --- 4. Entity Resolution Quality ---
    print("\n" + "=" * 60)
    print("4. ENTITY RESOLUTION QUALITY")
    print("=" * 60)

    total_raw_entities = len(raw_entities)
    total_resolved_entities = len(resolved_entities)
    total_merges = sum(len(v) for v in entity_mapping.values())
    merge_groups = len(entity_mapping)

    print(f"  Entities before resolution: {total_raw_entities}")
    print(f"  Entities after resolution:  {total_resolved_entities}")
    print(f"  Total entities merged:      {total_merges}")
    print(f"  Merge groups:               {merge_groups}")
    print(f"  Merge ratio:                {100*total_merges/total_raw_entities:.1f}%")

    print(f"\n  Merge groups detail:")
    for canonical, aliases in entity_mapping.items():
        print(f"    '{canonical}' <- {aliases}")

    # Check for potential false positives
    print(f"\n  Potential false positive merges (manual review):")
    for canonical, aliases in entity_mapping.items():
        for alias in aliases:
            # Flag if names look semantically different
            can_words = set(canonical.lower().split())
            ali_words = set(alias.lower().split())
            overlap = len(can_words & ali_words)
            total = len(can_words | ali_words)
            if total > 0 and overlap / total < 0.3:
                print(f"    [!] SUSPICIOUS: '{canonical}' <- '{alias}' (word overlap: {overlap}/{total})")

    # --- 5. Graph Structure Statistics ---
    print("\n" + "=" * 60)
    print("5. GRAPH STRUCTURE STATISTICS")
    print("=" * 60)

    import networkx as nx
    G = nx.read_gml(gml_path)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Convert to undirected for components
    G_und = G.to_undirected()
    n_components = nx.number_connected_components(G_und)
    largest_cc = max(nx.connected_components(G_und), key=len)
    
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0

    print(f"  Nodes:                {n_nodes}")
    print(f"  Edges:                {n_edges}")
    print(f"  Density:              {density:.4f}")
    print(f"  Connected components: {n_components}")
    print(f"  Largest component:    {len(largest_cc)} nodes ({100*len(largest_cc)/n_nodes:.1f}%)")
    print(f"  Average degree:       {avg_degree:.2f}")
    print(f"  Max degree:           {max_degree}")

    # Top-5 most connected nodes
    degree_list = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top-5 most connected nodes:")
    for node, deg in degree_list:
        print(f"    {node:<45} degree={deg}")

    # --- Build results dict ---
    results = {
        "triple_extraction": {
            "raw_triples": total_raw,
            "resolved_triples": total_resolved,
            "avg_confidence": round(avg_conf, 3),
            "high_confidence_pct": round(100 * high_conf / total_raw, 1),
        },
        "entity_resolution": {
            "entities_before": total_raw_entities,
            "entities_after": total_resolved_entities,
            "merges": total_merges,
            "merge_groups": merge_groups,
            "merge_ratio_pct": round(100 * total_merges / total_raw_entities, 1),
        },
        "graph_structure": {
            "nodes": n_nodes,
            "edges": n_edges,
            "density": round(density, 4),
            "connected_components": n_components,
            "largest_component_pct": round(100 * len(largest_cc) / n_nodes, 1),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree,
        },
        "entity_type_distribution": {k: v for k, v in sorted(entity_types_resolved.items())},
        "relation_type_distribution": rel_types,
    }

    out_path = os.path.join(config.DATA_DIR, "kg_quality_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] KG quality results saved to {out_path}")
    return results


if __name__ == "__main__":
    evaluate_kg_quality()
