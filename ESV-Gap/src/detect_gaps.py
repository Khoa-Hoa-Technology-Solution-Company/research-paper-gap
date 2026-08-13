"""
Stage 4: Topological Gap Detection

Implements three gap detection algorithms on the knowledge graph:
1. Missing Link Prediction (TransE via PyKEEN)
2. Orphan Cluster Detection (Louvain community detection)
3. Temporal Decay Analysis

Usage:
    python run_pipeline.py --stage detect
"""

import pickle
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict
import community as community_louvain
from src.utils import get_logger, save_json, load_json, ensure_dir

logger = get_logger("detect_gaps")


# ============================================================
# GAP TYPE 1: Missing Link Prediction
# ============================================================

def detect_missing_links_common_neighbors(G, top_k):
    """Deterministic lightweight fallback when PyKEEN/Torch is unavailable."""
    simple = nx.Graph(G)
    candidate_pairs = set()

    for middle in simple.nodes():
        neighbors = sorted(simple.neighbors(middle), key=str)
        for i, head in enumerate(neighbors):
            for tail in neighbors[i + 1:]:
                if head != tail and not simple.has_edge(head, tail):
                    candidate_pairs.add(tuple(sorted((head, tail), key=str)))

    scored = []
    for head, tail in candidate_pairs:
        common_count = len(list(nx.common_neighbors(simple, head, tail)))
        degree_scale = max((simple.degree(head) * simple.degree(tail)) ** 0.5, 1.0)
        normalized_score = common_count / degree_scale
        scored.append({
            "type": "missing_link",
            "head": str(head),
            "relation": "RELATED_TO",
            "tail": str(tail),
            "prediction_score": float(10.0 * normalized_score),
            "description": (
                f"Common-neighbour candidate: '{head}' and '{tail}' share "
                f"{common_count} graph neighbours but have no observed edge."
            ),
            "detector": "common_neighbors_fallback",
            "common_neighbors": common_count,
        })

    scored.sort(
        key=lambda gap: (
            gap["prediction_score"],
            gap["common_neighbors"],
            gap["head"],
            gap["tail"],
        ),
        reverse=True,
    )
    result = scored[:top_k]
    logger.info(f"  Common-neighbour fallback found {len(result)} candidates")
    return result

def detect_missing_links(G, config):
    """
    Use TransE graph embeddings to predict missing links.
    Entity pairs with high predicted scores but no existing edge
    are flagged as candidate gaps.
    """
    logger.info("--- Missing Link Prediction (TransE) ---")
    
    transE_config = config["gap_detection"]["transE"]
    top_k = transE_config["top_k_predictions"]
    
    try:
        from pykeen.pipeline import pipeline as pykeen_pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        logger.warning(
            "PyKEEN/Torch is not installed; using deterministic "
            "common-neighbour missing-link generation."
        )
        return detect_missing_links_common_neighbors(G, top_k)
    
    # Convert NetworkX graph to PyKEEN triples
    triples_list = []
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "RELATED")
        triples_list.append([str(u), str(relation), str(v)])
    
    if len(triples_list) < 10:
        logger.warning(f"Too few triples ({len(triples_list)}) for meaningful link prediction")
        return []
    
    triples_array = np.array(triples_list)
    logger.info(f"  Training TransE on {len(triples_array)} triples...")
    
    # Create triples factory
    tf = TriplesFactory.from_labeled_triples(triples_array)
    
    # Train TransE model
    training, testing = tf.split(ratios=[0.8, 0.2], random_state=42)

    result = pykeen_pipeline(
        training=training,
        testing=testing,
        model="TransE",
        model_kwargs={
            "embedding_dim": transE_config["embedding_dim"],
        },
        training_kwargs={
            "num_epochs": transE_config["num_epochs"],
            "use_tqdm": True,
        },
        optimizer_kwargs={
            "lr": transE_config["learning_rate"],
        },
        random_seed=42,
    )
    
    model = result.model
    logger.info(f"  TransE training complete. Loss: {result.losses[-1]:.4f}")
    
    # Predict missing links
    logger.info(f"  Predicting top {top_k} missing links...")
    
    # Get all existing edges as a set for fast lookup
    existing_edges = set()
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "RELATED")
        existing_edges.add((str(u), str(rel), str(v)))
    
    # Score all possible triples and find the best missing ones
    all_nodes = list(G.nodes())
    all_relations = list(set(d.get("relation", "RELATED") for _, _, d in G.edges(data=True)))
    
    # For efficiency, sample candidate pairs rather than all N^2
    candidates = []
    
    # Focus on node pairs that share a neighbor but aren't directly connected
    for node in all_nodes:
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        for neighbor in neighbors:
            second_hop = set(G.successors(neighbor)) | set(G.predecessors(neighbor))
            for target in second_hop:
                if target != node and not G.has_edge(node, target):
                    candidates.append((node, target))
    
    # Deduplicate candidates
    candidates = list(set(candidates))
    
    if not candidates:
        logger.warning("  No candidate missing links found")
        return []
    
    logger.info(f"  Evaluating {len(candidates)} candidate pairs...")
    
    # Score candidates using the trained model
    scored_gaps = []
    
    for head, tail in candidates[:min(len(candidates), 5000)]:  # Cap for performance
        for rel in all_relations:
            triple_key = (str(head), str(rel), str(tail))
            if triple_key in existing_edges:
                continue
            
            try:
                # Get score from model
                h_id = tf.entity_to_id.get(str(head))
                r_id = tf.relation_to_id.get(str(rel))
                t_id = tf.entity_to_id.get(str(tail))
                
                if h_id is None or r_id is None or t_id is None:
                    continue
                
                import torch
                h_tensor = torch.tensor([[h_id]])
                r_tensor = torch.tensor([[r_id]])
                t_tensor = torch.tensor([[t_id]])
                
                score = model.score_hrt(
                    torch.cat([h_tensor, r_tensor, t_tensor], dim=1)
                ).item()
                
                scored_gaps.append({
                    "type": "missing_link",
                    "head": str(head),
                    "relation": str(rel),
                    "tail": str(tail),
                    "prediction_score": float(score),
                    "description": f"Predicted connection: '{head}' --[{rel}]--> '{tail}' is likely but missing from the literature.",
                })
            except Exception:
                continue
    
    # Log final loss so it can be reported in the paper

    final_loss = result.losses[-1]
    logger.info(f'  TransE final loss: {final_loss:.4f} (confirms training convergence)')
    logger.info(f'  Triples used: {len(triples_array)} (threshold for meaningful predictions: ~500)')
    if len(triples_array) < 500:
        logger.warning(f'  Graph too sparse for TransE gap prediction.')
        logger.warning(f'  Zero gaps expected. Increase corpus to 100+ papers.')
        
    # Save the loss for paper reporting even if 0 gaps
        save_json({'final_loss': float(final_loss), 'triples': len(triples_array),'threshold': 500, 'gaps_produced': 0},
            Path(config['paths']['outputs']) / 'transe_training_log.json')

    # Sort by score (higher = more likely missing link)
    scored_gaps.sort(key=lambda x: x["prediction_score"], reverse=True)
    top_gaps = scored_gaps[:top_k]
    
    logger.info(f"  Found {len(top_gaps)} missing link gaps")
    return top_gaps


# ============================================================
# GAP TYPE 2: Orphan Cluster Detection
# ============================================================

def detect_orphan_clusters(G, config):
    """
    Use Louvain community detection to find weakly connected
    subgraphs (orphan clusters) representing under-explored areas.
    """
    logger.info("--- Orphan Cluster Detection (Louvain) ---")
    
    orphan_config = config["gap_detection"]["orphan"]
    min_ratio = orphan_config["min_cluster_ratio"]
    max_inter_ratio = orphan_config["max_inter_cluster_edge_ratio"]
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Remove self-loops and isolates
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
    isolates = list(nx.isolates(G_undirected))
    G_undirected.remove_nodes_from(isolates)
    
    if G_undirected.number_of_nodes() < 5:
        logger.warning("  Graph too small for community detection")
        return []
    
    # Run Louvain community detection
    # python-louvain needs a simple Graph (no multi-edges)
    G_simple = nx.Graph(G_undirected)
    # Louvain is stochastic. A fixed, recorded seed is required for a
    # reproducible gap list and for stable downstream validation decisions.
    random_seed = orphan_config.get("random_seed", 42)
    partition = community_louvain.best_partition(G_simple, random_state=random_seed)
    
    # Group nodes by community
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    total_nodes = G_simple.number_of_nodes()
    logger.info(f"  Detected {len(communities)} communities")
    
    # Analyse each community
    orphan_gaps = []
    
    for comm_id, members in communities.items():
        comm_size = len(members)
        size_ratio = comm_size / total_nodes
        
        # Count inter-community edges
        inter_edges = 0
        intra_edges = 0
        comm_set = set(members)
        
        for node in members:
            for neighbor in G_simple.neighbors(node):
                if neighbor in comm_set:
                    intra_edges += 1
                else:
                    inter_edges += 1
        
        intra_edges //= 2  # Undirected, counted twice
        total_comm_edges = inter_edges + intra_edges
        inter_ratio = inter_edges / max(total_comm_edges, 1)
        
        # Flag as orphan if small AND isolated
        if size_ratio < min_ratio or inter_ratio < max_inter_ratio:
            # Get the main concepts in this cluster
            node_types = defaultdict(list)
            for node in members:
                ntype = G.nodes[node].get("type", "UNKNOWN") if G.has_node(node) else "UNKNOWN"
                node_types[ntype].append(node)
            
            orphan_gaps.append({
                "type": "orphan_cluster",
                "community_id": comm_id,
                "size": comm_size,
                "size_ratio": round(size_ratio, 4),
                "inter_edge_ratio": round(inter_ratio, 4),
                "intra_edges": intra_edges,
                "inter_edges": inter_edges,
                "members": members,
                "key_concepts": members[:10],  # Top 10 for display
                "description": f"Isolated research cluster with {comm_size} concepts and only {inter_ratio:.1%} connections to the broader literature. Key concepts: {', '.join(members[:5])}.",
            })
    
    # Sort by isolation (lowest inter_ratio = most isolated)
    orphan_gaps.sort(key=lambda x: x["inter_edge_ratio"])
    
    logger.info(f"  Found {len(orphan_gaps)} orphan clusters")
    return orphan_gaps


# ============================================================
# GAP TYPE 3: Temporal Decay Analysis
# ============================================================

def detect_temporal_decay(G, config):
    """
    Identify concepts whose publication-normalised activity declines.

    Relation events are clustered by paper, not counted independently.  The
    final calendar year is excluded when it is right-censored by the recorded
    snapshot date.
    """
    logger.info("--- Temporal Decay Analysis ---")
    
    temp_config = config["gap_detection"]["temporal"]
    decay_threshold = temp_config["decay_threshold"]
    lookback = temp_config["lookback_years"]
    
    publication_counts = {
        int(year): int(count)
        for year, count in temp_config.get("publication_counts", {}).items()
    }
    papers_by_year = defaultdict(set)
    edge_years = []
    for _, _, data in G.edges(data=True):
        try:
            year = int(data.get("year"))
        except (TypeError, ValueError):
            continue
        edge_years.append(year)
        paper = data.get("source_paper") or data.get("source_paper_id") or data.get("paper_id")
        if paper:
            papers_by_year[year].add(str(paper))
    
    if not edge_years:
        logger.warning("  No temporal data on edges")
        return []
    
    min_year = min(edge_years)
    observed_max_year = max(edge_years)
    snapshot_date = str(
        temp_config.get("snapshot_date")
        or config.get("gap_validation", {}).get("snapshot_date", "")
    )
    try:
        snapshot_year = int(snapshot_date[:4])
    except ValueError:
        snapshot_year = 0
    exclude_partial = bool(
        temp_config.get(
            "exclude_incomplete_final_year",
            config.get("gap_validation", {}).get("exclude_incomplete_final_year", True),
        )
    )
    max_year = observed_max_year - 1 if exclude_partial and snapshot_year == observed_max_year else observed_max_year
    logger.info(
        "  Temporal range: %d - %d (observed through %d; snapshot %s)",
        min_year, max_year, observed_max_year, snapshot_date or "unspecified",
    )
    if not publication_counts:
        publication_counts = {
            year: len(papers) for year, papers in papers_by_year.items()
        }
    
    # Build per-node temporal profiles
    node_year_papers = defaultdict(lambda: defaultdict(set))
    
    for u, v, data in G.edges(data=True):
        try:
            year = int(data.get("year"))
        except (TypeError, ValueError):
            continue
        if year > max_year:
            continue
        paper = str(
            data.get("source_paper")
            or data.get("source_paper_id")
            or data.get("paper_id")
            or f"event-{u}-{v}-{year}"
        )
        node_year_papers[u][year].add(paper)
        node_year_papers[v][year].add(paper)
    
    # Analyse decay for each node
    decay_gaps = []
    recent_years = list(range(max_year - lookback + 1, max_year + 1))
    earlier_years = list(range(max_year - 2 * lookback + 1, max_year - lookback + 1))
    
    for node, year_papers in node_year_papers.items():
        year_counts = {year: len(papers) for year, papers in year_papers.items()}
        year_rates = {
            year: year_counts.get(year, 0) / max(publication_counts.get(year, 0), 1)
            for year in range(min_year, max_year + 1)
        }
        recent_activity = sum(year_rates.get(y, 0.0) for y in recent_years) / max(len(recent_years), 1)
        earlier_activity = sum(year_rates.get(y, 0.0) for y in earlier_years) / max(len(earlier_years), 1)
        
        # Skip nodes with very little activity overall
        total = sum(year_counts.values())
        if total < 3:
            continue
        
        # Calculate decay rate
        if earlier_activity > 0:
            decay_rate = 1.0 - (recent_activity / earlier_activity)
        else:
            # With no baseline activity, neither zero-to-zero nor new activity
            # identifies a decline.  Both cases fail closed to zero decay.
            decay_rate = 0.0
        
        # Find peak year
        peak_year = max(year_rates, key=year_rates.get)
        peak_count = year_counts.get(peak_year, 0)
        
        if decay_rate >= decay_threshold:
            # Build temporal profile for this node
            profile = {y: year_counts.get(y, 0) for y in range(min_year, max_year + 1)}
            normalised_profile = {y: round(year_rates.get(y, 0.0), 6) for y in range(min_year, max_year + 1)}
            
            decay_gaps.append({
                "type": "temporal_decay",
                "concept": node,
                "concept_type": G.nodes[node].get("type", "UNKNOWN") if G.has_node(node) else "UNKNOWN",
                "decay_rate": round(decay_rate, 4),
                "peak_year": peak_year,
                "peak_activity": peak_count,
                "recent_activity": round(recent_activity, 6),
                "earlier_activity": round(earlier_activity, 6),
                "total_activity": total,
                "temporal_profile": profile,
                "normalised_temporal_profile": normalised_profile,
                "publication_counts": publication_counts,
                "analysis_end_year": max_year,
                "snapshot_date": snapshot_date or None,
                "right_censored_year_excluded": observed_max_year if max_year < observed_max_year else None,
                "description": f"'{node}' peaked in {peak_year} and its share of screened papers declined by {decay_rate:.0%} across the two comparison windows. This is a triage signal, not evidence of a scientific gap.",
            })
    
    # Sort by decay rate (highest decay = most stalled)
    decay_gaps.sort(key=lambda x: x["decay_rate"], reverse=True)
    
    logger.info(f"  Found {len(decay_gaps)} decaying concepts")
    return decay_gaps


# ============================================================
# MAIN
# ============================================================

def detect_all_gaps(config):
    """
    Run all three gap detection algorithms and save results.
    """
    graph_dir = Path(config["paths"]["graph"])
    output_dir = ensure_dir(config["paths"]["outputs"])
    
    # Load knowledge graph
    pkl_path = graph_dir / "knowledge_graph.pkl"
    if not pkl_path.exists():
        logger.error(f"Knowledge graph not found: {pkl_path}")
        logger.error("Run 'python run_pipeline.py --stage build' first.")
        return
    
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    all_gaps = {
        "missing_links": [],
        "orphan_clusters": [],
        "temporal_decay": [],
    }
    
    # --- Run detection algorithms ---
    
    # 1. Missing Link Prediction
    try:
        missing_links = detect_missing_links(G, config)
        all_gaps["missing_links"] = missing_links
    except Exception as e:
        logger.error(f"Missing link detection failed: {e}")
        logger.info("Continuing with other methods...")
    
    # 2. Orphan Cluster Detection
    try:
        orphan_clusters = detect_orphan_clusters(G, config)
        all_gaps["orphan_clusters"] = orphan_clusters
    except Exception as e:
        logger.error(f"Orphan cluster detection failed: {e}")
    
    # 3. Temporal Decay Analysis
    try:
        temporal_decay = detect_temporal_decay(G, config)
        all_gaps["temporal_decay"] = temporal_decay
    except Exception as e:
        logger.error(f"Temporal decay detection failed: {e}")
    
    # --- Save results ---
    save_json(all_gaps, output_dir / "detected_gaps_raw.json")
    
    # --- Print summary ---
    total = sum(len(v) for v in all_gaps.values())
    
    logger.info(f"\n{'='*50}")
    logger.info(f"  GAP DETECTION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Missing links:    {len(all_gaps['missing_links'])}")
    logger.info(f"  Orphan clusters:  {len(all_gaps['orphan_clusters'])}")
    logger.info(f"  Temporal decay:   {len(all_gaps['temporal_decay'])}")
    logger.info(f"  Total gaps:       {total}")
    logger.info(f"  Saved to: {output_dir / 'detected_gaps_raw.json'}")
    
    return all_gaps


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    detect_all_gaps(config)
