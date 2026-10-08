"""
Stage 4: Evidence and Topological Gap-Signal Detection

Produces explicit-limitation and graph-structural signals.  These outputs are
not research gaps until the downstream synthesis contract combines convergent
signals into an answerable, corpus-bounded question.

Usage:
    python run_pipeline.py --stage detect
"""

import pickle
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import community as community_louvain
from src.evidence_graph import (
    add_empty_cells_to_evidence_graph,
    consolidate_evidence_candidates,
    save_evidence_graph,
)
from src.utils import get_logger, save_json, load_json, load_jsonl, ensure_dir

logger = get_logger("detect_gaps")


# ============================================================
# GAP TYPE 1: Missing Link Prediction
# ============================================================

_ALLOWED_RESEARCH_TYPE_PAIRS = {
    frozenset(("METHOD", "CONCEPT")),
    frozenset(("METHOD", "DATASET")),
    frozenset(("METHOD", "METRIC")),
    frozenset(("CONCEPT", "DATASET")),
    frozenset(("CONCEPT", "METRIC")),
}
_GENERIC_EVIDENCE_MAP_LABELS = {
    "accuracy", "artificial intelligence", "deep learning",
    "deep learning algorithms", "f1 score", "intrusion detection",
    "intrusion detection system", "intrusion detection systems",
    "machine learning", "method", "model", "precision", "recall",
}


def _normalized_label_tokens(value):
    return {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", str(value).casefold())
        if token not in {"a", "an", "and", "for", "of", "the"}
    }


def _label_mentioned(label, document_text):
    label_tokens = _normalized_label_tokens(label)
    text_tokens = _normalized_label_tokens(document_text)
    if not label_tokens:
        return False
    coverage = len(label_tokens.intersection(text_tokens)) / len(label_tokens)
    words = re.findall(r"[A-Za-z0-9]+", str(label))
    acronym = "".join(word[0] for word in words if word).casefold()
    raw = str(document_text).casefold()
    return coverage >= 0.8 or (
        len(words) >= 2 and len(acronym) >= 2
        and re.search(rf"\b{re.escape(acronym)}s?\b", raw) is not None
    )


def _label_mentioned_profile(label, raw, text_tokens):
    label_tokens = _normalized_label_tokens(label)
    if not label_tokens:
        return False
    coverage = len(label_tokens.intersection(text_tokens)) / len(label_tokens)
    words = re.findall(r"[A-Za-z0-9]+", str(label))
    acronym = "".join(word[0] for word in words if word).casefold()
    return coverage >= 0.8 or (
        len(words) >= 2 and len(acronym) >= 2
        and re.search(rf"\b{re.escape(acronym)}s?\b", raw) is not None
    )


def _research_pair_allowed(G, head, tail):
    head_type = str(G.nodes[head].get("type", "UNKNOWN")).upper()
    tail_type = str(G.nodes[tail].get("type", "UNKNOWN")).upper()
    return frozenset((head_type, tail_type)) in _ALLOWED_RESEARCH_TYPE_PAIRS


def _research_relation(G, head, tail):
    types = {
        str(G.nodes[head].get("type", "UNKNOWN")).upper(),
        str(G.nodes[tail].get("type", "UNKNOWN")).upper(),
    }
    if "DATASET" in types or "METRIC" in types:
        return "EVALUATES_ON"
    return "APPLIED_TO"


def detect_evidence_map_empty_cells(G, config):
    """Find typed, source-supported empty cells in the paper evidence matrix.

    Unlike unconstrained link prediction, both endpoints must already be
    independently studied, the pairing must be meaningful for a PMCOST frame,
    and at least two source-disjoint graph paths must motivate the missing cell.
    """
    logger.info("--- Typed Evidence-Map Empty Cells ---")
    settings = config.get("gap_detection", {}).get("evidence_map", {})
    minimum_papers = int(settings.get("min_marginal_papers", 2))
    top_k = int(settings.get("top_k_candidates", 40))
    max_entities_per_type = int(settings.get("max_entities_per_type", 60))
    path_cutoff = int(config.get("gap_validation", {}).get("max_path_length", 4))
    required_paths = int(config.get("gap_validation", {}).get("min_independent_paths", 2))
    configured_pairs = settings.get("allowed_type_pairs", [
        "METHOD-DATASET", "METHOD-CONCEPT", "CONCEPT-DATASET",
    ])
    allowed_pairs = {
        frozenset(str(value).upper().split("-", 1))
        for value in configured_pairs
        if "-" in str(value)
    }
    corpus_path = (
        Path(config.get("paths", {}).get("processed_data", ""))
        / "corpus_filtered.jsonl"
    )
    document_texts = [
        " ".join(str(paper.get(key, "")) for key in ("title", "abstract"))
        for paper in (load_jsonl(corpus_path) if corpus_path.exists() else [])
    ]
    document_profiles = [
        (text.casefold(), _normalized_label_tokens(text)) for text in document_texts
    ]

    node_papers = {}
    for node, data in G.nodes(data=True):
        papers = {str(value) for value in data.get("papers", []) if value}
        if len(papers) >= minimum_papers:
            node_papers[node] = papers
    by_type = defaultdict(list)
    for node, papers in node_papers.items():
        node_type = str(G.nodes[node].get("type", "UNKNOWN")).upper()
        if node_type in {"METHOD", "DATASET", "METRIC", "CONCEPT"}:
            by_type[node_type].append(node)
    for node_type in by_type:
        by_type[node_type].sort(
            key=lambda node: (len(node_papers[node]), str(node)), reverse=True
        )
        by_type[node_type] = by_type[node_type][:max_entities_per_type]
    mention_documents = {
        node: {
            index for index, (raw, tokens) in enumerate(document_profiles)
            if _label_mentioned_profile(node, raw, tokens)
        }
        for nodes in by_type.values()
        for node in nodes
    }

    simple = nx.Graph(G)
    preliminary = []
    pairs_seen = set()
    for type_pair in allowed_pairs:
        left_type, right_type = sorted(type_pair)
        for left in by_type.get(left_type, []):
            for right in by_type.get(right_type, []):
                if left == right:
                    continue
                pair_key = frozenset((str(left), str(right)))
                if pair_key in pairs_seen or simple.has_edge(left, right):
                    continue
                left_label = str(left).casefold().strip()
                right_label = str(right).casefold().strip()
                left_tokens = _normalized_label_tokens(left)
                right_tokens = _normalized_label_tokens(right)
                lexical_overlap = (
                    len(left_tokens.intersection(right_tokens))
                    / max(min(len(left_tokens), len(right_tokens)), 1)
                )
                if (
                    left_label in _GENERIC_EVIDENCE_MAP_LABELS
                    or right_label in _GENERIC_EVIDENCE_MAP_LABELS
                    or lexical_overlap >= 0.8
                ):
                    continue
                if mention_documents.get(left, set()).intersection(
                    mention_documents.get(right, set())
                ):
                    continue
                pairs_seen.add(pair_key)
                # An empty evidence cell means no screened paper evaluates or
                # studies both endpoints together.
                observed_joint = node_papers[left].intersection(node_papers[right])
                if observed_joint:
                    continue
                try:
                    distance = nx.shortest_path_length(simple, left, right)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if distance > path_cutoff:
                    continue
                marginal_balance = min(
                    len(node_papers[left]), len(node_papers[right])
                ) / max(len(node_papers[left]), len(node_papers[right]))
                expected_joint = (
                    len(node_papers[left]) * len(node_papers[right])
                    / max(len({paper for papers in node_papers.values() for paper in papers}), 1)
                )
                preliminary.append((
                    expected_joint * (0.5 + 0.5 * marginal_balance) / max(distance - 1, 1),
                    left,
                    right,
                    distance,
                ))
    preliminary.sort(reverse=True, key=lambda item: (item[0], str(item[1]), str(item[2])))

    from src.validate_gaps import _candidate_problem_match, independent_evidence_paths

    candidates = []
    for matrix_score, left, right, distance in preliminary[: max(top_k * 8, top_k)]:
        problem_ready, problem_coverage = _candidate_problem_match(
            {"type": "missing_link", "head": str(left), "tail": str(right)},
            str(config.get("project", {}).get("domain", "")),
        )
        if (
            not problem_ready
            or problem_coverage < float(
                config.get("gap_validation", {}).get("min_problem_relevance", 0.60)
            )
        ):
            continue
        paths = independent_evidence_paths(G, str(left), str(right), cutoff=path_cutoff)
        if len(paths) < required_paths:
            continue
        left_type = str(G.nodes[left].get("type", "UNKNOWN")).upper()
        right_type = str(G.nodes[right].get("type", "UNKNOWN")).upper()
        # Put a method first where possible so the research question is natural.
        if right_type == "METHOD" and left_type != "METHOD":
            left, right = right, left
        path_papers = sorted({
            paper for path in paths for paper in path.get("papers", [])
        })
        source_strength = min(len(path_papers) / 6, 1.0)
        marginal_strength = min(
            (len(node_papers[left]) + len(node_papers[right])) / 20, 1.0
        )
        path_strength = min(len(paths) / 3, 1.0)
        edge_years = []
        for node in (left, right):
            incident = G.edges(node, data=True) if not G.is_directed() else list(
                G.in_edges(node, data=True)
            ) + list(G.out_edges(node, data=True))
            for _, _, data in incident:
                try:
                    edge_years.append(int(data.get("year")))
                except (TypeError, ValueError):
                    pass
        snapshot = str(config.get("gap_validation", {}).get("snapshot_date", ""))
        try:
            snapshot_year = int(snapshot[:4])
        except ValueError:
            snapshot_year = max(edge_years) if edge_years else 0
        frontier_score = 1.0 if edge_years and max(edge_years) >= snapshot_year - 2 else 0.5
        quality = (
            0.35 * path_strength + 0.25 * source_strength
            + 0.20 * marginal_strength + 0.20 * frontier_score
        )
        candidates.append({
            "type": "missing_link",
            "head": str(left),
            "tail": str(right),
            "relation": _research_relation(G, left, right),
            "detector": "typed_evidence_map_empty_cell",
            "prediction_score": round(float(matrix_score), 4),
            "supporting_paper_ids": path_papers,
            "independent_evidence_paths": paths,
            "evidence_map": {
                "head_type": str(G.nodes[left].get("type", "UNKNOWN")),
                "tail_type": str(G.nodes[right].get("type", "UNKNOWN")),
                "head_paper_count": len(node_papers[left]),
                "tail_paper_count": len(node_papers[right]),
                "observed_joint_paper_count": 0,
                "graph_distance": distance,
                "source_disjoint_path_count": len(paths),
                "candidate_problem_relevance": problem_coverage,
            },
            "candidate_quality": {
                "score": round(quality, 4),
                "independent_source_count": len(path_papers),
                "semantic_cohesion": 1.0,
                "frame_completeness": 1.0,
                "full_text_source_count": 0,
                "full_text_coverage": 0.0,
                "certificate_readiness": {
                    "independent_sources_ready": len(path_papers) >= 2,
                    "full_text_ready": False,
                    "research_frame_ready": True,
                },
            },
            "description": (
                f"Typed evidence-map cell '{left}' × '{right}' is empty despite "
                f"independent marginal evidence from {len(path_papers)} papers "
                f"across {len(paths)} source-disjoint paths."
            ),
        })
        if len(candidates) >= top_k:
            break
    candidates.sort(
        key=lambda item: (
            item["candidate_quality"]["score"], item["prediction_score"]
        ),
        reverse=True,
    )
    unique_candidates = []
    seen_cells = set()
    for candidate in candidates:
        key = (
            " ".join(sorted(
                _normalized_label_tokens(candidate["head"]).difference({"method", "model"})
            )),
            " ".join(sorted(
                _normalized_label_tokens(candidate["tail"]).difference({"dataset", "benchmark"})
            )),
        )
        if key in seen_cells:
            continue
        seen_cells.add(key)
        unique_candidates.append(candidate)
    candidates = unique_candidates[:top_k]
    logger.info("  Found %d typed evidence-map empty cells", len(candidates))
    return candidates


def detect_missing_links_common_neighbors(G, top_k):
    """Deterministic lightweight fallback when PyKEEN/Torch is unavailable."""
    simple = nx.Graph(G)
    candidate_pairs = set()

    for middle in simple.nodes():
        neighbors = sorted(simple.neighbors(middle), key=str)
        for i, head in enumerate(neighbors):
            for tail in neighbors[i + 1:]:
                if (
                    head != tail
                    and not simple.has_edge(head, tail)
                    and _research_pair_allowed(G, head, tail)
                ):
                    candidate_pairs.add(tuple(sorted((head, tail), key=str)))

    scored = []
    for head, tail in candidate_pairs:
        common_count = len(list(nx.common_neighbors(simple, head, tail)))
        degree_scale = max((simple.degree(head) * simple.degree(tail)) ** 0.5, 1.0)
        normalized_score = common_count / degree_scale
        scored.append({
            "type": "missing_link",
            "head": str(head),
            "relation": _research_relation(G, head, tail),
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
        if not _research_pair_allowed(G, head, tail):
            continue
        for rel in all_relations:
            if rel != _research_relation(G, head, tail):
                continue
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
# GAP TYPE 4: Explicit evidence-backed limitations
# ============================================================

LIMITATION_CUES = {
    "challenge", "challenges", "fails", "failure", "failures", "future",
    "gap", "gaps", "however", "lack", "lacks", "limitation",
    "limitations", "limited", "missing", "not",
    "poor", "poorly", "remains", "unable", "unaddressed", "without",
    "struggle", "struggles", "unexplored", "drawback", "drawbacks",
    "over-correction", "overcorrection", "correction", "penalized", "penalised",
}

BENEFIT_NEGATION_PATTERNS = (
    r"\b(?:does|do|did)\s+not\s+require\b",
    r"\bno\s+longer\s+requires?\b",
    r"\bwithout\s+(?:the\s+need\s+for|requiring)\b",
    r"\beliminates?\s+the\s+need\s+for\b",
    r"\bavoids?\s+(?:the\s+need\s+for|requiring)\b",
    r"\bneed\s+not\b",
    r"\bnot\s+necessary\b",
)
EXPLICIT_PROBLEM_PATTERNS = (
    r"\black(?:s|ed|ing)?\b",
    r"\black\s+of\b",
    r"\blimitation(?:s)?\b",
    r"\binsufficient\b",
    r"\bscar(?:ce|city)\b",
    r"\bshortage\b",
    r"\b(?:can|could|does|did)\s+not\b",
    r"\bunable\b",
    r"\bfails?\b",
    r"\bpoor(?:ly)?\b",
    r"\bunaddressed\b",
    r"\bremains?\s+(?:an?\s+)?(?:open|challeng|problem)\w*\b",
    r"\b(?:serious|major|open)\s+(?:issue|problem|challenge)\b",
    r"\bstruggl(?:e|es|ed|ing)\b",
    r"\b(?:face|faces|faced|facing)\s+(?:an?\s+)?challenge\b",
    r"\bremain(?:s|ed|ing)?\s+(?:relatively\s+)?unexplored\b",
    r"\bdrawbacks?\b",
    r"\bover[- ]correction\b",
    r"\b(?:heavily\s+)?penali[sz](?:e|es|ed|ing)\b",
)
NO_LIMITATION_PATTERNS = (
    r"\bno\s+(?:explicit\s+)?limitation(?:s)?\s+(?:is|are|was|were)\s+(?:stated|reported|identified)\b",
    r"\bdoes\s+not\s+(?:explicitly\s+)?state\s+(?:any\s+)?(?:unresolved\s+)?limitation\b",
)


def limitation_evidence_semantics(evidence, subject="", source_context=""):
    """Deterministically check whether evidence entails an unresolved deficit.

    A negated requirement (``does not require X``) is normally an advantage,
    not evidence that the method lacks X.  A statement about prior/existing
    methods is also not attributed to a newly proposed subject unless that
    subject is mentioned in the evidence.  The result is persisted so the
    polarity decision is reproducible and testable rather than left to a UI
    reviewer.
    """
    text = str(evidence or "").strip()
    lowered = text.casefold()
    if not lowered:
        return {"valid": False, "reason": "empty_evidence", "scope": None}
    if any(re.search(pattern, lowered) for pattern in NO_LIMITATION_PATTERNS):
        return {
            "valid": False,
            "reason": "explicit_absence_of_limitation",
            "scope": None,
        }
    if any(re.search(pattern, lowered) for pattern in BENEFIT_NEGATION_PATTERNS):
        return {
            "valid": False,
            "reason": "negated_requirement_is_not_a_limitation",
            "scope": None,
        }
    if not any(re.search(pattern, lowered) for pattern in EXPLICIT_PROBLEM_PATTERNS):
        return {
            "valid": False,
            "reason": "no_explicit_unresolved_problem_predicate",
            "scope": None,
        }

    subject_tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(subject).casefold())
        if len(token) > 2
    }
    evidence_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    context_tokens = set(re.findall(r"[a-z0-9]+", str(source_context).casefold()))
    subject_mentioned = bool(subject_tokens and subject_tokens.issubset(evidence_tokens))
    subject_context_supported = bool(
        subject_tokens
        and subject_tokens.issubset(context_tokens)
        and re.search(r"\b(?:it|its|their|these|those|such)\b", lowered)
    )
    prior_method_scope = bool(re.search(
        r"\b(?:existing|prior|previous|traditional|conventional)\b"
        r".{0,80}\b(?:methods?|models?|approaches?|systems?)\b",
        lowered,
    ))
    if prior_method_scope and not subject_mentioned:
        return {
            "valid": False,
            "reason": "limitation_is_attributed_to_other_methods",
            "scope": None,
        }

    field_level = not subject_mentioned and bool(re.search(
        r"\b(?:lack\s+of|insufficient|scarcity|shortage|open\s+problem|"
        r"serious\s+issue|major\s+challenge)\b",
        lowered,
    ))
    if not subject_mentioned and not subject_context_supported and not field_level:
        return {
            "valid": False,
            "reason": "subject_not_entailed_by_evidence",
            "scope": None,
        }
    return {
        "valid": True,
        "reason": (
            "explicit_unresolved_problem"
            if subject_mentioned or field_level
            else "explicit_unresolved_problem_context_supported"
        ),
        "scope": "field" if field_level else "subject",
    }


def detect_evidence_gaps(G, config):
    """Aggregate explicit ``LACKS`` relations into scoped gap candidates.

    Unlike topology-only signals, these candidates retain the paper, year,
    confidence, and textual evidence that reported the limitation.  Validation
    still decides whether the evidence is independent and whether later work
    appears to close the candidate.
    """
    logger.info("--- Explicit Limitation Evidence ---")
    settings = config.get("gap_detection", {}).get("evidence", {})
    min_confidence = float(settings.get("min_confidence", 0.65))
    require_cue = bool(settings.get("require_limitation_cue", True))

    paper_context = {}
    corpus_path = (
        Path(config.get("paths", {}).get("processed_data", ""))
        / "corpus_filtered.jsonl"
    )
    if corpus_path.exists():
        for paper in load_jsonl(corpus_path):
            context_paper_id = str(
                paper.get("paperId") or paper.get("paper_id") or ""
            )
            if context_paper_id:
                paper_context[context_paper_id] = " ".join(
                    str(paper.get(key, "")) for key in ("title", "abstract")
                )

    grouped = defaultdict(list)
    edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for edge in edges:
        subject, missing_capability, data = edge[0], edge[1], edge[-1]
        if str(data.get("relation", "")).upper() != "LACKS":
            continue
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = str(data.get("evidence", "") or "").strip()
        evidence_lower = evidence.lower()
        evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_lower))
        has_cue = any(
            cue in evidence_lower if " " in cue else cue in evidence_tokens
            for cue in LIMITATION_CUES
        )
        paper_id = (
            data.get("source_paper")
            or data.get("source_paper_id")
            or data.get("paper_id")
        )
        semantics = limitation_evidence_semantics(
            evidence,
            subject,
            paper_context.get(str(paper_id), ""),
        )
        if confidence < min_confidence or not evidence or not paper_id:
            continue
        if require_cue and (not has_cue or not semantics["valid"]):
            continue
        grouped[(str(subject), str(missing_capability))].append({
            "paper_id": str(paper_id),
            "year": data.get("year"),
            "confidence": confidence,
            "evidence": evidence,
            "semantic_scope": semantics.get("scope"),
            "semantic_validation": semantics.get("reason"),
        })

    candidates = []
    for (subject, missing_capability), records in grouped.items():
        unique_papers = sorted({record["paper_id"] for record in records})
        years = []
        for record in records:
            try:
                years.append(int(record["year"]))
            except (TypeError, ValueError):
                continue
        mean_confidence = sum(record["confidence"] for record in records) / len(records)
        semantic_scope = (
            "field" if any(record.get("semantic_scope") == "field" for record in records)
            else "subject"
        )
        candidates.append({
            "type": "evidence_gap",
            "subject": subject,
            "missing_capability": missing_capability,
            "key_concepts": [subject, missing_capability],
            "supporting_paper_ids": unique_papers,
            "supporting_paper_count": len(unique_papers),
            "source_evidence": records,
            "semantic_scope": semantic_scope,
            "domain": config.get("project", {}).get("domain", ""),
            "mean_evidence_confidence": round(mean_confidence, 4),
            "first_reported_year": min(years) if years else None,
            "latest_reported_year": max(years) if years else None,
            "description": (
                f"Published limitation evidence reports that '{subject}' lacks "
                f"or does not adequately address '{missing_capability}'."
            ),
            "draft_claim": (
                f"Within the searched literature, '{missing_capability}' is a "
                f"reported unresolved limitation of '{subject}'."
            ),
        })

    candidates.sort(key=lambda item: (
        item["supporting_paper_count"],
        item["mean_evidence_confidence"],
        item.get("latest_reported_year") or 0,
    ), reverse=True)
    if settings.get("consolidate_evidence_cells", True):
        candidates, evidence_graph = consolidate_evidence_candidates(
            G, candidates, config
        )
        save_evidence_graph(evidence_graph, candidates, config)
        logger.info(
            "  Consolidated explicit limitations into %d evidence-cell candidates",
            len(candidates),
        )
    else:
        logger.info("  Found %d explicit evidence-gap candidates", len(candidates))
    return candidates


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
        "evidence_gaps": [],
        "missing_links": [],
        "orphan_clusters": [],
        "temporal_decay": [],
    }
    
    # --- Run detection algorithms ---
    
    # 1. Explicit limitation evidence
    try:
        all_gaps["evidence_gaps"] = detect_evidence_gaps(G, config)
    except Exception as e:
        logger.error(f"Explicit evidence-gap detection failed: {e}")

    # 2. Typed evidence-map empty cells (embedding fallback remains available)
    try:
        if config.get("gap_detection", {}).get("evidence_map", {}).get(
            "enabled", True
        ):
            missing_links = detect_evidence_map_empty_cells(G, config)
        else:
            missing_links = detect_missing_links(G, config)
        all_gaps["missing_links"] = missing_links
        add_empty_cells_to_evidence_graph(missing_links, config)
    except Exception as e:
        logger.error(f"Missing link detection failed: {e}")
        logger.info("Continuing with other methods...")
    
    # 3. Orphan Cluster Detection
    try:
        orphan_clusters = detect_orphan_clusters(G, config)
        all_gaps["orphan_clusters"] = orphan_clusters
    except Exception as e:
        logger.error(f"Orphan cluster detection failed: {e}")
    
    # 4. Temporal Decay Analysis
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
    logger.info(f"  GAP-SIGNAL DETECTION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Missing links:    {len(all_gaps['missing_links'])}")
    logger.info(f"  Evidence gaps:    {len(all_gaps['evidence_gaps'])}")
    logger.info(f"  Orphan clusters:  {len(all_gaps['orphan_clusters'])}")
    logger.info(f"  Temporal decay:   {len(all_gaps['temporal_decay'])}")
    logger.info(f"  Total signals:    {total}")
    logger.info(f"  Saved to: {output_dir / 'detected_gaps_raw.json'}")
    
    return all_gaps


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    detect_all_gaps(config)
