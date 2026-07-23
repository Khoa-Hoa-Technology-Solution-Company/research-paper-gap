import hashlib
import json
import os
from collections import Counter
from math import erfc, sqrt
import networkx as nx
from src import config

def build_graph(triples: list[dict]) -> nx.DiGraph:
    """
    Construct a directed graph from the resolved triples.
    """
    print("[*] Building NetworkX DiGraph from resolved triples...")
    G = nx.DiGraph()
    
    for t in triples:
        sub = t["subject"]
        obj = t["object"]
        
        # Add nodes with their types
        G.add_node(sub, type=t["subject_type"])
        G.add_node(obj, type=t["object_type"])
        
        # Aggregate every extracted relation event on a directed node pair.  A
        # DiGraph is used for topology, but `events` preserves relation, year,
        # confidence, and evidence rather than overwriting earlier triples.
        event = {
            "relation": t["relation"], "year": t["year"],
            "confidence": t["confidence"], "evidence_quote": t.get("evidence_quote", ""),
        }
        for key in (
            "paper_id", "paper_title", "chunk_id", "chunk_index",
            "section_label", "sentence_start", "sentence_end",
            "evidence_char_start", "evidence_char_end",
            "evidence_location_status", "model_reported_extraction_score",
            "model_name", "prompt_version",
        ):
            if key in t:
                event[key] = t[key]
        if G.has_edge(sub, obj):
            events = G[sub][obj].setdefault("events", [])
            events.append(event)
            G[sub][obj]["confidence"] = max(G[sub][obj]["confidence"], t["confidence"])
            G[sub][obj]["years"] = [item["year"] for item in events]
            G[sub][obj]["relations"] = sorted({item["relation"] for item in events})
        else:
            G.add_edge(
                sub, 
                obj, 
                relation=t["relation"], 
                confidence=t["confidence"], 
                years=[t["year"]],
                relations=[t["relation"]],
                events=[event],
                # Anchor year (usually the first publication year)
                year=t["year"]
            )
            
    print(f"[+] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def gml_export_copy(graph: nx.Graph) -> nx.Graph:
    """Return a GML-safe copy without mutating analysis provenance.

    NetworkX's GML writer accepts only scalar attributes.  The analysis graph
    deliberately keeps lists of relation events (including nullable provenance
    fields), so convert non-scalar attributes to deterministic strings only on
    the export copy.  JSON retains the event record for users opening the GML
    artifact while the in-memory graph keeps its native, auditable structure.
    """
    exported = graph.copy()

    def normalize_attributes(attributes: dict) -> None:
        for key, value in list(attributes.items()):
            if key == "years" and isinstance(value, list):
                attributes[key] = ",".join(map(str, value))
            elif key == "relations" and isinstance(value, list):
                attributes[key] = ",".join(map(str, value))
            elif value is None:
                attributes[key] = ""
            elif not isinstance(value, (str, int, float, bool)):
                attributes[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )

    for _, attributes in exported.nodes(data=True):
        normalize_attributes(attributes)
    for _, _, attributes in exported.edges(data=True):
        normalize_attributes(attributes)
    return exported


def louvain_partition(G: nx.DiGraph) -> dict:
    """Return a deterministic Louvain partition of the undirected projection."""
    G_undirected = G.to_undirected()
    try:
        import community as community_louvain
        return community_louvain.best_partition(
            G_undirected,
            random_state=config.LOUVAIN_RANDOM_STATE,
            resolution=config.LOUVAIN_RESOLUTION,
        )
    except ImportError:
        print("[!] python-louvain not installed. Falling back to greedy modularity communities.")
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G_undirected)
        return {
            node: cid
            for cid, community in enumerate(communities)
            for node in community
        }


def community_cut_edge_metrics(
    graph: nx.Graph, partition: dict
) -> list[dict]:
    """Return the one canonical cut-edge calculation used by all analyses.

    ``graph`` must be the simple undirected projection supplied to Louvain.
    Keeping this function in the production topology module prevents the main
    report, sensitivity analyses, and rewiring diagnostic from drifting into
    subtly different denominators or size filters.
    """
    total_nodes = graph.number_of_nodes()
    communities: dict[object, set] = {}
    for node, community_id in partition.items():
        communities.setdefault(community_id, set()).add(node)

    metrics: list[dict] = []
    for community_id, nodes in communities.items():
        internal_edges = 0
        cross_edges = 0
        for source, target in graph.edges():
            if source in nodes and target in nodes:
                internal_edges += 1
            elif (source in nodes) != (target in nodes):
                cross_edges += 1
        denominator = internal_edges + cross_edges
        metrics.append({
            "community_id": int(community_id) if isinstance(community_id, int) else str(community_id),
            "node_count": len(nodes),
            "size_ratio": len(nodes) / total_nodes if total_nodes else 0.0,
            "internal_edges": internal_edges,
            "cross_edges": cross_edges,
            "cut_edge_fraction": cross_edges / denominator if denominator else 0.0,
        })
    return metrics


def topology_configuration_hash() -> str:
    """Hash the parameters that define the reported topology statistic."""
    payload = {
        "projection": "simple_undirected",
        "louvain_implementation": "python-louvain.best_partition",
        "random_state": config.LOUVAIN_RANDOM_STATE,
        "resolution": config.LOUVAIN_RESOLUTION,
        "minimum_size_ratio": config.LOUVAIN_MIN_SIZE_RATIO,
        "maximum_cut_edge_fraction": config.LOUVAIN_MAX_BRIDGE_RATIO,
        "cut_edge_formula": "cross_edges/(internal_edges+cross_edges)",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_topology_run_config() -> str:
    """Persist all topology settings needed to reproduce a run."""
    path = os.path.join(config.GRAPH_DIR, "topology_run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "graph_projection_for_louvain": "undirected",
            "bridge_ratio_graph": "undirected projection (cut-edge ratio)",
            "edge_event_aggregation": "all relation/year/confidence/evidence events retained per directed node pair",
            "louvain_implementation": "python-louvain.best_partition",
            "louvain_random_state": config.LOUVAIN_RANDOM_STATE,
            "louvain_resolution": config.LOUVAIN_RESOLUTION,
            "topology_configuration_sha256": topology_configuration_hash(),
            "cut_edge_statistic_implementation": "src.graph_analysis.community_cut_edge_metrics",
            "min_size_ratio": config.LOUVAIN_MIN_SIZE_RATIO,
            "max_bridge_ratio": config.LOUVAIN_MAX_BRIDGE_RATIO,
            "temporal_decay_threshold": config.TEMPORAL_DECAY_THRESHOLD,
            "minimum_temporal_events": config.TEMPORAL_MIN_EVENTS,
            "generic_temporal_nodes_excluded": sorted(config.GENERIC_TEMPORAL_NODES),
            "compatibility_threshold": config.COMPATIBILITY_THRESHOLD,
            "compatibility_min_content_overlap": config.COMPATIBILITY_MIN_CONTENT_OVERLAP,
            "semantic_compatibility_threshold": config.SEMANTIC_COMPATIBILITY_THRESHOLD,
            "semantic_compatibility_model": config.SEMANTIC_COMPATIBILITY_MODEL,
            "temporal_min_distinct_years": config.TEMPORAL_MIN_DISTINCT_YEARS,
            "temporal_significance_level": config.TEMPORAL_SIGNIFICANCE_LEVEL,
            "temporal_fdr_significance_level": config.TEMPORAL_FDR_SIGNIFICANCE_LEVEL,
            "temporal_cutoff_year": config.TEMPORAL_CUTOFF_YEAR,
            "temporal_recent_window_covered_years": config.TEMPORAL_RECENT_WINDOW_YEARS,
            "temporal_min_negative_sen_slope": config.TEMPORAL_MIN_NEGATIVE_SLOPE,
            "temporal_zero_coverage_policy": (
                "Calendar years with zero relation-event coverage are recorded as "
                "missing and excluded from normalized activity, Mann--Kendall, "
                "Sen-slope, and recent/historical means."
            ),
            "temporal_activity_normalization": (
                "Node incident relation events divided by all retained relation events "
                "in the same covered year."
            ),
            "temporal_decay_definition": (
                "clip(1 - mean(normalized activity in recent covered years) / "
                "mean(normalized activity in historical covered years), 0, 1)"
            ),
            "temporal_multiple_testing_correction": (
                "Benjamini--Hochberg FDR across all eligible temporal concepts."
            ),
        }, f, ensure_ascii=False, indent=2)
    return path

def detect_orphan_clusters(G: nx.DiGraph) -> list[dict]:
    """
    Run Louvain community partitioning and identify isolated orphan clusters.
    """
    print("[*] Phase 3a: Running Louvain community detection...")

    partition = louvain_partition(G)
    G_undirected = G.to_undirected()

    # Group nodes by community ID
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
        
    orphan_clusters = []
    metrics_by_id = {
        item["community_id"]: item
        for item in community_cut_edge_metrics(G_undirected, partition)
    }

    for cid, nodes in communities.items():
        metrics = metrics_by_id[cid]
        r_size = metrics["size_ratio"]
        r_bridge = metrics["cut_edge_fraction"]
        
        print(f"[*] Community {cid}: nodes={len(nodes)}, R_size={r_size:.3f}, R_bridge={r_bridge:.3f}")
        
        # Check orphan cluster conditions
        if r_size >= config.LOUVAIN_MIN_SIZE_RATIO and r_bridge <= config.LOUVAIN_MAX_BRIDGE_RATIO:
            print(f"[!] Orphan Cluster detected: Community {cid}")
            orphan_clusters.append({
                "cluster_id": cid,
                "nodes": nodes,
                "representative_nodes": sorted(
                    nodes,
                    key=lambda node: (-G_undirected.degree(node), str(node).lower()),
                )[:config.COMPATIBILITY_TOP_NODES],
                "r_size": r_size,
                "r_bridge": r_bridge
            })
            
    return orphan_clusters

def _mann_kendall(values: list[float]) -> tuple[float, float]:
    """Return the Mann--Kendall z statistic and two-sided normal p-value."""
    n = len(values)
    statistic = sum(
        1 if values[j] > values[i] else -1 if values[j] < values[i] else 0
        for i in range(n - 1) for j in range(i + 1, n)
    )
    ties = Counter(values)
    variance = (n * (n - 1) * (2 * n + 5) - sum(
        count * (count - 1) * (2 * count + 5) for count in ties.values()
    )) / 18
    if variance <= 0:
        return 0.0, 1.0
    z_score = (statistic - 1) / sqrt(variance) if statistic > 0 else (
        (statistic + 1) / sqrt(variance) if statistic < 0 else 0.0
    )
    return z_score, erfc(abs(z_score) / sqrt(2))


def _sen_slope(years: list[int], values: list[float]) -> float:
    slopes = [
        (values[j] - values[i]) / (years[j] - years[i])
        for i in range(len(values) - 1) for j in range(i + 1, len(values))
        if years[j] != years[i]
    ]
    if not slopes:
        return 0.0
    slopes.sort()
    midpoint = len(slopes) // 2
    return slopes[midpoint] if len(slopes) % 2 else (slopes[midpoint - 1] + slopes[midpoint]) / 2


def _coerce_year(value) -> int | None:
    """Return an integer publication year when the value is parseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _edge_event_years(data: dict) -> list[int]:
    """Read every event year, with support for pre-event graph artifacts."""
    events = data.get("events") or []
    values = [event.get("year") for event in events]
    if not values:
        values = data.get("years", [data.get("year")])
    return [year for value in values if (year := _coerce_year(value)) is not None]


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return monotone Benjamini--Hochberg adjusted p-values in input order."""
    if not p_values:
        return []

    total = len(p_values)
    ordered = sorted(
        enumerate(p_values),
        key=lambda item: (item[1], item[0]),
    )
    adjusted = [1.0] * total
    running_minimum = 1.0
    for rank in range(total, 0, -1):
        index, p_value = ordered[rank - 1]
        candidate = min(1.0, max(0.0, p_value) * total / rank)
        running_minimum = min(running_minimum, candidate)
        adjusted[index] = running_minimum
    return adjusted


def _empty_temporal_report(G: nx.DiGraph, cutoff_year: int) -> dict:
    """Create a schema-stable temporal-screening report for every outcome."""
    return {
        "method_version": "temporal-screen-v2",
        "cutoff_year": cutoff_year,
        "recent_window_covered_years": config.TEMPORAL_RECENT_WINDOW_YEARS,
        "zero_coverage_policy": (
            "Calendar years with zero relation-event coverage are missing and are "
            "excluded from normalized activity, trend tests, and decay means."
        ),
        "normalization": (
            "Per-node incident relation events divided by all retained relation events "
            "in the same covered year."
        ),
        "normalized_decay_definition": (
            "clip(1 - mean(recent normalized activity) / "
            "mean(historical normalized activity), 0, 1); a zero historical mean "
            "does not support a decline and receives a score of 0."
        ),
        "multiple_testing": {
            "method": "Benjamini--Hochberg",
            "family": "all eligible nodes in this run",
            "fdr_significance_level": config.TEMPORAL_FDR_SIGNIFICANCE_LEVEL,
            "raw_p_reporting_level": config.TEMPORAL_SIGNIFICANCE_LEVEL,
        },
        "thresholds": {
            "minimum_events": config.TEMPORAL_MIN_EVENTS,
            "minimum_distinct_years": config.TEMPORAL_MIN_DISTINCT_YEARS,
            "minimum_negative_sen_slope": config.TEMPORAL_MIN_NEGATIVE_SLOPE,
            "normalized_decay_threshold": config.TEMPORAL_DECAY_THRESHOLD,
        },
        "coverage": {
            "calendar_years": [],
            "covered_years": [],
            "zero_coverage_years": [],
            "relation_events_by_year": {},
            "events_excluded_after_cutoff": 0,
        },
        "pipeline_counts": {
            "total_nodes": G.number_of_nodes(),
            "non_generic_nodes": 0,
            "nodes_with_minimum_events": 0,
            "nodes_with_minimum_distinct_years": 0,
            "eligible_nodes_tested": 0,
            "nodes_with_negative_sen_slope": 0,
            "nodes_with_raw_p_below_alpha": 0,
            "nodes_with_negative_slope_and_raw_p_below_alpha": 0,
            "nodes_with_fdr_q_below_alpha": 0,
            "nodes_with_negative_slope_and_fdr_q_below_alpha": 0,
            "nodes_meeting_normalized_decay_threshold": 0,
            "final_signals": 0,
        },
        "eligible_node_statistics": [],
        "status": "not_run",
    }


def compute_temporal_decay(
    G: nx.DiGraph,
    cutoff_year: int | None = None,
    return_report: bool = False,
):
    """Screen for declining normalized concept activity in completed covered years.

    The primary output remains the signal list for backwards compatibility. Set
    ``return_report=True`` to also obtain an auditable report containing coverage,
    eligibility flow counts, raw p-values, and Benjamini--Hochberg q-values.
    """
    cutoff = config.TEMPORAL_CUTOFF_YEAR if cutoff_year is None else int(cutoff_year)
    report = _empty_temporal_report(G, cutoff)
    print(f"[*] Phase 3b: Running Temporal Decay Analysis (cutoff year={cutoff})...")

    node_event_years = {node: [] for node in G.nodes()}
    retained_years = []
    excluded_after_cutoff = 0
    for source, target, data in G.edges(data=True):
        for year in _edge_event_years(data):
            if year > cutoff:
                excluded_after_cutoff += 1
                continue
            retained_years.append(year)
            node_event_years[source].append(year)
            if target != source:
                node_event_years[target].append(year)

    report["coverage"]["events_excluded_after_cutoff"] = excluded_after_cutoff
    if not retained_years:
        report["status"] = "no_completed_temporal_events"
        print("[!] No completed-year temporal data found. Skipping trend analysis.")
        return ([], report) if return_report else []

    corpus_events_by_year = Counter(retained_years)
    first_year, last_year = min(corpus_events_by_year), max(corpus_events_by_year)
    calendar_years = list(range(first_year, last_year + 1))
    covered_years = [year for year in calendar_years if corpus_events_by_year[year] > 0]
    zero_coverage_years = [year for year in calendar_years if corpus_events_by_year[year] == 0]
    report["coverage"].update({
        "calendar_years": calendar_years,
        "covered_years": covered_years,
        "zero_coverage_years": zero_coverage_years,
        "relation_events_by_year": {
            str(year): corpus_events_by_year[year] for year in calendar_years
        },
    })

    if len(covered_years) <= config.TEMPORAL_RECENT_WINDOW_YEARS:
        report["status"] = "insufficient_covered_years_for_historical_comparison"
        print("[!] Insufficient covered years for a historical-versus-recent trend test.")
        return ([], report) if return_report else []

    recent_years = covered_years[-config.TEMPORAL_RECENT_WINDOW_YEARS:]
    historical_years = covered_years[:-config.TEMPORAL_RECENT_WINDOW_YEARS]
    report["coverage"]["recent_covered_years"] = recent_years
    report["coverage"]["historical_covered_years"] = historical_years
    print(
        "[*] Covered complete years: "
        f"{covered_years[0]}--{covered_years[-1]}; recent window R={recent_years}; "
        f"zero-coverage calendar years={zero_coverage_years}"
    )

    counts = report["pipeline_counts"]
    candidates = []
    for node in sorted(G.nodes(), key=lambda value: str(value).casefold()):
        if str(node).strip().lower() in config.GENERIC_TEMPORAL_NODES:
            continue
        counts["non_generic_nodes"] += 1

        event_years = node_event_years[node]
        total_events = len(event_years)
        if total_events < config.TEMPORAL_MIN_EVENTS:
            continue
        counts["nodes_with_minimum_events"] += 1

        annual_events = Counter(event_years)
        observed_years = [year for year in covered_years if annual_events[year] > 0]
        if len(observed_years) < config.TEMPORAL_MIN_DISTINCT_YEARS:
            continue
        counts["nodes_with_minimum_distinct_years"] += 1

        annual_activity = [
            annual_events[year] / corpus_events_by_year[year]
            for year in covered_years
        ]
        z_score, p_value = _mann_kendall(annual_activity)
        slope = _sen_slope(covered_years, annual_activity)
        historical_activity = annual_activity[:-config.TEMPORAL_RECENT_WINDOW_YEARS]
        recent_activity = annual_activity[-config.TEMPORAL_RECENT_WINDOW_YEARS:]
        historical_mean = sum(historical_activity) / len(historical_activity)
        recent_mean = sum(recent_activity) / len(recent_activity)
        normalized_decay = (
            max(0.0, min(1.0, 1.0 - recent_mean / historical_mean))
            if historical_mean > 0 else 0.0
        )

        candidates.append({
            "node": node,
            "node_type": G.nodes[node].get("type", "CONCEPT"),
            "total_events": total_events,
            "recent_events": sum(annual_events[year] for year in recent_years),
            "observed_event_years": observed_years,
            "annual_events": {str(year): annual_events[year] for year in covered_years},
            "annual_share": {
                str(year): annual_events[year] / corpus_events_by_year[year]
                for year in covered_years
            },
            "historical_normalized_mean": historical_mean,
            "recent_normalized_mean": recent_mean,
            "decay_rate": normalized_decay,
            "decay_rate_definition": "normalized recent-versus-historical activity decline",
            "mann_kendall_z": z_score,
            "mann_kendall_p": p_value,
            "sen_slope": slope,
        })

    counts["eligible_nodes_tested"] = len(candidates)
    q_values = _benjamini_hochberg([candidate["mann_kendall_p"] for candidate in candidates])
    stagnant_concepts = []
    for candidate, q_value in zip(candidates, q_values):
        candidate["mann_kendall_q"] = q_value
        candidate["passes_negative_slope"] = (
            candidate["sen_slope"] < -config.TEMPORAL_MIN_NEGATIVE_SLOPE
        )
        candidate["passes_raw_p"] = (
            candidate["mann_kendall_p"] < config.TEMPORAL_SIGNIFICANCE_LEVEL
        )
        candidate["passes_fdr_q"] = (
            q_value < config.TEMPORAL_FDR_SIGNIFICANCE_LEVEL
        )
        candidate["passes_normalized_decay"] = (
            candidate["decay_rate"] >= config.TEMPORAL_DECAY_THRESHOLD
        )

        counts["nodes_with_negative_sen_slope"] += int(candidate["passes_negative_slope"])
        counts["nodes_with_raw_p_below_alpha"] += int(candidate["passes_raw_p"])
        counts["nodes_with_negative_slope_and_raw_p_below_alpha"] += int(
            candidate["passes_negative_slope"] and candidate["passes_raw_p"]
        )
        counts["nodes_with_fdr_q_below_alpha"] += int(candidate["passes_fdr_q"])
        counts["nodes_with_negative_slope_and_fdr_q_below_alpha"] += int(
            candidate["passes_negative_slope"] and candidate["passes_fdr_q"]
        )
        counts["nodes_meeting_normalized_decay_threshold"] += int(
            candidate["passes_normalized_decay"]
        )

        if (
            candidate["passes_normalized_decay"]
            and candidate["passes_negative_slope"]
            and candidate["passes_fdr_q"]
        ):
            candidate["trend_interpretation"] = (
                "FDR-significant declining normalized relation share"
            )
            stagnant_concepts.append(candidate)
            print(
                f"[!] Declining temporal signal: '{candidate['node']}' "
                f"(normalized decay={candidate['decay_rate']:.2f}, "
                f"q={candidate['mann_kendall_q']:.3f}, "
                f"slope={candidate['sen_slope']:.4f}, "
                f"events={candidate['total_events']})"
            )

    stagnant_concepts.sort(
        key=lambda item: (item["mann_kendall_q"], item["mann_kendall_p"], str(item["node"]).casefold())
    )
    counts["final_signals"] = len(stagnant_concepts)
    report["eligible_node_statistics"] = candidates
    report["status"] = "completed"
    return (stagnant_concepts, report) if return_report else stagnant_concepts

def run_analysis():
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    if not os.path.exists(resolved_path):
        print(f"[!] File {resolved_path} not found. Please run entity_resolution first.")
        return
        
    with open(resolved_path, "r", encoding="utf-8") as f:
        resolved_triples = json.load(f)
        
    # Build Graph
    G = build_graph(resolved_triples)
    
    # Save Graph
    gml_path = os.path.join(config.GRAPH_DIR, "knowledge_graph.gml")
    nx.write_gml(gml_export_copy(G), gml_path)
    print(f"[+] Saved knowledge graph structure to {gml_path}")
    
    # Louvain communities
    orphans = detect_orphan_clusters(G)
    orphans_path = os.path.join(config.GRAPH_DIR, "orphan_clusters.json")
    with open(orphans_path, "w", encoding="utf-8") as f:
        json.dump(orphans, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved orphan clusters report to {orphans_path}")

    run_config_path = save_topology_run_config()
    print(f"[+] Saved topology configuration to {run_config_path}")
    
    # Temporal decay. The signal list remains a compact pipeline artifact;
    # the companion report preserves every eligibility and FDR decision.
    stagnant, temporal_report = compute_temporal_decay(G, return_report=True)
    stagnant_path = os.path.join(config.GRAPH_DIR, "temporal_decay.json")
    with open(stagnant_path, "w", encoding="utf-8") as f:
        json.dump(stagnant, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved temporal decay report to {stagnant_path}")

    screening_path = os.path.join(config.GRAPH_DIR, "temporal_screening_report.json")
    with open(screening_path, "w", encoding="utf-8") as f:
        json.dump(temporal_report, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved temporal screening audit to {screening_path}")

if __name__ == "__main__":
    run_analysis()
