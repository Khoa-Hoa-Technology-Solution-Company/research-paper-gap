import json
import os
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
        
        # Add edge with attributes
        # Since it is a multigraph conceptually, if edge exists, we can keep the one with higher confidence
        # or aggregate them. Here we keep the highest confidence, and keep a list of years.
        if G.has_edge(sub, obj):
            existing_years = G[sub][obj].get("years", [])
            if t["year"] not in existing_years:
                existing_years.append(t["year"])
            G[sub][obj]["confidence"] = max(G[sub][obj]["confidence"], t["confidence"])
            G[sub][obj]["years"] = sorted(existing_years)
        else:
            G.add_edge(
                sub, 
                obj, 
                relation=t["relation"], 
                confidence=t["confidence"], 
                years=[t["year"]],
                # Anchor year (usually the first publication year)
                year=t["year"]
            )
            
    print(f"[+] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def detect_orphan_clusters(G: nx.DiGraph) -> list[dict]:
    """
    Run Louvain community partitioning and identify isolated orphan clusters.
    """
    print("[*] Phase 3a: Running Louvain community detection...")
    
    # Louvain requires an undirected graph
    G_undirected = G.to_undirected()
    
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_undirected)
    except ImportError:
        # Fallback to networkx modularity-based communities if python-louvain is missing
        print("[!] python-louvain not installed. Falling back to greedy_modularity_communities.")
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G_undirected)
        partition = {}
        for cid, c in enumerate(communities):
            for node in c:
                partition[node] = cid

    # Group nodes by community ID
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
        
    total_nodes = G.number_of_nodes()
    orphan_clusters = []
    
    for cid, nodes in communities.items():
        # Calculate R_size
        r_size = len(nodes) / total_nodes
        
        # Calculate R_bridge: outbound edges from cluster / total edges originating in cluster
        outbound_edges = 0
        total_cluster_out_edges = 0
        
        for u in nodes:
            for v in G.successors(u):
                total_cluster_out_edges += 1
                if v not in nodes:
                    outbound_edges += 1
                    
        r_bridge = 0.0
        if total_cluster_out_edges > 0:
            r_bridge = outbound_edges / total_cluster_out_edges
        
        print(f"[*] Community {cid}: nodes={len(nodes)}, R_size={r_size:.3f}, R_bridge={r_bridge:.3f}")
        
        # Check orphan cluster conditions
        if r_size >= config.LOUVAIN_MIN_SIZE_RATIO and r_bridge <= config.LOUVAIN_MAX_BRIDGE_RATIO:
            print(f"[!] Orphan Cluster detected: Community {cid}")
            orphan_clusters.append({
                "cluster_id": cid,
                "nodes": nodes,
                "r_size": r_size,
                "r_bridge": r_bridge
            })
            
    return orphan_clusters

def compute_temporal_decay(G: nx.DiGraph) -> list[dict]:
    """
    Calculate temporal concept decay. Flags stalled concepts.
    """
    print("[*] Phase 3b: Running Temporal Decay Analysis...")
    
    # 1. Collect all edge years to find the max year in the dataset
    all_years = []
    for _, _, data in G.edges(data=True):
        all_years.extend(data.get("years", [data.get("year")]))
    
    if not all_years:
        print("[!] No temporal data found on edges. Skipping decay analysis.")
        return []
        
    max_year = max(all_years)
    recent_period = [max_year - 1, max_year]
    print(f"[*] Dataset temporal range peak: {max_year}. Recent lookback window R: {recent_period}")
    
    stagnant_concepts = []
    
    for v in G.nodes():
        # Count edge events (incoming and outgoing) containing years
        edge_years = []
        for u, w, data in G.edges(data=True):
            if u == v or w == v:
                edge_years.extend(data.get("years", [data.get("year")]))
                
        total_events = len(edge_years)
        if total_events < 3:
            # Stalled concepts must have at least 3 historical connections to be significant
            continue
            
        recent_events = sum(1 for y in edge_years if y in recent_period)
        
        # Decay: 1.0 - (recent_activity / total_activity)
        decay_rate = 1.0 - (recent_events / total_events)
        
        if decay_rate >= config.TEMPORAL_DECAY_THRESHOLD:
            print(f"[!] Stagnant concept flagged: '{v}' (Decay: {decay_rate:.2f}, Events: {total_events})")
            stagnant_concepts.append({
                "node": v,
                "node_type": G.nodes[v].get("type", "CONCEPT"),
                "total_events": total_events,
                "recent_events": recent_events,
                "decay_rate": decay_rate
            })
            
    return stagnant_concepts

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
    # NetworkX GML writer requires string attributes and lists to be serialized
    # Let's clean edge properties for standard GML output compatibility
    G_export = G.copy()
    for u, v, data in G_export.edges(data=True):
        if "years" in data:
            data["years"] = ",".join(map(str, data["years"]))
            
    nx.write_gml(G_export, gml_path)
    print(f"[+] Saved knowledge graph structure to {gml_path}")
    
    # Louvain communities
    orphans = detect_orphan_clusters(G)
    orphans_path = os.path.join(config.GRAPH_DIR, "orphan_clusters.json")
    with open(orphans_path, "w", encoding="utf-8") as f:
        json.dump(orphans, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved orphan clusters report to {orphans_path}")
    
    # Temporal decay
    stagnant = compute_temporal_decay(G)
    stagnant_path = os.path.join(config.GRAPH_DIR, "temporal_decay.json")
    with open(stagnant_path, "w", encoding="utf-8") as f:
        json.dump(stagnant, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved temporal decay report to {stagnant_path}")

if __name__ == "__main__":
    run_analysis()
