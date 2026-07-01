import json
import os
from src import config
from src.llm_client import get_llm_client, get_llm_model, call_llm

# 3-shot Examples for TABI inference
TABI_EXAMPLES = """
Example 1:
Topological Evidence: Zero bridge relations between Community A (Istio, Envoy Proxy, mTLS, JWT) and Community B (Autoencoders, Anomaly detection, Drift detection).
JSON Output:
{
  "Grounds": "Modularity clustering isolated Community A (Envoy Proxy, mTLS, JWT) and Community B (Autoencoders, Drift detection). There are zero bridge relations between these nodes in the literature graph.",
  "Claim": "Integrating unsupervised autoencoders directly into Envoy proxy sidecars for real-time traffic drift and anomaly detection.",
  "Warrant": "Sidecars inspect all ingress/egress traffic. Standard JWT and mTLS only authenticate identity, but cannot detect malicious behavior post-auth. Deploying lightweight anomaly detection models inside the proxy layer allows immediate mitigation of insider threats without inflating network latency.",
  "Bucket": "more_probable"
}

Example 2:
Topological Evidence: Zero bridge relations between Community A (Search-based software testing, Genetic algorithms) and Community B (Smart contracts, Solidity, Gas optimization).
JSON Output:
{
  "Grounds": "Modularity clustering isolated Community A (Search-based software testing, Genetic algorithms) and Community B (Solidity, Gas optimization). There are zero bridge relations between these nodes in the literature graph.",
  "Claim": "Applying search-based test generation algorithms (like genetic programming) to optimize gas consumption test suites for EVM smart contracts.",
  "Warrant": "Existing smart contract test generators focus on coverage or vulnerability detection, but do not optimize for gas usage profiles. Bridging these communities helps developers automatically identify high-gas execution paths before deployment.",
  "Bucket": "more_probable"
}

Example 3:
Topological Evidence: Stagnant Concept 'Serverless cold start' has a temporal decay rate of 0.85 and no links to 'Static code analysis'.
JSON Output:
{
  "Grounds": "Stagnant concept 'Serverless cold start' has a temporal decay rate of 0.85. In the literature graph, it lacks connections to 'Static code analysis'.",
  "Claim": "Static analysis of serverless deployment configurations (IaC) using AST parsers to predict cold-start latency patterns.",
  "Warrant": "Serverless performance suffers from cold starts. Current static analyzers only check for security flaws, not performance anti-patterns. Bridging these areas allows identifying heavy imports during compile-time.",
  "Bucket": "least_probable"
}
"""

def infer_gaps_from_clusters(orphans: list[dict]) -> list[dict]:
    """
    Generate TABI gaps for disconnected Louvain orphan clusters.
    """
    if len(orphans) < 2:
        print("[*] Fewer than 2 orphan clusters detected. No cross-community gaps to deduce.")
        return []
        
    client = get_llm_client()
    model = get_llm_model()
    gaps = []
    
    # Compare every pair of orphan clusters to deduce missing bridges
    for i in range(len(orphans)):
        cluster_a = orphans[i]
        nodes_a = ", ".join(cluster_a["nodes"][:8]) # Limit to 8 nodes to avoid token bloat
        
        for j in range(i + 1, len(orphans)):
            cluster_b = orphans[j]
            nodes_b = ", ".join(cluster_b["nodes"][:8])
            
            print(f"[*] Inferring bridge gap between Community {cluster_a['cluster_id']} and Community {cluster_b['cluster_id']}...")
            
            prompt = f"""
            Act as a Senior Software Engineering Researcher. Based on the topological analysis of a scientific knowledge graph, the algorithm found a structural hole: 
            There are ZERO bridge relations between Community A (Nodes: {nodes_a}) and Community B (Nodes: {nodes_b}).
            
            Deduce an implicit research gap addressing this missing connection using the TABI framework.
            
            Here are three reference examples of how to output under the TABI framework:
            {TABI_EXAMPLES}
            
            Generate the output for the current case. Output strictly as a JSON object with no markdown formatting (like ```json), backticks, or explanation surrounding the JSON.
            JSON Schema:
            {{
              "Grounds": "Describe the graph evidence (disconnection between these two communities and their concepts).",
              "Claim": "Formulate a clear research gap statement combining concepts/methods from both communities.",
              "Warrant": "Explain the technical justification: Why is bridging these two communities important for the field of Software Engineering?",
              "Bucket": "Classify as 'more_probable' (highly feasible/immediate next step) or 'least_probable' (speculative/long-term gap)."
            }}
            """
            
            try:
                content = call_llm(prompt, temperature=0.2)
                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[0].startswith("```json") or lines[0].startswith("```"):
                        content = "\n".join(lines[1:-1])
                        
                gap_data = json.loads(content)
                gap_data["type"] = "orphan_cluster"
                gap_data["source"] = f"Community {cluster_a['cluster_id']} vs Community {cluster_b['cluster_id']}"
                gaps.append(gap_data)
                print(f"[+] Inferred Gap Claim: {gap_data.get('Claim')}")
            except Exception as e:
                print(f"[!] Error inferring gap for communities {cluster_a['cluster_id']} & {cluster_b['cluster_id']}: {e}")
                
    return gaps

def infer_gaps_from_decay(stagnant_concepts: list[dict]) -> list[dict]:
    """
    Generate TABI gaps for stagnant nodes that have stalled over time.
    """
    if not stagnant_concepts:
        print("[*] No stagnant concepts to analyze.")
        return []
        
    client = get_llm_client()
    model = get_llm_model()
    gaps = []
    
    for sc in stagnant_concepts:
        concept = sc["node"]
        decay_rate = sc["decay_rate"]
        print(f"[*] Inferring gap for stagnant concept '{concept}'...")
        
        prompt = f"""
        Act as a Senior Software Engineering Researcher. Based on the temporal analysis of a scientific knowledge graph, the algorithm found a stagnant concept:
        The concept '{concept}' has a high temporal decay rate of {decay_rate:.2f} (meaning new research activity connected to it has declined by {decay_rate*100:.0f}% in recent years).
        
        Deduce an implicit research gap addressing this stalled thread to revive or pivot research on it using the TABI framework.
        
        Here are reference examples of how to output under the TABI framework:
        {TABI_EXAMPLES}
        
        Generate the output for the current case. Output strictly as a JSON object with no markdown formatting (like ```json), backticks, or explanation surrounding the JSON.
        JSON Schema:
        {{
          "Grounds": "Describe the temporal decay evidence (concept name, decay rate, and general decline in connections).",
          "Claim": "Formulate a clear research gap statement pointing out what is missing to revive/pivot research on this concept.",
          "Warrant": "Explain the technical justification: Why is reviving or pivoting research on this concept important for Software Engineering?",
          "Bucket": "Classify as 'more_probable' (highly feasible/immediate next step) or 'least_probable' (speculative/long-term gap)."
        }}
        """
        
        try:
            content = call_llm(prompt, temperature=0.2)
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
                    
            gap_data = json.loads(content)
            gap_data["type"] = "temporal_decay"
            gap_data["source"] = f"Stagnant Concept: {concept}"
            gaps.append(gap_data)
            print(f"[+] Inferred Gap Claim: {gap_data.get('Claim')}")
        except Exception as e:
            print(f"[!] Error inferring gap for stagnant concept '{concept}': {e}")
            
    return gaps

def run_inference():
    orphans_path = os.path.join(config.GRAPH_DIR, "orphan_clusters.json")
    stagnant_path = os.path.join(config.GRAPH_DIR, "temporal_decay.json")
    
    if not os.path.exists(orphans_path) or not os.path.exists(stagnant_path):
        print("[!] Topologic reports not found. Please run graph_analysis first.")
        return
        
    with open(orphans_path, "r", encoding="utf-8") as f:
        orphans = json.load(f)
        
    with open(stagnant_path, "r", encoding="utf-8") as f:
        stagnant = json.load(f)
        
    print("[*] Phase 4: Starting TABI Inference on topologic gaps...")
    cluster_gaps = infer_gaps_from_clusters(orphans)
    decay_gaps = infer_gaps_from_decay(stagnant)
    
    all_gaps = cluster_gaps + decay_gaps
    
    output_path = os.path.join(config.GAPS_DIR, "kgtabi_gaps.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved {len(all_gaps)} inferred gaps to {output_path}")

if __name__ == "__main__":
    run_inference()
