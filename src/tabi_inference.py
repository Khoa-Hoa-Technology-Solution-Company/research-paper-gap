import json
import os
from src import config
from src.compatibility import score_cluster_pair
from src.llm_client import LLMCallError, call_llm_with_provenance

# 3-shot Examples for TABI inference
TABI_EXAMPLES = """
Example 1:
Topological Evidence: Zero bridge relations between Community A (Istio, Envoy Proxy, mTLS, JWT) and Community B (Autoencoders, Anomaly detection, Drift detection).
JSON Output:
{
  "Grounds": "Modularity clustering isolated Community A (Envoy Proxy, mTLS, JWT) and Community B (Autoencoders, Drift detection), with zero bridge relations between their member nodes in the literature graph under seed 42. This structural isolation does not prove a scientific research gap exists.",
  "Claim": "Evidence remains insufficient on whether unsupervised autoencoders integrated directly into Envoy proxy sidecars can detect real-time traffic drift without inflating routing latency or CPU utilization.",
  "Warrant": "Envoy proxy sidecars inspect all traffic, but standard JWT and mTLS only authenticate identity rather than behavioral anomalies. If lightweight drift detection can run in the proxy layer, anomaly mitigation could occur immediately; this remains an open question requiring empirical validation.",
  "Bucket": "near_term_feasible"
}

Example 2:
Topological Evidence: Zero bridge relations between Community A (Search-based software testing, Genetic algorithms) and Community B (Smart contracts, Solidity, Gas optimization).
JSON Output:
{
  "Grounds": "Modularity clustering isolated Community A (Search-based software testing, Genetic algorithms) and Community B (Solidity, Gas optimization), with zero bridge relations between their member nodes in the literature graph under seed 42. This does not establish that no prior work exists.",
  "Claim": "To what extent can genetic programming test generation optimize gas consumption validation profiles for EVM smart contracts compared to traditional coverage-focused test generation?",
  "Warrant": "Existing smart contract test generators target coverage or security flaws rather than gas optimization. If search-based testing can be adapted to target gas consumption metrics, developers can locate expensive execution paths; this hypothesis has not been independently verified.",
  "Bucket": "near_term_feasible"
}

Example 3:
Topological Evidence: Stagnant Concept 'Serverless cold start' has a temporal decay rate of 0.85 and no links to 'Static code analysis'.
JSON Output:
{
  "Grounds": "In the exploratory configuration, stagnant concept 'Serverless cold start' has a normalized decline score of 0.85, a negative Sen slope, and adjusted q < 0.05. It lacks event connections to 'Static code analysis' in the literature graph. This signal does not establish absolute absence of research.",
  "Claim": "To what extent can static code analysis of Infrastructure-as-Code (IaC) serverless configurations predict cold-start latency patterns without raising deployment-time verification overhead?",
  "Warrant": "If serverless configurations can be statically parsed to identify heavy import hierarchies during compile-time, developers might mitigate cold starts. This possibility warrants evaluation, but the graph signal itself is unverified by external source texts.",
  "Bucket": "long_term_or_speculative"
}
"""

TABI_BUCKETS = {"near_term_feasible", "long_term_or_speculative"}


def _cluster_prompt_nodes(cluster: dict, limit: int = 8) -> list[str]:
    """Return the same representative labels used by the compatibility gate."""
    candidates = cluster.get("representative_nodes") or cluster.get("nodes") or []
    return list(dict.fromkeys(str(node) for node in candidates))[:limit]


def _validate_tabi_output(payload: dict) -> dict:
    """Validate the public TABI contract before persisting model output."""
    if not isinstance(payload, dict):
        raise ValueError("TABI output must be a JSON object.")

    normalized = dict(payload)

    for field in ("Grounds", "Claim", "Warrant"):
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"TABI field {field!r} must be a non-empty string.")
        normalized[field] = value.strip()
    if normalized.get("Bucket") not in TABI_BUCKETS:
        raise ValueError(
            "TABI Bucket must be 'near_term_feasible' or "
            "'long_term_or_speculative'."
        )
    return normalized

def infer_gaps_from_clusters(orphans: list[dict]) -> list[dict]:
    """
    Generate TABI gaps for disconnected Louvain orphan clusters.
    """
    if len(orphans) < 2:
        print("[*] Fewer than 2 orphan clusters detected. No cross-community gaps to deduce.")
        return []
        
    gaps = []
    
    # Only forward compatible pairs to the LLM. Disconnection alone is not a
    # sufficient reason to synthesize a cross-community hypothesis.
    for i in range(len(orphans)):
        cluster_a = orphans[i]
        nodes_a = ", ".join(_cluster_prompt_nodes(cluster_a))
        
        for j in range(i + 1, len(orphans)):
            cluster_b = orphans[j]
            nodes_b = ", ".join(_cluster_prompt_nodes(cluster_b))
            compatibility = score_cluster_pair(cluster_a, cluster_b)

            if not compatibility["passed"]:
                print(
                     f"[*] Skipping Communities {cluster_a['cluster_id']} and "
                     f"{cluster_b['cluster_id']}: compatibility "
                     f"{compatibility['score']:.3f}; content overlap "
                     f"{compatibility['overlap'] if 'overlap' in compatibility else compatibility.get('lexical_overlap', 0):.3f} must be >= "
                     f"{compatibility['minimum_content_overlap'] if 'minimum_content_overlap' in compatibility else 0.1:.3f}"
                )
                continue

            print(f"[*] Inferring bridge gap between Community {cluster_a['cluster_id']} and Community {cluster_b['cluster_id']}...")
            
            prompt = f"""
            Act as a Senior Software Engineering Researcher. Based on a scientific knowledge graph, two isolated communities passed a pre-generation compatibility screen.
            Community A: {nodes_a}
            Community B: {nodes_b}
            Compatibility evidence: score={compatibility['score']:.2f}, shared terms={compatibility.get('shared_terms', [])}, structural isolation={compatibility.get('structural_evidence', 0.0):.2f}.

            Deduce an implicit research gap addressing this missing connection using the TABI framework.
            
            CONSTRAINTS:
            1. DO NOT use overclaiming language in "Grounds" (such as "indicating a complete decline in research", "proves", "shows no research exists", "demonstrates a research gap"). "Grounds" must strictly report the graph metrics and isolation under seed 42, and explicitly state that this does not prove a scientific research gap exists.
            2. "Claim" must be formulated as a testable follow-up hypothesis or question, not a solution project proposal (e.g. do not say "Developing X...", say "Evidence remains insufficient on whether X...", or "To what extent can X...").
            3. "Warrant" must use conditional language, identifying the rationale as an LLM-proposed unverified hypothesis rather than verified facts.

            Here are three reference examples of how to output under the TABI framework:
            {TABI_EXAMPLES}
            
            Generate the output for the current case. Output strictly as a JSON object with no markdown formatting (like ```json), backticks, or explanation surrounding the JSON.
            JSON Schema:
            {{
              "Grounds": "Describe the structural signal and the compatibility-screen result without claiming absence of prior work.",
              "Claim": "Formulate a testable hypothesis about a potential research gap or follow-up question combining the compatible concepts.",
              "Warrant": "Explain a conditional technical rationale, clearly distinguishing it from independently verified evidence.",
              "Bucket": "Classify feasibility only: 'near_term_feasible' or 'long_term_or_speculative'. This field is not a probability that the Claim is true."
            }}
            """
            
            try:
                result = call_llm_with_provenance(
                    prompt,
                    temperature=0.2,
                    prompt_version=config.TABI_INFERENCE_PROMPT_VERSION,
                    purpose="tabi-orphan-community-inference",
                )
                content = result.text
                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[0].startswith("```json") or lines[0].startswith("```"):
                        content = "\n".join(lines[1:-1])
                        
                gap_data = _validate_tabi_output(json.loads(content))
                gap_data["type"] = "orphan_cluster"
                gap_data["source"] = f"Community {cluster_a['cluster_id']} vs Community {cluster_b['cluster_id']}"
                gap_data["compatibility"] = compatibility
                gap_data["tabi_call"] = result.provenance
                gaps.append(gap_data)
                print(f"[+] Inferred Gap Claim: {gap_data.get('Claim')}")
            except LLMCallError as e:
                print(f"[!] Error inferring gap for communities {cluster_a['cluster_id']} & {cluster_b['cluster_id']}: {e}")
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

    gaps = []
    
    for sc in stagnant_concepts:
        concept = sc["node"]
        decay_rate = sc["decay_rate"]
        sen_slope = sc.get("sen_slope", 0)
        p_value = sc.get("mann_kendall_p", 1)
        q_value = sc.get("mann_kendall_q", 1)
        print(f"[*] Inferring gap for stagnant concept '{concept}'...")
        
        prompt = f"""
        Act as a Senior Software Engineering Researcher. Based on the temporal analysis of a scientific knowledge graph, the algorithm found a declining signal:
        The concept '{concept}' has a normalized recent-versus-historical decline score of {decay_rate:.2f}, a normalized-share Sen slope of {sen_slope:.4f}, Mann--Kendall p={p_value:.4f}, and Benjamini--Hochberg q={q_value:.4f}.
        
        Formulate a cautious, testable hypothesis for expert inspection using the TABI framework.
        
        CONSTRAINTS:
            1. DO NOT use overclaiming language in "Grounds" (such as "indicating a complete decline in research", "proves", "shows no research exists", "demonstrates a research gap"). "Grounds" must strictly report the temporal statistics (decline score, Sen slope, p/q values under the recorded configuration) and explicitly state that this signal does not prove the topic is neglected or unexplored.
        2. "Claim" must be formulated as a testable follow-up hypothesis or question, not a solution project proposal (e.g. do not say "Developing X...", say "Evidence remains insufficient on whether X...", or "To what extent can X...").
        3. "Warrant" must use conditional language, identifying the rationale as an LLM-proposed unverified hypothesis rather than verified facts.

        Here are reference examples of how to output under the TABI framework:
        {TABI_EXAMPLES}
        
        Generate the output for the current case. Output strictly as a JSON object with no markdown formatting (like ```json), backticks, or explanation surrounding the JSON.
        JSON Schema:
        {{
          "Grounds": "Describe the normalized temporal trend statistics without inferring absence of work.",
          "Claim": "Formulate a testable hypothesis about a potential research gap or scoping question for the concept.",
          "Warrant": "Give a conditional technical rationale, not independently verified evidence.",
          "Bucket": "Classify feasibility only: 'near_term_feasible' or 'long_term_or_speculative'. This field is not a probability that the Claim is true."
        }}
        """
        
        try:
            result = call_llm_with_provenance(
                prompt,
                temperature=0.2,
                prompt_version=config.TABI_INFERENCE_PROMPT_VERSION,
                purpose="tabi-temporal-inference",
            )
            content = result.text
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
                    
            gap_data = _validate_tabi_output(json.loads(content))
            gap_data["type"] = "temporal_decay"
            gap_data["source"] = f"Stagnant Concept: {concept}"
            gap_data["tabi_call"] = result.provenance
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
