"""
Ablation Study - Measures contribution of each component.
Some variants are offline (filter existing results), others need API calls.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.graph_analysis import build_graph, detect_orphan_clusters, compute_temporal_decay
from src.tabi_inference import infer_gaps_from_clusters, infer_gaps_from_decay
from src.evaluate import count_unique_gaps, evaluate_nli_entailment_rate


def ablation_no_louvain():
    """Remove orphan cluster detection. Only use temporal decay gaps."""
    gaps_path = os.path.join(config.GAPS_DIR, "kgtabi_gaps.json")
    with open(gaps_path, "r", encoding="utf-8") as f:
        all_gaps = json.load(f)
    # Filter to only decay-based gaps
    return [g for g in all_gaps if g.get("type") == "temporal_decay"]


def ablation_no_decay():
    """Remove temporal decay. Only use orphan cluster gaps."""
    gaps_path = os.path.join(config.GAPS_DIR, "kgtabi_gaps.json")
    with open(gaps_path, "r", encoding="utf-8") as f:
        all_gaps = json.load(f)
    # Filter to only cluster-based gaps
    return [g for g in all_gaps if g.get("type") == "orphan_cluster"]


def ablation_no_entity_resolution():
    """
    Rebuild graph from raw_triples (no entity resolution), re-run detection + TABI.
    REQUIRES API CALLS for TABI inference.
    """
    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_triples = json.load(f)

    G = build_graph(raw_triples)
    orphans = detect_orphan_clusters(G)
    stagnant = compute_temporal_decay(G)

    cluster_gaps = infer_gaps_from_clusters(orphans)
    decay_gaps = infer_gaps_from_decay(stagnant)
    return cluster_gaps + decay_gaps


def ablation_no_tabi_3shot():
    """
    Use zero-shot TABI (no 3-shot examples). REQUIRES API CALLS.
    """
    from src.llm_client import call_llm

    orphans_path = os.path.join(config.GRAPH_DIR, "orphan_clusters.json")
    stagnant_path = os.path.join(config.GRAPH_DIR, "temporal_decay.json")

    with open(orphans_path, "r", encoding="utf-8") as f:
        orphans = json.load(f)
    with open(stagnant_path, "r", encoding="utf-8") as f:
        stagnant = json.load(f)

    gaps = []

    # Zero-shot cluster gaps
    for i in range(len(orphans)):
        for j in range(i + 1, len(orphans)):
            nodes_a = ", ".join(orphans[i]["nodes"][:8])
            nodes_b = ", ".join(orphans[j]["nodes"][:8])

            prompt = f"""Act as a Senior Researcher. The knowledge graph analysis found ZERO bridge relations between Community A (Nodes: {nodes_a}) and Community B (Nodes: {nodes_b}).
Deduce an implicit research gap. Output strictly as JSON (no markdown):
{{"Grounds": "...", "Claim": "...", "Warrant": "...", "Bucket": "more_probable or least_probable"}}"""

            try:
                content = call_llm(prompt, temperature=0.2)
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1])
                gap = json.loads(content)
                gap["type"] = "orphan_cluster"
                gap["source"] = f"Community {orphans[i]['cluster_id']} vs {orphans[j]['cluster_id']}"
                gaps.append(gap)
            except Exception as e:
                print(f"[!] Zero-shot cluster gap error: {e}")

    # Zero-shot decay gaps
    for sc in stagnant:
        prompt = f"""Act as a Senior Researcher. The concept '{sc["node"]}' has temporal decay rate {sc["decay_rate"]:.2f}.
Deduce an implicit research gap. Output strictly as JSON (no markdown):
{{"Grounds": "...", "Claim": "...", "Warrant": "...", "Bucket": "more_probable or least_probable"}}"""

        try:
            content = call_llm(prompt, temperature=0.2)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            gap = json.loads(content)
            gap["type"] = "temporal_decay"
            gap["source"] = f"Stagnant: {sc['node']}"
            gaps.append(gap)
        except Exception as e:
            print(f"[!] Zero-shot decay gap error: {e}")

    return gaps


def ablation_no_kg():
    """No KG at all = B3 GAPMAP baseline (already computed)."""
    gapmap_path = os.path.join(config.GAPS_DIR, "baseline_gapmap.json")
    if os.path.exists(gapmap_path):
        with open(gapmap_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def evaluate_variant(name, gaps):
    """Compute metrics for an ablation variant."""
    total = len(gaps)
    unique = count_unique_gaps(gaps)
    claim_lens = [len(g.get("Claim", "").split()) for g in gaps if g.get("Claim")]
    avg_words = sum(claim_lens) / len(claim_lens) if claim_lens else 0
    nli_rate = evaluate_nli_entailment_rate(gaps, name) if gaps else 0.0
    return {
        "method": name,
        "total_gaps": total,
        "unique_gaps": unique,
        "avg_words": round(avg_words, 1),
        "nli_entailment": round(nli_rate * 100, 1),
    }


def run_ablation(skip_api=False):
    """Run all ablation variants."""
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    results = []

    # Full KG-TABI (reference)
    full_path = os.path.join(config.GAPS_DIR, "kgtabi_gaps.json")
    with open(full_path, "r", encoding="utf-8") as f:
        full_gaps = json.load(f)
    results.append(evaluate_variant("Full KG-TABI", full_gaps))

    # Offline variants
    print("\n[*] Ablation: -Louvain (decay only)")
    results.append(evaluate_variant("-Louvain", ablation_no_louvain()))

    print("\n[*] Ablation: -Decay (clusters only)")
    results.append(evaluate_variant("-Decay", ablation_no_decay()))

    print("\n[*] Ablation: -KG (text-only TABI = B3)")
    results.append(evaluate_variant("-KG (B3 GAPMAP)", ablation_no_kg()))

    if not skip_api:
        print("\n[*] Ablation: -Entity Resolution (needs API)")
        no_er_gaps = ablation_no_entity_resolution()
        out = os.path.join(config.GAPS_DIR, "ablation_no_er.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(no_er_gaps, f, ensure_ascii=False, indent=2)
        results.append(evaluate_variant("-Entity Resolution", no_er_gaps))

        print("\n[*] Ablation: -3-shot (zero-shot TABI, needs API)")
        no_3shot_gaps = ablation_no_tabi_3shot()
        out = os.path.join(config.GAPS_DIR, "ablation_no_3shot.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(no_3shot_gaps, f, ensure_ascii=False, indent=2)
        results.append(evaluate_variant("-3-shot Examples", no_3shot_gaps))
    else:
        print("\n[*] Skipping API-dependent ablations (-ER, -3-shot). Use --run-api to enable.")

    # Print summary table
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<25} {'Total':>6} {'Unique':>7} {'AvgW':>6} {'NLI%':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['method']:<25} {r['total_gaps']:>6} {r['unique_gaps']:>7} {r['avg_words']:>6} {r['nli_entailment']:>5}%")

    out_path = os.path.join(config.DATA_DIR, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Ablation results saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-api", action="store_true", help="Run API-dependent ablations")
    args = parser.parse_args()
    run_ablation(skip_api=not args.run_api)
