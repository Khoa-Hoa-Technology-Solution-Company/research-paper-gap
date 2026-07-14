"""
Independent NLI Evaluation using DeBERTa-v3-large-mnli-fever-anli.
Replaces LLM-as-judge with a dedicated NLI model.
Also computes Cohen's Kappa between LLM-judge and DeBERTa.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config


def evaluate_nli_deberta(gaps, method_name):
    """
    Use DeBERTa NLI model to evaluate Grounds + Warrant => Claim entailment.
    Returns entailment rate and per-gap labels.
    """
    from transformers import pipeline

    print(f"[*] Loading DeBERTa NLI model...")
    nli = pipeline("text-classification",
                   model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                   device=-1)  # CPU; use device=0 for GPU

    print(f"[*] Evaluating {len(gaps)} gaps for {method_name}...")
    labels = []
    entailed = 0
    total = 0

    for g in gaps:
        grounds = g.get("Grounds", "")
        warrant = g.get("Warrant", "")
        claim = g.get("Claim", "")

        if not grounds or not claim:
            labels.append("SKIP")
            continue

        premise = f"{grounds} {warrant}".strip()
        # DeBERTa NLI input format: premise [SEP] hypothesis
        try:
            result = nli(f"{premise} [SEP] {claim}", truncation=True, max_length=512)
            label = result[0]["label"].upper()
            # Normalize labels (model may output ENTAILMENT/NEUTRAL/CONTRADICTION)
            if "ENTAIL" in label:
                label = "ENTAILMENT"
                entailed += 1
            elif "CONTRA" in label:
                label = "CONTRADICTION"
            else:
                label = "NEUTRAL"
            labels.append(label)
            total += 1
        except Exception as e:
            print(f"  [!] Error: {e}")
            labels.append("ERROR")

    rate = entailed / total if total > 0 else 0.0
    print(f"[+] DeBERTa NLI for {method_name}: {rate*100:.1f}% ({entailed}/{total})")
    return rate, labels


def compute_cohens_kappa(labels_a, labels_b):
    """Compute Cohen's Kappa between two lists of labels."""
    assert len(labels_a) == len(labels_b)

    # Filter out SKIPs/ERRORs
    valid = [(a, b) for a, b in zip(labels_a, labels_b)
             if a not in ("SKIP", "ERROR") and b not in ("SKIP", "ERROR")]

    if not valid:
        return 0.0

    n = len(valid)
    categories = list(set([a for a, _ in valid] + [b for _, b in valid]))

    # Observed agreement
    agree = sum(1 for a, b in valid if a == b)
    po = agree / n

    # Expected agreement
    pe = 0
    for cat in categories:
        count_a = sum(1 for a, _ in valid if a == cat) / n
        count_b = sum(1 for _, b in valid if b == cat) / n
        pe += count_a * count_b

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa


def evaluate_llm_judge_labels(gaps, method_name):
    """
    Re-run LLM-as-judge to get labels (for Kappa comparison).
    Uses existing evaluate.py logic.
    """
    from src.llm_client import call_llm

    labels = []
    for g in gaps:
        grounds = g.get("Grounds", "")
        warrant = g.get("Warrant", "")
        claim = g.get("Claim", "")

        if not grounds or not claim:
            labels.append("SKIP")
            continue

        prompt = f"""Evaluate if the Claim is logically entailed by the Premises.
Premises:
- Grounds: {grounds}
- Warrant: {warrant}

Hypothesis:
- Claim: {claim}

Output strictly as JSON (no markdown):
{{"logical_relationship": "ENTAILMENT, NEUTRAL, or CONTRADICTION", "reason": "brief reason"}}"""

        try:
            content = call_llm(prompt, temperature=0.0)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            result = json.loads(content)
            label = result.get("logical_relationship", "NEUTRAL").strip().upper()
            if "ENTAIL" in label:
                label = "ENTAILMENT"
            elif "CONTRA" in label:
                label = "CONTRADICTION"
            else:
                label = "NEUTRAL"
            labels.append(label)
        except Exception:
            labels.append("ERROR")

    return labels


def run_independent_nli(skip_llm_judge=False):
    """Run DeBERTa NLI on all methods and compute Kappa vs LLM-judge."""
    print("=" * 60)
    print("INDEPENDENT NLI EVALUATION (DeBERTa)")
    print("=" * 60)

    gap_files = {
        "KG-TABI": "kgtabi_gaps.json",
        "B1 Mulla RAG": "baseline_mulla_rag.json",
        "B2 Simple LLM": "baseline_simple_llm.json",
        "B3 GAPMAP": "baseline_gapmap.json",
        "B4 GraphRAG": "baseline_graphrag.json",
        "B5 LightRAG": "baseline_lightrag.json",
        "B6 HippoRAG": "baseline_hipporag.json",
    }

    results = []

    for method, filename in gap_files.items():
        filepath = os.path.join(config.GAPS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[!] {filepath} not found, skipping {method}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            gaps = json.load(f)

        deberta_rate, deberta_labels = evaluate_nli_deberta(gaps, method)

        kappa = None
        if not skip_llm_judge:
            print(f"[*] Running LLM-judge for Kappa comparison on {method}...")
            llm_labels = evaluate_llm_judge_labels(gaps, method)
            kappa = compute_cohens_kappa(deberta_labels, llm_labels)
            print(f"[+] Cohen's Kappa ({method}): {kappa:.3f}")

        results.append({
            "method": method,
            "deberta_entailment_rate": round(deberta_rate * 100, 1),
            "cohens_kappa": round(kappa, 3) if kappa is not None else "N/A",
        })

    print("\n" + "=" * 60)
    print("INDEPENDENT NLI RESULTS")
    print("=" * 60)
    print(f"  {'Method':<20} {'DeBERTa NLI%':>13} {'Kappa':>8}")
    for r in results:
        print(f"  {r['method']:<20} {r['deberta_entailment_rate']:>12}% {str(r['cohens_kappa']):>8}")

    out_path = os.path.join(config.DATA_DIR, "nli_independent_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Results saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm-judge", action="store_true",
                        help="Skip LLM-judge comparison (no API calls needed)")
    args = parser.parse_args()
    run_independent_nli(skip_llm_judge=args.skip_llm_judge)
