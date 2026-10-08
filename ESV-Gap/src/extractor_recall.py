"""Extractor recall quantification on manually annotated gold standard.

Absence claims in a knowledge graph assume that missing edges reflect real
gaps in scientific literature rather than extraction errors. Without measuring
the extractor's recall on human-verified gold triples, any claim of "absence"
is uninterpretable.

This module evaluates relation extraction on a curated ground-truth benchmark:
- True Positives (TP): Gold triples correctly identified by the extractor.
- False Negatives (FN): Real relations present in text but missed by extractor.
- False Positives (FP): Extracted triples not present in the gold benchmark.
- Extractor Recall = TP / (TP + FN)
- Extractor Miss Rate = FN / (TP + FN) = 1 - Recall

The resulting miss rate bounds the epistemic confidence of any graph-derived gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.entity_normalization import (
    canonical_entity_key,
    canonical_entity_label,
    entity_tokens,
    mentions_entity,
)
from src.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("extractor_recall")

DEFAULT_BENCHMARK_PATH = Path("benchmarks/gold_triples_benchmark.json")


def _fuzzy_token_match(phrase_a: str, phrase_b: str, threshold: float = 0.60) -> bool:
    """Check if two entity mentions match via token overlap or substring."""
    key_a = canonical_entity_key(phrase_a)
    key_b = canonical_entity_key(phrase_b)
    if key_a == key_b or key_a in key_b or key_b in key_a:
        return True

    tokens_a = entity_tokens(phrase_a)
    tokens_b = entity_tokens(phrase_b)
    if not tokens_a or not tokens_b:
        return False

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union if union else 0.0
    if jaccard >= threshold:
        return True

    return (
        mentions_entity(tokens_a, tokens_b, threshold)
        or mentions_entity(tokens_b, tokens_a, threshold)
    )


def match_triple(
    gold: dict[str, Any],
    pred: dict[str, Any],
    entity_threshold: float = 0.60,
    strict_relation: bool = True,
) -> bool:
    """Determine whether an extracted triple satisfies a gold triple."""
    # Standardize field names from diverse extraction outputs
    pred_rel = str(pred.get("relation") or pred.get("relation_type") or "").strip().upper()
    gold_rel = str(gold.get("relation") or gold.get("relation_type") or "").strip().upper()

    if strict_relation:
        if pred_rel != gold_rel:
            return False
    else:
        # Relaxed: treat USES/APPLIED_TO or IMPROVES/ADDRESSES as semantically adjacent
        related_groups = [
            {"USES", "APPLIED_TO", "COMBINES"},
            {"IMPROVES", "ADDRESSES", "EXTENDS"},
        ]
        rel_match = (pred_rel == gold_rel) or any(
            pred_rel in g and gold_rel in g for g in related_groups
        )
        if not rel_match:
            return False

    # Extract subject/object phrases
    pred_subj = pred.get("subject")
    if isinstance(pred_subj, dict):
        pred_subj = pred_subj.get("name", "")
    pred_subj = str(pred_subj or "")

    pred_obj = pred.get("object")
    if isinstance(pred_obj, dict):
        pred_obj = pred_obj.get("name", "")
    pred_obj = str(pred_obj or "")

    gold_subj = str(gold.get("subject") or "")
    gold_obj = str(gold.get("object") or "")

    subj_ok = _fuzzy_token_match(pred_subj, gold_subj, threshold=entity_threshold)
    obj_ok = _fuzzy_token_match(pred_obj, gold_obj, threshold=entity_threshold)

    return subj_ok and obj_ok


def evaluate_extracted_triples(
    gold_papers: list[dict[str, Any]],
    extracted_triples: list[dict[str, Any]],
    entity_threshold: float = 0.60,
    strict_relation: bool = True,
) -> dict[str, Any]:
    """Calculate recall, precision, miss rate and breakdowns against gold data."""
    # Index extracted triples by paper_id if available, otherwise flat pool
    triples_by_paper: dict[str, list[dict[str, Any]]] = {}
    for t in extracted_triples:
        pid = str(t.get("source_paper_id") or t.get("paper_id") or "")
        triples_by_paper.setdefault(pid, []).append(t)

    all_extracted_flat = extracted_triples

    total_gold = 0
    true_positives = 0
    false_negatives = 0
    matched_pred_indices: set[int] = set()

    relation_breakdown: dict[str, dict[str, int]] = {}
    missed_triples: list[dict[str, Any]] = []

    for paper in gold_papers:
        pid = str(paper.get("paper_id", ""))
        paper_gold = paper.get("gold_triples", [])
        candidate_preds = triples_by_paper.get(pid, all_extracted_flat)

        for g in paper_gold:
            rel = str(g.get("relation", "")).upper()
            relation_breakdown.setdefault(
                rel, {"gold": 0, "tp": 0, "fn": 0}
            )
            relation_breakdown[rel]["gold"] += 1
            total_gold += 1

            found_match = False
            for idx, pred in enumerate(candidate_preds):
                if match_triple(g, pred, entity_threshold, strict_relation):
                    found_match = True
                    matched_pred_indices.add(id(pred))
                    break

            if found_match:
                true_positives += 1
                relation_breakdown[rel]["tp"] += 1
            else:
                false_negatives += 1
                relation_breakdown[rel]["fn"] += 1
                missed_triples.append({
                    "paper_id": pid,
                    "paper_title": paper.get("title", ""),
                    "gold_triple": g,
                    "reason": "not_extracted",
                })

    total_preds = len(extracted_triples)
    false_positives = max(0, total_preds - len(matched_pred_indices))

    recall = round(true_positives / total_gold, 4) if total_gold > 0 else 0.0
    miss_rate = round(false_negatives / total_gold, 4) if total_gold > 0 else 0.0
    precision = round(true_positives / total_preds, 4) if total_preds > 0 else 0.0
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )

    per_relation_metrics = {}
    for rel, counts in relation_breakdown.items():
        g_count = counts["gold"]
        tp_count = counts["tp"]
        fn_count = counts["fn"]
        r = round(tp_count / g_count, 4) if g_count > 0 else 0.0
        mr = round(fn_count / g_count, 4) if g_count > 0 else 0.0
        per_relation_metrics[rel] = {
            "gold_count": g_count,
            "tp": tp_count,
            "fn": fn_count,
            "recall": r,
            "miss_rate": mr,
        }

    return {
        "summary": {
            "gold_triples_total": total_gold,
            "extracted_triples_evaluated": total_preds,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "recall": recall,
            "miss_rate": miss_rate,
            "precision": precision,
            "f1_score": f1,
            "interpretation": (
                f"The extractor achieves a recall of {recall*100:.1f}%. "
                f"When a relation is absent from the knowledge graph, there is a "
                f"{miss_rate*100:.1f}% base rate that the absence is an extractor omission "
                f"rather than true literature absence."
            ),
        },
        "per_relation": per_relation_metrics,
        "missed_triples": missed_triples,
    }


def evaluate_extractor_recall(
    config: dict[str, Any],
    benchmark_path: str | Path | None = None,
    extracted_triples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage entry point: quantify extractor recall and miss rate."""
    output_dir = ensure_dir(config.get("paths", {}).get("outputs", "outputs"))

    # Resolve benchmark path
    if benchmark_path is None:
        cfg_bench = config.get("extractor_recall", {}).get("benchmark_path")
        benchmark_path = Path(cfg_bench) if cfg_bench else DEFAULT_BENCHMARK_PATH
    else:
        benchmark_path = Path(benchmark_path)

    if not benchmark_path.exists():
        logger.warning("Benchmark file not found at %s. Creating default.", benchmark_path)
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    benchmark_data = load_json(benchmark_path)
    gold_papers = benchmark_data.get("papers", [])

    # Resolve extracted triples
    if extracted_triples is None:
        cfg_extracted = config.get("extractor_recall", {}).get("extracted_triples_path")
        if cfg_extracted and Path(cfg_extracted).exists():
            extracted_triples = load_json(Path(cfg_extracted)) or []
        else:
            triples_path = Path(config.get("paths", {}).get("triples", "data/triples")) / "all_triples.json"
            if not triples_path.exists():
                # Check for alternative extracted_triples.json
                triples_path = Path(config.get("paths", {}).get("triples", "data/triples")) / "extracted_triples.json"

            if triples_path.exists():
                extracted_triples = load_json(triples_path) or []
            else:
                # Fallback: extract directly or simulate using deterministic patterns if available
                logger.info("No extracted triples found at %s; generating triples from benchmark text...", triples_path)
                from src.extract_triples import deterministic_limitation_triples
                extracted_triples = []
                for paper in gold_papers:
                    t_list = deterministic_limitation_triples("XAI", paper.get("title", ""), paper.get("text", ""))
                    for item in t_list:
                        extracted_triples.append({
                            "subject": item.get("subject", ""),
                            "relation": item.get("relation", ""),
                            "object": item.get("object", ""),
                            "source_paper_id": paper.get("paper_id", ""),
                        })

    threshold = float(config.get("extractor_recall", {}).get("entity_match_threshold", 0.60))
    strict_rel = bool(config.get("extractor_recall", {}).get("strict_relation", True))

    results = evaluate_extracted_triples(
        gold_papers=gold_papers,
        extracted_triples=extracted_triples or [],
        entity_threshold=threshold,
        strict_relation=strict_rel,
    )

    report_path = output_dir / "extractor_recall_report.json"
    save_json(results, report_path)
    logger.info("Extractor recall evaluation saved to %s: Recall=%.2f%%, Miss Rate=%.2f%%",
                report_path, results["summary"]["recall"] * 100, results["summary"]["miss_rate"] * 100)
    return results


if __name__ == "__main__":
    import yaml
    with open("config.yaml", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    evaluate_extractor_recall(conf)
