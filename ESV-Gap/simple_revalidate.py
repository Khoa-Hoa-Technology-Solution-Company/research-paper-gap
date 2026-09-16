#!/usr/bin/env python3
"""Simple revalidation using validate_candidate directly."""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_json, save_json, get_logger, load_jsonl
from src.validate_gaps import validate_candidate
import yaml

logger = get_logger("simple_revalidate")


def main():
    base_dir = Path(__file__).parent
    run_dir = base_dir / "runs" / "security_vulnerabilities_in_mo_20260824_133527"
    config_path = base_dir / "config_relaxed.yaml"

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    val_config = config["gap_validation"]

    logger.info(f"Revalidating: {run_dir.name}")
    logger.info("Relaxed thresholds:")
    logger.info(f"  min_supporting_papers: {val_config['min_supporting_papers']}")
    logger.info(f"  min_independent_paths: {val_config['min_independent_paths']}")
    logger.info(f"  min_specificity: {val_config['min_specificity']}")
    logger.info(f"  min_stability: {val_config['min_stability']}")

    # Load data
    detected_gaps_data = load_json(run_dir / "outputs" / "detected_gaps_raw.json")

    # Extract all candidates from different gap types
    detected_gaps = []
    for key in ["evidence_gaps", "unrealized_connections", "conceptual_voids"]:
        if key in detected_gaps_data:
            detected_gaps.extend(detected_gaps_data[key])

    logger.info(f"Loaded {len(detected_gaps)} candidates")

    graph_path = run_dir / "data" / "graph" / "knowledge_graph.pkl"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    corpus_path = run_dir / "data" / "processed" / "corpus_filtered.jsonl"
    if not corpus_path.exists():
        corpus_path = run_dir / "data" / "processed" / "corpus.jsonl"

    corpus = list(load_jsonl(corpus_path))
    logger.info(f"Corpus: {len(corpus)} papers")

    # Build document index
    doc_index = {}
    for doc in corpus:
        if "doi" in doc and doc["doi"]:
            doc_index[doc["doi"].lower()] = doc
        if "title" in doc and doc["title"]:
            doc_index[doc["title"].lower()] = doc

    # Validate each candidate
    validated = []
    review_required = []
    rejected = []

    for i, cand in enumerate(detected_gaps, 1):
        logger.info(f"\nCandidate {i}/{len(detected_gaps)}: {cand.get('type', 'unknown')}")

        result = validate_candidate(
            G=graph,
            candidate=cand,
            config=config,
            documents=corpus,
        )

        decision = result.get("status", "rejected")
        logger.info(f"  Decision: {decision}")
        logger.info(f"  Specificity: {result.get('metrics', {}).get('specificity', 0):.3f}")
        logger.info(f"  Papers: {result.get('supporting_paper_count', 0)}")
        logger.info(f"  Paths: {result.get('independent_evidence_path_count', 0)}")

        if decision == "automatically_eligible":
            validated.append(result)
        elif decision == "review_required":
            review_required.append(result)
        else:
            rejected.append(result)

    # Save results
    output_dir = run_dir / "outputs_relaxed"
    output_dir.mkdir(exist_ok=True)

    save_json(validated, output_dir / "validated_gaps.json")
    save_json(review_required, output_dir / "review_required_gaps.json")
    save_json(rejected, output_dir / "rejected_gaps.json")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Validated:       {len(validated)}")
    logger.info(f"Review required: {len(review_required)}")
    logger.info(f"Rejected:        {len(rejected)}")
    logger.info(f"\nResults: {output_dir}")

    if validated or review_required:
        logger.info("\n🎉 Found gaps!")
        return True
    else:
        logger.info("\n⚠️  No gaps passing relaxed validation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
