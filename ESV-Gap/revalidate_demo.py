#!/usr/bin/env python3
"""Re-validate existing run with relaxed thresholds to demonstrate gap detection."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_json, save_json, get_logger
from src.validate_gaps import validate_all_gaps
import yaml
import shutil

logger = get_logger("revalidate_demo")


def revalidate_with_relaxed_config(run_dir: Path, config_path: Path):
    """Re-run validation on existing candidates with relaxed settings."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Revalidating run: {run_dir.name}")
    logger.info(f"Using config: {config_path.name}")

    # Load existing detected gaps
    detected_gaps_path = run_dir / "outputs" / "detected_gaps_raw.json"
    if not detected_gaps_path.exists():
        logger.error(f"No detected gaps found at {detected_gaps_path}")
        return None

    detected_gaps = load_json(detected_gaps_path)
    logger.info(f"Loaded {len(detected_gaps)} raw candidates")

    # Load graph and corpus
    graph_path = run_dir / "data" / "graph" / "knowledge_graph.pkl"
    if not graph_path.exists():
        graph_path = run_dir / "data" / "graph" / "knowledge_graph.gpickle"

    corpus_path = run_dir / "data" / "processed" / "corpus.jsonl"

    if not graph_path.exists():
        logger.error(f"Graph not found: {graph_path}")
        return None

    logger.info("Loading knowledge graph...")
    import pickle
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Create temporary run directory for validate_all_gaps
    temp_run_dir = run_dir.parent / f"{run_dir.name}_relaxed_temp"
    temp_run_dir.mkdir(exist_ok=True)

    # Copy necessary files
    for subdir in ["data/graph", "data/processed", "outputs"]:
        src = run_dir / subdir
        dst = temp_run_dir / subdir
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Update config with paths
    temp_config = config.copy()
    temp_config["paths"]["outputs"] = str(temp_run_dir / "outputs")
    temp_config["gap_validation"]["corpus_path"] = str(corpus_path)

    # Run validation with relaxed config
    logger.info("Running validation with RELAXED thresholds...")
    logger.info(f"  - min_supporting_papers: {config['gap_validation']['min_supporting_papers']}")
    logger.info(f"  - min_independent_paths: {config['gap_validation']['min_independent_paths']}")
    logger.info(f"  - min_specificity: {config['gap_validation']['min_specificity']}")
    logger.info(f"  - min_stability: {config['gap_validation']['min_stability']}")

    results = validate_all_gaps(temp_config)

    # Copy results to output_relaxed directory
    output_dir = run_dir / "outputs_relaxed"
    output_dir.mkdir(exist_ok=True)

    # Copy all output files from temp run
    temp_outputs = temp_run_dir / "outputs"
    if temp_outputs.exists():
        for file in temp_outputs.glob("*.json"):
            shutil.copy(file, output_dir / file.name)
        for file in temp_outputs.glob("*.csv"):
            shutil.copy(file, output_dir / file.name)

    # Load audit for summary
    audit_path = output_dir / "gap_validation_audit.json"
    if audit_path.exists():
        audit = load_json(audit_path)
        summary = audit.get("summary", {})

        logger.info("\n" + "=" * 60)
        logger.info("REVALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Raw candidates:         {summary.get('raw_candidates', 0)}")
        logger.info(f"Automatically eligible: {summary.get('automatically_eligible', 0)}")
        logger.info(f"Review required:        {summary.get('review_required', 0)}")
        logger.info(f"Rejected:               {summary.get('rejected', 0)}")
        logger.info(f"\n✅ Results saved to: {output_dir}")

        # Cleanup temp directory
        shutil.rmtree(temp_run_dir, ignore_errors=True)

        if summary.get('automatically_eligible', 0) > 0 or summary.get('review_required', 0) > 0:
            logger.info("\n🎉 Found gaps! Check validated_gaps.json and review_required_gaps.json")
            return True
        else:
            logger.info("\n⚠️  Still no gaps passing validation. Corpus may be too small.")
            return False
    else:
        logger.error("Validation did not produce audit file")
        return False


if __name__ == "__main__":
    # Use the run with most papers
    base_dir = Path(__file__).parent

    # Try the security_vulnerabilities run (29 papers)
    run_dir = base_dir / "runs" / "security_vulnerabilities_in_mo_20260824_133527"

    if not run_dir.exists():
        # Fallback to any existing run
        runs_dir = base_dir / "runs"
        runs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not runs:
            print("❌ No existing runs found. Run the pipeline first.")
            sys.exit(1)
        run_dir = runs[-1]

    config_path = base_dir / "config_relaxed.yaml"

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    success = revalidate_with_relaxed_config(run_dir, config_path)
    sys.exit(0 if success else 1)
