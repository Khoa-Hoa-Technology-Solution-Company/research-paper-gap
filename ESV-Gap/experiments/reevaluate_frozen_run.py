"""Re-evaluate one completed run without overwriting its archived artifacts.

The script reuses its graph, corpus, raw detector output, and completed closure
search cache.  It is intended for before/after contract evaluation after a
validation-rule change; no collection, LLM extraction, or network retrieval is
performed when the cache is complete.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _code_root() -> Path:
    return Path(__file__).resolve().parents[1]


CODE_ROOT = _code_root()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.synthesise_research_gap import synthesise_research_gap  # noqa: E402
from src.validate_gaps import validate_all_gaps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to an existing completed run")
    parser.add_argument(
        "--output-name",
        default="contract_revalidation",
        help="Name of the child output directory (default: contract_revalidation)",
    )
    parser.add_argument(
        "--redetect",
        action="store_true",
        help="Regenerate candidates with the current detector in the child output",
    )
    parser.add_argument(
        "--min-problem-relevance",
        type=float,
        default=None,
        help=(
            "Override the predeclared problem-alignment threshold for a "
            "frozen sensitivity analysis. No collection, extraction, or "
            "external retrieval is performed."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = run_dir / "outputs" / args.output_name
    raw_gaps = run_dir / "outputs" / "detected_gaps_raw.json"
    closure_cache = run_dir / "outputs" / "gap_closure_search.json"
    metadata_path = run_dir / "run_metadata.json"
    if not raw_gaps.exists() or not (run_dir / "data" / "graph" / "knowledge_graph.pkl").exists():
        raise FileNotFoundError("The run needs detected_gaps_raw.json and knowledge_graph.pkl")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    config = yaml.safe_load((CODE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    if args.min_problem_relevance is not None:
        if not 0.0 <= args.min_problem_relevance <= 1.0:
            raise ValueError("--min-problem-relevance must be in [0, 1]")
        config["gap_validation"]["min_problem_relevance"] = args.min_problem_relevance
    config["project"]["domain"] = str(metadata.get("topic") or config["project"].get("domain", ""))
    # The base config is unrelated to an archived run. Reconstruct the actual
    # query plan from collection provenance so the certificate report does not
    # claim it used the template's default queries.
    raw_corpus = run_dir / "data" / "raw" / "all_papers_raw.jsonl"
    recorded_queries: set[str] = set()
    if raw_corpus.exists():
        for line in raw_corpus.read_text(encoding="utf-8").splitlines():
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue
            for query in paper.get("_matched_queries", []) or []:
                if str(query).strip():
                    recorded_queries.add(str(query).strip())
    if recorded_queries:
        config["collection"]["queries"] = sorted(recorded_queries)
    created_at = str(metadata.get("created_at") or "")
    if created_at:
        config["gap_validation"]["snapshot_date"] = datetime.fromisoformat(created_at).date().isoformat()
    graph_path = run_dir / "data" / "graph"
    if args.redetect:
        graph_path = output_dir / "graph"
    config["paths"] = {
        "raw_data": str(run_dir / "data" / "raw"),
        "processed_data": str(run_dir / "data" / "processed"),
        "triples": str(run_dir / "data" / "triples"),
        "graph": str(graph_path),
        "outputs": str(output_dir),
        "figures": str(output_dir / "figures"),
        "prompts": str(CODE_ROOT / "prompts"),
    }
    # A frozen re-evaluation must not alter evidence by issuing fresh searches.
    # Completed cached searches are still loaded below and their prior results
    # continue to participate in local/external closure checks.
    config["gap_validation"]["external_closure"]["enabled"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.redetect:
        graph_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            run_dir / "data" / "graph" / "knowledge_graph.pkl",
            graph_path / "knowledge_graph.pkl",
        )
        from src.detect_gaps import detect_all_gaps
        detect_all_gaps(config)
        evidence_graph_path = graph_path / "gap_evidence_graph.pkl"
        if evidence_graph_path.exists():
            import pickle
            from src.visualise import create_gap_evidence_map
            with open(evidence_graph_path, "rb") as stream:
                evidence_graph = pickle.load(stream)
            create_gap_evidence_map(
                evidence_graph, output_dir / "gap_evidence_graph.html"
            )
    else:
        shutil.copyfile(raw_gaps, output_dir / "detected_gaps_raw.json")
    if closure_cache.exists():
        shutil.copyfile(closure_cache, output_dir / "gap_closure_search.json")

    validate_all_gaps(config)
    synthesis = synthesise_research_gap(config)
    top_candidates = []
    for item in sorted(
        synthesis.get("candidate_audit", []),
        key=lambda value: value.get("source_candidate", {}).get(
            "candidate_quality", {}
        ).get("score", 0.0),
        reverse=True,
    )[:10]:
        source = item.get("source_candidate", {})
        top_candidates.append({
            "type": source.get("type"),
            "candidate": source.get("missing_capability") or (
                f"{source.get('head', '')} x {source.get('tail', '')}"
            ),
            "quality_score": source.get("candidate_quality", {}).get("score", 0.0),
            "independent_source_count": source.get("candidate_quality", {}).get(
                "independent_source_count", 0
            ),
            "failed_synthesis_gates": [
                gate for gate, passed in item.get("decision", {}).get(
                    "hard_gates", {}
                ).items() if not passed
            ],
            "failed_certificate_gates": item.get("gap_certificate", {}).get(
                "failed_gates", []
            ),
        })
    summary = {
        "source_run": run_dir.name,
        "output_dir": str(output_dir),
        "snapshot_date": config["gap_validation"]["snapshot_date"],
        "min_problem_relevance": config["gap_validation"]["min_problem_relevance"],
        "validation": json.loads((output_dir / "gap_validation_audit.json").read_text(encoding="utf-8")).get("summary", {}),
        "synthesis": {
            "evaluated_candidate_count": synthesis["evaluated_candidate_count"],
            "qualified_gap_count": synthesis["qualified_gap_count"],
            "certified_gap_count": synthesis.get("certified_gap_count", 0),
            "rejection_gate_counts": synthesis["rejection_gate_counts"],
            "certificate_rejection_gate_counts": synthesis.get(
                "certificate_rejection_gate_counts", {}
            ),
            "top_candidate_audit": top_candidates,
        },
    }
    (output_dir / "contract_revalidation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
