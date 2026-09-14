"""Run leakage-controlled temporal backtesting on a completed ESV-Gap run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.temporal_backtest import aggregate_results, run_cutoff_backtest  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether pre-cutoff candidates anticipate post-cutoff research outcomes."
    )
    parser.add_argument("run_dir", help="Completed run containing corpus and extracted triples")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--output-name", default="temporal_backtest")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    corpus_path = run_dir / "data" / "processed" / "corpus_filtered.jsonl"
    triples_path = run_dir / "data" / "triples" / "all_triples.json"
    metadata_path = run_dir / "run_metadata.json"
    if not corpus_path.exists() or not triples_path.exists():
        raise FileNotFoundError("The run must contain corpus_filtered.jsonl and all_triples.json")

    documents = load_jsonl(corpus_path)
    triples = json.loads(triples_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["project"]["domain"] = str(metadata.get("topic") or config["project"].get("domain", ""))

    available_years = sorted({int(document.get("year") or 0) for document in documents})
    cutoffs = [
        cutoff for cutoff in sorted(set(args.cutoffs))
        if any(year <= cutoff for year in available_years)
        and any(year > cutoff for year in available_years)
    ]
    if not cutoffs:
        raise ValueError("No cutoff has both pre-cutoff and post-cutoff papers")

    output_dir = run_dir / "outputs" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_cutoff_backtest(documents, triples, config, cutoff, output_dir)
        for cutoff in cutoffs
    ]
    report = {
        "schema_version": 1,
        "source_run": run_dir.name,
        "label_standard": "outcome_based_silver_standard",
        "interpretation": (
            "Post-cutoff limitation-plus-resolution events and future typed relations are positive controls; "
            "pre-cutoff source-resolved limitations are negative controls; unmatched candidates remain unlabelled."
        ),
        "limitations": [
            "The benchmark measures anticipatory candidate utility, not global novelty.",
            "Post-cutoff publication is an observable outcome, not proof that a paper was globally first.",
            "Certificate precision is not estimable without historical full-text and external-search snapshots.",
        ],
        "aggregate": aggregate_results(results),
        "cutoff_results": [
            {key: value for key, value in result.items() if key not in {
                "positive_controls", "negative_controls", "ranked_candidates"
            }}
            for result in results
        ],
    }
    (output_dir / "temporal_backtest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

