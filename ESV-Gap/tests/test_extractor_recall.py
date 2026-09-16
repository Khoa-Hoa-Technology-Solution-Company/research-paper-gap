"""Unit tests for Component C: Extractor recall quantification."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.extractor_recall import (
    evaluate_extracted_triples,
    evaluate_extractor_recall,
    match_triple,
)
from src.utils import save_json


def test_match_triple_exact():
    gold = {"subject": "SHAP", "relation": "ADDRESSES", "object": "fidelity"}
    pred = {"subject": "SHAP", "relation": "ADDRESSES", "object": "fidelity"}
    assert match_triple(gold, pred) is True


def test_match_triple_fuzzy_and_case():
    gold = {"subject": "Counterfactual Explanations", "relation": "IMPROVES", "object": "Algorithmic Recourse"}
    pred = {"subject": "counterfactual explanation", "relation": "IMPROVES", "object": "recourse for algorithmic systems"}
    assert match_triple(gold, pred, entity_threshold=0.50) is True


def test_match_triple_wrong_relation():
    gold = {"subject": "SHAP", "relation": "LACKS", "object": "computational efficiency"}
    pred = {"subject": "SHAP", "relation": "IMPROVES", "object": "computational efficiency"}
    assert match_triple(gold, pred, strict_relation=True) is False


def test_evaluate_extracted_triples_perfect():
    gold_papers = [
        {
            "paper_id": "p1",
            "title": "Paper 1",
            "gold_triples": [
                {"subject": "A", "relation": "USES", "object": "B"},
                {"subject": "A", "relation": "LACKS", "object": "C"},
            ],
        }
    ]
    extracted = [
        {"subject": "A", "relation": "USES", "object": "B", "source_paper_id": "p1"},
        {"subject": "A", "relation": "LACKS", "object": "C", "source_paper_id": "p1"},
    ]
    res = evaluate_extracted_triples(gold_papers, extracted)
    assert res["summary"]["recall"] == 1.0
    assert res["summary"]["miss_rate"] == 0.0
    assert res["summary"]["true_positives"] == 2
    assert res["summary"]["false_negatives"] == 0


def test_evaluate_extracted_triples_miss_rate():
    gold_papers = [
        {
            "paper_id": "p1",
            "title": "Paper 1",
            "gold_triples": [
                {"subject": "A", "relation": "USES", "object": "B"},
                {"subject": "A", "relation": "LACKS", "object": "C"},
                {"subject": "B", "relation": "ADDRESSES", "object": "D"},
                {"subject": "C", "relation": "IMPROVES", "object": "E"},
            ],
        }
    ]
    # Only 1 of 4 triples extracted
    extracted = [
        {"subject": "A", "relation": "USES", "object": "B", "source_paper_id": "p1"},
        {"subject": "X", "relation": "USES", "object": "Y", "source_paper_id": "p1"},  # FP
    ]
    res = evaluate_extracted_triples(gold_papers, extracted)
    assert res["summary"]["gold_triples_total"] == 4
    assert res["summary"]["true_positives"] == 1
    assert res["summary"]["false_negatives"] == 3
    assert res["summary"]["recall"] == 0.25
    assert res["summary"]["miss_rate"] == 0.75
    assert len(res["missed_triples"]) == 3
    assert "USES" in res["per_relation"]
    assert res["per_relation"]["USES"]["recall"] == 1.0


def test_evaluate_extractor_recall_pipeline_flow(tmp_path: Path):
    bench_file = tmp_path / "bench.json"
    save_json(
        {
            "papers": [
                {
                    "paper_id": "t1",
                    "title": "Test Paper",
                    "gold_triples": [
                        {"subject": "XAI", "relation": "USES", "object": "Decision Trees"}
                    ],
                }
            ]
        },
        bench_file,
    )
    triples_dir = tmp_path / "triples"
    triples_dir.mkdir()
    save_json(
        [{"subject": "XAI", "relation": "USES", "object": "Decision Trees", "source_paper_id": "t1"}],
        triples_dir / "all_triples.json",
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    cfg = {
        "paths": {"outputs": str(outputs), "triples": str(triples_dir)},
        "extractor_recall": {
            "benchmark_path": str(bench_file),
            "entity_match_threshold": 0.6,
            "strict_relation": True,
        },
    }
    res = evaluate_extractor_recall(cfg)
    assert res["summary"]["recall"] == 1.0
    assert (outputs / "extractor_recall_report.json").exists()
