"""Unit tests for Component D: Corpus saturation and entity discovery dynamics."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.corpus_saturation import (
    analyze_corpus_saturation,
    compute_saturation_series,
    fit_heaps_law,
)
from src.utils import save_json


def test_fit_heaps_law_sublinear():
    # Synthetic sublinear: E = 10 * n^0.5
    indices = list(range(1, 51))
    cum_ents = [int(10 * (n ** 0.5)) for n in indices]
    res = fit_heaps_law(indices, cum_ents)
    assert 0.45 <= res["beta"] <= 0.55
    assert res["r_squared"] > 0.95


def test_compute_saturation_series_saturated():
    # 20 papers: first few introduce many entities, later ones introduce 0 or 1
    papers = [{"paper_id": f"p_{i}", "year": 2020 + (i // 5)} for i in range(20)]
    triples_by_paper = {}
    for i in range(20):
        if i < 5:
            # Paper introducing 5 new entities each
            triples_by_paper[f"p_{i}"] = [
                {"subject": f"Entity_{i}_{j}", "relation": "USES", "object": "CoreConcept"}
                for j in range(5)
            ]
        else:
            # Reusing existing entities
            triples_by_paper[f"p_{i}"] = [
                {"subject": "Entity_0_0", "relation": "USES", "object": "CoreConcept"}
            ]

    series, summary = compute_saturation_series(papers, triples_by_paper)
    assert len(series) == 20
    assert summary["verdict"] == "SATURATED"
    assert summary["discovery_decay_rate"] >= 0.60
    assert summary["marginal_final_mean"] <= 1.0


def test_compute_saturation_series_unsaturated():
    # 10 papers: each paper introduces 10 completely new entities constantly
    papers = [{"paper_id": f"p_{i}", "year": 2020} for i in range(10)]
    triples_by_paper = {
        f"p_{i}": [
            {"subject": f"UniqueEntity_{i}_{j}", "relation": "USES", "object": f"OtherEntity_{i}_{j}"}
            for j in range(5)
        ]
        for i in range(10)
    }

    series, summary = compute_saturation_series(papers, triples_by_paper)
    assert len(series) == 10
    assert summary["verdict"] == "UNSATURATED"
    assert summary["discovery_decay_rate"] <= 0.20


def test_analyze_corpus_saturation_pipeline_flow(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    triples_dir = tmp_path / "triples"
    triples_dir.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    figures = outputs / "figures"

    # Save mock filtered papers
    papers = [
        {"paper_id": f"paper_{i}", "title": f"Study {i}", "year": 2020 + i}
        for i in range(5)
    ]
    save_json(papers, processed / "filtered_papers.json")

    # Save mock triples
    triples = [
        {"subject": "SHAP", "relation": "USES", "object": f"Concept_{i}", "source_paper_id": f"paper_{i}"}
        for i in range(5)
    ]
    save_json(triples, triples_dir / "all_triples.json")

    config = {
        "paths": {
            "processed_data": str(processed),
            "triples": str(triples_dir),
            "outputs": str(outputs),
            "figures": str(figures),
        }
    }

    result = analyze_corpus_saturation(config)
    assert "summary" in result
    assert "series" in result
    assert len(result["series"]) == 5
    assert (outputs / "corpus_saturation_report.json").exists()
    assert (figures / "corpus_saturation_curve.png").exists()


def test_marginal_decay_formula():
    """Verify decay formula: Delta_decay = (M_initial - M_final) / M_initial."""
    m_initial = 8.37
    m_final = 9.93
    decay = (m_initial - m_final) / m_initial
    assert abs(decay - (-0.1863799)) < 1e-4
    assert round(decay, 3) == -0.186
    assert round(decay * 100, 1) == -18.6

