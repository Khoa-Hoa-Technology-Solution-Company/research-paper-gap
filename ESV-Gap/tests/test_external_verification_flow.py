"""End-to-end test for external verification + author corroboration + synthesis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.author_stated_gaps import mine_author_stated_gaps
from src.external_verification import verify_against_external_indices
from src.synthesize_rankings import synthesize_final_rankings


@pytest.fixture
def mock_config(tmp_path: Path) -> dict:
    """Minimal config pointing to temp directories."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    return {
        "paths": {"outputs": str(outputs), "graph": str(graph_dir), "processed_data": str(processed)},
        "external_verification": {
            "enabled": True,
            "sources": ["openalex"],
            "openalex_contact_email": "",
            "request_timeout_seconds": 5,
            "max_retries": 1,
            "max_records_per_probe": 2,
            "max_queries_per_candidate": 2,
            "co_mention_token_coverage": 0.6,
            "temporal_decay_threshold": 0.3,
            "temporal_lookback_years": 2,
            "orphan_probe_pairs": 2,
        },
        "author_stated_gaps": {
            "enabled": True,
            "co_mention_token_coverage": 0.6,
        },
        "synthesis": {"rank_by": "confidence_score"},
    }


@pytest.fixture
def mock_validation_outputs(mock_config: dict) -> None:
    """Create fake validation outputs for the pipeline to consume."""
    import pickle
    import networkx as nx
    from src.utils import save_json

    outputs = Path(mock_config["paths"]["outputs"])
    graph_dir = Path(mock_config["paths"]["graph"])

    # Create minimal graph
    G = nx.MultiDiGraph()
    G.add_node("SHAP", type="concept")
    G.add_node("fairness", type="concept")
    G.add_node("counterfactual explanations", type="concept")
    G.add_edge("SHAP", "fairness", type="LACKS")
    with open(graph_dir / "knowledge_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    # Evidence-clear candidates
    eligible = {
        "missing_link": [
            {
                "type": "missing_link",
                "head": "SHAP",
                "tail": "fairness",
                "concept": None,
                "community_id": None,
                "corpus_coverage_fraction": 0.20,
                "mean_citation_count": 80,
                "publication_recency_score": 0.85,
            }
        ],
        "orphan_cluster": [
            {
                "type": "orphan_cluster",
                "head": None,
                "tail": None,
                "concept": "counterfactual explanations",
                "key_concepts": ["counterfactual explanations"],
                "community_id": 5,
                "corpus_coverage_fraction": 0.12,
                "mean_citation_count": 30,
                "publication_recency_score": 0.70,
            }
        ],
    }
    save_json(eligible, outputs / "evidence_clear_candidates.json")

    # Review-required (none for simplicity)
    save_json({}, outputs / "review_required_gaps.json")

    # Screened corpus for author mining
    corpus = [
        {
            "paperId": "P1",
            "title": "SHAP for Model Interpretability",
            "abstract": "We lack systematic methods to evaluate fairness using SHAP values.",
            "year": 2023,
        },
        {
            "paperId": "P2",
            "title": "Counterfactual Explanations Survey",
            "abstract": "Despite progress, counterfactual explanations remain under-explored in production systems.",
            "year": 2024,
        },
    ]
    save_json(corpus, outputs.parent / "processed" / "filtered_corpus.json")


def test_external_verification_stage(mock_config: dict, mock_validation_outputs: None) -> None:
    """Stage A should produce external_verification.json."""
    result = verify_against_external_indices(mock_config)
    assert result["verified"] == 2
    assert "candidates" not in result  # summary only

    outputs = Path(mock_config["paths"]["outputs"])
    assert (outputs / "external_verification.json").exists()


def test_author_stated_gaps_stage(mock_config: dict, mock_validation_outputs: None) -> None:
    """Stage B should produce author_stated_gaps.json."""
    result = mine_author_stated_gaps(mock_config)
    assert result["verified"] == 2
    assert "candidates" not in result  # summary only

    outputs = Path(mock_config["paths"]["outputs"])
    assert (outputs / "author_stated_gaps.json").exists()


def test_synthesis_stage(mock_config: dict, mock_validation_outputs: None) -> None:
    """Stage C should merge signals and produce final_rankings.json."""
    # Run A and B first
    verify_against_external_indices(mock_config)
    mine_author_stated_gaps(mock_config)

    # Run synthesis
    summary = synthesize_final_rankings(mock_config)
    assert summary["total"] == 2
    assert "approved" in summary
    assert "review_required" in summary
    assert "excluded" in summary

    outputs = Path(mock_config["paths"]["outputs"])
    rankings_path = outputs / "final_rankings.json"
    assert rankings_path.exists()

    from src.utils import load_json

    rankings = load_json(rankings_path)
    assert "summary" in rankings
    assert "approved" in rankings
    assert "review_required" in rankings
    assert "excluded" in rankings

    # Check that each candidate has scores and disposition
    for cand in rankings["approved"] + rankings["review_required"] + rankings["excluded"]:
        assert "disposition" in cand
        assert "impact_score" in cand
        assert "novelty_score" in cand
        assert "confidence_score" in cand
        assert "rank" in cand
        assert "rank_group" in cand


def test_full_flow_integration(mock_config: dict, mock_validation_outputs: None) -> None:
    """Run A → B → C and verify final output structure."""
    verify_against_external_indices(mock_config)
    mine_author_stated_gaps(mock_config)
    summary = synthesize_final_rankings(mock_config)

    outputs = Path(mock_config["paths"]["outputs"])
    assert (outputs / "external_verification.json").exists()
    assert (outputs / "author_stated_gaps.json").exists()
    assert (outputs / "final_rankings.json").exists()

    assert summary["total"] == summary["approved"] + summary["review_required"] + summary["excluded"]
