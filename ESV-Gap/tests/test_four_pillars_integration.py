"""Integration test for the complete Four Evidence Pillars framework (A + B + C + D)."""

from __future__ import annotations

import pickle
from pathlib import Path
import networkx as nx
import pytest

from src.author_stated_gaps import mine_author_stated_gaps
from src.corpus_saturation import analyze_corpus_saturation
from src.external_verification import verify_all_candidates
from src.extractor_recall import evaluate_extractor_recall
from src.synthesize_rankings import synthesize_final_rankings
from src.utils import load_json, save_json


def test_four_pillars_full_pipeline_flow(tmp_path: Path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    figures = outputs / "figures"
    figures.mkdir()
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    triples_dir = tmp_path / "triples"
    triples_dir.mkdir()
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()

    # 1. Setup Mock Knowledge Graph with LACKS edge for Pillar B
    G = nx.MultiDiGraph()
    G.add_node("SHAP", type="METHOD")
    G.add_node("fairness", type="CONCEPT")
    G.add_node("counterfactual explanations", type="METHOD")
    G.add_edge("SHAP", "fairness", relation_type="LACKS", context="SHAP lacks fairness guarantees across demographic groups", source_papers=["p_external"])
    with open(graph_dir / "knowledge_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    # 2. Setup Screened Corpus Papers
    corpus_papers = [
        {
            "paper_id": "p1",
            "title": "A Unified Approach to Interpreting Model Predictions",
            "year": 2020,
            "abstract": "We introduce SHAP. While SHAP addresses local fidelity, it remains unexplored whether fairness constraints hold.",
        },
        {
            "paper_id": "p2",
            "title": "Fairness and Interpretability in Machine Learning",
            "year": 2021,
            "abstract": "We evaluate fairness. Few studies have examined SHAP with fairness in high-stakes domains.",
        },
    ]
    save_json(corpus_papers, processed / "filtered_papers.json")

    # 3. Setup Triples for Extractor Recall (C) and Saturation (D)
    triples = [
        {"subject": "SHAP", "relation": "ADDRESSES", "object": "local fidelity", "source_paper_id": "p1"},
        {"subject": "SHAP", "relation": "LACKS", "object": "fairness", "source_paper_id": "p2"},
    ]
    save_json(triples, triples_dir / "all_triples.json")

    # 4. Setup Gold Benchmark for Extractor Recall (C)
    gold_benchmark = {
        "papers": [
            {
                "paper_id": "p1",
                "title": "A Unified Approach to Interpreting Model Predictions",
                "gold_triples": [
                    {"subject": "SHAP", "relation": "ADDRESSES", "object": "local fidelity"},
                    {"subject": "SHAP", "relation": "LACKS", "object": "computational speed"},
                ],
            }
        ]
    }
    gold_bench_path = benchmarks_dir / "gold_triples.json"
    save_json(gold_benchmark, gold_bench_path)

    # 5. Setup Validation Outputs (candidates surviving the gate)
    candidates = {
        "missing_link": [
            {
                "type": "missing_link",
                "head": "SHAP",
                "tail": "fairness",
                "concept": None,
                "community_id": None,
                "corpus_coverage_fraction": 0.50,
                "mean_citation_count": 120,
                "publication_recency_score": 0.90,
            }
        ]
    }
    save_json(candidates, outputs / "evidence_clear_candidates.json")
    save_json({}, outputs / "review_required_gaps.json")

    # Pipeline Config
    config = {
        "paths": {
            "outputs": str(outputs),
            "figures": str(figures),
            "graph": str(graph_dir),
            "processed_data": str(processed),
            "triples": str(triples_dir),
        },
        "external_verification": {
            "enabled": True,
            "sources": ["mock_source"],
            "max_queries_per_candidate": 2,
            "co_mention_token_coverage": 0.60,
        },
        "author_stated_gaps": {
            "enabled": True,
            "co_mention_token_coverage": 0.60,
        },
        "extractor_recall": {
            "enabled": True,
            "benchmark_path": str(gold_bench_path),
            "entity_match_threshold": 0.60,
            "strict_relation": True,
        },
        "corpus_saturation": {
            "enabled": True,
        },
        "synthesis": {
            "rank_by": "confidence_score",
        },
    }

    # Run Pillar A: External Verification with Mock Searcher (simulates 0 hits -> absence corroborated)
    mock_searchers = {
        "mock_source": lambda q, mode="search": {"raw_count": 0, "records": []}
    }
    res_a = verify_all_candidates(config, searchers=mock_searchers)
    assert res_a["enabled"] is True
    assert (outputs / "external_verification.json").exists()

    # Run Pillar B: Author-Stated Gaps
    res_b = mine_author_stated_gaps(config, graph=G, corpus_documents=corpus_papers)
    assert res_b["enabled"] is True
    assert (outputs / "author_stated_gaps.json").exists()

    # Run Pillar C: Extractor Recall
    res_c = evaluate_extractor_recall(config, benchmark_path=gold_bench_path, extracted_triples=triples)
    assert res_c["summary"]["recall"] == 0.5  # 1 of 2 gold triples found
    assert res_c["summary"]["miss_rate"] == 0.5
    assert (outputs / "extractor_recall_report.json").exists()

    # Run Pillar D: Corpus Saturation
    res_d = analyze_corpus_saturation(config)
    assert len(res_d["series"]) == 2
    assert (outputs / "corpus_saturation_report.json").exists()
    assert (figures / "corpus_saturation_curve.png").exists()

    # Run Synthesis: Combines all signals
    res_syn = synthesize_final_rankings(config)
    assert res_syn["total"] == 1
    assert "evidence_pillars" in res_syn
    pillars = res_syn["evidence_pillars"]
    assert "A_external_verification" in pillars
    assert "B_author_stated_gaps" in pillars
    assert "C_extractor_recall" in pillars
    assert "D_corpus_saturation" in pillars

    # Verify final rankings file
    final_rankings = load_json(outputs / "final_rankings.json")
    assert len(final_rankings["approved"]) == 1 or len(final_rankings["review_required"]) == 1
