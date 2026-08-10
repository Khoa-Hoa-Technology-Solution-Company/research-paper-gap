"""Run detector-plus-validation diagnostics on an existing resolved-triple corpus.

This adapter accepts both the compact triple schema used by the original
engine and the provenance-rich schema used by later experimental snapshots.
It uses a deterministic common-neighbour missing-link generator so all three
signal families are exercised on CPU.  This is not presented as a substitute
for a fully tuned embedding model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detect_gaps import detect_orphan_clusters, detect_temporal_decay
from src.entity_normalization import canonical_entity_key, canonical_entity_label
from src.validate_gaps import validate_candidate


def _entity(triple, role):
    value = triple.get(role)
    if isinstance(value, dict):
        return value.get("name", ""), value.get("type", "UNKNOWN")
    return str(value or ""), triple.get(f"{role}_type", "UNKNOWN")


def build_graph(triples):
    graph = nx.MultiDiGraph()
    labels_by_key = {}

    def canonical(value):
        label = canonical_entity_label(value)
        key = canonical_entity_key(label)
        if not key:
            return ""
        labels_by_key.setdefault(key, label)
        return labels_by_key[key]

    for triple in triples:
        head, head_type = _entity(triple, "subject")
        tail, tail_type = _entity(triple, "object")
        head, tail = canonical(head), canonical(tail)
        if not head or not tail or canonical_entity_key(head) == canonical_entity_key(tail):
            continue
        paper = triple.get("paper_id") or triple.get("source_paper_id") or ""
        for node, node_type in ((head, head_type), (tail, tail_type)):
            if not graph.has_node(node):
                graph.add_node(node, type=node_type, papers=set())
            if paper:
                graph.nodes[node]["papers"].add(str(paper))
        graph.add_edge(
            head,
            tail,
            relation=triple.get("relation", "RELATED"),
            confidence=float(triple.get("confidence", 0.5)),
            source_paper=str(paper),
            year=triple.get("year") or triple.get("source_year"),
            evidence=triple.get("evidence_quote") or triple.get("evidence", ""),
        )
    for node in graph:
        graph.nodes[node]["papers"] = sorted(graph.nodes[node]["papers"])
    centrality = nx.degree_centrality(nx.DiGraph(graph))
    nx.set_node_attributes(graph, centrality, "degree_centrality")
    return graph


def settings():
    return {
        "gap_detection": {
            "orphan": {
                "min_cluster_ratio": 0.05,
                "max_inter_cluster_edge_ratio": 0.10,
                "random_seed": 42,
            },
            "temporal": {
                "decay_threshold": 0.30,
                "lookback_years": 2,
                "snapshot_date": "2026-07-20",
                "exclude_incomplete_final_year": True,
            },
        },
        "gap_validation": {
            "min_supporting_papers": 2,
            "min_independent_paths": 2,
            "min_surviving_paths": 1,
            "max_path_length": 4,
            "min_specificity": 0.55,
            "bootstrap_repeats": 50,
            "edge_keep_probability": 0.95,
            "paper_keep_probability": 0.95,
            "plausible_edge_add_probability": 0.50,
            "min_stability": 0.70,
            "orphan_isolation_threshold": 0.90,
            "temporal_decay_threshold": 0.30,
            "temporal_lookback_years": 2,
            "closure_token_coverage": 0.60,
            "snapshot_date": "2026-07-20",
            "exclude_incomplete_final_year": True,
            "random_seed": 42,
            "max_closure_hits_to_record": 10,
            "weights": {
                "provenance": 0.25,
                "specificity": 0.15,
                "path_diversity": 0.20,
                "stability": 0.30,
                "closure_clearance": 0.10,
            },
        },
    }


def detect_structural_missing_links(graph, top_k=50):
    """Generate deterministic two-hop candidates using common neighbours."""
    simple = nx.Graph(graph)
    scores = {}
    for middle in sorted(simple.nodes(), key=str):
        neighbours = sorted(simple.neighbors(middle), key=str)
        for i, head in enumerate(neighbours):
            for tail in neighbours[i + 1:]:
                if head == tail or simple.has_edge(head, tail):
                    continue
                pair = tuple(sorted((str(head), str(tail))))
                scores[pair] = scores.get(pair, 0) + 1
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        {
            "type": "missing_link",
            "head": head,
            "tail": tail,
            "relation": "RELATED",
            "prediction_score": float(score),
            "generator": "deterministic_common_neighbour",
        }
        for (head, tail), score in ranked
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="outputs/corpus_validation.json")
    args = parser.parse_args()

    triples = json.loads(Path(args.triples).read_text(encoding="utf-8"))
    documents = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    graph = build_graph(triples)
    cfg = settings()
    publication_counts = {}
    for document in documents:
        try:
            year = int(document.get("year") or document.get("publication_year"))
        except (TypeError, ValueError):
            continue
        publication_counts[year] = publication_counts.get(year, 0) + 1
    cfg["gap_detection"]["temporal"]["publication_counts"] = publication_counts
    candidates = (
        detect_structural_missing_links(graph)
        + detect_orphan_clusters(graph, cfg)
        + detect_temporal_decay(graph, cfg)
    )
    audited = [
        {**candidate, "validation": validate_candidate(graph, candidate, cfg, documents)}
        for candidate in candidates
    ]
    status_counts = {
        status: sum(item["validation"]["status"] == status for item in audited)
        for status in ("automatically_eligible", "review_required", "rejected")
    }
    report = {
        "graph": {
            "papers": len({paper for _, data in graph.nodes(data=True) for paper in data.get("papers", [])}),
            "triples": len(triples),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "components": nx.number_weakly_connected_components(graph),
        },
        "raw_candidates": len(candidates),
        "raw_missing_links": sum(item["type"] == "missing_link" for item in candidates),
        "raw_orphan_clusters": sum(item["type"] == "orphan_cluster" for item in candidates),
        "raw_temporal_decay": sum(item["type"] == "temporal_decay" for item in candidates),
        "validation_status": status_counts,
        "scope": (
            "Retrospective diagnostic on an existing abstract-only corpus. "
            "Automatically eligible means the evidence contract passed, not that novelty or importance was established."
        ),
        "candidates": audited,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
