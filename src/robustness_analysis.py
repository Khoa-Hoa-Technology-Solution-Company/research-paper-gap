"""Offline robustness analysis for the topology stage.

This module deliberately separates robustness checks from the main pipeline.
It does not alter the pilot graph, thresholds, or candidate outputs.  It
reports how much the Louvain partition and the orphan-cluster decision change
across random seeds, resolution values, and two transparent graph variants:

* ``unweighted``: one undirected edge per observed node pair;
* ``source_count_weighted``: edge weight is the number of uniquely recovered
  source papers supporting that node pair, with a conservative minimum of one
  for events whose source cannot be recovered from legacy artifacts.

The latter is a sensitivity variant, not a claim that all evidence has been
verified.  The historical triples did not retain source identifiers, so source
papers are recovered only when an evidence quote has one exact normalized
substring match in a retained chunk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from src import config


DEFAULT_RESOLUTIONS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
DEFAULT_SEEDS = tuple(range(100))


def _normalise_text(value: object) -> str:
    """Case- and whitespace-normalise a quote for conservative matching."""
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class ProvenanceMatch:
    """Recovered source information for one legacy raw triple."""

    status: str
    paper_id: str | None = None
    chunk_index: int | None = None


def recover_quote_provenance(raw_triples: list[dict], chunks: list[dict]) -> list[ProvenanceMatch]:
    """Recover a source only for a unique exact normalised quote match.

    No fuzzy matching is used: an unmatched or ambiguous quote is explicitly
    reported as unavailable rather than assigned to a plausible paper.
    """
    normalised_chunks = [_normalise_text(chunk.get("text", "")) for chunk in chunks]
    matches: list[ProvenanceMatch] = []
    for triple in raw_triples:
        quote = _normalise_text(triple.get("evidence_quote", ""))
        candidates = [
            index for index, text in enumerate(normalised_chunks)
            if quote and quote in text
        ]
        if len(candidates) == 1:
            chunk = chunks[candidates[0]]
            matches.append(ProvenanceMatch(
                status="unique_normalized_exact_quote_match",
                paper_id=str(chunk.get("paperId", "")) or None,
                chunk_index=int(chunk.get("chunk_index", 0)),
            ))
        elif not candidates:
            matches.append(ProvenanceMatch(status="unmatched_quote"))
        else:
            matches.append(ProvenanceMatch(status="ambiguous_quote"))
    return matches


def _indices_align(raw_triples: list[dict], resolved_triples: list[dict]) -> bool:
    """Check whether source metadata can safely transfer by retained index."""
    return (
        len(raw_triples) == len(resolved_triples)
        and all(
            raw.get("evidence_quote", "") == resolved.get("evidence_quote", "")
            and raw.get("year") == resolved.get("year")
            for raw, resolved in zip(raw_triples, resolved_triples)
        )
    )


def build_projection(
    resolved_triples: list[dict],
    provenance: list[ProvenanceMatch],
    *,
    weighted: bool,
) -> nx.Graph:
    """Build a simple undirected evidence graph for a robustness variant."""
    if len(resolved_triples) != len(provenance):
        raise ValueError("Resolved triples and provenance records must align.")

    graph = nx.Graph()
    for triple, source in zip(resolved_triples, provenance):
        subject = triple["subject"]
        obj = triple["object"]
        graph.add_node(subject, type=triple.get("subject_type", "CONCEPT"))
        graph.add_node(obj, type=triple.get("object_type", "CONCEPT"))
        if subject == obj:
            continue
        if not graph.has_edge(subject, obj):
            graph.add_edge(
                subject,
                obj,
                extraction_event_count=0,
                provenance_supported_events=0,
                unavailable_provenance_events=0,
                source_paper_ids=set(),
            )
        edge = graph[subject][obj]
        edge["extraction_event_count"] += 1
        if source.paper_id:
            edge["source_paper_ids"].add(source.paper_id)
            edge["provenance_supported_events"] += 1
        else:
            edge["unavailable_provenance_events"] += 1

    for _, _, edge in graph.edges(data=True):
        unique_sources = len(edge["source_paper_ids"])
        # A legacy event with unavailable source cannot justify a zero-weight
        # edge.  Giving it one unit keeps the variant connected while avoiding
        # any invented multiplicity.
        edge["weight"] = float(max(1, unique_sources)) if weighted else 1.0
        edge["unique_source_paper_count"] = unique_sources
        edge["source_paper_ids"] = sorted(edge["source_paper_ids"])
    return graph


def _partition(graph: nx.Graph, seed: int, resolution: float, weight: str | None) -> dict[str, int]:
    try:
        import community as community_louvain
    except ImportError as error:  # pragma: no cover - requirements declare it
        raise RuntimeError("python-louvain is required for this robustness analysis.") from error
    return community_louvain.best_partition(
        graph,
        random_state=seed,
        resolution=resolution,
        weight=weight,
    )


def _communities(partition: dict[str, int]) -> list[set[str]]:
    by_id: dict[int, set[str]] = defaultdict(set)
    for node, community_id in partition.items():
        by_id[community_id].add(node)
    return list(by_id.values())


def adjusted_rand_index(left: dict[str, int], right: dict[str, int]) -> float:
    """Compute ARI without adding scikit-learn as a project dependency."""
    if set(left) != set(right):
        raise ValueError("Partitions must cover the same node set.")
    count = len(left)
    if count < 2:
        return 1.0
    contingency: Counter[tuple[int, int]] = Counter(
        (left[node], right[node]) for node in left
    )
    left_sizes = Counter(left.values())
    right_sizes = Counter(right.values())

    def choose_two(number: int) -> float:
        return number * (number - 1) / 2

    agree_pairs = sum(choose_two(value) for value in contingency.values())
    left_pairs = sum(choose_two(value) for value in left_sizes.values())
    right_pairs = sum(choose_two(value) for value in right_sizes.values())
    total_pairs = choose_two(count)
    expected = left_pairs * right_pairs / total_pairs
    maximum = (left_pairs + right_pairs) / 2
    if maximum == expected:
        return 1.0 if left == right else 0.0
    return (agree_pairs - expected) / (maximum - expected)


def _partition_fingerprint(partition: dict[str, int]) -> str:
    """Hash memberships (not arbitrary Louvain numeric labels)."""
    signature = "|".join(
        ",".join(sorted(group, key=str.casefold))
        for group in sorted(
            _communities(partition),
            key=lambda group: tuple(sorted(group, key=str.casefold)),
        )
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def _orphan_metrics(
    graph: nx.Graph,
    partition: dict[str, int],
    min_size_ratio: float,
    max_bridge_ratio: float,
    weight: str | None,
) -> list[dict]:
    """Apply the same cut-edge fraction definition to a graph variant."""
    total_nodes = graph.number_of_nodes()
    result = []
    for nodes in _communities(partition):
        internal = 0.0
        cross = 0.0
        for source, target, data in graph.edges(data=True):
            value = float(data.get(weight, 1.0)) if weight else 1.0
            if source in nodes and target in nodes:
                internal += value
            elif (source in nodes) != (target in nodes):
                cross += value
        ratio = cross / (internal + cross) if internal + cross else 0.0
        if len(nodes) / total_nodes >= min_size_ratio and ratio <= max_bridge_ratio:
            result.append({
                "node_count": len(nodes),
                "size_ratio": len(nodes) / total_nodes,
                "bridge_ratio": ratio,
                "internal_edge_mass": internal,
                "cross_edge_mass": cross,
            })
    return result


def _summary(values: Iterable[float]) -> dict[str, float]:
    numbers = list(values)
    if not numbers:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": round(min(numbers), 6),
        "median": round(statistics.median(numbers), 6),
        "max": round(max(numbers), 6),
    }


def _modularity(graph: nx.Graph, partition: dict[str, int], weight: str | None) -> float:
    return nx.algorithms.community.quality.modularity(
        graph,
        _communities(partition),
        weight=weight,
    )


def run_robustness_analysis(
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    resolutions: Iterable[float] = DEFAULT_RESOLUTIONS,
    output: str | None = None,
    append: bool = False,
) -> dict:
    """Run the pre-specified offline seed/resolution/weight robustness sweep."""
    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    raw_triples = _load_json(raw_path)
    resolved_triples = _load_json(resolved_path)
    chunks = _load_json(chunks_path)
    aligned = _indices_align(raw_triples, resolved_triples)
    if not aligned:
        raise RuntimeError(
            "Legacy raw/resolved triples are not index-aligned; refusing to transfer provenance."
        )
    provenance = recover_quote_provenance(raw_triples, chunks)
    provenance_counts = Counter(match.status for match in provenance)

    variants = {
        # python-louvain 0.16 expects a string attribute name rather than
        # ``None``. Every edge has unit weight in this variant, so passing the
        # common attribute still implements an unweighted partition.
        "unweighted": (build_projection(resolved_triples, provenance, weighted=False), "weight"),
        "source_count_weighted": (
            build_projection(resolved_triples, provenance, weighted=True), "weight"
        ),
    }
    seed_values = tuple(int(seed) for seed in seeds)
    resolution_values = tuple(float(value) for value in resolutions)
    if config.LOUVAIN_RANDOM_STATE not in seed_values:
        raise ValueError("The seed list must include the configured reference seed.")

    results: dict[str, dict[str, dict]] = {}
    for variant_name, (graph, weight) in variants.items():
        print(
            f"[*] {variant_name}: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} undirected edges"
        )
        variant_results: dict[str, dict] = {}
        for resolution in resolution_values:
            run_rows = []
            partitions: dict[int, dict[str, int]] = {}
            for seed in seed_values:
                partition = _partition(graph, seed, resolution, weight)
                partitions[seed] = partition
                orphan_clusters = _orphan_metrics(
                    graph,
                    partition,
                    config.LOUVAIN_MIN_SIZE_RATIO,
                    config.LOUVAIN_MAX_BRIDGE_RATIO,
                    weight,
                )
                run_rows.append({
                    "seed": seed,
                    "communities": len(_communities(partition)),
                    "modularity": round(_modularity(graph, partition, weight), 8),
                    "orphan_clusters": len(orphan_clusters),
                    "partition_fingerprint": _partition_fingerprint(partition),
                })
            reference = partitions[config.LOUVAIN_RANDOM_STATE]
            aris = [adjusted_rand_index(reference, partitions[seed]) for seed in seed_values]
            orphan_counts = [row["orphan_clusters"] for row in run_rows]
            variant_results[f"{resolution:.2f}"] = {
                "resolution": resolution,
                "runs": run_rows,
                "community_count": _summary(row["communities"] for row in run_rows),
                "modularity": _summary(row["modularity"] for row in run_rows),
                "orphan_count": _summary(orphan_counts),
                "zero_orphan_rate": round(
                    sum(value == 0 for value in orphan_counts) / len(orphan_counts), 6
                ),
                "ari_to_seed_42": _summary(aris),
                "distinct_partition_fingerprints": len({row["partition_fingerprint"] for row in run_rows}),
            }
            print(
                f"[+] {variant_name} r={resolution:.2f}: "
                f"zero-orphan={variant_results[f'{resolution:.2f}']['zero_orphan_rate']:.2%}, "
                f"ARI median={variant_results[f'{resolution:.2f}']['ari_to_seed_42']['median']:.3f}"
            )
        results[variant_name] = variant_results

    weighted_graph = variants["source_count_weighted"][0]
    weights = [data["weight"] for _, _, data in weighted_graph.edges(data=True)]
    report = {
        "scope": (
            "Offline partition and isolation robustness only. This does not evaluate "
            "TABI generation, novelty, or research-gap quality."
        ),
        "method": {
            "algorithm": "python-louvain.best_partition",
            "seeds": list(seed_values),
            "resolutions": list(resolution_values),
            "reference_seed_for_ari": config.LOUVAIN_RANDOM_STATE,
            "orphan_definition": (
                "community size ratio >= %.3f and cut-edge fraction <= %.3f"
                % (config.LOUVAIN_MIN_SIZE_RATIO, config.LOUVAIN_MAX_BRIDGE_RATIO)
            ),
            "unweighted_variant": "one undirected edge per observed node pair",
            "source_count_weighted_variant": (
                "weight=max(1, unique source paper IDs recovered by a unique "
                "case/whitespace-normalized exact evidence-quote match)"
            ),
        },
        "legacy_provenance_recovery": {
            "raw_resolved_index_alignment": aligned,
            "raw_triples": len(raw_triples),
            "unique_exact_quote_matches": provenance_counts[
                "unique_normalized_exact_quote_match"
            ],
            "unmatched_quotes": provenance_counts["unmatched_quote"],
            "ambiguous_quotes": provenance_counts["ambiguous_quote"],
            "matched_fraction": round(
                provenance_counts["unique_normalized_exact_quote_match"] / len(provenance), 6
            ),
            "model_version_recoverable_from_legacy_artifacts": False,
        },
        "weighted_edge_support": {
            "edges": weighted_graph.number_of_edges(),
            "min_weight": min(weights) if weights else 0,
            "median_weight": statistics.median(weights) if weights else 0,
            "max_weight": max(weights) if weights else 0,
            "edges_with_weight_gt_one": sum(weight > 1 for weight in weights),
        },
        "results": results,
    }
    destination = output or os.path.join(config.DATA_DIR, "louvain_robustness_results.json")
    if append and os.path.exists(destination):
        with open(destination, encoding="utf-8") as handle:
            existing = json.load(handle)
        # Preserve earlier resolution cells from the same deterministic setup.
        # New cells intentionally replace cells with the same variant/resolution.
        for variant_name, variant_result in results.items():
            existing.setdefault("results", {}).setdefault(variant_name, {}).update(variant_result)
        report = existing
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"[+] Saved Louvain robustness report to {destination}")
    return report


def _parse_seeds(value: str) -> tuple[int, ...]:
    """Parse ``0:99`` (inclusive) or comma-separated seeds."""
    if ":" in value:
        start, end = (int(part) for part in value.split(":", 1))
        if end < start:
            raise argparse.ArgumentTypeError("seed range end must be >= start")
        return tuple(range(start, end + 1))
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline Louvain robustness analyses.")
    parser.add_argument("--seeds", default="0:99", help="Inclusive range (e.g. 0:99) or comma list.")
    parser.add_argument(
        "--resolutions",
        default=",".join(str(value) for value in DEFAULT_RESOLUTIONS),
        help="Comma-separated Louvain resolution values.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge newly computed resolution cells into an existing output JSON.",
    )
    args = parser.parse_args()
    run_robustness_analysis(
        seeds=_parse_seeds(args.seeds),
        resolutions=tuple(float(item) for item in args.resolutions.split(",") if item.strip()),
        output=args.output,
        append=args.append,
    )


if __name__ == "__main__":
    main()
