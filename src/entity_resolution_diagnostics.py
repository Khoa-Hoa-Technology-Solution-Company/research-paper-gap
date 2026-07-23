"""Audit legacy entity-resolution artifacts without rewriting historical data.

The historical mapping predates the forward resolver and has no decision-level
provenance. This utility reports only observable risks in that stored mapping:
alias-component size/cohesion under the current lexical scorer and surface
labels whose raw occurrences received multiple entity types. It never changes
the legacy mapping and does not claim merge correctness.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src import config
from src.entity_resolution import _lexical_similarity, _normalise_label


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_legacy_resolution_diagnostic(
    raw_triples: list[dict[str, Any]], mapping: dict[str, list[str]]
) -> dict[str, Any]:
    """Summarize observable cohesion and type-fragmentation risks."""
    types_by_surface: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    for triple in raw_triples:
        for label_field, type_field in (("subject", "subject_type"), ("object", "object_type")):
            label = _normalise_label(triple.get(label_field, ""))
            entity_type = str(triple.get(type_field, "")).strip().upper()
            if label and entity_type:
                types_by_surface[label].add(entity_type)
                occurrences[label] += 1

    conflicted = sorted(label for label, types in types_by_surface.items() if len(types) > 1)
    type_pairs: Counter[tuple[str, str]] = Counter()
    for label in conflicted:
        types = sorted(types_by_surface[label])
        for index, left in enumerate(types):
            for right in types[index + 1:]:
                type_pairs[(left, right)] += 1

    components = []
    scorer_backend = "not_used"
    for canonical, aliases in sorted(mapping.items(), key=lambda item: _normalise_label(item[0])):
        labels = [canonical, *aliases]
        normalised = sorted({_normalise_label(label) for label in labels if _normalise_label(label)})
        pair_scores: list[float] = []
        canonical_scores: list[float] = []
        for index, left in enumerate(normalised):
            for right in normalised[index + 1:]:
                score, scorer_backend = _lexical_similarity(left, right)
                pair_scores.append(score)
        canonical_normalised = _normalise_label(canonical)
        for alias in normalised:
            if alias == canonical_normalised:
                continue
            score, scorer_backend = _lexical_similarity(canonical_normalised, alias)
            canonical_scores.append(score)
        components.append({
            "canonical": canonical,
            "typed_labels_unavailable_in_legacy_mapping": True,
            "surface_label_count": len(normalised),
            "alias_count": max(0, len(normalised) - 1),
            "minimum_pairwise_lexical_similarity": round(min(pair_scores), 4) if pair_scores else None,
            "minimum_canonical_alias_lexical_similarity": round(min(canonical_scores), 4) if canonical_scores else None,
        })

    low_cohesion = [
        row for row in components
        if row["minimum_canonical_alias_lexical_similarity"] is not None
        and row["minimum_canonical_alias_lexical_similarity"] < config.FUZZY_MATCH_THRESHOLD
    ]
    return {
        "schema_version": "legacy-entity-resolution-diagnostic-v1",
        "scope": (
            "Retrospective diagnostic of a stored legacy mapping. It does not replay "
            "the unknown historical resolver, validate merges, or alter historical triples."
        ),
        "lexical_cohesion_reference": {
            "threshold": config.FUZZY_MATCH_THRESHOLD,
            "backend": scorer_backend,
            "interpretation": (
                "Scores are a current lexical audit reference only; the historical mapping's "
                "decision rule and semantic model are unrecoverable."
            ),
        },
        "legacy_mapping_components": {
            "component_count": len(components),
            "largest_surface_label_count": max((row["surface_label_count"] for row in components), default=0),
            "components_with_more_than_two_aliases": sum(row["alias_count"] > 2 for row in components),
            "components_below_current_canonical_lexical_threshold": len(low_cohesion),
            "lowest_cohesion_examples": sorted(
                low_cohesion,
                key=lambda row: (row["minimum_canonical_alias_lexical_similarity"], _normalise_label(row["canonical"])),
            )[:10],
        },
        "raw_type_conflicts": {
            "surface_labels_with_multiple_types": len(conflicted),
            "endpoint_occurrences_under_type_conflict": sum(occurrences[label] for label in conflicted),
            "top_type_conflict_pairs": [
                {"types": list(pair), "surface_labels": count}
                for pair, count in type_pairs.most_common(10)
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a stored legacy entity-resolution mapping.")
    parser.add_argument("--raw", type=Path, default=Path(config.TRIPLES_DIR) / "raw_triples.json")
    parser.add_argument("--mapping", type=Path, default=Path(config.TRIPLES_DIR) / "entity_mapping.json")
    parser.add_argument(
        "--output", type=Path, default=Path(config.DATA_DIR) / "entity_resolution_legacy_diagnostic.json"
    )
    args = parser.parse_args()
    report = build_legacy_resolution_diagnostic(_load(args.raw), _load(args.mapping))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Saved legacy entity-resolution diagnostic to {args.output}")


if __name__ == "__main__":
    main()
