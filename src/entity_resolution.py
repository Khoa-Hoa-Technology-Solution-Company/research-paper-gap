"""Deterministic, type-constrained entity resolution for forward KG-TABI runs.

The resolver has two ordered passes. It first proposes same-type lexical
matches and then proposes semantic matches between mean-embedding component
descriptors. Every proposed union must satisfy complete-link cohesion: every
cross-component label pair must reach the active threshold. Thus union--find
is used for deterministic closure without accepting an A--B--C chain whose
endpoints fail to match. It is deliberately not a substitute for the
outstanding human merge audit.

The historical pilot's resolved graph is retained as an immutable diagnostic.
This implementation and its manifest describe forward reruns only.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TypeAlias

from src import config


RESOLUTION_ALGORITHM_VERSION = "entity-resolution-v4-complete-link-component-semantic"
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
EntityKey: TypeAlias = tuple[str, str]  # (normalised label, normalised entity type)


def _normalise_label(value: object) -> str:
    """Case-fold and collapse whitespace without changing the displayed label."""
    return " ".join(str(value).strip().casefold().split())


def _normalise_type(value: object) -> str:
    return " ".join(str(value).strip().upper().split())


def _display_sort_key(value: str) -> tuple[int, str, str]:
    """Stable canonical-label tie-breaker: shortest, then lexical form."""
    stripped = value.strip()
    return (len(_normalise_label(stripped)), _normalise_label(stripped), stripped)


def _canonical_key_sort_key(key: EntityKey) -> tuple[int, str, str]:
    return (len(key[0]), key[0], key[1])


def _lexical_similarity(left: str, right: str) -> tuple[float, str]:
    """Return deterministic token-sort similarity and the backend used."""
    try:
        from rapidfuzz import fuzz

        return float(fuzz.token_sort_ratio(left, right)), "rapidfuzz.token_sort_ratio"
    except ImportError:
        # Keep the pass operational in a minimal environment.  The backend is
        # recorded in the manifest because its scale is not interchangeable
        # with rapidfuzz for a threshold-calibrated scientific rerun.
        return 100.0 * SequenceMatcher(None, left, right).ratio(), "difflib.SequenceMatcher-fallback"


def find_root(mapping: dict[Any, Any], name: Any) -> Any:
    """Find a union--find root with path compression (works for any hashable key)."""
    path: list[Any] = []
    current = name
    while mapping[current] != current:
        path.append(current)
        current = mapping[current]
    for item in path:
        mapping[item] = current
    return current


def _union(mapping: dict[EntityKey, EntityKey], left: EntityKey, right: EntityKey) -> bool:
    """Union components, selecting the documented stable canonical root."""
    left_root = find_root(mapping, left)
    right_root = find_root(mapping, right)
    if left_root == right_root:
        return False
    if _canonical_key_sort_key(left_root) <= _canonical_key_sort_key(right_root):
        mapping[right_root] = left_root
    else:
        mapping[left_root] = right_root
    return True


def _component_members(
    mapping: dict[EntityKey, EntityKey],
    entity_keys: list[EntityKey],
) -> dict[EntityKey, list[EntityKey]]:
    """Return deterministic component membership for the current union state."""
    members: dict[EntityKey, list[EntityKey]] = defaultdict(list)
    for key in entity_keys:
        members[find_root(mapping, key)].append(key)
    return members


def _complete_link_guarded_union(
    mapping: dict[EntityKey, EntityKey],
    left: EntityKey,
    right: EntityKey,
    entity_keys: list[EntityKey],
    similarity,
    threshold: float,
) -> tuple[bool, bool, float | None]:
    """Union only when every cross-component pair passes the active rule.

    This is a complete-link cohesion rule. Unlike representative-only checks,
    it rejects A--B--C whenever any member of the proposed merged component
    (for example A) fails the threshold against a member of the other side
    (for example C). The returned minimum is retained for auditability.
    """
    left_root = find_root(mapping, left)
    right_root = find_root(mapping, right)
    if left_root == right_root:
        return False, False, None
    members = _component_members(mapping, entity_keys)
    scores = [
        float(similarity(first, second))
        for first in members[left_root]
        for second in members[right_root]
    ]
    minimum = min(scores) if scores else None
    if minimum is None or minimum < threshold:
        return False, True, minimum
    return _union(mapping, left_root, right_root), False, minimum


def _validate_and_index_entities(
    triples: list[dict[str, Any]],
) -> tuple[dict[EntityKey, str], dict[str, set[str]], list[tuple[EntityKey, EntityKey]]]:
    """Build stable entity keys and validate the minimal triple schema."""
    display_candidates: dict[EntityKey, list[str]] = defaultdict(list)
    types_by_label: dict[str, set[str]] = defaultdict(set)
    endpoints: list[tuple[EntityKey, EntityKey]] = []
    required = ("subject", "subject_type", "object", "object_type")

    for index, triple in enumerate(triples):
        missing = [field for field in required if field not in triple]
        if missing:
            raise ValueError(f"Triple {index} is missing required entity fields: {', '.join(missing)}")
        subject_label = _normalise_label(triple["subject"])
        object_label = _normalise_label(triple["object"])
        subject_type = _normalise_type(triple["subject_type"])
        object_type = _normalise_type(triple["object_type"])
        if not subject_label or not object_label or not subject_type or not object_type:
            raise ValueError(f"Triple {index} has an empty entity label or entity type.")
        subject_key = (subject_label, subject_type)
        object_key = (object_label, object_type)
        display_candidates[subject_key].append(str(triple["subject"]).strip())
        display_candidates[object_key].append(str(triple["object"]).strip())
        types_by_label[subject_label].add(subject_type)
        types_by_label[object_label].add(object_type)
        endpoints.append((subject_key, object_key))

    # Selecting a representative independently from list insertion order is
    # important because source retrieval can arrive in a different order.
    display_by_key = {
        key: min(candidates, key=_display_sort_key)
        for key, candidates in display_candidates.items()
    }
    return display_by_key, types_by_label, endpoints


def _type_safe_display(
    key: EntityKey, display_by_key: dict[EntityKey, str], types_by_label: dict[str, set[str]]
) -> str:
    """Avoid graph-node collisions when one label is assigned multiple types."""
    display = display_by_key[key]
    return f"{display} [{key[1]}]" if len(types_by_label[key[0]]) > 1 else display


def _resolve_entities_with_audit(
    triples: list[dict[str, Any]],
    *,
    enable_lexical: bool = True,
    enable_semantic: bool = True,
    semantic_model_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    """Resolve entities and return output, alias groups, and a run manifest payload."""
    display_by_key, types_by_label, endpoints = _validate_and_index_entities(triples)
    entity_keys = sorted(display_by_key, key=_canonical_key_sort_key)
    mapping: dict[EntityKey, EntityKey] = {key: key for key in entity_keys}
    lexical_backend = "disabled"
    lexical_pairs_passing = 0
    semantic_pairs_passing = 0
    lexical_cohesion_rejections = 0
    semantic_cohesion_rejections = 0
    semantic_status = "disabled"
    semantic_error: str | None = None

    if enable_lexical:
        print("[*] Pass 1/2: deterministic lexical entity matching...")
        for left_index, left in enumerate(entity_keys):
            for right in entity_keys[left_index + 1:]:
                # Same normalised type is a hard constraint in both passes.
                if left[1] != right[1]:
                    continue
                score, backend = _lexical_similarity(left[0], right[0])
                lexical_backend = backend
                if score >= config.FUZZY_MATCH_THRESHOLD:
                    lexical_pairs_passing += 1
                    left_root = find_root(mapping, left)
                    right_root = find_root(mapping, right)
                    if left_root == right_root:
                        continue
                    _, rejected, _ = _complete_link_guarded_union(
                        mapping,
                        left_root,
                        right_root,
                        entity_keys,
                        lambda first, second: _lexical_similarity(first[0], second[0])[0],
                        config.FUZZY_MATCH_THRESHOLD,
                    )
                    if rejected:
                        lexical_cohesion_rejections += 1

    if enable_semantic:
        semantic_model = semantic_model_name or getattr(
            config, "ENTITY_RESOLUTION_MODEL", DEFAULT_SEMANTIC_MODEL
        )
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            print("[*] Pass 2/2: semantic matching of post-lexical component descriptors...")
            # Every typed label is embedded. A component descriptor is the
            # normalized mean of all current member embeddings, so a short
            # canonical label does not discard informative aliases.
            canonical_keys = sorted(
                {find_root(mapping, key) for key in entity_keys}, key=_canonical_key_sort_key
            )
            if len(canonical_keys) > 1:
                model = SentenceTransformer(semantic_model)
                labels = [display_by_key[key] for key in entity_keys]
                embeddings = model.encode(labels, show_progress_bar=False)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # A zero vector cannot establish semantic equivalence.
                safe_norms = np.where(norms == 0, 1.0, norms)
                normalised = embeddings / safe_norms
                embedding_by_key = {
                    key: normalised[index] for index, key in enumerate(entity_keys)
                }

                current_members = _component_members(mapping, entity_keys)
                descriptor_cache = {}

                def get_descriptor(root: EntityKey):
                    r = find_root(mapping, root)
                    if r in descriptor_cache:
                        return descriptor_cache[r]
                    members = current_members[r]
                    vector = np.mean([embedding_by_key[key] for key in members], axis=0)
                    norm = float(np.linalg.norm(vector))
                    desc = vector / norm if norm else vector
                    descriptor_cache[r] = desc
                    return desc

                def semantic_similarity(left: EntityKey, right: EntityKey) -> float:
                    return float(np.dot(embedding_by_key[left], embedding_by_key[right]))

                for left_index, left in enumerate(canonical_keys):
                    for right_index in range(left_index + 1, len(canonical_keys)):
                        right = canonical_keys[right_index]
                        left_root = find_root(mapping, left)
                        right_root = find_root(mapping, right)
                        if left_root == right_root or left_root[1] != right_root[1]:
                            continue
                        if float(np.dot(get_descriptor(left_root), get_descriptor(right_root))) >= config.COSINE_SIMILARITY_THRESHOLD:
                            semantic_pairs_passing += 1
                            _, rejected, _ = _complete_link_guarded_union(
                                mapping,
                                left_root,
                                right_root,
                                entity_keys,
                                semantic_similarity,
                                config.COSINE_SIMILARITY_THRESHOLD,
                            )
                            if not rejected:
                                new_root = find_root(mapping, left_root)
                                old_root = right_root if new_root == left_root else left_root
                                current_members[new_root].extend(current_members[old_root])
                                current_members[old_root] = []
                                if old_root in descriptor_cache:
                                    del descriptor_cache[old_root]
                                if new_root in descriptor_cache:
                                    del descriptor_cache[new_root]
                            else:
                                semantic_cohesion_rejections += 1
            semantic_status = "completed"
        except ImportError:
            semantic_status = "skipped_missing_sentence_transformers"
            print("[!] sentence-transformers not installed. Semantic pass skipped.")
        except Exception as error:  # model availability is recorded, never hidden
            semantic_status = "skipped_error"
            semantic_error = f"{type(error).__name__}: {error}"
            print(f"[!] Semantic pass skipped: {semantic_error}")

    final_key_mapping = {key: find_root(mapping, key) for key in entity_keys}
    display_for_key = {
        key: _type_safe_display(key, display_by_key, types_by_label) for key in entity_keys
    }
    resolved_triples: list[dict[str, Any]] = []
    removed_self_loops = 0
    for triple, (subject_key, object_key) in zip(triples, endpoints):
        canonical_subject = final_key_mapping[subject_key]
        canonical_object = final_key_mapping[object_key]
        if canonical_subject == canonical_object:
            removed_self_loops += 1
            continue
        # Preserve all source/provenance fields so that newly added extraction
        # metadata survives resolution without a brittle allow-list.
        resolved = dict(triple)
        resolved["subject"] = display_for_key[canonical_subject]
        resolved["subject_type"] = canonical_subject[1]
        resolved["object"] = display_for_key[canonical_object]
        resolved["object_type"] = canonical_object[1]
        resolved_triples.append(resolved)

    grouped_aliases: dict[str, list[str]] = defaultdict(list)
    for key in entity_keys:
        canonical = final_key_mapping[key]
        if key != canonical:
            grouped_aliases[display_for_key[canonical]].append(display_for_key[key])
    entity_mapping = {
        canonical: sorted(set(aliases), key=lambda label: (_normalise_label(label), label))
        for canonical, aliases in sorted(
            grouped_aliases.items(), key=lambda item: (_normalise_label(item[0]), item[0])
        )
    }
    type_conflicts = sorted(
        label for label, entity_types in types_by_label.items() if len(entity_types) > 1
    )
    endpoint_occurrences_by_label: dict[str, int] = defaultdict(int)
    type_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for triple in triples:
        for label_field in ("subject", "object"):
            endpoint_occurrences_by_label[_normalise_label(triple[label_field])] += 1
    for label in type_conflicts:
        entity_types = sorted(types_by_label[label])
        for index, left_type in enumerate(entity_types):
            for right_type in entity_types[index + 1:]:
                type_pair_counts[(left_type, right_type)] += 1
    component_sizes: dict[EntityKey, int] = defaultdict(int)
    for root in final_key_mapping.values():
        component_sizes[root] += 1
    component_size_values = list(component_sizes.values())
    audit = {
        "schema_version": "entity-resolution-run-manifest-v2",
        "algorithm_version": RESOLUTION_ALGORITHM_VERSION,
        "scope": (
            "Forward-run entity resolution only. It does not retroactively make the "
            "historical pilot end-to-end reproducible or validate merge correctness."
        ),
        "decision_rule": {
            "operator": "lexical OR semantic threshold pass, subject to same-type constraint",
            "pass_order": ["lexical", "semantic"],
            "transitive_closure": (
                "union-find over accepted same-type pairs, with deterministic "
                "complete-link cohesion validation before every union"
            ),
            "cohesion_guard": (
                "every cross-component typed-label pair must pass the active lexical "
                "or semantic threshold; this blocks A--B--C chaining when A and C "
                "are not sufficiently similar"
            ),
            "semantic_component_descriptor": (
                "L2-normalized mean of all current typed-label embeddings; proposed "
                "descriptor matches still require complete-link member validation"
            ),
            "canonical_selection": "shortest normalised label, then lexical label, then type",
            "type_conflict_policy": (
                "never merge across types; type-qualify output node labels when one "
                "normalised label occurs with multiple entity types"
            ),
        },
        "parameters": {
            "fuzzy_match_threshold": config.FUZZY_MATCH_THRESHOLD,
            "cosine_similarity_threshold": config.COSINE_SIMILARITY_THRESHOLD,
            "semantic_model": semantic_model_name
            or getattr(config, "ENTITY_RESOLUTION_MODEL", DEFAULT_SEMANTIC_MODEL),
        },
        "execution": {
            "lexical_enabled": enable_lexical,
            "lexical_backend": lexical_backend,
            "semantic_enabled": enable_semantic,
            "semantic_status": semantic_status,
            "semantic_error": semantic_error,
        },
        "counts": {
            "input_triples": len(triples),
            "unique_typed_entities_before": len(entity_keys),
            "unique_typed_entities_after": len(set(final_key_mapping.values())),
            "entities_merged": len(entity_keys) - len(set(final_key_mapping.values())),
            "lexical_pairs_passing_threshold": lexical_pairs_passing,
            "semantic_component_descriptor_pairs_passing_threshold": semantic_pairs_passing,
            "lexical_cohesion_rejections": lexical_cohesion_rejections,
            "semantic_cohesion_rejections": semantic_cohesion_rejections,
            "same-label_type-conflicts": len(type_conflicts),
            "self_loops_removed_after_resolution": removed_self_loops,
            "output_triples": len(resolved_triples),
        },
        "type_conflict_labels": type_conflicts,
        "type_conflict_summary": {
            "surface_labels_with_multiple_types": len(type_conflicts),
            "endpoint_occurrences_under_type_conflict": sum(
                endpoint_occurrences_by_label[label] for label in type_conflicts
            ),
            "top_type_conflict_pairs": [
                {"types": list(pair), "surface_labels": count}
                for pair, count in sorted(
                    type_pair_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
        "component_summary": {
            "largest_component_typed_labels": max(component_size_values, default=0),
            "components_with_more_than_two_typed_labels": sum(
                size > 2 for size in component_size_values
            ),
            "cohesion_rejections": lexical_cohesion_rejections + semantic_cohesion_rejections,
        },
    }
    print(
        "[+] Entity resolution complete. "
        f"Merged {audit['counts']['entities_merged']} typed entities; "
        f"removed {removed_self_loops} self-loops."
    )
    return resolved_triples, entity_mapping, audit


def resolve_entities(
    triples: list[dict[str, Any]],
    *,
    enable_lexical: bool = True,
    enable_semantic: bool = True,
    semantic_model_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Backward-compatible public resolver returning triples and alias groups."""
    resolved, entity_mapping, _ = _resolve_entities_with_audit(
        triples,
        enable_lexical=enable_lexical,
        enable_semantic=enable_semantic,
        semantic_model_name=semantic_model_name,
    )
    return resolved, entity_mapping


def run_resolution() -> None:
    raw_path = Path(config.TRIPLES_DIR) / "raw_triples.json"
    if not raw_path.exists():
        print(f"[!] File {raw_path} not found. Please run extract_triples first.")
        return
    raw_triples = json.loads(raw_path.read_text(encoding="utf-8"))
    resolved_triples, entity_mapping, manifest = _resolve_entities_with_audit(raw_triples)

    triples_dir = Path(config.TRIPLES_DIR)
    (triples_dir / "entity_mapping.json").write_text(
        json.dumps(entity_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (triples_dir / "resolved_triples.json").write_text(
        json.dumps(resolved_triples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = triples_dir / "entity_resolution_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Saved resolved triples to {triples_dir / 'resolved_triples.json'}")
    print(f"[+] Saved mapping rules to {triples_dir / 'entity_mapping.json'}")
    print(f"[+] Saved resolution manifest to {manifest_path}")


if __name__ == "__main__":
    run_resolution()
