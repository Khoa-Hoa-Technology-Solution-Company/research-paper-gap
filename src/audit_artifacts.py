"""Generate reproducible, non-judgmental screening and KG audit artifacts.

The pilot artifacts predate full per-triple provenance.  This tool never
rewrites them and never fabricates an annotation.  It creates separate audit
packets plus machine-readable reports that distinguish what can be recovered
from legacy files from what requires new human labels or a fresh extraction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import random
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Iterable

from src import config
from src.fetch_papers import screen_relevance_lexical
from src.robustness_analysis import _indices_align, recover_quote_provenance


AUDIT_SEED = 42
LEXICAL_RULE_VERSION = "microservices-security-lexical-v1 (source-hash recorded)"
ARCHITECTURE_TERMS = ("microservice", "micro-service", "service mesh", "cloud-native")
SECURITY_TERMS = (
    "security", "secure", "authentication", "authorization", "access control",
    "zero trust", "vulnerability", "attack", "threat", "privacy", "encryption",
    "identity", "policy", "oauth", "mTLS", "mtls",
)


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)


def _csv_value(value: object) -> object:
    """Avoid spreadsheet formula execution when a CSV is opened interactively."""
    if value is None:
        return ""
    text = str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@")) else text


def _write_csv(path: str, rows: list[dict], fieldnames: Iterable[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames})


def _lexical_features(paper: dict) -> tuple[list[str], list[str]]:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    architecture = [term for term in ARCHITECTURE_TERMS if term in text]
    security = [term for term in SECURITY_TERMS if term.lower() in text]
    return architecture, security


def create_screening_audit(
    papers: list[dict], stored_screened: list[dict], audit_dir: str, seed: int
) -> dict:
    """Create blinded labels and a reproducibility comparison for screening."""
    # Execute the exact currently checked-in deterministic function, retaining
    # all records rather than applying a study-size cap.
    recomputed = screen_relevance_lexical(deepcopy(papers), max_papers=len(papers))
    stored_ids = {str(item.get("paperId")) for item in stored_screened}
    lexical_ids = {str(item.get("paperId")) for item in recomputed}
    all_ids = {str(item.get("paperId")) for item in papers}
    if len(all_ids) != len(papers):
        raise RuntimeError("Cannot produce a reliable screening audit with duplicate/missing paper IDs.")

    source_text = inspect.getsource(screen_relevance_lexical)
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    stored_only = stored_ids - lexical_ids
    lexical_only = lexical_ids - stored_ids
    neither = all_ids - (stored_ids | lexical_ids)

    ordered = list(papers)
    random.Random(seed).shuffle(ordered)
    packet_rows: list[dict] = []
    key_rows: list[dict] = []
    for index, paper in enumerate(ordered, start=1):
        item_id = f"screen-{index:03d}"
        paper_id = str(paper.get("paperId"))
        architecture, security = _lexical_features(paper)
        packet_rows.append({
            "item_id": item_id,
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "publication_year": paper.get("year", ""),
            "reviewer_1_decision": "",
            "reviewer_2_decision": "",
            "adjudicated_decision": "",
            "exclusion_reason": "",
            "comments": "",
        })
        key_rows.append({
            "item_id": item_id,
            "paper_id": paper_id,
            "stored_pilot_retained": paper_id in stored_ids,
            "recomputed_current_lexical_retained": paper_id in lexical_ids,
            "architecture_terms_matched": "; ".join(architecture),
            "security_terms_matched": "; ".join(security),
        })

    packet_path = os.path.join(audit_dir, "screening_audit_packet_blinded.csv")
    key_path = os.path.join(audit_dir, "screening_audit_key.csv")
    _write_csv(packet_path, packet_rows, packet_rows[0].keys())
    _write_csv(key_path, key_rows, key_rows[0].keys())

    stored_method_counts = Counter(
        str(item.get("screening_method")) if item.get("screening_method") else "missing"
        for item in stored_screened
    )
    stored_reason_counts = Counter(
        "transparent_lexical_reason"
        if str(item.get("relevance_reason", "")).startswith("Transparent lexical retrieval screen;")
        else "other_or_missing_reason"
        for item in stored_screened
    )
    report = {
        "scope": (
            "Reproducibility comparison only. The stored pilot decision is not a gold "
            "standard and this report does not estimate screening precision or recall."
        ),
        "current_lexical_rule": {
            "version": LEXICAL_RULE_VERSION,
            "function": "src.fetch_papers.screen_relevance_lexical",
            "function_sha256": source_hash,
            "input": "lower-cased title concatenated with abstract",
            "include_rule": "at least one architecture term AND at least one security term",
            "architecture_terms": list(ARCHITECTURE_TERMS),
            "security_terms": list(SECURITY_TERMS),
            "ranking_after_inclusion": "relevance score descending, then citation count descending",
            "cap_for_this_comparison": len(papers),
        },
        "counts": {
            "retrieved_records": len(papers),
            "stored_pilot_retained": len(stored_ids),
            "recomputed_current_lexical_retained": len(lexical_ids),
            "both_retained": len(stored_ids & lexical_ids),
            "stored_only": len(stored_only),
            "lexical_only": len(lexical_only),
            "neither": len(neither),
            "set_jaccard_agreement": round(
                len(stored_ids & lexical_ids) / len(stored_ids | lexical_ids), 6
            ),
            "raw_decision_agreement": round(
                (len(stored_ids & lexical_ids) + len(neither)) / len(all_ids), 6
            ),
        },
        "stored_pilot_metadata": {
            "screening_method_field": dict(stored_method_counts),
            "relevance_reason_format": dict(stored_reason_counts),
            "conclusion": (
                "Retained fields do not establish that the stored pilot selection was "
                "produced by the current lexical rule; its historical procedure is not "
                "reconstructible from these metadata alone."
            ),
        },
        "human_audit_protocol": {
            "packet": os.path.basename(packet_path),
            "key": os.path.basename(key_path),
            "records": len(packet_rows),
            "recommended_labels": ["include", "exclude", "uncertain"],
            "recommended_process": "two independent reviewers followed by adjudication",
        },
    }
    _write_json(os.path.join(audit_dir, "screening_reproducibility_report.json"), report)
    return report


def _confidence_bin(value: object) -> str:
    score = float(value or 0)
    if score < 0.50:
        return "low_[0.30,0.50)"
    if score < 0.70:
        return "medium_[0.50,0.70)"
    return "high_[0.70,1.00]"


def _stratified_indices(triples: list[dict], size: int, seed: int) -> list[int]:
    """Coverage-first sampling across relation, score bin, and year strata."""
    groups: dict[tuple[str, str, object], list[int]] = defaultdict(list)
    for index, triple in enumerate(triples):
        groups[(
            str(triple.get("relation", "")),
            _confidence_bin(triple.get("confidence")),
            triple.get("year"),
        )].append(index)
    randomizer = random.Random(seed)
    keys = list(groups)
    randomizer.shuffle(keys)
    for key in keys:
        randomizer.shuffle(groups[key])

    selected: list[int] = []
    positions = {key: 0 for key in keys}
    # First pass gives every non-empty relation/score/year stratum coverage.
    for key in keys:
        if len(selected) >= size:
            break
        selected.append(groups[key][0])
        positions[key] = 1
    # Further passes retain breadth instead of letting high-frequency relations
    # consume the whole packet.
    while len(selected) < min(size, len(triples)):
        advanced = False
        for key in keys:
            position = positions[key]
            if position < len(groups[key]):
                selected.append(groups[key][position])
                positions[key] += 1
                advanced = True
                if len(selected) >= min(size, len(triples)):
                    break
        if not advanced:
            break
    return selected


def create_provenance_and_triple_audit(
    raw_triples: list[dict],
    resolved_triples: list[dict],
    chunks: list[dict],
    audit_dir: str,
    size: int,
    seed: int,
) -> dict:
    """Recover conservative legacy provenance and create a 200-item audit packet."""
    aligned = _indices_align(raw_triples, resolved_triples)
    if not aligned:
        raise RuntimeError("Raw/resolved index alignment failed; provenance is not transferred.")
    matches = recover_quote_provenance(raw_triples, chunks)
    by_paper_id = {str(chunk.get("paperId")): chunk for chunk in chunks}
    status_counts = Counter(match.status for match in matches)
    recovery_rows: list[dict] = []
    for index, (triple, match) in enumerate(zip(raw_triples, matches)):
        chunk = by_paper_id.get(match.paper_id or "")
        recovery_rows.append({
            "legacy_triple_index": index,
            "subject": triple.get("subject", ""),
            "relation": triple.get("relation", ""),
            "object": triple.get("object", ""),
            "publication_year": triple.get("year", ""),
            "evidence_quote": triple.get("evidence_quote", ""),
            "provenance_recovery_status": match.status,
            "paper_id": match.paper_id or "",
            "chunk_index": match.chunk_index if match.chunk_index is not None else "",
            "source_title": chunk.get("title", "") if chunk else "",
            "source_modality": "abstract" if chunk else "",
            "sentence_offset": "",  # not retained by the legacy extraction
            "model_name": "",  # not recoverable from the legacy artifacts
            "prompt_version": "",  # not recoverable from the legacy artifacts
        })
    recovery_path = os.path.join(audit_dir, "legacy_triple_provenance_recovery.csv")
    _write_csv(recovery_path, recovery_rows, recovery_rows[0].keys())

    sample_indices = _stratified_indices(raw_triples, size, seed)
    packet_rows: list[dict] = []
    sample_strata = Counter()
    for packet_index, triple_index in enumerate(sample_indices, start=1):
        triple = raw_triples[triple_index]
        match = matches[triple_index]
        chunk = by_paper_id.get(match.paper_id or "")
        stratum = (
            triple.get("relation"),
            _confidence_bin(triple.get("confidence")),
            triple.get("year"),
        )
        sample_strata[stratum] += 1
        packet_rows.append({
            "item_id": f"triple-{packet_index:03d}",
            "legacy_triple_index": triple_index,
            "subject": triple.get("subject", ""),
            "subject_type": triple.get("subject_type", ""),
            "relation": triple.get("relation", ""),
            "object": triple.get("object", ""),
            "object_type": triple.get("object_type", ""),
            "publication_year": triple.get("year", ""),
            "model_reported_extraction_score": triple.get("confidence", ""),
            "evidence_quote": triple.get("evidence_quote", ""),
            "provenance_recovery_status": match.status,
            "paper_id": match.paper_id or "",
            "chunk_index": match.chunk_index if match.chunk_index is not None else "",
            "source_title": chunk.get("title", "") if chunk else "",
            "source_modality": "abstract" if chunk else "",
            "source_chunk_text": chunk.get("text", "") if chunk else "",
            "subject_correct": "",
            "object_correct": "",
            "relation_correct": "",
            "entity_types_correct": "",
            "evidence_quote_supports_triple": "",
            "source_metadata_correct": "",
            "triple_specific_enough": "",
            "reviewer_id": "",
            "comments": "",
        })
    packet_path = os.path.join(audit_dir, "triple_extraction_audit_packet.csv")
    _write_csv(packet_path, packet_rows, packet_rows[0].keys())

    report = {
        "scope": (
            "Legacy provenance recovery and an unlabelled human-audit packet. "
            "No human correctness labels are inferred by this tool."
        ),
        "raw_resolved_index_alignment": aligned,
        "recovery_method": "unique case/whitespace-normalized exact evidence-quote substring match",
        "counts": {
            "raw_triples": len(raw_triples),
            "unique_exact_quote_matches": status_counts["unique_normalized_exact_quote_match"],
            "unmatched_quotes": status_counts["unmatched_quote"],
            "ambiguous_quotes": status_counts["ambiguous_quote"],
            "recovered_fraction": round(
                status_counts["unique_normalized_exact_quote_match"] / len(matches), 6
            ),
        },
        "not_recoverable_from_legacy_files": [
            "model name/version", "prompt version", "sentence offsets", "source paper for unmatched quotes",
        ],
        "human_audit_protocol": {
            "packet": os.path.basename(packet_path),
            "sample_size": len(packet_rows),
            "seed": seed,
            "design": "coverage-first strata: relation type x model-reported score bin x publication year",
            "available_score_bins": sorted({_confidence_bin(item.get("confidence")) for item in raw_triples}),
            "represented_strata": len(sample_strata),
            "recommended_labels": [
                "subject_correct", "object_correct", "relation_correct", "entity_types_correct",
                "evidence_quote_supports_triple", "source_metadata_correct", "triple_specific_enough",
            ],
        },
    }
    _write_json(os.path.join(audit_dir, "triple_provenance_recovery_report.json"), report)
    return report


def _similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz import fuzz
        return float(fuzz.token_sort_ratio(left.casefold(), right.casefold()))
    except ImportError:  # pragma: no cover - requirements declare rapidfuzz
        return 100 * SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def create_entity_resolution_audit(
    raw_triples: list[dict], entity_mapping: dict[str, list[str]], audit_dir: str, seed: int
) -> dict:
    """Create blinded merge and non-merge comparison packets; no labels inferred."""
    entity_types: dict[str, str] = {}
    for triple in raw_triples:
        entity_types[str(triple["subject"])] = str(triple["subject_type"])
        entity_types[str(triple["object"])] = str(triple["object_type"])
    canonical_of = {name: name for name in entity_types}
    merged_pairs: list[dict] = []
    for canonical, aliases in entity_mapping.items():
        canonical_of.setdefault(canonical, canonical)
        for alias in aliases:
            canonical_of[alias] = canonical
            merged_pairs.append({
                "left_entity": canonical,
                "right_entity": alias,
                "entity_type": entity_types.get(canonical, entity_types.get(alias, "")),
                "pair_kind": "merged_by_pipeline",
                "lexical_similarity": round(_similarity(canonical, alias), 3),
            })

    names = sorted(entity_types, key=str.casefold)
    nonmerged_pairs: list[dict] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            if entity_types[left] != entity_types[right]:
                continue
            if canonical_of.get(left, left) == canonical_of.get(right, right):
                continue
            nonmerged_pairs.append({
                "left_entity": left,
                "right_entity": right,
                "entity_type": entity_types[left],
                "pair_kind": "high_lexical_similarity_nonmerge",
                "lexical_similarity": round(_similarity(left, right), 3),
            })
    nonmerged_pairs.sort(
        key=lambda row: (-float(row["lexical_similarity"]), row["left_entity"].casefold(), row["right_entity"].casefold())
    )
    randomizer = random.Random(seed)
    randomizer.shuffle(merged_pairs)
    chosen = merged_pairs[:100] + nonmerged_pairs[:100]
    randomizer.shuffle(chosen)

    packet_rows = []
    key_rows = []
    for index, row in enumerate(chosen, start=1):
        item_id = f"entity-pair-{index:03d}"
        packet_rows.append({
            "item_id": item_id,
            "entity_a": row["left_entity"],
            "entity_b": row["right_entity"],
            "entity_type": row["entity_type"],
            "same_entity": "",
            "related_but_distinct": "",
            "unrelated": "",
            "uncertain": "",
            "reviewer_id": "",
            "comments": "",
        })
        key_rows.append({"item_id": item_id, **row})
    packet_path = os.path.join(audit_dir, "entity_resolution_audit_packet_blinded.csv")
    key_path = os.path.join(audit_dir, "entity_resolution_audit_key.csv")
    _write_csv(packet_path, packet_rows, packet_rows[0].keys())
    _write_csv(key_path, key_rows, key_rows[0].keys())
    report = {
        "scope": "Unlabelled pair sampling only; no merge precision or false-negative rate is estimated yet.",
        "packet": os.path.basename(packet_path),
        "key": os.path.basename(key_path),
        "merged_pairs_available": len(merged_pairs),
        "merged_pairs_sampled": min(100, len(merged_pairs)),
        "nonmerged_pairs_sampled": min(100, len(nonmerged_pairs)),
        "nonmerged_pair_selection": "top same-type lexical token-sort similarity candidates not merged by the current pipeline",
        "lowest_nonmerged_similarity_in_packet": (
            min((row["lexical_similarity"] for row in nonmerged_pairs[:100]), default=None)
        ),
        "recommended_process": "two independent reviewers followed by adjudication; calculate merge precision and false-negative controls after labels exist.",
    }
    _write_json(os.path.join(audit_dir, "entity_resolution_audit_packet_report.json"), report)
    return report


def run_audit_artifacts(*, triple_sample_size: int = 200, seed: int = AUDIT_SEED) -> dict:
    """Write audit packets below the active run namespace and return a manifest."""
    audit_dir = os.path.join(config.DATA_DIR, "audits")
    os.makedirs(audit_dir, exist_ok=True)
    papers = _load_json(os.path.join(config.RAW_PAPERS_DIR, "papers_metadata.json"))
    screened = _load_json(os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json"))
    raw_triples = _load_json(os.path.join(config.TRIPLES_DIR, "raw_triples.json"))
    resolved_triples = _load_json(os.path.join(config.TRIPLES_DIR, "resolved_triples.json"))
    chunks = _load_json(os.path.join(config.RAW_PAPERS_DIR, "chunks.json"))
    entity_mapping = _load_json(os.path.join(config.TRIPLES_DIR, "entity_mapping.json"))
    screening = create_screening_audit(papers, screened, audit_dir, seed)
    triple = create_provenance_and_triple_audit(
        raw_triples, resolved_triples, chunks, audit_dir, triple_sample_size, seed
    )
    entity = create_entity_resolution_audit(raw_triples, entity_mapping, audit_dir, seed)
    manifest = {
        "scope": "Generated offline packets only; all reviewer and adjudication fields remain blank.",
        "seed": seed,
        "audit_directory": audit_dir,
        "screening": screening,
        "triple_provenance": triple,
        "entity_resolution": entity,
    }
    _write_json(os.path.join(audit_dir, "audit_artifact_manifest.json"), manifest)
    print(f"[+] Wrote audit artifacts to {audit_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create non-judgmental screening and KG audit packets.")
    parser.add_argument("--triple-sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
    args = parser.parse_args()
    run_audit_artifacts(triple_sample_size=args.triple_sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
