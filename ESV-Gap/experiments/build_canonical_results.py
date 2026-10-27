"""Rebuild paper_v2/canonical_results.json from the frozen run's cached artefacts.

The AMI 2027 manuscript (paper_v2/main_springer_ami.tex, Tables 3 and 5) reports
numbers that were stored only in paper_v2/canonical_results.json.  This script
regenerates every field it can from local caches and compares the result field
by field with the committed file.  It never overwrites the committed file and
never edits the manuscript.

Inputs (all local; no network access is performed):

- runs/deep_learning_iot_intrusion_de_20260831_114802/   frozen run (gitignored)
    data/raw, data/processed                              corpus counts
    data/triples/all_triples.json                         relation events
    data/graph/knowledge_graph.pkl                        canonical graph
    outputs/corpus_saturation_report.json                 raw-mention saturation
    outputs/external_verification.json                    Pillar A Stage A1
    outputs/author_stated_gaps.json                       Pillar B
- benchmarks/gold_triples_benchmark.json,
  benchmarks/gold_benchmark_extracted.json                Pillar C
- scratch/external_refuting_papers.json,
  scratch/central_evidence_abstracts.json                 cached abstracts (span checks)
- scratch/c4_s2_50.json                                   K_fetch sensitivity records

Fields that no code or cache can regenerate (notably the Stage A2 semantic
counterevidence labels) are emitted as {"_not_reproducible": reason} and are
reported as NOT_REPRODUCIBLE rather than silently copied from the committed file.

Usage (from ESV-Gap/):

    python experiments/build_canonical_results.py
    python experiments/build_canonical_results.py --output outputs/canonical_results_rebuilt.json \
        --report outputs/canonical_results_diff.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any


def _code_root() -> Path:
    return Path(__file__).resolve().parents[1]


CODE_ROOT = _code_root()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.autonomous_reasoning import decide_hypothesis  # noqa: E402
from src.corpus_saturation import fit_heaps_law  # noqa: E402
from src.entity_normalization import entity_tokens, mentions_entity  # noqa: E402
from src.extractor_recall import evaluate_extracted_triples  # noqa: E402

RUN_ID = "deep_learning_iot_intrusion_de_20260831_114802"
CANDIDATE_IDS = ["C1", "C2", "C3", "C4", "C5"]
FLOAT_TOL = 1e-4

NO_STAGE_A2 = (
    "Stage A2 (LLM semantic relation classification) has no implementation in src/, "
    "experiments/ or scratch/, and the frozen run stores no Stage A2 labels; "
    "only Stage A1 lexical co-mention hits are cached."
)


def nr(reason: str) -> dict[str, str]:
    return {"_not_reproducible": reason}


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 4), round(centre + half, 4)]


# --------------------------------------------------------------------------- corpus


def build_corpus(run: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    raw_query_records = sum(
        len(load_jsonl(p)) for p in sorted((run / "data" / "raw").glob("query_*_raw.jsonl"))
    )
    raw_unique = load_jsonl(run / "data" / "raw" / "all_papers_raw.jsonl")
    screened = load_jsonl(run / "data" / "processed" / "all_screened.jsonl")
    retained = load_jsonl(run / "data" / "processed" / "corpus_filtered.jsonl")
    years = [int(p["year"]) for p in retained if p.get("year")]
    return {
        "collected_records": metadata.get("collection_pool_size"),
        "screened_records": len(screened),
        "retained_papers": len(retained),
        "year_range": [min(years), max(years)],
        "target_domain": nr("Free-text label; the run stores topic only as "
                            f"'{metadata.get('topic')}'."),
        "_raw_records_returned_by_queries": raw_query_records,
        "_raw_unique_records": len(raw_unique),
        "_note": "collected_records is the configured collection_pool_size from run_metadata.json, "
                 "not a count of returned records.",
    }


# ------------------------------------------------------------------------ multigraph


def build_multigraph(run: Path, saturation: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    with open(run / "data" / "graph" / "knowledge_graph.pkl", "rb") as f:
        graph = pickle.load(f)
    triples = load_json(run / "data" / "triples" / "all_triples.json")
    summary = saturation["summary"]
    return {
        "canonical_entities": graph.number_of_nodes(),
        "relation_events": len(triples),
        "cumulative_unique_entities": summary["total_unique_entities"],
        "cumulative_unique_relations": summary["total_unique_relations"],
        "_canonical_graph_edges": graph.number_of_edges(),
    }, graph


# ------------------------------------------------------------------- gold benchmark


def build_gold(root: Path) -> dict[str, Any]:
    bench = load_json(root / "benchmarks" / "gold_triples_benchmark.json")
    extracted = load_json(root / "benchmarks" / "gold_benchmark_extracted.json")
    res = evaluate_extracted_triples(bench["papers"], extracted, entity_threshold=0.60, strict_relation=True)
    s = res["summary"]
    tp, fn, fp = s["true_positives"], s["false_negatives"], s["false_positives"]
    gold_n, pred_n = s["gold_triples_total"], s["extracted_triples_evaluated"]
    per_rel = {
        rel: {"gold": v["gold_count"], "tp": v["tp"], "fn": v["fn"], "recall": round(v["recall"], 4)}
        for rel, v in res["per_relation"].items()
    }
    lacks = res["per_relation"].get("LACKS", {"tp": 0, "gold_count": 0})
    return {
        "abstracts_count": len(bench["papers"]),
        "gold_triples_total": gold_n,
        "extracted_triples_evaluated": pred_n,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "recall": s["recall"],
        "recall_wilson_ci_95": wilson(tp, gold_n),
        "miss_rate": s["miss_rate"],
        "miss_rate_wilson_ci_95": wilson(fn, gold_n),
        "precision": s["precision"],
        "precision_wilson_ci_95": wilson(tp, pred_n),
        "f1_score": s["f1_score"],
        "lacks_recall": round(lacks["tp"] / lacks["gold_count"], 4) if lacks["gold_count"] else 0.0,
        "lacks_recall_wilson_ci_95": wilson(lacks["tp"], lacks["gold_count"]),
        "per_relation": per_rel,
        "_provenance_warning": (
            "benchmarks/gold_benchmark_extracted.json is not extractor output: "
            "scratch/setup_exact_benchmark.py writes it by removing a hand-picked set of 8 gold "
            "triples and appending 4 hand-written false positives. The metrics below are "
            "therefore reproducible arithmetic over a constructed file, not a measured "
            "extractor recall."
        ),
    }


# -------------------------------------------------------------------- saturation


def _mean(values: list[int]) -> float:
    return sum(values) / len(values)


def build_saturation(saturation: dict[str, Any], graph: Any) -> dict[str, Any]:
    raw = saturation["summary"]
    raw_decay = raw["discovery_decay_rate"]

    # Same procedure as scratch/render_canonical_saturation.py, without side effects.
    node_papers: dict[str, set] = {}
    for node, data in graph.nodes(data=True):
        for pid in data.get("papers", []):
            node_papers.setdefault(pid, set()).add(node)
            if len(pid) >= 12:
                node_papers.setdefault(pid[:12], set()).add(node)
    seen: set = set()
    marginal: list[int] = []
    cumulative: list[int] = []
    for entry in saturation["series"]:
        ents = node_papers.get(entry["paper_id"], set())
        marginal.append(len(ents - seen))
        seen.update(ents)
        cumulative.append(len(seen))
    heaps = fit_heaps_law(list(range(1, len(cumulative) + 1)), cumulative)
    q = 30
    m_init, m_final = _mean(marginal[:q]), _mean(marginal[-q:])
    decay = round((m_init - m_final) / m_init, 4)
    canonical = {
        "total_papers": len(cumulative),
        "total_unique_entities": len(seen),
        "total_unique_relations": graph.number_of_edges(),
        "heaps_law": heaps,
        "marginal_initial_mean": round(m_init, 2),
        "marginal_final_mean": round(m_final, 2),
        "discovery_decay_rate": decay,
        "discovery_decay_percent": round(decay * 100, 1),
        "verdict": "UNSATURATED",
        "reason": (f"Canonical vocabulary continues active discovery ({m_final:.2f}/paper, "
                   f"beta={heaps['beta']:.3f}, decay={decay * 100:.1f}%)."),
    }
    return {
        "total_papers": raw["total_papers"],
        "marginal_initial_mean": raw["marginal_initial_mean"],
        "marginal_final_mean": raw["marginal_final_mean"],
        "discovery_decay_rate": raw_decay,
        "discovery_decay_percent": round(raw_decay * 100, 1),
        "heaps_beta": raw["heaps_law"]["beta"],
        "heaps_k": raw["heaps_law"]["K"],
        "heaps_r_squared": raw["heaps_law"]["r_squared"],
        "verdict": raw["verdict"],
        "rationale": nr("Hand-written prose; the run's report stores a different reason string."),
        "canonical": canonical,
        "total_unique_canonical_entities": len(seen),
        "canonical_heaps_beta": heaps["beta"],
        "canonical_heaps_k": heaps["K"],
        "canonical_heaps_r_squared": heaps["r_squared"],
        "canonical_marginal_initial_mean": round(m_init, 2),
        "canonical_marginal_final_mean": round(m_final, 2),
        "canonical_decay_percent": round(decay * 100, 1),
        "_note": "The canonical verdict string 'UNSATURATED' is a constant in "
                 "scratch/render_canonical_saturation.py, not a thresholded decision.",
    }


# -------------------------------------------------------------------- candidates


def _norm(text: str) -> str:
    text = text.replace("…", "...")
    return re.sub(r"\s+", " ", text).strip().lower()


def span_check(span: str, abstracts: list[str]) -> str:
    """Check that every '...'-separated segment of a span occurs verbatim in one abstract."""
    segments = [s.strip(" .") for s in _norm(span).split("...") if s.strip(" .")]
    for abstract in abstracts:
        a = _norm(abstract)
        if segments and all(seg in a for seg in segments):
            return "VERBATIM_IN_CACHED_ABSTRACT" if len(segments) == 1 else "VERBATIM_SEGMENTS_IN_CACHED_ABSTRACT"
    return "NOT_FOUND_IN_CACHED_ABSTRACT"


def _cached_abstracts_by_doi(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in ("external_refuting_papers.json", "central_evidence_abstracts.json"):
        path = root / "scratch" / name
        if not path.exists():
            continue
        for item in load_json(path).values():
            doi = str(item.get("doi") or "").lower()
            if not doi:
                continue
            entry = out.setdefault(doi, {"abstracts": [], "titles": [], "authors": None,
                                         "year": None, "venue": None, "sources": []})
            if item.get("abstract"):
                entry["abstracts"].append(item["abstract"])
            if item.get("title"):
                entry["titles"].append(item["title"])
            entry["authors"] = entry["authors"] or item.get("authors")
            entry["year"] = entry["year"] or item.get("year")
            entry["venue"] = entry["venue"] or item.get("venue")
            entry["sources"].append(f"scratch/{name}")
    return out


def _corpus_index(run: Path) -> dict[str, dict[str, Any]]:
    return {p["paperId"]: p for p in load_jsonl(run / "data" / "processed" / "corpus_filtered.jsonl")}


def _author_names(paper: dict[str, Any]) -> list[str]:
    return [a.get("name", "") if isinstance(a, dict) else str(a) for a in paper.get("authors") or []]


def _format_authors(names: list[str]) -> str:
    return " and ".join(names) if len(names) == 2 else ", ".join(names)


def build_candidates(run: Path, root: Path, committed: dict[str, Any],
                     saturation_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ext = load_json(run / "outputs" / "external_verification.json")["candidates"]
    auth = {c["evidence_cell_id"]: c["author_corroboration"]
            for c in load_json(run / "outputs" / "author_stated_gaps.json")["candidates"]}
    raw_gaps = {g["evidence_cell_id"]: g
                for g in load_json(run / "outputs" / "detected_gaps_raw.json").get("evidence_gaps", [])}
    cached = _cached_abstracts_by_doi(root)
    corpus = _corpus_index(run)
    committed_by_id = {c["candidate_id"]: c for c in committed.get("candidates", [])}

    rebuilt, policy_checks = [], []
    for cid, cand in zip(CANDIDATE_IDS, ext):
        ev = cand["external_verification"]
        probes = [p for p in ev["probes"] if p.get("ok")]
        hits = [h for p in probes for h in p.get("confirmed_hits", [])]
        hit_dois = {str(h.get("doi") or "").lower() for h in hits}
        author = auth.get(cand["evidence_cell_id"], {})
        co_mentions = sum(int(p.get("confirmed_hit_count", 0)) for p in probes)
        origin = raw_gaps.get(cand["evidence_cell_id"], {}).get("supporting_paper_ids", [])

        # Disposition exactly as the current code computes it from the cached run.
        code_as_is = decide_hypothesis(cand, external_record=cand, author_record={"author_corroboration": author},
                                       saturation_summary=saturation_summary, timestamp="frozen")

        row: dict[str, Any] = {
            "candidate_id": cid,
            "subject": cand["subject"],
            "capability_limitation": cand["missing_capability"],
            "raw_index_count": sum(int(p.get("raw_count", 0)) for p in probes),
            "fetched_record_count": sum(int(p.get("inspected_records", 0)) for p in probes),
            "inspected_record_count": sum(int(p.get("inspected_records", 0)) for p in probes),
            "co_mention_count": co_mentions,
            "semantic_verified_counterevidence_count": nr(NO_STAGE_A2),
            "independent_corroboration_count": int(author.get("corroboration_count", 0)),
            "circular_corroboration_count": int(author.get("circular_count", 0)),
            "_origin_paper_ids": origin,
            "_circular_source_papers": sorted({p for c in author.get("circular", []) for p in c.get("source_papers", [])}),
            "_stage_a1_hits": [{"title": h.get("title"), "doi": h.get("doi"), "year": h.get("year")} for h in hits],
            "_disposition_code_as_is": code_as_is.epistemic_disposition.name,
            "_action_code_as_is": code_as_is.action.name,
        }
        if co_mentions == 0:
            # Stage A2 has nothing to classify, so the disposition does not depend on it.
            row["disposition"] = code_as_is.epistemic_disposition.name
            row["action"] = code_as_is.action.name
        else:
            reason = ("Depends on Stage A2 labels. " + NO_STAGE_A2 +
                      f" Current code on the cached run gives {code_as_is.epistemic_disposition.name}"
                      " (lexical co-mention treated as refutation).")
            row["disposition"] = nr(reason)
            row["action"] = nr(reason)
        row["abstention_reason"] = nr("Hand-written prose.")

        # Conditional check: given the committed (unreproduced) Stage A2 counts, does the
        # code's decision policy yield the committed disposition?
        committed_row = committed_by_id.get(cid, {})
        a2 = committed_row.get("semantic_verified_counterevidence_count")
        if a2 is not None:
            conditional_ext = {"verdict": "refuted" if a2 > 0 else "absence_corroborated",
                               "refuting_papers": [], "confirmed_hit_count": a2}
            cond = decide_hypothesis(cand, external_record={"external_verification": conditional_ext},
                                     author_record={"author_corroboration": author},
                                     saturation_summary=saturation_summary, timestamp="frozen")
            policy_checks.append({
                "candidate_id": cid,
                "committed_stage_a2_count": a2,
                "policy_disposition": cond.epistemic_disposition.name,
                "policy_action": cond.action.name,
                "committed_disposition": committed_row.get("disposition"),
                "committed_action": committed_row.get("action"),
                "consistent": (cond.epistemic_disposition.name == committed_row.get("disposition")
                               and cond.action.name == committed_row.get("action")),
                "policy_reason": cond.decision_reason,
            })

        pce = committed_row.get("primary_counterevidence")
        if pce:
            doi = pce["doi"].lower()
            c = cached.get(doi, {})
            first_author = (c.get("authors") or [None])[0]
            row["primary_counterevidence"] = {
                "title": next((h["title"] for h in hits if str(h.get("doi") or "").lower() == doi),
                              nr("DOI is not among the frozen run's Stage A1 hits.")),
                "authors": f"{first_author} et al." if first_author else nr("No cached author list."),
                "year": next((h["year"] for h in hits if str(h.get("doi") or "").lower() == doi), c.get("year")),
                "venue": c.get("venue") or nr("No cached venue."),
                "doi": pce["doi"] if doi in hit_dois else nr("DOI is not among the frozen run's Stage A1 hits."),
                "evidence_span": {"_span_check": span_check(pce["evidence_span"], c.get("abstracts", []))},
                "_selection": nr("Choice of primary record among Stage A1 hits is a Stage A2 output. " + NO_STAGE_A2),
                "_cache_sources": c.get("sources", []),
            }

        sources = []
        for item in author.get("corroborating", []):
            for pid in item.get("source_papers", []):
                paper = corpus.get(pid, {})
                names = _author_names(paper)
                sources.append({
                    "paper_id": pid,
                    "authors": _format_authors(names),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "venue": paper.get("venue") or (paper.get("journal") or {}).get("name"),
                    "doi": (paper.get("externalIds") or {}).get("DOI"),
                    "evidence_span": {"_pillar_b_sentence": item.get("sentence", "")},
                    "_author_disjoint_from_origin": not (
                        set(names) & {n for o in origin for n in _author_names(corpus.get(o, {}))}),
                })
        if sources or committed_row.get("corroborating_sources"):
            row["corroborating_sources"] = sources
        rebuilt.append(row)
    return rebuilt, policy_checks


# --------------------------------------------------------------------- K_fetch


STRICT_GROUPS = (["machine learning-driven intrusion detection frameworks"], ["computational burdens"])
EXPANDED_GROUPS = (
    ["machine learning-driven intrusion detection frameworks", "machine learning", "intrusion detection"],
    ["computational burdens", "computational overhead", "resource-constrained", "edge"],
)


def _co_mentions(records: list[dict[str, Any]], groups: tuple[list[str], list[str]]) -> int:
    left = [entity_tokens(p) for p in groups[0]]
    right = [entity_tokens(p) for p in groups[1]]
    hits = 0
    for r in records:
        toks = entity_tokens(f"{r.get('title') or ''} {r.get('abstract') or ''}".lower())
        if any(mentions_entity(toks, lt, 0.6) for lt in left if lt) and \
                any(mentions_entity(toks, rt, 0.6) for rt in right if rt):
            hits += 1
    return hits


def build_k_fetch(root: Path) -> dict[str, Any]:
    data = load_json(root / "scratch" / "c4_s2_50.json")
    records = data.get("data", [])
    hardcoded = nr("Hard-coded constant in scratch/eval_k_fetch.py (verified_counter = 0, "
                   "disposition = 'EVIDENCE_SUPPORTED_BUT_OPEN'); not computed. " + NO_STAGE_A2)
    out: dict[str, Any] = {
        "provider": "Semantic Scholar Graph API",
        "ranking": "Default relevance",
        "candidate_id": "C4",
        "evaluated_depths": [5, 10, 20, 50],
        "_records_available": len(records),
        "_provider_total": data.get("total"),
        "_note": "scratch/c4_s2_50.json was queried on 2026-09-15, after the frozen run's "
                 "2026-09-14 probe; its record list is not the frozen run's Stage A1 pool.",
    }
    for key, groups in (("strict_canonical_protocol", STRICT_GROUPS), ("expanded_synonym_protocol", EXPANDED_GROUPS)):
        out[key] = {
            f"K={k}": {"lexical_co_mentions": _co_mentions(records[:k], groups),
                       "verified_counterevidence": hardcoded, "disposition": hardcoded}
            for k in (5, 10, 20, 50)
        }
    return out


# -------------------------------------------------------------------- comparison


def _walk(value: Any, path: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            out.extend(_walk(v, f"{path}.{k}" if path else k))
        return out
    if isinstance(value, list) and value and isinstance(value[0], dict):
        out = []
        for i, v in enumerate(value):
            out.extend(_walk(v, f"{path}[{i}]"))
        return out
    return [(path, value)]


def _lookup(tree: Any, path: str) -> tuple[bool, Any]:
    node = tree
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if isinstance(node, dict) and "_not_reproducible" in node:
            return True, node
        if part.startswith("["):
            idx = int(part[1:-1])
            if not isinstance(node, list) or idx >= len(node):
                return False, None
            node = node[idx]
        else:
            if not isinstance(node, dict) or part not in node:
                return False, None
            node = node[part]
    return True, node


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= FLOAT_TOL
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def compare(committed: dict[str, Any], rebuilt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path, expected in _walk(committed):
        found, actual = _lookup(rebuilt, path)
        row = {"field": path, "committed": expected}
        if not found:
            row.update(status="MISSING_IN_REBUILD", rebuilt=None)
        elif isinstance(actual, dict) and "_not_reproducible" in actual:
            row.update(status="NOT_REPRODUCIBLE", rebuilt=None, reason=actual["_not_reproducible"])
        elif isinstance(actual, dict) and "_span_check" in actual:
            status = "MATCH" if actual["_span_check"].startswith("VERBATIM") else "MISMATCH"
            row.update(status=status, rebuilt=actual["_span_check"])
        elif isinstance(actual, dict) and "_pillar_b_sentence" in actual:
            sentence = actual["_pillar_b_sentence"]
            ok = _norm(str(expected)) in _norm(sentence)
            row.update(status="MATCH" if ok else "MISMATCH",
                       rebuilt="committed span is a substring of the Pillar B sentence" if ok else sentence)
        elif _equal(expected, actual):
            row.update(status="MATCH", rebuilt=actual)
        else:
            row.update(status="MISMATCH", rebuilt=actual)
        rows.append(row)
    return rows


# -------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", default=str(CODE_ROOT / "runs" / RUN_ID))
    parser.add_argument("--committed", default=str(CODE_ROOT / "paper_v2" / "canonical_results.json"))
    parser.add_argument("--output", default=str(CODE_ROOT / "outputs" / "canonical_results_rebuilt.json"))
    parser.add_argument("--report", default=str(CODE_ROOT / "outputs" / "canonical_results_diff.json"))
    args = parser.parse_args()

    run = Path(args.run_dir)
    if not run.exists():
        sys.exit(f"Frozen run not found: {run} (runs/ is gitignored; restore it from the archive).")
    committed_path = Path(args.committed)
    if Path(args.output).resolve() == committed_path.resolve():
        sys.exit("Refusing to overwrite the committed canonical_results.json.")
    committed = load_json(committed_path)

    metadata = load_json(run / "run_metadata.json")
    saturation = load_json(run / "outputs" / "corpus_saturation_report.json")
    multigraph, graph = build_multigraph(run, saturation)
    candidates, policy_checks = build_candidates(run, CODE_ROOT, committed, saturation["summary"])

    rebuilt = {
        "corpus": build_corpus(run, metadata),
        "multigraph": multigraph,
        "gold_benchmark": build_gold(CODE_ROOT),
        "corpus_saturation": build_saturation(saturation, graph),
        "candidates": candidates,
        "k_fetch_sensitivity": build_k_fetch(CODE_ROOT),
        "_conditional_policy_check": {
            "description": "decide_hypothesis() applied to the cached run with Stage A1 hits replaced by "
                           "the committed (unreproduced) Stage A2 counts. Shows whether the code's policy "
                           "is consistent with the committed dispositions; it does not reproduce them.",
            "rows": policy_checks,
        },
        "_source_run": RUN_ID,
    }

    rows = compare(committed, rebuilt)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rebuilt, f, indent=2, ensure_ascii=False)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"summary": counts, "fields": rows}, f, indent=2, ensure_ascii=False)

    print(f"Rebuilt: {args.output}\nDiff:    {args.report}\n")
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for status in ("MISMATCH", "MISSING_IN_REBUILD", "NOT_REPRODUCIBLE"):
        bad = [r for r in rows if r["status"] == status]
        if bad:
            print(f"\n{status} ({len(bad)}):")
            for r in bad:
                shown = r.get("rebuilt") if status != "NOT_REPRODUCIBLE" else "-"
                print(f"  {r['field']}: committed={json.dumps(r['committed'], ensure_ascii=False)[:90]} "
                      f"rebuilt={json.dumps(shown, ensure_ascii=False)[:90]}")
    print("\nConditional policy check (committed Stage A2 counts -> code policy):")
    for c in policy_checks:
        print(f"  {c['candidate_id']}: A2={c['committed_stage_a2_count']} -> {c['policy_disposition']}/"
              f"{c['policy_action']} (committed {c['committed_disposition']}/{c['committed_action']}) "
              f"{'consistent' if c['consistent'] else 'INCONSISTENT'}")


if __name__ == "__main__":
    main()
