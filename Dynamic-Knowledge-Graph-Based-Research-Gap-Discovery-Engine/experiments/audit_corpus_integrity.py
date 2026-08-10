"""Audit whether a retrospective corpus and its triples are source-traceable."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


REDIRECT_CODES = {301, 302, 303, 307, 308}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def resolve_doi(doi: str) -> tuple[str, int]:
    request = urllib.request.Request(
        "https://doi.org/" + doi,
        method="HEAD",
        headers={"User-Agent": "ESV-Gap-corpus-audit/1.0"},
    )
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=20) as response:
            return doi, int(response.status)
    except urllib.error.HTTPError as error:
        return doi, int(error.code)
    except Exception:
        return doi, 0


def audit(papers: list[dict], triples: list[dict], resolve_dois: bool) -> dict:
    paper_ids = [str(item.get("paperId") or item.get("paper_id") or "") for item in papers]
    paper_by_id = {paper_id: item for paper_id, item in zip(paper_ids, papers) if paper_id}
    triple_paper_ids = [
        str(item.get("paper_id") or item.get("source_paper_id") or "")
        for item in triples
    ]
    exact_evidence = 0
    nonempty_evidence = 0
    location_status = Counter()
    for triple, paper_id in zip(triples, triple_paper_ids):
        evidence = str(triple.get("evidence_quote") or triple.get("evidence") or "")
        if evidence:
            nonempty_evidence += 1
            abstract = str(paper_by_id.get(paper_id, {}).get("abstract") or "")
            if evidence.casefold() in abstract.casefold():
                exact_evidence += 1
        location_status[str(triple.get("evidence_location_status") or "unrecorded")] += 1

    dois = [str(item.get("externalIds", {}).get("DOI") or "") for item in papers]
    dois = [doi for doi in dois if doi]
    result = {
        "papers": len(papers),
        "unique_paper_ids": len(set(filter(None, paper_ids))),
        "papers_with_abstract": sum(bool(item.get("abstract")) for item in papers),
        "papers_with_doi": len(dois),
        "papers_with_semantic_scholar_corpus_id": sum(
            bool(item.get("externalIds", {}).get("CorpusId")) for item in papers
        ),
        "duplicate_titles": len(papers) - len({str(item.get("title") or "").casefold() for item in papers}),
        "triples": len(triples),
        "triples_with_paper_id": sum(bool(value) for value in triple_paper_ids),
        "unique_triple_paper_ids": len(set(filter(None, triple_paper_ids))),
        "triple_paper_ids_not_in_corpus": sorted(set(filter(None, triple_paper_ids)) - set(paper_by_id)),
        "triples_with_evidence": nonempty_evidence,
        "evidence_exact_substring_of_source_abstract": exact_evidence,
        "evidence_exact_match_rate": round(exact_evidence / max(len(triples), 1), 4),
        "evidence_location_status": dict(sorted(location_status.items())),
    }
    if resolve_dois:
        with ThreadPoolExecutor(max_workers=16) as executor:
            resolved = list(executor.map(resolve_doi, dois))
        statuses = Counter(status for _, status in resolved)
        nonresolving = [doi for doi, status in resolved if status not in REDIRECT_CODES]
        result["doi_resolution"] = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "attempted": len(resolved),
            "resolving_redirects": len(resolved) - len(nonresolving),
            "nonresolving_dois": nonresolving,
            "status_counts": {str(key): value for key, value in sorted(statuses.items())},
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--triples", required=True)
    parser.add_argument("--output", default="outputs/corpus_integrity_audit.json")
    parser.add_argument("--resolve-dois", action="store_true")
    args = parser.parse_args()
    papers = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    triples = json.loads(Path(args.triples).read_text(encoding="utf-8"))
    result = audit(papers, triples, args.resolve_dois)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
