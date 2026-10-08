"""Build the GapClose-SciFact benchmark from the official SciFact release.

Mapping (fixed before any system was run):
  * every SciFact claim c is read as a candidate research question
    "is there published evidence on whether c?";
  * gold status CLOSED  <=> c has >=1 SUPPORT/CONTRADICT rationale in the corpus;
    gold status OPEN    <=> c is NEI (no rationale anywhere in the 5,183-abstract corpus);
  * gold witnesses are the annotated (doc, rationale-sentence) pairs;
  * hard negatives are cited documents that contain no rationale (topically
    related, expert-checked, but not closing the question).
SciFact `train` is used only for threshold calibration; SciFact `dev` is the
held-out test split (the official test split has no public labels).
"""
from __future__ import annotations

import sys
from collections import Counter

from common import DATA, OUT, read_jsonl, sha256, write_json

SRC = DATA / "scifact" / "data"


def convert(claims: list[dict]) -> list[dict]:
    out = []
    for c in claims:
        ev = c.get("evidence", {}) or {}
        wit = {}
        labels = Counter()
        for did, sets in ev.items():
            sents = sorted({s for es in sets for s in es["sentences"]})
            wit[str(did)] = sents
            for es in sets:
                labels[es["label"]] += 1
        cited = [str(d) for d in c.get("cited_doc_ids", [])]
        out.append({
            "id": c["id"],
            "claim": c["claim"],
            "status": "CLOSED" if wit else "OPEN",
            "witness": wit,
            "evidence_label": (labels.most_common(1)[0][0] if labels else "NEI"),
            "cited_doc_ids": cited,
            "hard_negatives": [d for d in cited if d not in wit],
        })
    return out


def main() -> None:
    corpus = read_jsonl(SRC / "corpus.jsonl")
    train = convert(read_jsonl(SRC / "claims_train.jsonl"))
    dev = convert(read_jsonl(SRC / "claims_dev.jsonl"))
    docs = {str(d["doc_id"]): {"title": d["title"], "sentences": d["abstract"]} for d in corpus}
    for split in (train, dev):
        for c in split:
            for d in list(c["witness"]) + c["cited_doc_ids"]:
                assert d in docs, (c["id"], d)

    def stats(split):
        st = Counter(c["status"] for c in split)
        lab = Counter(c["evidence_label"] for c in split)
        hn = sum(len(c["hard_negatives"]) for c in split if c["status"] == "OPEN")
        return {"claims": len(split), "status": dict(st), "label": dict(lab),
                "open_claims_with_hard_negative": sum(1 for c in split if c["status"] == "OPEN" and c["hard_negatives"]),
                "hard_negative_docs_for_open": hn,
                "witness_docs": sum(len(c["witness"]) for c in split)}

    write_json(OUT / "bench" / "calib.json", train)
    write_json(OUT / "bench" / "test.json", dev)
    write_json(OUT / "bench" / "corpus.json", docs)
    manifest = {
        "source": "SciFact release (Wadden et al., EMNLP 2020)",
        "files": {p.name: sha256(p) for p in sorted(SRC.glob("*.jsonl"))},
        "corpus_docs": len(docs),
        "calib": stats(train),
        "test": stats(dev),
    }
    write_json(OUT / "bench" / "manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    sys.exit(main())
