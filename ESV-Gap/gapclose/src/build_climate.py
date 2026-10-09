"""Build GapClose-ClimateFEVER, a second-domain benchmark, in the same format as GapClose-SciFact.

Climate-FEVER (Diggelmann et al., 2020) pairs 1,535 real-world climate claims with five Wikipedia
sentences each, every sentence labelled by annotators as SUPPORTS / REFUTES / NOT_ENOUGH_INFO.
Mapping (fixed before any system was run):
  * one document per evidence sentence, titled by its Wikipedia article;
  * CLOSED  <=> claim label SUPPORTS or REFUTES and at least one sentence with that label (witnesses);
  * OPEN    <=> claim label NOT_ENOUGH_INFO and no sentence labelled SUPPORTS/REFUTES;
  * DISPUTED claims are excluded;
  * the five retrieved sentences of an OPEN claim are its hard negatives;
  * a fixed random 50/50 split gives the calibration and test halves.
Run with GAPCLOSE_OUT=<dir> so that outputs do not overwrite the SciFact files.
"""
from __future__ import annotations

import json
import random
from collections import Counter

from common import DATA, OUT, write_json

SRC = DATA / "climatefever" / "climate-fever.jsonl"


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    corpus, claims = {}, []
    for r in rows:
        for e in r["evidences"]:
            corpus.setdefault(e["evidence_id"], {"title": e["article"], "sentences": [e["evidence"]]})
    for r in rows:
        lab = r["claim_label"]
        if lab == "DISPUTED":
            continue
        wit = {e["evidence_id"]: [0] for e in r["evidences"] if e["evidence_label"] in ("SUPPORTS", "REFUTES")}
        cited = [e["evidence_id"] for e in r["evidences"]]
        if lab in ("SUPPORTS", "REFUTES") and wit:
            status = "CLOSED"
        elif lab == "NOT_ENOUGH_INFO" and not wit:
            status = "OPEN"
        else:
            continue  # label/evidence disagreement: dropped
        claims.append({"id": int(r["claim_id"]), "claim": r["claim"], "status": status, "witness": wit,
                       "evidence_label": lab, "cited_doc_ids": cited,
                       "hard_negatives": [d for d in cited if d not in wit]})
    rng = random.Random(0)
    rng.shuffle(claims)
    half = len(claims) // 2
    calib, test = sorted(claims[:half], key=lambda c: c["id"]), sorted(claims[half:], key=lambda c: c["id"])
    write_json(OUT / "bench" / "calib.json", calib)
    write_json(OUT / "bench" / "test.json", test)
    write_json(OUT / "bench" / "corpus.json", corpus)
    man = {"source": "Climate-FEVER (Diggelmann et al., 2020)", "corpus_docs": len(corpus),
           "excluded_disputed": sum(r["claim_label"] == "DISPUTED" for r in rows),
           "dropped_inconsistent": len(rows) - sum(r["claim_label"] == "DISPUTED" for r in rows) - len(claims)}
    for name, split in (("calib", calib), ("test", test)):
        man[name] = {"claims": len(split), "status": dict(Counter(c["status"] for c in split)),
                     "witness_docs": len({d for c in split for d in c["witness"]})}
    write_json(OUT / "bench" / "manifest.json", man)
    print(man)


if __name__ == "__main__":
    main()
