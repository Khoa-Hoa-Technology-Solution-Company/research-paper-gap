"""Build manual-audit sheets for the authors (no labels are produced automatically).

annotation/witness_audit.csv  : 100 CLOSED decisions of ESV-Scope (a=1, k=10) on the test split,
                                 balanced between quoted sentences that match a gold rationale and
                                 sentences that do not. Question: is the quoted sentence valid
                                 evidence that answers the question?
annotation/nei_external_audit.csv : the 60 SciFact NEI ("open") questions with the strongest external
                                 OpenAlex evidence. Question: does the retrieved sentence answer the
                                 question (support / refute / no)?
"""
from __future__ import annotations

import csv
import json
import random

from common import OUT, ROOT, content_terms
from evaluate import load
from external import CACHE
from methods import esv_scope

ANN = ROOT / "annotation"
ANN.mkdir(exist_ok=True)
rng = random.Random(7)

# ---------------------------------------------------------------- witnesses
ctx, claims = load("test", "_large")
gold, other = [], []
for c in claims:
    s, w = esv_scope(ctx, c["id"], alpha=1.0, k=10)
    if w is None or s < 0.5:
        continue
    d, si = w
    row = {"question_id": c["id"], "question": c["claim"], "gold_status": c["status"],
           "doc_id": d, "doc_title": ctx.corpus[d]["title"], "witness_sentence": ctx.corpus[d]["sentences"][si],
           "esv_score": round(s, 3),
           "matches_gold_rationale": int(d in c["witness"] and si in c["witness"].get(d, [])),
           "VALID_EVIDENCE (yes/partial/no)": "", "DIRECTION (support/refute)": "", "notes": ""}
    (gold if row["matches_gold_rationale"] else other).append(row)
rng.shuffle(gold)
rng.shuffle(other)
sample = gold[:50] + other[:50]
rng.shuffle(sample)
with open(ANN / "witness_audit.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(sample[0]))
    w.writeheader()
    w.writerows(sample)
print(f"witness_audit.csv: {len(sample)} rows ({min(50, len(gold))} gold-matching, {min(50, len(other))} other)")

# ---------------------------------------------------------------- NEI questions with external evidence
rows = []
for split in ("calib", "test"):
    cl = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
    cache = json.load(open(CACHE / f"openalex_{split}.json", encoding="utf-8"))
    nli = json.load(open(CACHE / f"nli_{split}_large.json", encoding="utf-8"))
    for c in cl:
        k = str(c["id"])
        if c["status"] != "OPEN" or k not in nli:
            continue
        q = content_terms(c["claim"])
        best = (0.0, None)
        for di, d in enumerate(cache[k]["docs"]):
            tt = content_terms(d["title"])
            for si, (pe, pc, _) in enumerate(nli[k].get(str(di), [])):
                cov = len(q & (content_terms(d["sentences"][si]) | tt)) / len(q) if q else 1.0
                sc = max(pe, pc) * cov
                if sc > best[0]:
                    best = (sc, (d, si, "entail" if pe >= pc else "contradict"))
        if best[1]:
            d, si, lab = best[1]
            rows.append({"split": split, "question_id": c["id"], "question": c["claim"],
                         "openalex_id": d["id"], "year": d["year"], "title": d["title"],
                         "sentence": d["sentences"][si], "nli_direction": lab, "external_score": round(best[0], 3),
                         "ANSWERS_QUESTION (support/refute/no)": "", "SAME_SCOPE (yes/no)": "", "notes": ""})
rows.sort(key=lambda r: -r["external_score"])
rows = rows[:60]
with open(ANN / "nei_external_audit.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"nei_external_audit.csv: {len(rows)} rows")
