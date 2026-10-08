"""Summarise the manual audits (annotation/*_A.csv, *_B.csv; a single unsuffixed file also works)."""
from __future__ import annotations

import csv
from collections import Counter

from common import ROOT

ANN = ROOT / "annotation"


def read(name):
    out = {}
    for tag in ("A", "B", ""):
        p = ANN / (f"{name}_{tag}.csv" if tag else f"{name}.csv")
        if p.exists():
            out[tag or "single"] = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    return out


def col(rows, prefix):
    key = next(k for k in rows[0] if k.startswith(prefix))
    return [r[key].strip().lower() for r in rows]


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    po = sum(x == y for x, y in pairs) / len(pairs)
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / len(pairs) ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


w = read("witness_audit")
for who, rows in w.items():
    lab = col(rows, "VALID_EVIDENCE")
    for g in ("1", "0"):
        sub = [l for r, l in zip(rows, lab) if r["matches_gold_rationale"] == g and l]
        if sub:
            c = Counter(sub)
            print(f"[witness/{who}] gold-match={g}: n={len(sub)} yes={c['yes']/len(sub):.2f} "
                  f"partial={c['partial']/len(sub):.2f} no={c['no']/len(sub):.2f}")
if "A" in w and "B" in w:
    print("witness kappa:", kappa(col(w["A"], "VALID_EVIDENCE"), col(w["B"], "VALID_EVIDENCE")))

n = read("nei_external_audit")
for who, rows in n.items():
    ans, scope = col(rows, "ANSWERS_QUESTION"), col(rows, "SAME_SCOPE")
    done = [(a, s) for a, s in zip(ans, scope) if a]
    if done:
        answered = sum(a in ("support", "refute") for a, _ in done)
        same = sum(a in ("support", "refute") and s == "yes" for a, s in done)
        print(f"[nei/{who}] n={len(done)} answered elsewhere={answered/len(done):.2f} "
              f"answered within the same scope={same/len(done):.2f}")
if "A" in n and "B" in n:
    print("nei kappa:", kappa(col(n["A"], "ANSWERS_QUESTION"), col(n["B"], "ANSWERS_QUESTION")))
