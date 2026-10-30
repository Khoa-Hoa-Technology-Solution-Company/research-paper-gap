"""Two extra analyses for the main setting.

1. Fill rate of gaps by INISCOM call area (concepts are grouped by the section of the
   lexicon they appear in), pooled over the test cutoffs.
2. Calibration of the learned forecaster's probabilities on the pooled test cutoffs.

    python src/extra.py
"""
import argparse
import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression

import gaps
from gaps import FIRST_TRAIN, ROOT, TEST_CUTOFFS
from lexicon import CONCEPTS

AREA_NAMES = ["Wireless systems", "Industrial and intelligent", "Security and privacy", "HW/SW design",
              "Signal processing and data", "Quantum"]


def concept_areas():
    """Area index of each concept, read from the numbered section comments in lexicon.py."""
    area, out = -1, {}
    for line in open(ROOT / "src" / "lexicon.py", encoding="utf-8"):
        m = re.match(r"\s+# (\d)\.", line)
        if m:
            area = int(m.group(1)) - 1
        m = re.match(r'\s+"([^"]+)": \[', line)
        if m and area >= 0:
            out[m.group(1)] = area
    return [out[c] for c in CONCEPTS]


def reliability(p, y):
    bins = np.quantile(p, np.linspace(0, 1, 6))
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (p >= lo) & (p <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(s.sum()), "mean_p": float(p[s].mean()),
                    "observed": float(y[s].mean())})
    return out


def recalibrated(tables, T, h, rows):
    """Temporal Platt scaling: train the forecaster on cutoffs up to T-H-1, fit a one-feature
    logistic map from its score to the outcome on cutoff T-H (whose window has ended by T), apply at T."""
    train = [r for Tp in range(FIRST_TRAIN, T - h) for r in tables[Tp]]
    sc, m = gaps.fit(train)
    held = tables[T - h]
    s_held = m.decision_function(sc.transform(gaps.matrix(held)))
    platt = LogisticRegression().fit(s_held[:, None], [r["filled"] for r in held])
    return platt.predict_proba(m.decision_function(sc.transform(gaps.matrix(rows)))[:, None])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-occ", type=int, default=50)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--h", type=int, default=3)
    args = ap.parse_args()
    years, docs, occ, co, concepts = gaps.load()
    sem = gaps.semantic_sim(concepts)
    tables = {T: gaps.cutoff_table(T, years, docs, occ, co, sem, args)
              for T in range(FIRST_TRAIN, TEST_CUTOFFS[-1] + 1)}
    area = concept_areas()
    A = len(AREA_NAMES)
    n, f = np.zeros((A, A)), np.zeros((A, A))
    probs, ys, cal = [], [], []
    for T in TEST_CUTOFFS:
        train = [r for Tp in range(FIRST_TRAIN, T - args.h + 1) for r in tables[Tp]]
        sc, m = gaps.fit(train)
        rows = tables[T]
        probs.append(m.predict_proba(sc.transform(gaps.matrix(rows)))[:, 1])
        cal.append(recalibrated(tables, T, args.h, rows))
        ys.append(np.array([r["filled"] for r in rows]))
        for r in rows:
            i, j = sorted((area[r["a"]], area[r["b"]]))
            n[i, j] += 1
            f[i, j] += r["filled"]
    p, y, pc = np.concatenate(probs), np.concatenate(ys), np.concatenate(cal)
    rel, rel_cal = reliability(p, y), reliability(pc, y)
    base = y.mean()
    within = np.trace(f) / np.trace(n)
    across = (f.sum() - np.trace(f)) / (n.sum() - np.trace(n))
    out = {"areas": AREA_NAMES, "concepts_per_area": [area.count(i) for i in range(A)],
           "n": n.tolist(), "filled": f.tolist(), "within_rate": float(within), "across_rate": float(across),
           "within_n": int(np.trace(n)), "across_n": int(n.sum() - np.trace(n)),
           "brier": float(np.mean((p - y) ** 2)), "brier_base": float(np.mean((base - y) ** 2)),
           "pooled_n": int(len(y)), "pooled_rate": float(base), "reliability": rel,
           "brier_cal": float(np.mean((pc - y) ** 2)), "reliability_cal": rel_cal}
    json.dump(out, open(ROOT / "outputs" / "extra.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("n", "filled")}, indent=1))


if __name__ == "__main__":
    main()
