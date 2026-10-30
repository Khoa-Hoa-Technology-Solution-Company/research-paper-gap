"""Drop-one feature ablation of the learned scorer and worked examples at the last test cutoff.

    python src/ablation.py   (after gaps.py has built data/index.npz and data/semantic.npy)
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import gaps
from gaps import FEATURES, FIRST_TRAIN, ROOT, TEST_CUTOFFS


def fit_eval(train, rows, feats):
    X = np.array([[r[k] for k in feats] for r in train], float)
    y = np.array([r["filled"] for r in train])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    Xt = np.array([[r[k] for k in feats] for r in rows], float)
    s = m.decision_function(sc.transform(Xt))
    yt = np.array([r["filled"] for r in rows])
    return roc_auc_score(yt, s), average_precision_score(yt, s), s


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
    groups = {"all": FEATURES,
              "-expected,min_occ": [f for f in FEATURES if f not in ("expected", "min_occ")],
              "-growth": [f for f in FEATURES if not f.startswith("growth")],
              "-profile_cos": [f for f in FEATURES if f != "profile_cos"],
              "-recent_cos": [f for f in FEATURES if f != "recent_cos"],
              "-semantic": [f for f in FEATURES if f != "semantic"],
              "only growth": ["growth_min", "growth_max"],
              "only popularity": ["expected", "min_occ"]}
    out = {}
    for name, feats in groups.items():
        aucs, aps = [], []
        for T in TEST_CUTOFFS:
            train = [r for Tp in range(FIRST_TRAIN, T - args.h + 1) for r in tables[Tp]]
            a, p, _ = fit_eval(train, tables[T], feats)
            aucs.append(a)
            aps.append(p)
        out[name] = {"auc": float(np.mean(aucs)), "ap": float(np.mean(aps)), "auc_by_cutoff": aucs}
        print(f"{name:22s} AUC {np.mean(aucs):.3f} AP {np.mean(aps):.3f}", flush=True)
    # worked examples at the last test cutoff
    T = TEST_CUTOFFS[-1]
    train = [r for Tp in range(FIRST_TRAIN, T - args.h + 1) for r in tables[Tp]]
    _, _, s = fit_eval(train, tables[T], FEATURES)
    rows = tables[T]
    order = np.argsort(-s)
    ex = lambda i: {"pair": [concepts[rows[i]["a"]], concepts[rows[i]["b"]]], "future_co": rows[i]["future_co"],
                    "filled": rows[i]["filled"], "rank": int(np.where(order == i)[0][0]) + 1}
    out["examples"] = {"T": T, "n": len(rows), "top": [ex(i) for i in order[:10]],
                       "bottom": [ex(i) for i in order[::-1][:10]]}
    json.dump(out, open(ROOT / "outputs" / "ablation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
