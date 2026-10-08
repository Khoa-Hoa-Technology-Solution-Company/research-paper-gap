"""Development protocol: 5-fold cross-validation on the calibration split ONLY.

The variant grid below is fixed before any run of the stronger verifier.  For a
given score file (``--tag``) we report CV macro-F1 of every baseline and every
ESV variant, select the ESV configuration with the best mean CV macro-F1, and
write it to ``outputs/results/esv_config<tag>.json`` for evaluate.py.  The test
split is never loaded here.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import OUT, write_json
from evaluate import best_threshold, load, macro_f1
from methods import METHODS, ESVLearned, esv_scope

ALPHAS = [0.0, 0.25, 0.5, 1.0]
KS = [3, 5, 10]
FOLDS = 5


def folds(n, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::FOLDS] for i in range(FOLDS)]


def cv_threshold_f1(scores, y):
    out = []
    for te in folds(len(y)):
        tr = np.setdiff1d(np.arange(len(y)), te)
        thr = best_threshold(scores[tr], y[tr])
        out.append(macro_f1(y[te], (scores[te] >= thr).astype(int)))
    return float(np.mean(out)), float(np.std(out))


def cv_learned_f1(ctx, claims, y, k):
    out = []
    for te in folds(len(y)):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = ESVLearned(k).fit(ctx, [claims[i] for i in tr], y[tr])
        s_tr = np.array([m(ctx, claims[i]["id"])[0] for i in tr])
        thr = best_threshold(s_tr, y[tr])
        s_te = np.array([m(ctx, claims[i]["id"])[0] for i in te])
        out.append(macro_f1(y[te], (s_te >= thr).astype(int)))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ctx, claims = load("calib", args.tag)
    y = np.array([c["status"] == "CLOSED" for c in claims], int)

    res = {"baselines": {}, "esv_scope": {}, "esv_learned": {}}
    for name, fn in METHODS.items():
        if name == "ESV-Scope":
            continue
        s = np.array([fn(ctx, c["id"])[0] for c in claims])
        res["baselines"][name] = cv_threshold_f1(s, y)
        print(f"{name:12s} CV-F1={res['baselines'][name][0]:.3f} ±{res['baselines'][name][1]:.3f}", flush=True)
    for a in ALPHAS:
        for k in KS:
            s = np.array([esv_scope(ctx, c["id"], alpha=a, k=k)[0] for c in claims])
            res["esv_scope"][f"a={a},k={k}"] = cv_threshold_f1(s, y)
            print(f"ESV-Scope a={a:<4} k={k:<2} CV-F1={res['esv_scope'][f'a={a},k={k}'][0]:.3f}", flush=True)
    for k in KS:
        res["esv_learned"][f"k={k}"] = cv_learned_f1(ctx, claims, y, k)
        print(f"ESV-Learned k={k:<2} CV-F1={res['esv_learned'][f'k={k}'][0]:.3f}", flush=True)

    cands = [({"kind": "scope", "alpha": a, "k": k}, res["esv_scope"][f"a={a},k={k}"][0])
             for a in ALPHAS for k in KS]
    cands += [({"kind": "learned", "k": k}, res["esv_learned"][f"k={k}"][0]) for k in KS]
    best, f1 = max(cands, key=lambda x: x[1])
    res["selected"] = {**best, "cv_macro_f1": f1}
    print("selected:", res["selected"])
    write_json(OUT / "results" / f"cv_dev{args.tag}.json", res)
    write_json(OUT / "results" / f"esv_config{args.tag}.json", res["selected"])


if __name__ == "__main__":
    main()
