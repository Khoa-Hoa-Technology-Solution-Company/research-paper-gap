"""Uncertified accuracy of the extra baselines (BGE dense, BGE reranker, local LLM judge).

Threshold and Platt calibration are fitted on the calibration split and applied once to the
test split, exactly as in evaluate.py. Also reports the AUC that separates closed from open
calibration questions after deleting every witness document.
Writes <OUT>/results/extra_uncertified.json.
"""
from __future__ import annotations

import json

import numpy as np

from common import OUT, write_json
from evaluate import LogisticRegression, aurc, best_threshold, bootstrap, macro_f1

SOURCES = {
    "BGE-dense": ("bge", "bge_dense_top"),
    "BGE-rerank": ("bge", "bge_rerank"),
    "LLM-judge": ("llm", None),
}


def first_available(lst, removed=frozenset()):
    for d, s in sorted(lst, key=lambda x: -x[1]):
        if d not in removed:
            return s
    return 0.0


def main():
    out = {}
    for name, (prefix, key) in SOURCES.items():
        files = {s: OUT / "scores" / f"{prefix}_{s}.json" for s in ("calib", "test")}
        if not all(f.exists() for f in files.values()):
            continue
        data = {}
        for s, f in files.items():
            sc = json.load(open(f, encoding="utf-8"))
            cl = json.load(open(OUT / "bench" / f"{s}.json", encoding="utf-8"))
            data[s] = (cl, {k: (v[key] if key else v) for k, v in sc.items()})
        yc = np.array([c["status"] == "CLOSED" for c in data["calib"][0]], int)
        yt = np.array([c["status"] == "CLOSED" for c in data["test"][0]], int)
        allw = frozenset(d for c in data["calib"][0] for d in c["witness"])
        get = lambda s, rem=frozenset(): np.array([first_available(data[s][1][str(c["id"])], rem) for c in data[s][0]])
        s_c, s_t = get("calib"), get("test")
        thr = best_threshold(s_c, yc)
        yh = (s_t >= thr).astype(int)
        p = LogisticRegression().fit(s_c.reshape(-1, 1), yc).predict_proba(s_t.reshape(-1, 1))[:, 1]
        s_del, y = get("calib", allw), yc.astype(bool)
        out[name] = {
            "macro_f1": macro_f1(yt, yh), "macro_f1_ci": bootstrap(lambda i: macro_f1(yt[i], yh[i]), len(yt)),
            "false_novelty_rate": float(np.mean(yh[yt == 1] == 0)), "false_closure_rate": float(np.mean(yh[yt == 0] == 1)),
            "witness_sent_precision": None, "aurc": aurc(np.maximum(p, 1 - p), (p >= 0.5).astype(int) == yt)[0],
            "auc_complete_calib": float(np.mean([(a > b) + 0.5 * (a == b) for a in s_c[y] for b in s_c[~y]])),
            "auc_after_deletion_calib": float(np.mean([(a > b) + 0.5 * (a == b) for a in s_del[y] for b in s_del[~y]])),
        }
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()})
    write_json(OUT / "results" / "extra_uncertified.json", out)


if __name__ == "__main__":
    main()
