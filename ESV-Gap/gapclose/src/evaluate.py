"""Stage 2: calibrate on SciFact-train, evaluate on SciFact-dev, stress-test.

All thresholds / calibrators are fitted on the calibration split only and then
frozen; the test split is touched once per configuration.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
from collections import defaultdict

import numpy as np

from common import OUT, write_json
from methods import METHODS, Ctx, ESVLearned, esv_nearmiss, esv_scope

B = 2000  # bootstrap resamples
P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = range(5)
ABSTAIN_TARGET = 0.80  # coverage used for the fixed-coverage selective comparison


def load(split, tag=""):
    corpus = json.load(open(OUT / "bench" / "corpus.json", encoding="utf-8"))
    claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
    rows = json.load(open(OUT / "scores" / f"{split}{tag}.json", encoding="utf-8"))["rows"]
    return Ctx(corpus, claims, rows), claims


class LogisticRegression:
    """1-D L2-regularised logistic regression (C=1, unpenalised intercept), Newton steps.

    Matches sklearn's LogisticRegression defaults for a single feature; kept local to
    avoid a scipy dependency.
    """

    def fit(self, X, y, C=1.0, iters=100):
        x, y = np.asarray(X, float).ravel(), np.asarray(y, float)
        w = np.zeros(2)  # [intercept, slope]
        A = np.stack([np.ones_like(x), x], 1)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y) + np.array([0, w[1] / C])
            H = (A * (p * (1 - p))[:, None]).T @ A + np.diag([1e-9, 1 / C])
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-10:
                break
        self.w = w
        return self

    def predict_proba(self, X):
        x = np.asarray(X, float).ravel()
        p = 1 / (1 + np.exp(-(self.w[0] + self.w[1] * x)))
        return np.stack([1 - p, p], 1)


def macro_f1(y, yhat):
    f = []
    for cls in (0, 1):
        tp = np.sum((yhat == cls) & (y == cls))
        fp = np.sum((yhat == cls) & (y != cls))
        fn = np.sum((yhat != cls) & (y == cls))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def best_threshold(s, y):
    cands = np.unique(s)
    best = (-1, None)
    for t in cands:
        m = macro_f1(y, (s >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return best[1]


def witness_ok(claim, wit, strict=False):
    if wit is None:
        return False
    d, si = wit
    if d not in claim["witness"]:
        return False
    if strict:
        return si is not None and si in claim["witness"][d]
    return True


def point_metrics(claims, y, yhat, wits):
    gc, go = y == 1, y == 0
    pc = yhat == 1
    m = {
        "macro_f1": macro_f1(y, yhat),
        "accuracy": float(np.mean(y == yhat)),
        "false_novelty_rate": float(np.mean(yhat[gc] == 0)),   # gold CLOSED -> OPEN
        "false_closure_rate": float(np.mean(yhat[go] == 1)),   # gold OPEN   -> CLOSED
    }
    tpc = [i for i in range(len(y)) if pc[i] and gc[i]]
    m["witness_doc_precision"] = float(np.mean([witness_ok(claims[i], wits[i]) for i in tpc])) if tpc else None
    # sentence-level witnesses only exist for sentence-level methods; a missing witness counts as wrong
    m["witness_sent_precision"] = (float(np.mean([witness_ok(claims[i], wits[i], True) for i in tpc]))
                                   if tpc and any(wits[i] and wits[i][1] is not None for i in tpc) else None)
    # grounded closure: CLOSED decision whose witness is a gold rationale document
    m["grounded_closure_recall"] = float(np.mean([pc[i] and witness_ok(claims[i], wits[i])
                                                  for i in range(len(y)) if gc[i]]))
    fc = [i for i in range(len(y)) if pc[i] and go[i]]
    m["false_closures_on_hard_negative"] = (float(np.mean([wits[i] is not None and wits[i][0] in claims[i]["hard_negatives"]
                                                           for i in fc])) if fc else None)
    return m


def bootstrap(fn, n, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def aurc(conf, correct):
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    risks = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(np.mean(risks)), risks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="score-file suffix; uses esv_config<tag>.json if present")
    args = ap.parse_args()
    cctx, cclaims = load("calib", args.tag)
    tctx, tclaims = load("test", args.tag)
    yc = np.array([c["status"] == "CLOSED" for c in cclaims], int)
    yt = np.array([c["status"] == "CLOSED" for c in tclaims], int)
    cfg_path = OUT / "results" / f"esv_config{args.tag}.json"
    cfg = json.load(open(cfg_path, encoding="utf-8")) if cfg_path.exists() else {"kind": "scope", "alpha": 0.5, "k": 5}
    if cfg["kind"] == "learned":
        METHODS["ESV-Scope"] = ESVLearned(cfg["k"]).fit(cctx, cclaims, yc)
    else:
        METHODS["ESV-Scope"] = functools.partial(esv_scope, alpha=cfg["alpha"], k=cfg["k"])
    print("ESV config:", cfg, flush=True)

    report = {"n_calib": len(cclaims), "n_test": len(tclaims), "esv_config": cfg, "methods": {}}
    frozen, platts = {}, {}
    test_out = {}
    for name, fn in METHODS.items():
        sc = np.array([fn(cctx, c["id"])[0] for c in cclaims])
        thr = best_threshold(sc, yc)
        platt = LogisticRegression().fit(sc.reshape(-1, 1), yc)
        frozen[name] = {"threshold": thr}
        platts[name] = platt
        res = [fn(tctx, c["id"]) for c in tclaims]
        st = np.array([r[0] for r in res])
        wits = [r[1] for r in res]
        yhat = (st >= thr).astype(int)
        m = point_metrics(tclaims, yt, yhat, wits)
        n = len(yt)
        m["macro_f1_ci"] = bootstrap(lambda idx: macro_f1(yt[idx], yhat[idx]), n)
        m["false_novelty_ci"] = bootstrap(
            lambda idx: np.mean(yhat[idx][yt[idx] == 1] == 0) if np.any(yt[idx] == 1) else None, n)
        m["false_closure_ci"] = bootstrap(
            lambda idx: np.mean(yhat[idx][yt[idx] == 0] == 1) if np.any(yt[idx] == 0) else None, n)
        p = platt.predict_proba(st.reshape(-1, 1))[:, 1]
        conf = np.maximum(p, 1 - p)
        m["aurc"], _ = aurc(conf, (p >= 0.5).astype(int) == yt)
        report["methods"][name] = m
        test_out[name] = {"score": st, "yhat": yhat, "wits": wits, "p": p}
        print(f"{name:12s} F1={m['macro_f1']:.3f} FNov={m['false_novelty_rate']:.3f} "
              f"FClo={m['false_closure_rate']:.3f} wdoc={m['witness_doc_precision']} AURC={m['aurc']:.3f}")

    # paired differences vs ESV-Scope
    ref = test_out["ESV-Scope"]["yhat"]
    report["paired_vs_esv"] = {}
    for name in METHODS:
        if name == "ESV-Scope":
            continue
        o = test_out[name]["yhat"]
        report["paired_vs_esv"][name] = {
            "delta_macro_f1": macro_f1(yt, ref) - macro_f1(yt, o),
            "ci": bootstrap(lambda idx: macro_f1(yt[idx], ref[idx]) - macro_f1(yt[idx], o[idx]), len(yt), 1),
            "mcnemar_b_c": [int(np.sum((ref == yt) & (o != yt))), int(np.sum((ref != yt) & (o == yt)))],
        }

    report["frozen_thresholds"] = frozen
    write_json(OUT / "results" / f"main{args.tag}.json", report)
    report["stress"] = stress_test(cctx, cclaims, yc, tctx, tclaims, yt, frozen, platts)
    write_json(OUT / "results" / f"main{args.tag}.json", report)
    return report


# ---------------------------------------------------------------- abstention
def fit_policies(cctx, cclaims, frozen, platts):
    """Fit abstention thresholds on calib so that ABSTAIN_TARGET of calib claims get a decision.

    * conf-abstain: generic selective prediction, abstain on lowest Platt confidence;
    * ESV-Abstain (ESV-Scope only): never abstain on CLOSED; withhold OPEN when the
      question sits in a well-studied neighbourhood (high near-miss relevance).
    """
    pol = {}
    n_abs = int(round((1 - ABSTAIN_TARGET) * len(cclaims)))
    for name, fn in METHODS.items():
        sc = np.array([fn(cctx, c["id"])[0] for c in cclaims])
        p = platts[name].predict_proba(sc.reshape(-1, 1))[:, 1]
        conf = np.sort(np.maximum(p, 1 - p))
        pol[name] = {"conf_tau": float(conf[n_abs]) if n_abs else -1.0}
    sc = np.array([METHODS["ESV-Scope"](cctx, c["id"])[0] for c in cclaims])
    nm = np.array([esv_nearmiss(cctx, c["id"]) for c in cclaims])
    below = np.sort(nm[sc < frozen["ESV-Scope"]["threshold"]])[::-1]
    pol["ESV-Scope"]["nearmiss_tau"] = float(below[n_abs - 1]) if 0 < n_abs <= len(below) else float(below[-1])
    return pol


def decide(name, policy, fn, ctx, cid, thr, platt, pol, removed):
    """Return 1 (CLOSED), 0 (OPEN) or -1 (ABSTAIN)."""
    s = fn(ctx, cid, removed)[0]
    closed = s >= thr
    if policy == "none":
        return int(closed)
    if policy == "conf":
        p = platt.predict_proba([[s]])[0, 1]
        return -1 if max(p, 1 - p) < pol[name]["conf_tau"] else int(closed)
    if policy == "nearmiss":
        if closed:
            return 1
        return -1 if esv_nearmiss(ctx, cid, removed) >= pol[name]["nearmiss_tau"] else 0
    raise ValueError(policy)


def stress_test(cctx, cclaims, yc, tctx, tclaims, yt, frozen, platts):
    """Corpus-incompleteness stress test: delete each gold witness document with prob p.

    The deletion is global (a document missing from the corpus is missing for every
    question).  Thresholds and abstention policies stay frozen from calibration.
    """
    pol = fit_policies(cctx, cclaims, frozen, platts)
    wit_docs = sorted({d for c in tclaims for d in c["witness"]})
    configs = [(n, "none") for n in METHODS] + [(n, "conf") for n in METHODS] + [("ESV-Scope", "nearmiss")]
    out = {"policies": pol, "n_witness_docs": len(wit_docs), "grid": {}}
    gc, go = yt == 1, yt == 0
    for p in P_GRID:
        agg = defaultdict(lambda: defaultdict(list))
        for seed in (SEEDS if 0 < p < 1 else [0]):
            rng = random.Random(seed)
            removed = frozenset(d for d in wit_docs if rng.random() < p)
            for name, policy in configs:
                fn = METHODS[name]
                yh = np.array([decide(name, policy, fn, tctx, c["id"], frozen[name]["threshold"],
                                      platts[name], pol, removed) for c in tclaims])
                said_open = yh == 0
                a = agg[f"{name}|{policy}"]
                a["false_novelty"].append(float(np.mean(yh[gc] == 0)))
                a["closed_recall"].append(float(np.mean(yh[gc] == 1)))
                a["abstain_closed"].append(float(np.mean(yh[gc] == -1)))
                a["abstain_open"].append(float(np.mean(yh[go] == -1)))
                a["open_recall"].append(float(np.mean(yh[go] == 0)))
                a["open_precision"].append(float(np.mean(yt[said_open] == 0)) if said_open.any() else float("nan"))
                a["coverage"].append(float(np.mean(yh != -1)))
                dec = yh != -1
                a["selective_acc"].append(float(np.mean(yh[dec] == yt[dec])) if dec.any() else float("nan"))
        out["grid"][str(p)] = {k: {m: [float(np.nanmean(v)), float(np.nanstd(v))] for m, v in d.items()}
                               for k, d in agg.items()}
        print(f"p={p}: " + "  ".join(f"{k}:FNov={np.mean(d['false_novelty']):.3f}"
                                    for k, d in agg.items()), flush=True)
    return out


if __name__ == "__main__":
    main()
