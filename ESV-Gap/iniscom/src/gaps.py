"""Retrospective gap-closure experiment.

At a cutoff year T, a concept pair (a, b) is a *gap* if both concepts are established
(>= MIN_OCC works up to T) and they co-occur in at most G works up to T. The gap is
*filled* if they co-occur in >= K works in the window T+1..T+H. Every score uses
counts up to T only; the learned model at test cutoff T is trained on cutoffs whose
outcome window ends at or before T, so no future information leaks into training.

    python src/gaps.py [--min-occ 50] [--g 0] [--k 3] [--h 3]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TEST_CUTOFFS = [2019, 2020, 2021, 2022]
FIRST_TRAIN = 2013
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
FEATURES = ["expected", "min_occ", "growth_min", "growth_max", "profile_cos", "recent_cos", "semantic", "co_prior"]
POP_GROWTH = ["expected", "min_occ", "growth_min", "growth_max"]
SCORES = ["random", "expected", "growth_min", "profile_cos", "recent_cos", "adamic_adar", "semantic",
          "lr_pop_growth", "gbm", "learned"]


def load():
    d = np.load(ROOT / "data" / "index.npz")
    load.co_sent = d["co_sent"] if "co_sent" in d.files else None
    load.co_sent_ns = d["co_sent_ns"] if "co_sent_ns" in d.files else None
    return d["years"], d["docs"], d["occ"], d["co"], list(d["concepts"])


def semantic_sim(concepts):
    cache = ROOT / "data" / "semantic.npy"
    if cache.exists():
        return np.load(cache)
    import sys
    sys.path.insert(0, str(ROOT.parent / "gapclose" / "src"))
    from onnx_models import Encoder

    from lexicon import LEXICON
    texts = [f"{c}: " + ", ".join(p.replace("\\b", "").replace("\\w*", "").replace("\\w+", "")
                                   for p in LEXICON[c]) for c in concepts]
    e = Encoder("sentence-transformers/all-MiniLM-L6-v2").encode(texts)
    s = e @ e.T
    np.save(cache, s)
    return s


def ppmi_cos(co_sum, occ_sum, n):
    """Cosine between concepts' PPMI co-occurrence profiles (distributional similarity)."""
    expct = np.outer(occ_sum, occ_sum) / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(co_sum / expct)
    p = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    np.fill_diagonal(p, 0.0)
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    u = p / np.clip(norm, 1e-12, None)
    return u @ u.T


def cutoff_table(T, years, docs, occ, co, sem, args, with_outcome=True, lab=None):
    """Candidate gaps at cutoff T, their features and (optionally) the outcome.

    lab: yearly pair counts used for the outcome (default: same-document co-occurrence)."""
    lab = co if lab is None else lab
    t = int(np.where(years == T)[0][0])
    vy = getattr(args, "vocab_year", 0)
    A0 = occ[: int(np.where(years == vy)[0][0]) + 1].sum(0) if vy else None
    n = docs[: t + 1].sum()
    A = occ[: t + 1].sum(0)
    B = co[: t + 1].sum(0)
    rec = slice(max(0, t - 2), t + 1)
    prev = slice(max(0, t - 5), max(0, t - 2))
    growth = np.log((occ[rec].sum(0) + 1) / (occ[prev].sum(0) + 1))
    pc = ppmi_cos(B.astype(float), A.astype(float), n)
    rc = ppmi_cos(co[rec].sum(0).astype(float), occ[rec].sum(0).astype(float), docs[rec].sum())
    C = len(A)
    nb = (B > 0) & ~np.eye(C, dtype=bool)
    inv_log_deg = 1.0 / np.log(np.maximum(nb.sum(1), 2))
    rows = []
    for a in range(C):
        for b in range(a + 1, C):
            if A[a] < args.min_occ or A[b] < args.min_occ or B[a, b] > args.g:
                continue
            if A0 is not None and (A0[a] < args.min_occ or A0[b] < args.min_occ):
                continue
            f = {"a": a, "b": b,
                 "expected": np.log(A[a] * A[b] / n),
                 "min_occ": np.log(min(A[a], A[b])),
                 "growth_min": min(growth[a], growth[b]), "growth_max": max(growth[a], growth[b]),
                 "profile_cos": pc[a, b], "recent_cos": rc[a, b], "semantic": sem[a, b],
                 "co_prior": float(B[a, b]),
                 "adamic_adar": float(inv_log_deg[nb[a] & nb[b]].sum())}
            if with_outcome:
                w = lab[t + 1: t + 1 + args.h, a, b].sum()
                f["future_co"] = int(w)
                f["filled"] = int(w >= args.k)
            rows.append(f)
    return rows


def matrix(rows):
    return np.array([[r[k] for k in FEATURES] for r in rows], float)


def matrix_of(rows, feats):
    return np.array([[r[k] for k in feats] for r in rows], float)


def fit(rows, feats=FEATURES):
    X, y = matrix_of(rows, feats), np.array([r["filled"] for r in rows])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return sc, m


def fit_models(train):
    y = np.array([r["filled"] for r in train])
    gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)
    gbm.fit(matrix(train), y)
    return {"learned": fit(train), "lr_pop_growth": fit(train, POP_GROWTH), "gbm": gbm}


def score(rows, name, models=None, rng=None):
    if name == "random":
        return rng.random(len(rows))
    if name == "learned":
        sc, m = models["learned"] if isinstance(models, dict) else models
        return m.decision_function(sc.transform(matrix(rows)))
    if name == "lr_pop_growth":
        sc, m = models[name]
        return m.decision_function(sc.transform(matrix_of(rows, POP_GROWTH)))
    if name == "gbm":
        return models[name].predict_proba(matrix(rows))[:, 1]
    return np.array([r[name] for r in rows])


def prec_at(y, s, k):
    k = min(k, len(y))
    return float(y[np.argsort(-s)[:k]].mean())


def boot_auc_diff(y, s1, s2, rng, R=BOOTSTRAP_RESAMPLES):
    n, out = len(y), []
    for _ in range(R):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def concept_boot_auc_diff(rows, y, s1, s2, n_concepts, rng, R=BOOTSTRAP_RESAMPLES):
    """Cluster bootstrap over concepts: resample concepts with replacement and weight each gap
    pair by the product of its two concepts' multiplicities (pairs sharing a concept move together)."""
    a = np.array([r["a"] for r in rows])
    b = np.array([r["b"] for r in rows])
    out = []
    for _ in range(R):
        cnt = np.bincount(rng.integers(0, n_concepts, n_concepts), minlength=n_concepts)
        w = cnt[a] * cnt[b]
        keep = w > 0
        if keep.sum() < 10 or y[keep].min() == y[keep].max():
            continue
        out.append(roc_auc_score(y[keep], s1[keep], sample_weight=w[keep])
                   - roc_auc_score(y[keep], s2[keep], sample_weight=w[keep]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def run(args):
    years, docs, occ, co, concepts = load()
    sem = semantic_sim(concepts)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lab = {"doc": None, "sentence": load.co_sent, "sentence_ns": load.co_sent_ns}[args.label]
    tables = {T: cutoff_table(T, years, docs, occ, co, sem, args, lab=lab)
              for T in range(FIRST_TRAIN, TEST_CUTOFFS[-1] + 1)}
    res = {"args": vars(args), "bootstrap": {
        "requested_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
        "aa_seed_rule": "test cutoff year (independent generator)", "n_concepts": len(concepts),
        "interval": "pointwise 95% percentile, no multiplicity adjustment",
        "refit_models": False, "resample_test_cutoffs": False,
        "concept_pair_weight": "multiplicity[a] * multiplicity[b]",
        "discard_draw_if": "fewer than 10 positive-weight pairs or only one outcome class"
    }, "cutoffs": {}}
    for T in TEST_CUTOFFS:
        train = [r for Tp in range(FIRST_TRAIN, T - args.h + 1) for r in tables[Tp]]
        model = fit_models(train)
        rows = tables[T]
        y = np.array([r["filled"] for r in rows])
        entry = {"gaps": len(rows), "filled_rate": float(y.mean()), "train_rows": len(train), "methods": {}}
        S = {nm: score(rows, nm, model, rng) for nm in SCORES}
        for nm, s in S.items():
            entry["methods"][nm] = {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
                                    "p@20": prec_at(y, s, 20), "open@20": prec_at(1 - y, -s, 20)}
        entry["ci_learned_minus_expected"] = boot_auc_diff(y, S["learned"], S["expected"], rng)
        entry["ci_learned_minus_recent"] = boot_auc_diff(y, S["learned"], S["recent_cos"], rng)
        entry["ci_cluster_learned_minus_expected"] = concept_boot_auc_diff(rows, y, S["learned"], S["expected"],
                                                                           len(concepts), rng)
        entry["ci_cluster_learned_minus_lrpg"] = concept_boot_auc_diff(rows, y, S["learned"], S["lr_pop_growth"],
                                                                       len(concepts), rng)
        # separate generator so that adding this comparison leaves the draws above unchanged
        entry["ci_cluster_learned_minus_aa"] = concept_boot_auc_diff(rows, y, S["learned"], S["adamic_adar"],
                                                                     len(concepts), np.random.default_rng(T))
        entry["coef"] = dict(zip(FEATURES, map(float, model["learned"][1].coef_[0])))
        # how the classic "largest expected but absent" ranking fares: top-decile fill rate
        e = S["expected"]
        entry["fill_rate_by_expected_quintile"] = [float(y[(e >= np.quantile(e, q)) & (e <= np.quantile(e, q + .2))].mean())
                                                   for q in (0, .2, .4, .6, .8)]
        res["cutoffs"][T] = entry
        print(T, json.dumps({k: (round(v["auc"], 3), round(v["ap"], 3)) for k, v in entry["methods"].items()}),
              f"gaps={len(rows)} fill={y.mean():.3f}", flush=True)
    # mean over cutoffs
    res["mean"] = {nm: {m: float(np.mean([res["cutoffs"][T]["methods"][nm][m] for T in TEST_CUTOFFS]))
                        for m in ("auc", "ap", "p@20", "open@20")} for nm in SCORES}
    return res, tables, (years, docs, occ, co, concepts, sem)


def survival(years, docs, occ, co, args, T0=2013):
    """Share of the gaps open at T0 that are filled (>= K co-occurrences) by each later year."""
    t = int(np.where(years == T0)[0][0])
    A, B = occ[: t + 1].sum(0), co[: t + 1].sum(0)
    C = len(A)
    pairs = [(a, b) for a in range(C) for b in range(a + 1, C)
             if A[a] >= args.min_occ and A[b] >= args.min_occ and B[a, b] <= args.g]
    cum = np.cumsum(co[t + 1:], axis=0)
    out = {}
    for j, y in enumerate(years[t + 1:]):
        out[int(y)] = float(np.mean([cum[j, a, b] >= args.k for a, b in pairs]))
    return {"T0": T0, "gaps": len(pairs), "filled_by_year": out}


def forecast(tables, data, args, top=12):
    """Train on every cutoff with a complete window, score the gaps open at the last year."""
    years, docs, occ, co, concepts, sem = data
    T = int(years[-1])
    rows = cutoff_table(T, years, docs, occ, co, sem, args, with_outcome=False)
    lab = {"doc": None, "sentence": load.co_sent, "sentence_ns": load.co_sent_ns}[args.label]
    for Tp in range(FIRST_TRAIN, T - args.h + 1):  # every cutoff whose outcome window has ended
        if not tables.get(Tp):
            tables[Tp] = cutoff_table(Tp, years, docs, occ, co, sem, args, lab=lab)
    # temporal Platt scaling did not improve calibration on the test cutoffs (extra.py), so the
    # forecast uses the model trained on every complete window and reports its raw closure score
    train = [r for Tp in range(FIRST_TRAIN, T - args.h + 1) for r in tables[Tp]]
    sc0, m0 = fit(train)
    p = m0.predict_proba(sc0.transform(matrix(rows)))[:, 1]
    order = np.argsort(-p)
    fmt = lambda i: {"pair": [concepts[rows[i]["a"]], concepts[rows[i]["b"]]], "p_fill": round(float(p[i]), 3),
                     "co_prior": rows[i]["co_prior"]}
    return {"T": T, "gaps": len(rows), "train_rows": len(train),
            "likely_filled": [fmt(i) for i in order[:top]],
            "persistent": [fmt(i) for i in order[::-1][:top]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-occ", type=int, default=50)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--label", choices=["doc", "sentence", "sentence_ns"], default="doc")
    ap.add_argument("--vocab-year", type=int, default=0,
                    help="only concepts already established (>= min-occ works) by this year")
    args = ap.parse_args()
    res, tables, data = run(args)
    res["survival"] = survival(*data[:4], args)
    for T in range(TEST_CUTOFFS[-1] + 1, int(data[0][-1]) + 1):
        tables[T] = []  # windows not complete, never used for training
    res["forecast"] = forecast(tables, data, args)
    out = ROOT / "outputs" / f"results_{args.tag}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("mean", json.dumps({k: round(v["auc"], 3) for k, v in res["mean"].items()}))
    print("wrote", out)
