"""Certified gap declaration under corpus incompleteness (see THEORY.md).

Pools both SciFact splits (scored with the same verifier) and repeats R random 50/50
calibration/test splits.  In every repetition a global deletion pattern removes each gold
witness document with probability p_true.  Thresholds compared:

  f1-frozen  : macro-F1-optimal threshold on the complete calibration half (usual practice)
  conformal  : split-conformal threshold on the complete calibration half (assumes no gaps)
  in-situ    : conformal, calibration questions scored on the same incomplete corpus (Thm 1)
  transfer   : conformal on the complete half with simulated deletion at p_hat, where p_hat is
               a Clopper-Pearson upper bound from m sampled reference documents (Thm 2 + Cor.)
  worst-case : transfer with p_hat = 1

All scores are monotone max-over-available-documents scores, computed from cached Stage-1
results, so the score under any deletion set is the best surviving entry of a ranked list.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from common import OUT, write_json
from evaluate import best_threshold, load
from external import CACHE, external_scores
from methods import Ctx, content_terms, scope_factor

P_TRUE = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


# ---------------------------------------------------------------- ranked lists
def ranked_lists(ctx: Ctx, cid) -> dict[str, list[tuple[str, float]]]:
    row = ctx.rows[cid]
    out = {"BM25": [tuple(x) for x in row["bm25_top"]],
           "Dense": [tuple(x) for x in row["dense_top"]]}
    q = content_terms(ctx.claims[cid]["claim"])
    cnt: dict[str, int] = {}
    for t in q:
        for d in ctx.inv.get(t, ()):
            cnt[d] = cnt.get(d, 0) + 1
    out["CoMention"] = sorted(((d, n / len(q)) for d, n in cnt.items()), key=lambda x: -x[1])[:200] if q else []
    nli, esv = [], []
    for d, _ in row["hybrid_top"]:
        sents = row["nli"].get(d, [])
        if not sents:
            continue
        nli.append((d, max(max(pe, pc) for pe, pc, _ in sents)))
        esv.append((d, max(max(pe, pc) * scope_factor(ctx, cid, d, si) for si, (pe, pc, _) in enumerate(sents))))
    out["NLI-verify"] = sorted(nli, key=lambda x: -x[1])
    out["ESV-Scope"] = sorted(esv, key=lambda x: -x[1])
    return out


def score(lst, removed) -> float:
    for d, s in lst:
        if d not in removed:
            return s
    return 0.0


# ---------------------------------------------------------------- statistics
def binom_cdf(x, m, p):
    return sum(math.comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(x + 1))


def cp_upper(x, m, delta):
    """One-sided (1-delta) Clopper-Pearson upper bound for a binomial proportion."""
    if x >= m:
        return 1.0
    lo, hi = x / m, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(x, m, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def conformal_tau(closed_scores, alpha):
    s = np.sort(closed_scores)
    k = int(math.floor(alpha * (len(s) + 1)))
    return s[k - 1] if k >= 1 else -np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_large")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--m", type=int, default=100, help="reference documents sampled to estimate p")
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--splits", default="calib,test", help="comma-separated SciFact splits to load")
    ap.add_argument("--protocol", choices=["pooled", "holdout"], default="holdout",
                    help="holdout: calibrate on a random half of the first split, evaluate on the whole second split")
    ap.add_argument("--deletion", choices=["mcar", "targeted"], default="mcar",
                    help="targeted: stronger witnesses are more likely to be missing (not at random)")
    args = ap.parse_args()

    lists, closed, wit, ext, part = [], [], [], [], []
    splits = args.splits.split(",")
    use_ext = all((CACHE / f"nli_{s}{args.tag}.json").exists() for s in splits)
    for split in splits:
        ctx, claims = load(split, args.tag)
        es = external_scores(split, args.tag) if use_ext else {}
        if use_ext:  # keep only questions that have external results (API budget may cut some)
            have = json.load(open(CACHE / f"nli_{split}{args.tag}.json", encoding="utf-8"))
            dropped = [c["id"] for c in claims if str(c["id"]) not in have]
            claims = [c for c in claims if str(c["id"]) in have]
            print(f"{split}: {len(claims)} questions with external evidence, {len(dropped)} dropped", flush=True)
        bge_path = OUT / "scores" / f"bge_{split}.json"
        bge = json.load(open(bge_path, encoding="utf-8")) if bge_path.exists() else {}
        for c in claims:
            lists.append(ranked_lists(ctx, c["id"]))
            if bge:  # modern retrieval baselines (BGE-large dense, BGE cross-encoder reranker)
                b = bge[str(c["id"])]
                lists[-1]["BGE-dense"] = [tuple(x) for x in b["bge_dense_top"]]
                lists[-1]["BGE-rerank"] = sorted((tuple(x) for x in b["bge_rerank"]), key=lambda x: -x[1])
            ext.append(es.get(str(c["id"]), 0.0))
            closed.append(c["status"] == "CLOSED")
            wit.append(list(c["witness"]))
            part.append(split)
    closed = np.array(closed)
    part = np.array(part)
    n = len(closed)
    methods = list(lists[0])
    if use_ext:
        # external evidence (OpenAlex) is unaffected by local deletion: max(local, ext) stays monotone
        methods += [f"{m}+OA" for m in ("CoMention", "Dense", "ESV-Scope", "BGE-rerank") if m in lists[0]]
    ext = np.array(ext)
    all_wit = sorted({d for w in wit for d in w})
    widx = {d: i for i, d in enumerate(all_wit)}
    print(f"{n} questions ({closed.sum()} CLOSED), {len(all_wit)} witness docs, methods={methods}", flush=True)

    def scores(name, idx, removed):
        if name.endswith("+OA"):
            base = name[:-3]
            return np.array([max(score(lists[i][base], removed), ext[i]) for i in idx])
        return np.array([score(lists[i][name], removed) for i in idx])

    # deletion probabilities per witness document
    if args.deletion == "targeted":
        strength = {}
        for i, w in enumerate(wit):
            local = dict(lists[i]["ESV-Scope"])
            for d in w:
                strength[d] = max(strength.get(d, 0.0), local.get(d, 0.0))
        wgt = np.array([strength[d] + 0.05 for d in all_wit])
        wgt = wgt / wgt.mean()
    else:
        wgt = np.ones(len(all_wit))
    cal_pool = np.where(part == splits[0])[0]
    eval_idx = np.where(part == splits[-1])[0]

    rng = np.random.default_rng(0)
    res = {"alpha": args.alpha, "delta": args.delta, "m": args.m, "R": args.R, "n": int(n), "grid": {}}
    for p in P_TRUE:
        acc = {}
        for r in range(args.R):
            if args.protocol == "holdout":
                A, B = rng.permutation(cal_pool)[: len(cal_pool) // 2], eval_idx
            else:
                perm = rng.permutation(n)
                A, B = perm[: n // 2], perm[n // 2:]
            Ac, Bc, Bo = A[closed[A]], B[closed[B]], B[~closed[B]]
            gone = rng.random(len(all_wit)) < np.minimum(1.0, p * wgt)  # the real incomplete corpus
            removed = frozenset(d for d, g in zip(all_wit, gone) if g)
            # p_hat: sample m reference documents among the calibration witnesses, count missing
            ref = rng.choice([widx[d] for i in Ac for d in wit[i]], size=args.m, replace=True)
            p_hat = cp_upper(int(gone[ref].sum()), args.m, args.delta)
            sim_hat = frozenset(d for d in all_wit if rng.random() < p_hat)
            for name in methods:
                full_A = scores(name, A, frozenset())
                taus = {
                    "f1-frozen": best_threshold(full_A, closed[A].astype(int)),
                    "conformal": conformal_tau(scores(name, Ac, frozenset()), args.alpha),
                    "in-situ": conformal_tau(scores(name, Ac, removed), args.alpha),
                    "transfer": conformal_tau(scores(name, Ac, sim_hat), args.alpha),
                    "worst-case": conformal_tau(scores(name, Ac, frozenset(all_wit)), args.alpha),
                }
                sBc, sBo = scores(name, Bc, removed), scores(name, Bo, frozenset())
                if name.endswith("+OA"):  # external calls are only needed when the local score is below tau
                    loc = np.concatenate([scores(name[:-3], Bc, removed), scores(name[:-3], Bo, frozenset())])
                for rule, tau in taus.items():
                    a = acc.setdefault(f"{name}|{rule}", {"fnov": [], "power": [], "p_hat": [], "ext_calls": [],
                                                          "true_rate": []})
                    a["true_rate"].append(float(gone.mean()))
                    a["fnov"].append(float(np.mean(sBc < tau)))
                    a["power"].append(float(np.mean(sBo < tau)))
                    a["p_hat"].append(p_hat)
                    a["ext_calls"].append(float(np.mean(loc < tau)) if name.endswith("+OA") else 0.0)
        res["grid"][str(p)] = {k: {"fnov": float(np.mean(v["fnov"])),
                                   "fnov_q95": float(np.quantile(v["fnov"], 0.95)),
                                   "power": float(np.mean(v["power"])),
                                   "power_sd": float(np.std(v["power"])),
                                   "p_hat": float(np.mean(v["p_hat"])),
                                   "ext_calls": float(np.mean(v["ext_calls"])),
                                   "true_rate": float(np.mean(v["true_rate"]))} for k, v in acc.items()}
        print(f"p_true={p}: mean p_hat={np.mean(acc[f'{methods[0]}|transfer']['p_hat']):.3f}", flush=True)
    res.update(splits=splits, protocol=args.protocol, deletion=args.deletion,
               n_calibration_pool=int(len(cal_pool)), n_eval=int(len(eval_idx)))
    suffix = ("" if splits == ["calib", "test"] else "_" + "_".join(splits)) + f"_{args.protocol}_{args.deletion}"
    write_json(OUT / "results" / f"certify{args.tag}{suffix}.json", res)

    rules = ["f1-frozen", "conformal", "in-situ", "transfer", "worst-case"]
    for name in methods:
        print(f"\n{name}  (false novelty / power; target false novelty <= {args.alpha})")
        print(f"  {'rule':11s} " + " ".join(f"p={p:<11}" for p in P_TRUE))
        for rule in rules:
            g = [res["grid"][str(p)][f"{name}|{rule}"] for p in P_TRUE]
            print(f"  {rule:11s} " + " ".join(f"{x['fnov']:.3f}/{x['power']:.3f}  " for x in g))


if __name__ == "__main__":
    main()
