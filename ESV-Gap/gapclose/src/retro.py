"""Retrospective gap certification: natural incompleteness from time.

At a cutoff year Y the system sees only documents published up to Y, in the local corpus and in the
external OpenAlex results (results published after Y are discarded, so the papers that later answer a
question cannot leak).  Ground truth comes from the future:

  CLOSED at Y      : a SciFact question with a witness published <= Y
  FILLED gap at Y  : a SciFact question whose witnesses were all published after Y (open at Y, answered later)
  OPEN (NEI)       : a SciFact question with no witness in the corpus (open at Y as far as we know)

Calibration uses a random half of the CLOSED-at-Y and NEI questions (status known at time Y); the
evaluation set is the other half plus every FILLED gap. Rules: F1-optimal threshold and in-situ
conformal (Theorem 1).  Writes <OUT>/results/retro.json and gap dossiers for one run.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from certify import conformal_tau, ranked_lists, score
from common import OUT, content_terms, write_json
from evaluate import best_threshold, load
from external import CACHE


def ext_lists(split):
    """Per question: [(openalex_id, score, year, sentence)] from cached OpenAlex results, best sentence per work."""
    claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
    cache = json.load(open(CACHE / f"openalex_{split}.json", encoding="utf-8"))
    nli = json.load(open(CACHE / f"nli_{split}_large.json", encoding="utf-8"))
    out = {}
    for c in claims:
        k = str(c["id"])
        q = content_terms(c["claim"])
        lst = []
        for di, d in enumerate(cache.get(k, {}).get("docs", [])):
            tt = content_terms(d["title"])
            best = (0.0, "")
            for si, (pe, pc, _) in enumerate(nli.get(k, {}).get(str(di), [])):
                cov = len(q & (content_terms(d["sentences"][si]) | tt)) / len(q) if q else 1.0
                v = max(pe, pc) * cov
                if v > best[0]:
                    best = (v, d["sentences"][si])
            lst.append((d["id"], best[0], d.get("year"), best[1], d["title"]))
        out[k] = sorted(lst, key=lambda x: -x[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2008,2010,2012")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--R", type=int, default=500)
    args = ap.parse_args()
    meta = json.load(open(OUT / "bench" / "doc_meta.json", encoding="utf-8"))
    year = {d: m["year"] for d, m in meta.items()}

    Q, lists, ext, ctxs = [], [], [], {}
    for split in ("calib", "test"):
        ctx, claims = load(split, "_large")
        ctxs[split] = ctx
        bge = json.load(open(OUT / "scores" / f"bge_{split}.json", encoding="utf-8"))
        llm_p = OUT / "scores" / f"llm_{split}.json"
        llm = json.load(open(llm_p, encoding="utf-8")) if llm_p.exists() else {}
        el = ext_lists(split)
        for c in claims:
            if c["status"] == "CLOSED" and any(year.get(d) is None for d in c["witness"]):
                continue  # witness year unknown: cannot place the question in time
            L = ranked_lists(ctx, c["id"])
            L["BGE-rerank"] = sorted((tuple(x) for x in bge[str(c["id"])]["bge_rerank"]), key=lambda x: -x[1])
            if llm:
                L["LLM-judge"] = sorted((tuple(x) for x in llm[str(c["id"])]), key=lambda x: -x[1])
            Q.append({"split": split, **c})
            lists.append(L)
            ext.append(el[str(c["id"])])
    first = np.array([min(year[d] for d in c["witness"]) if c["status"] == "CLOSED" else 10 ** 4 for c in Q])
    nei = np.array([c["status"] == "OPEN" for c in Q])
    base = [m for m in ("Dense", "BGE-rerank", "LLM-judge", "ESV-Scope") if m in lists[0]]
    methods = base + [f"{m}+OA" for m in ("BGE-rerank", "ESV-Scope")]
    print(f"{len(Q)} questions placed in time; methods {methods}", flush=True)

    inv_cache = {}

    def s_at(i, m, Y):
        if Y not in inv_cache:
            inv_cache[Y] = frozenset(d for d, y in year.items() if y is None or y > Y)
        invisible = inv_cache[Y]
        loc = score(lists[i][m.replace("+OA", "")], invisible)
        if m.endswith("+OA"):
            e = max([v for _, v, y, _, _ in ext[i] if y is not None and y <= Y], default=0.0)
            return max(loc, e)
        return loc

    rng = np.random.default_rng(0)
    res = {"alpha": args.alpha, "R": args.R, "grid": {}}
    dossiers = {}
    for Y in [int(y) for y in args.years.split(",")]:
        closed_y = np.where(first <= Y)[0]
        filled = np.where((first > Y) & (first < 10 ** 4))[0]
        open_nei = np.where(nei)[0]
        S = {m: np.array([s_at(i, m, Y) for i in range(len(Q))]) for m in methods}
        acc = {}
        for r in range(args.R):
            pc, po = rng.permutation(closed_y), rng.permutation(open_nei)
            cal_c, ev_c = pc[: len(pc) // 2], pc[len(pc) // 2:]
            cal_o, ev_o = po[: len(po) // 2], po[len(po) // 2:]
            for m in methods:
                s = S[m]
                ycal = np.r_[np.ones(len(cal_c)), np.zeros(len(cal_o))].astype(int)
                taus = {"f1-frozen": best_threshold(np.r_[s[cal_c], s[cal_o]], ycal),
                        "in-situ": conformal_tau(s[cal_c], args.alpha)}
                for rule, tau in taus.items():
                    a = acc.setdefault(f"{m}|{rule}", {"fn": [], "pow_filled": [], "pow_nei": [], "prec": []})
                    dec_c, dec_f, dec_o = s[ev_c] < tau, s[filled] < tau, s[ev_o] < tau
                    a["fn"].append(float(dec_c.mean()))
                    a["pow_filled"].append(float(dec_f.mean()) if len(filled) else float("nan"))
                    a["pow_nei"].append(float(dec_o.mean()))
                    n_dec = dec_c.sum() + dec_f.sum() + dec_o.sum()
                    a["prec"].append(float((dec_f.sum() + dec_o.sum()) / n_dec) if n_dec else float("nan"))
                    if r == 0 and rule == "in-situ" and m == "ESV-Scope+OA":
                        dossiers[str(Y)] = {"tau": float(tau), "items": [int(i) for i in filled[dec_f]]}
        res["grid"][str(Y)] = {"n_closed": int(len(closed_y)), "n_filled": int(len(filled)), "n_nei": int(len(open_nei)),
                               "visible_share": float(np.mean([y is not None and y <= Y for y in year.values()])),
                               "methods": {k: {kk: float(np.nanmean(vv)) for kk, vv in v.items()} for k, v in acc.items()}}
        g = res["grid"][str(Y)]
        print(f"\nY={Y}: closed {g['n_closed']}, filled gaps {g['n_filled']}, NEI {g['n_nei']}, visible {g['visible_share']:.2f}")
        for k, v in g["methods"].items():
            print(f"  {k:24s} FN={v['fn']:.3f} power(filled)={v['pow_filled']:.3f} power(NEI)={v['pow_nei']:.3f} precision={v['prec']:.3f}")
    write_json(OUT / "results" / "retro.json", res)

    # ------------------------------------------------------------ gap dossiers (Y = middle cutoff, run 0)
    out = {}
    for Ys, d in dossiers.items():
        Y = int(Ys)
        items = []
        for i in d["items"]:
            c = Q[i]
            ctx = ctxs[c["split"]]
            invisible = frozenset(dd for dd, y in year.items() if y is None or y > Y)
            # nearest visible local evidence (ESV sentence level)
            best = (0.0, None)
            row = ctx.rows[c["id"]]
            from methods import scope_factor
            for dd, _ in row["hybrid_top"]:
                if dd in invisible:
                    continue
                for si, (pe, pc, _) in enumerate(row["nli"].get(dd, [])):
                    v = max(pe, pc) * scope_factor(ctx, c["id"], dd, si)
                    if v > best[0]:
                        best = (v, (dd, si))
            near_ext = max(([x for x in ext[i] if x[2] is not None and x[2] <= Y]), key=lambda x: x[1], default=None)
            filler = sorted(c["witness"], key=lambda dd: year[dd])[0]
            items.append({
                "question": c["claim"], "cutoff_year": Y, "tau": round(d["tau"], 3),
                "nearest_local": None if best[1] is None else {
                    "title": ctx.corpus[best[1][0]]["title"], "year": year.get(best[1][0]), "doi": meta[best[1][0]]["doi"],
                    "sentence": ctx.corpus[best[1][0]]["sentences"][best[1][1]], "score": round(best[0], 3)},
                "nearest_external": None if near_ext is None else {
                    "openalex": near_ext[0], "title": near_ext[4], "year": near_ext[2], "sentence": near_ext[3],
                    "score": round(near_ext[1], 3)},
                "filled_by": {"title": ctx.corpus[filler]["title"], "year": year[filler], "doi": meta[filler]["doi"],
                              "rationale": [ctx.corpus[filler]["sentences"][k] for k in c["witness"][filler]],
                              "label": c["evidence_label"]},
                "lag_years": year[filler] - Y})
        out[Ys] = items
        lags = [x["lag_years"] for x in items]
        if lags:
            print(f"dossiers Y={Ys}: {len(items)} certified filled gaps, lag median {np.median(lags):.0f} years")
    write_json(OUT / "results" / "gap_dossiers.json", out)


if __name__ == "__main__":
    main()
