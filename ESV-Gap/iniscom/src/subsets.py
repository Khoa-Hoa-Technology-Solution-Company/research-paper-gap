"""Lexicon sensitivity: rerun the main experiment on random subsets of the concept lexicon.

Each subset keeps a fixed share of the concepts (drawn with a fixed seed). The corpus is held
fixed, the count tensors are restricted to the selected concepts, and graph features, candidate
gaps and models are recomputed. The best single score is selected retrospectively within each
subset as a descriptive comparator. This is vocabulary sensitivity, not external validation.

    python src/subsets.py [--share 0.67] [--n 20]
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import gaps
from gaps import FIRST_TRAIN, ROOT, TEST_CUTOFFS

SINGLE = ["expected", "growth_min", "profile_cos", "recent_cos", "adamic_adar", "semantic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=2 / 3)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--min-occ", type=int, default=50)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--h", type=int, default=3)
    args = ap.parse_args()
    if not 0 < args.share <= 1 or args.n < 1:
        ap.error("--share must be in (0, 1] and --n must be positive")
    years, docs, occ, co, concepts = gaps.load()
    sem = gaps.semantic_sim(concepts)
    C = len(concepts)
    rng = np.random.default_rng(args.seed)
    runs = []
    for i in range(args.n):
        sub = np.sort(rng.choice(C, int(round(args.share * C)), replace=False))
        o, c, s = occ[:, sub], co[:, sub][:, :, sub], sem[np.ix_(sub, sub)]
        tables = {T: gaps.cutoff_table(T, years, docs, o, c, s, args) for T in range(FIRST_TRAIN, TEST_CUTOFFS[-1] + 1)}
        learned, reduced, single, n_gaps, per_cutoff = [], [], {k: [] for k in SINGLE}, [], {}
        for T in TEST_CUTOFFS:
            train = [r for Tp in range(FIRST_TRAIN, T - args.h + 1) for r in tables[Tp]]
            rows = tables[T]
            y = np.array([r["filled"] for r in rows])
            if len(np.unique(y)) < 2:
                raise ValueError(f"Subset {i}, cutoff {T}: AUC needs both outcome classes")
            n_gaps.append(len(rows))
            learned.append(roc_auc_score(y, gaps.score(rows, "learned", gaps.fit(train))))
            reduced_models = {"lr_pop_growth": gaps.fit(train, gaps.POP_GROWTH)}
            reduced.append(roc_auc_score(y, gaps.score(rows, "lr_pop_growth", reduced_models)))
            for k in SINGLE:
                single[k].append(roc_auc_score(y, gaps.score(rows, k)))
            per_cutoff[T] = {"gaps": len(rows), "filled": int(y.sum()), "train_rows": len(train),
                             "last_train_cutoff": T - args.h,
                             "learned": float(learned[-1]), "lr_pop_growth": float(reduced[-1]),
                             "single_auc": {k: float(single[k][-1]) for k in SINGLE}}
        best = max(SINGLE, key=lambda k: np.mean(single[k]))
        runs.append({"concepts": int(len(sub)), "concept_ids": sub.tolist(),
                     "concept_names": [str(concepts[j]) for j in sub], "cutoffs": per_cutoff,
                     "mean_gaps": float(np.mean(n_gaps)), "learned": float(np.mean(learned)),
                     "lr_pop_growth": float(np.mean(reduced)),
                     "best_single": float(np.mean(single[best])), "best_single_name": best})
        print(i, json.dumps({k: v for k, v in runs[-1].items()
                            if k not in ("concept_ids", "concept_names", "cutoffs")}), flush=True)
    L = np.array([r["learned"] for r in runs])
    B = np.array([r["best_single"] for r in runs])
    P = np.array([r["lr_pop_growth"] for r in runs])
    out = {"share": args.share, "n": args.n, "seed": args.seed, "args": vars(args), "runs": runs,
           "learned_median": float(np.median(L)), "learned_min": float(L.min()), "learned_max": float(L.max()),
           "best_median": float(np.median(B)), "best_min": float(B.min()), "best_max": float(B.max()),
           "lrpg_median": float(np.median(P)), "lrpg_min": float(P.min()), "lrpg_max": float(P.max()),
           "learned_minus_best_min": float((L - B).min()), "learned_minus_best_max": float((L - B).max()),
           "learned_minus_lrpg_median": float(np.median(L - P)),
           "learned_minus_lrpg_min": float((L - P).min()), "learned_minus_lrpg_max": float((L - P).max()),
           "learned_wins_lrpg": int((L > P).sum()),
           "learned_wins": int((L > B).sum()), "gap_min": float(min(r["mean_gaps"] for r in runs)),
           "gap_max": float(max(r["mean_gaps"] for r in runs))}
    json.dump(out, open(ROOT / "outputs" / "subsets.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
