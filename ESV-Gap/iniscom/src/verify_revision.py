"""Check analysis artifacts used by the revised submission against the fixed snapshot.

    python src/verify_revision.py

The full bootstrap and subset experiments are run by gaps.py and subsets.py. This check
independently recomputes non-random scores and verifies temporal boundaries, subset draws,
case-study membership and reference metadata. It does not validate scientific label precision.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import gaps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def main():
    result = read("results_main.json")
    args = argparse.Namespace(**result["args"])
    years, docs, occ, co, concepts = gaps.load()
    sem = gaps.semantic_sim(concepts)
    assert len(concepts) == 90 and int(docs.sum()) == 194221
    assert np.array_equal(years, np.arange(2010, 2026))
    assert np.array_equal(co, co.transpose(0, 2, 1))
    assert np.array_equal(np.diagonal(co, axis1=1, axis2=2), occ)
    assert np.all(gaps.load.co_sent <= co)
    tables = {T: gaps.cutoff_table(T, years, docs, occ, co, sem, args)
              for T in range(gaps.FIRST_TRAIN, gaps.TEST_CUTOFFS[-1] + 1)}
    checks = []
    for T in gaps.TEST_CUTOFFS:
        cutoffs = list(range(gaps.FIRST_TRAIN, T - args.h + 1))
        assert max(cutoffs) + args.h == T
        train = [r for t in cutoffs for r in tables[t]]
        rows = tables[T]
        saved = result["cutoffs"][str(T)]
        assert len(train) == saved["train_rows"] and len(rows) == saved["gaps"]
        y = np.array([r["filled"] for r in rows])
        close(y.mean(), saved["filled_rate"])
        models = gaps.fit_models(train)
        if T == 2022:
            sc_l, m_l = models["learned"]
            X_22 = sc_l.transform(gaps.matrix(rows))
            w_22 = m_l.coef_[0]
            fp = {tuple( (concepts[r["a"]], concepts[r["b"]]) ): {
                f: round(float(xi * wi), 2) for f, xi, wi in zip(gaps.FEATURES, x, w_22) if f != "co_prior"
            } for r, x in zip(rows, X_22)}
            assert fp[("intrusion detection", "semantic communication")]["profile_cos"] == 1.93
            assert fp[("intrusion detection", "semantic communication")]["growth_max"] == 1.57
            assert fp[("intrusion detection", "semantic communication")]["expected"] == -1.07
            assert fp[("RIS", "wearable")]["growth_max"] == 1.85
            assert fp[("RIS", "wearable")]["expected"] == 1.20
        for name in (n for n in gaps.SCORES if n != "random"):
            score = gaps.score(rows, name, models)
            metrics = saved["methods"][name]
            close(roc_auc_score(y, score), metrics["auc"])
            close(average_precision_score(y, score), metrics["ap"])
            close(gaps.prec_at(y, score, 20), metrics["p@20"])
            close(gaps.prec_at(1 - y, -score, 20), metrics["open@20"])
        # Perturb all future-year counts: candidate features and completed training
        # windows must remain unchanged. Retrospective vocabulary/embeddings stay fixed.
        future = years > T
        docs2, occ2, co2 = docs.copy(), occ.copy(), co.copy()
        docs2[future] *= 3
        occ2[future] *= 2
        co2[future] *= 2
        assert gaps.cutoff_table(T, years, docs2, occ2, co2, sem, args, with_outcome=False) == [
            {k: v for k, v in row.items() if k not in ("future_co", "filled")} for row in rows]
        for t in cutoffs:
            assert gaps.cutoff_table(t, years, docs2, occ2, co2, sem, args) == tables[t]
        checks.append({"cutoff": T, "gaps": len(rows), "train_rows": len(train),
                       "last_training_cutoff": max(cutoffs), "last_training_outcome_year": T,
                       "recomputed_nonrandom_methods": len(gaps.SCORES) - 1,
                       "future_count_perturbation": "passed"})
    for name, metrics in result["mean"].items():
        for metric, value in metrics.items():
            close(value, np.mean([result["cutoffs"][str(t)]["methods"][name][metric]
                                  for t in gaps.TEST_CUTOFFS]))

    subsets = read("subsets.json")
    rng = np.random.default_rng(subsets["seed"])
    for run in subsets["runs"]:
        expected = np.sort(rng.choice(len(concepts), run["concepts"], replace=False))
        assert expected.tolist() == run["concept_ids"]
        assert [str(concepts[i]) for i in expected] == run["concept_names"]
        assert len(set(run["concept_ids"])) == 60
        for T, saved in run["cutoffs"].items():
            assert 0 < saved["filled"] < saved["gaps"]
            assert saved["last_train_cutoff"] + args.h == int(T)
        close(run["learned"], np.mean([v["learned"] for v in run["cutoffs"].values()]))
        close(run["lr_pop_growth"], np.mean([v["lr_pop_growth"] for v in run["cutoffs"].values()]))
        means = {k: np.mean([v["single_auc"][k] for v in run["cutoffs"].values()])
                 for k in next(iter(run["cutoffs"].values()))["single_auc"]}
        assert run["best_single_name"] == max(means, key=means.get)
        close(run["best_single"], max(means.values()))
    assert len(subsets["runs"]) == subsets["n"] == 20
    close(subsets["learned_median"], np.median([r["learned"] for r in subsets["runs"]]))

    evidence = read("example_evidence.json")
    assert evidence["unique_corpus_works"] == int(docs.sum())
    counts = []
    records = {}
    for group, expected in zip(evidence["pairs"], [(26, 23), (28, 15), (16, 12)]):
        assert group["prior_count"] == len(group["prior_records"]) == 0
        assert len(group["window_records"]) == expected[0] == group["window_count"]
        assert sum(w["same_sentence"] for w in group["window_records"]) == expected[1]
        records.update({w["id"]: w for w in group["window_records"]})
        counts.append({"pair": group["pair"], "joint_mentions": expected[0], "same_sentence": expected[1]})
    refs = read("reference_metadata.json")["records"]
    bib = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    for ref in refs:
        assert ref["openalex_id"] in records and "error" not in ref
        metadata = ref["metadata"]
        close_title = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        assert close_title(metadata["title"][0]) == close_title(records[ref["openalex_id"]]["title"])
        assert metadata["DOI"].lower() in bib.lower()
        assert ref["bibkey"] in tex and ref["openalex_id"] in tex
    defined = set(re.findall(r"@\w+\s*\{\s*([^,]+)", bib))
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    assert cited <= defined
    assert "We release the benchmark" not in tex and "leakage-free" not in tex.lower()

    tracked = [ROOT / "data" / "index.npz", ROOT / "data" / "semantic.npy", ROOT / "src" / "lexicon.py",
               ROOT / "src" / "gaps.py", ROOT / "src" / "subsets.py", ROOT / "src" / "audit_examples.py",
               ROOT / "src" / "verify_revision.py", ROOT / "paper" / "main.tex", ROOT / "paper" / "references.bib",
               OUT / "results_main.json", OUT / "subsets.json", OUT / "example_evidence.json",
               OUT / "reference_metadata.json"]
    report = {"status": "passed", "snapshot_works": int(docs.sum()), "concepts": len(concepts),
              "cutoffs": checks, "subset_draws_verified": len(subsets["runs"]),
              "case_counts": counts, "corpus_references_verified": len(refs),
              "citation_keys_defined": len(cited),
              "aa_positive_pointwise_intervals": sum(result["cutoffs"][str(t)]["ci_cluster_learned_minus_aa"][0] > 0
                                                      for t in gaps.TEST_CUTOFFS),
              "scope": "Consistency and count separation; not independent label or utility validation",
              "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}}
    (OUT / "revision_analysis_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
