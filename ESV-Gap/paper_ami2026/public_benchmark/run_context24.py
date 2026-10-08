"""Leave-one-paper-out retrieval on 42 expert-labelled Context24 examples.

No official test labels, model API, generated gold labels, or scope closure used.
"""
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT))
from src.context_retrieval import context_prior, retrieve_context, word_overlap

METHODS = ("bm25", "tfidf", "prior_only", "bm25_diverse", "context", "context_diverse")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    data = BASE / "context24"
    claims_path = data / "task2-train-dev.json"
    texts_path = data / "full_texts-2024-04-25-update.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    sources = json.loads(texts_path.read_text(encoding="utf-8"))
    if len(claims) != 42 or len({r["id"] for r in claims}) != len(claims):
        raise ValueError("Unexpected labelled example set")
    predictions = {method: [] for method in METHODS}
    rows, priors, missing = [], {}, []
    for claim in claims:
        key = claim["citekey"]
        training = [row for row in claims if row["citekey"] != key]
        assert all(row["citekey"] != key for row in training)
        if key not in priors:
            priors[key] = context_prior(training, sources)
        source = sources.get(key, "")
        if not source:
            missing.append(claim["id"])
        row = {"claim_id": claim["id"], "citekey": key, "dataset": claim["dataset"],
               "training_claims": len(training), "training_papers": len({r["citekey"] for r in training}),
               "held_paper_excluded_from_prior": True, "metrics": {}}
        for method in METHODS:
            spans = retrieve_context(source, claim["claim"], prior=priors[key], method=method)
            assert sum(span["word_count"] for span in spans) <= 512
            assert all(source[s["start"]:s["end"]] == s["text"] and not s["reviewed"] for s in spans)
            prediction = " ".join(span["text"] for span in spans)
            row["metrics"][method] = word_overlap(prediction, " ".join(claim["context"]))
            row["metrics"][method]["output_words"] = sum(s["word_count"] for s in spans)
            predictions[method].append({"id": claim["id"], "citekey": key,
                                        "context": [span["text"] for span in spans],
                                        "source_spans": spans,
                                        "source_text_sha256": hashlib.sha256(source.encode()).hexdigest()})
        rows.append(row)
    names = ("rouge1_f1", "rouge2_f1", "rougel_f1", "unigram_precision", "unigram_recall", "output_words")
    def aggregate(selected):
        return {method: {name: sum(r["metrics"][method][name] for r in selected)/len(selected)
                         for name in names} for method in METHODS}
    paper_differences = defaultdict(list)
    for row in rows:
        paper_differences[row["citekey"]].append(row["metrics"]["context_diverse"]["rougel_f1"]-
                                                row["metrics"]["bm25"]["rougel_f1"])
    differences = [sum(values)/len(values) for _, values in sorted(paper_differences.items())]
    rng = random.Random(20261001)
    samples = sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(2000))
    report = {"status": "exploratory_leave_one_paper_out_expert_grounding_context_retrieval",
              "interpretation": "Expert labels are Context24 method-context quotes; scores are lexical retrieval proxies, not scope closure or gap novelty.",
              "claims": len(rows), "papers": len(paper_differences), "missing_text_claims": missing,
              "domains": dict(Counter(row["dataset"] for row in rows)),
              "word_budget": 512, "window_size": 96, "stride": 80,
              "same_budget_and_source_span_checks_passed": True,
              "held_paper_prior_exclusion_checks_passed": True,
              "macro_metrics": aggregate(rows),
              "domain_metrics": {domain: aggregate([r for r in rows if r["dataset"] == domain])
                                 for domain in sorted({r["dataset"] for r in rows})},
              "paper_macro_paired_rougel_delta_vs_bm25": sum(differences)/len(differences),
              "paired_paper_bootstrap_95_percentile": [samples[49], samples[1949]],
              "bootstrap_replicates": 2000, "seed": 20261001,
              "protocol_sha256": digest(BASE/"CONTEXT24_PROTOCOL.md"),
              "input_sha256": {"claims": digest(claims_path), "texts": digest(texts_path)},
              "code_sha256": {"retriever": digest(PROJECT/"src/context_retrieval.py"), "runner": digest(__file__)},
              "dataset_revision": "457d3b5cb4bb8ade34e37458f4900c6eae0959bb",
              "python": platform.python_version(), "elapsed_seconds": time.monotonic()-start,
              "official_test_evaluation": False, "new_human_annotation_performed": False,
              "scope_controller_accuracy_measured": False, "rows": rows}
    output = BASE / "results"
    output.mkdir(exist_ok=True)
    for method, values in predictions.items():
        (output/f"predictions_{method}.json").write_text(json.dumps(values,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"context24_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["claims","papers","macro_metrics","paper_macro_paired_rougel_delta_vs_bm25","paired_paper_bootstrap_95_percentile","elapsed_seconds"]},indent=2))


if __name__ == "__main__":
    main()
