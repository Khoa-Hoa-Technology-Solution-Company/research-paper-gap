"""Check hashes and aggregate arithmetic only; not full-pipeline reproduction."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def equal(value, expected, message):
    require(math.isclose(value, expected, abs_tol=0.00005), message)


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        require(path.is_relative_to(ROOT), "Manifest path escapes bundle")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"],
                "Hash mismatch: " + row["path"])

    replay = load("multidomain_summary.json")
    sources = replay["sources"]
    folds = [r for d in sources for r in d["cutoffs"] if r["status"] == "completed"]
    scored = [r for r in folds if r["positive_controls"]]
    require(len(sources) == replay["domain_count"] == 4, "Domain count")
    require(sum(d["retained_papers"] for d in sources) == replay["source_rows"] == 378,
            "Record count")
    require(sum(d["triples"] for d in sources) == 3188, "Triple count")
    require(len(folds) == replay["completed_domain_cutoffs"] == 10, "Fold count")
    require(len(scored) == replay["cutoffs_with_positive_controls"] == 9, "Scored count")
    require(sum(r["positive_controls"] for r in folds) ==
            replay["total_positive_controls_across_cutoffs_not_independent"] == 64,
            "Control count")
    require(sum(r["candidates"] for r in folds) ==
            replay["total_candidate_rows_across_cutoffs_not_independent"] == 39,
            "Candidate count")
    equal(sum(r["recall_at_20"] for r in scored) / len(scored),
          replay["macro_recall_at_20_scored_cutoffs"], "ESV macro recall")
    equal(replay["macro_recall_at_20_scored_cutoffs"], 0.0139, "ESV paper value")
    require(sum(round(r["recall_at_20"] * r["positive_controls"]) for r in scored) == 1,
            "ESV matched count")
    require(all(r["leakage_check_passed"] for r in folds), "Recorded ID checks")

    ablation = load("candidate_source_ablation.json")
    require(ablation["evaluations"] == len(ablation["rows"]) == 10, "Ablation folds")
    for field, total in [("candidate_count", "candidate_rows_not_independent"),
                         ("explicit_limitations", "explicit_limitation_rows_not_independent"),
                         ("typed_evidence_map_cells", "typed_evidence_map_rows_not_independent")]:
        require(sum(r[field] for r in ablation["rows"]) == ablation[total], total)
    require(ablation["explicit_limitation_rows_not_independent"] == 39 and
            ablation["typed_evidence_map_rows_not_independent"] == 0, "Ablation paper values")

    baseline = load("bm25_lacks_baseline.json")
    rows = baseline["rows"]
    positive = [r for r in rows if r["positive_controls"]]
    require(len(rows) == baseline["completed_evaluations"] == 10 and
            len(positive) == baseline["scored_evaluations"] == 9, "BM25 folds")
    require(sum(r["positive_controls"] for r in rows) ==
            baseline["positive_control_rows_not_independent"] == 64, "BM25 controls")
    require(sum(r["candidate_count"] for r in rows) == 474, "LACKS pool count")
    for field, aggregate in [("recall_at_k", "macro_recall_at_k"),
                             ("confidence_lacks_recall_at_k", "macro_confidence_lacks_recall_at_k"),
                             ("esv_gap_recall_at_k", "macro_esv_gap_recall_at_k")]:
        for k in ("5", "10", "20"):
            equal(sum(r[field][k] for r in positive) / len(positive),
                  baseline[aggregate][k], aggregate + "@" + k)
    require(sum(r["matched_control_count_at_20"] for r in rows) ==
            baseline["matched_control_rows_at_20_not_independent"] == 5, "BM25 matches")
    for r in positive:
        equal(r["recall_at_k"]["20"], r["matched_control_count_at_20"] / r["positive_controls"],
              "BM25 row count/recall")
    for k, value in [("5", 0.0556), ("10", 0.0694), ("20", 0.0933)]:
        equal(baseline["macro_recall_at_k"][k], value, "BM25 paper value")

    sensitivity = load("bm25_query_sensitivity.json")
    srows = sensitivity["rows"]
    sp = [r for r in srows if r["positive_controls"]]
    require(len(srows) == sensitivity["completed_evaluations"] == 10 and
            len(sp) == sensitivity["scored_evaluations"] == 9, "Sensitivity folds")
    bm_by_fold = {(r["domain"], r["cutoff"]): r for r in rows}
    for q, count, r20 in [("Q1", 5, 0.0933), ("Q2", 4, 0.0853), ("Q3", 4, 0.0853)]:
        summary = sensitivity["summary"][q]
        require(sum(r["queries"][q]["matched_control_count_at_20"] for r in srows) ==
                summary["matched_control_rows_at_20_not_independent"] == count, q + " matches")
        for k in ("5", "10", "20"):
            equal(sum(r["queries"][q]["recall_at_k"][k] for r in sp) / len(sp),
                  summary["macro_recall_at_k"][k], q + " macro@" + k)
        equal(summary["macro_recall_at_k"]["20"], r20, q + " paper value")
        for r in sp:
            v = r["queries"][q]
            equal(v["recall_at_k"]["20"], v["matched_control_count_at_20"] / r["positive_controls"],
                  q + " row count/recall")
    for r in srows:
        b = bm_by_fold[(r["domain"], r["cutoff"])]
        require(r["queries"]["Q1"]["recall_at_k"] == b["recall_at_k"], "Q1/BM25 mismatch")
    print("PASS: file fingerprints and aggregate arithmetic agree with the manuscript.")
    print("Not verified: omitted inputs, underlying matches, human labels, or gap closure.")


if __name__ == "__main__":
    main()
