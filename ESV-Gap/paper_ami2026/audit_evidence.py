"""Recount preserved artifacts and test omission of the uncertain MongoDB domain.

This is aggregate sensitivity, not a corrected-metadata pipeline replay.
"""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
ARCHIVE = PROJECT / "paper_icai2026/expanded_run"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    replay_path = ARCHIVE / "multidomain_summary.json"
    bm25_path = ARCHIVE / "bm25_lacks_baseline.json"
    replay, bm25 = load(replay_path), load(bm25_path)
    record_count = triple_count = 0
    verified_inputs = {}
    for source in replay["sources"]:
        run = PROJECT / "runs" / source["source_run"]
        corpus = run / "data/processed/corpus_filtered.jsonl"
        triples = run / "data/triples/all_triples.json"
        records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        triples_data = load(triples)
        assert len(records) == source["retained_papers"]
        assert len(triples_data) == source["triples"]
        hashes = {"corpus": digest(corpus), "triples": digest(triples)}
        assert hashes == replay["fingerprints"]["inputs"][source["domain"]]
        verified_inputs[source["domain"]] = hashes
        record_count += len(records); triple_count += len(triples_data)
    scored = [row for row in bm25["rows"] if row["positive_controls"]]
    eligible = [row for row in scored if row["domain"] != "mongodb_security"]
    def summary(rows):
        return {"scored_evaluations": len(rows),
                "control_rows_not_independent": sum(r["positive_controls"] for r in rows),
                "esv_macro_proxy_recall_at_20": sum(r["esv_gap_recall_at_k"]["20"] for r in rows) / len(rows),
                "bm25_macro_proxy_recall_at_20": sum(r["recall_at_k"]["20"] for r in rows) / len(rows)}
    primary = PROJECT / "runs/deep_learning_iot_intrusion_de_20260831_114802"
    diagnostics = load(primary / "data/processed/screening_diagnostics.json")
    validation = load(primary / "outputs/gap_validation_audit.json")
    report = {"status": "verified_recount_and_posthoc_domain_omission",
              "records": record_count, "triples": triple_count,
              "input_hashes_match_preserved_replay": True,
              "all_domains": summary(scored),
              "without_mongodb_domain": summary(eligible),
              "interpretation": "Entire-domain omission checks aggregate dependence on the uncertain MongoDB record; it does not repair labels, resolve metadata or rerun candidate generation.",
              "primary_iot_screening": {k: diagnostics.get(k) for k in
                                        ["raw_paper_count", "screened_paper_count", "retained_paper_count"]},
              "primary_iot_validation_summary": validation["summary"],
              "verified_input_sha256": verified_inputs,
              "report_sha256": {"replay": digest(replay_path), "bm25": digest(bm25_path)},
              "script_sha256": digest(__file__)}
    (BASE / "evidence_recount.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["records", "triples", "all_domains", "without_mongodb_domain"]}, indent=2))


if __name__ == "__main__":
    main()
