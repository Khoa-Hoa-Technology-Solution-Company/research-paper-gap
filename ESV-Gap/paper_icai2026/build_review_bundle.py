"""Build an allowlisted LOCAL bundle; omit inputs and project report identifiers."""
import copy
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import yaml


PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
WORKSPACE = PROJECT.parent
DEST = PAPER / "release_candidate"
ARCHIVE = WORKSPACE / "output/artifacts/ESV-Gap_ICAI2026_Review_Bundle.zip"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_secret_fields_empty(value, path="config"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + "." + str(key)
            sensitive = (str(key) == "api_keys" or
                         re.search(r"api[_-]?key|password|secret|credential|access[_-]?token", str(key), re.I))
            if sensitive and child:
                values = child.values() if isinstance(child, dict) else [child]
                if any(values):
                    raise ValueError("Nonempty credential field: " + child_path)
            ensure_secret_fields_empty(child, child_path)
    elif isinstance(value, list):
        for child in value:
            ensure_secret_fields_empty(child, path)


def main():
    if DEST.exists() or ARCHIVE.exists():
        raise FileExistsError("Refusing to overwrite an existing review bundle")
    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    ensure_secret_fields_empty(config)
    DEST.mkdir()
    entries = []

    def record(target, source, note):
        entries.append({"path": target.relative_to(DEST).as_posix(),
                        "sha256": digest(target), "source": source.relative_to(PROJECT).as_posix(),
                        "source_sha256": digest(source), "treatment": note})

    def original(relative, target=None):
        source = PROJECT / relative
        destination = DEST / (target or relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        record(destination, source, "byte-preserved")

    for filename in ["__init__.py", "temporal_backtest.py", "detect_gaps.py", "evidence_graph.py", "utils.py"]:
        original("src/" + filename)
    original("config.yaml")
    original("requirements-experiments.lock")
    original("tests/test_temporal_backtest.py")
    for filename in ["run_multidomain_backtest.py", "analyze_candidate_sources.py",
                     "run_bm25_lacks_baseline.py", "run_bm25_query_sensitivity.py"]:
        original("paper_icai2026/expanded_run/" + filename)
    for filename in ["multidomain_summary.json", "candidate_source_ablation.json"]:
        original("paper_icai2026/expanded_run/" + filename, "reports/" + filename)

    for filename in ["bm25_lacks_baseline.json", "bm25_query_sensitivity.json"]:
        source = PAPER / "expanded_run" / filename
        report = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
        if filename == "bm25_lacks_baseline.json":
            for row in report["rows"]:
                row["matched_control_count_at_20"] = len(set(row.pop("matched_control_ids_at_20")))
                row.pop("matched_pairs_at_20")
            note = "Removed matched_pairs_at_20; replaced matched_control_ids_at_20 by distinct-ID count."
        else:
            for row in report["rows"]:
                for query in row["queries"].values():
                    query["matched_control_count_at_20"] = len(set(query.pop("matched_control_ids_at_20")))
            note = "Replaced query-level matched_control_ids_at_20 by distinct-ID count."
        target = DEST / "reports" / filename
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record(target, source, note)
    original("paper_icai2026/ARTIFACT_README.md", "README.md")
    original("paper_icai2026/verify_aggregate_bundle.py", "verify_aggregate_bundle.py")
    lock = DEST / "requirements-experiments.lock"
    with lock.open("a", encoding="utf-8") as stream:
        stream.write("\nPyYAML==6.0.3\n")
    for entry in entries:
        if entry["path"] == "requirements-experiments.lock":
            entry["sha256"] = digest(lock)
            entry["treatment"] = "Appended PyYAML==6.0.3 for the original runner's yaml import."
    manifest = {"status": "local_unpublished_review_candidate",
                "license": "not selected; publication requires author approval and rights review",
                "verification_scope": "file identity and aggregate arithmetic, not pipeline or scientific reproduction",
                "excluded": ["corpus abstracts", "raw triples", "raw fold replay", "graph files",
                             "candidate/control text and IDs", "internal inspection notes", "credentials"],
                "files": sorted(entries, key=lambda entry: entry["path"])}
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                archive.write(path, "ESV-Gap_ICAI2026_Review_Bundle/" + path.relative_to(DEST).as_posix())
    print("Created local bundle:", ARCHIVE)
    print("SHA-256:", digest(ARCHIVE))


if __name__ == "__main__":
    main()
