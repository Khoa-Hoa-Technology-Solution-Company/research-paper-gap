"""Offline replay of the frozen IoT temporal backtest into an isolated audit folder.

No network/API calls. Historical run files are read but never changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.temporal_backtest import aggregate_results, run_cutoff_backtest  # noqa: E402


RUN = PROJECT / "runs" / "deep_learning_iot_intrusion_de_20260831_114802"
CORPUS = RUN / "data" / "processed" / "corpus_filtered.jsonl"
TRIPLES = RUN / "data" / "triples" / "all_triples.json"
OUTPUT = Path(__file__).resolve().parent / "replay"
EXPECTED_HASHES = {
    CORPUS: "852b63bf0b7f3fb641046b416c45723156d5663b56f87352b9145efb22282583",
    TRIPLES: "a352427bffb0d6d63bce589ec13df54e0f71b615bd6f2f9470d694fee4e21a31",
}
CODE_FILES = (
    PROJECT / "src" / "temporal_backtest.py",
    PROJECT / "src" / "detect_gaps.py",
    PROJECT / "src" / "evidence_graph.py",
    PROJECT / "config.yaml",
    Path(__file__).resolve(),
)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    for path, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {path}: {actual}")
    documents = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    triples = json.loads(TRIPLES.read_text(encoding="utf-8"))
    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((RUN / "run_metadata.json").read_text(encoding="utf-8"))
    config["project"]["domain"] = str(metadata["topic"])
    results = [run_cutoff_backtest(documents, triples, config, cutoff, OUTPUT) for cutoff in (2022, 2023, 2024)]
    summary = {
        "source_run": RUN.name,
        "input_sha256": {str(path.relative_to(PROJECT)): value for path, value in EXPECTED_HASHES.items()},
        "replay_environment": {
            "python_version": sys.version,
            "packages": {name: package_version(name) for name in ("networkx", "PyYAML", "numpy", "scipy", "python-louvain")},
            "code_sha256": {
                str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in CODE_FILES
            },
        },
        "aggregate": aggregate_results(results),
        "cutoff_results": [
            {key: value for key, value in result.items() if key not in ("positive_controls", "negative_controls", "ranked_candidates")}
            for result in results
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
