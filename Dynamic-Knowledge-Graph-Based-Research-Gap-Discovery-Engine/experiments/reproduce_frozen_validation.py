"""Reproduce a frozen ESV-Gap validation gate from an archival config."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def _code_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, script_dir.parent):
        if (candidate / "src" / "validate_gaps.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate bundled src/validate_gaps.py")


CODE_ROOT = _code_root()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.validate_gaps import validate_all_gaps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Frozen validation YAML")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    release_root = config_path.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in ("graph", "processed_data", "outputs"):
        value = Path(config["paths"][key])
        if not value.is_absolute():
            config["paths"][key] = str((release_root / value).resolve())
    corpus_path = Path(config["gap_validation"]["corpus_path"])
    if not corpus_path.is_absolute():
        config["gap_validation"]["corpus_path"] = str((release_root / corpus_path).resolve())

    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_detector_output = release_root / "inputs" / "outputs" / "detected_gaps_raw.json"
    shutil.copyfile(frozen_detector_output, output_dir / "detected_gaps_raw.json")

    validate_all_gaps(config)
    report_path = Path(config["paths"]["outputs"]) / "gap_validation_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
