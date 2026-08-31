"""Discovery and reconstruction helpers for saved ESV-Gap runs."""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _line_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0


def _topic_for_run(run_dir: Path, metadata: dict | None = None) -> str:
    metadata = metadata or _read_json(run_dir / "run_metadata.json")
    if metadata.get("topic"):
        return str(metadata["topic"])
    return run_dir.name.rsplit("_", 2)[0].replace("_", " ")


def _artifact_activity_time(run_dir: Path) -> float:
    candidates = [run_dir]
    for relative in (
        "run_metadata.json",
        "data/raw",
        "data/processed",
        "data/triples",
        "data/graph",
        "outputs",
        "outputs/gaps_ranked_top.json",
    ):
        path = run_dir / relative
        if path.exists():
            candidates.append(path)
    timestamps = []
    for path in candidates:
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(timestamps) if timestamps else 0.0


def inspect_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    metadata = _read_json(run_dir / "run_metadata.json")
    outputs_dir = run_dir / "outputs"
    corpus_path = run_dir / "data" / "processed" / "corpus_filtered.jsonl"
    raw_path = run_dir / "data" / "raw" / "all_papers_raw.jsonl"
    progress_path = run_dir / "data" / "triples" / "extraction_progress.json"
    progress = _read_json(progress_path)
    completed = (outputs_dir / "gaps_ranked_top.json").exists()
    total_papers = _line_count(corpus_path)
    collected_papers = _line_count(raw_path)
    extraction_done = len(set(progress.get("completed_ids", [])))

    if completed:
        stage = "completed"
    elif total_papers:
        stage = "extracting"
    elif raw_path.exists():
        stage = "screening"
    else:
        stage = "collecting"

    activity_time = _artifact_activity_time(run_dir)
    created_at = str(metadata.get("created_at") or "")
    try:
        created = dt.datetime.fromisoformat(created_at)
    except ValueError:
        created = dt.datetime.fromtimestamp(activity_time) if activity_time else None

    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "topic": _topic_for_run(run_dir, metadata),
        "created_at": created_at,
        "created": created,
        "completed": completed,
        "stage": stage,
        "activity_time": activity_time,
        "active": bool(activity_time and dt.datetime.now().timestamp() - activity_time < 120),
        "total_papers": total_papers,
        "collected_papers": collected_papers,
        "requested_papers": int(metadata.get("max_papers", total_papers) or total_papers),
        "extraction_done": extraction_done,
        "triple_count": int(progress.get("total_triples", 0) or 0),
    }


def list_run_history(runs_root: str | Path = "runs") -> list[dict]:
    root = Path(runs_root)
    if not root.exists():
        return []
    runs = [inspect_run(path) for path in root.iterdir() if path.is_dir()]
    return sorted(runs, key=lambda item: item["activity_time"], reverse=True)


def config_for_run(base_config: dict, run_info: dict, groq_keys: object = None) -> dict:
    """Point a copied base config at one existing run without modifying it."""
    from src.groq_key_pool import normalise_groq_keys

    cfg = copy.deepcopy(base_config)
    run_dir = Path(run_info["run_dir"])
    keys = normalise_groq_keys(groq_keys, include_environment=True)
    cfg["project"]["domain"] = run_info["topic"]
    cfg.setdefault("api_keys", {})["groq_keys"] = keys
    cfg["api_keys"]["groq"] = keys[0] if keys else ""
    target = int(run_info.get("total_papers", 0) or run_info.get("requested_papers", 0) or 0)
    cfg["collection"]["max_papers"] = target
    cfg["filtering"]["target_corpus_size"] = target
    created_at = str(run_info.get("created_at") or "")
    if created_at:
        try:
            snapshot_date = dt.datetime.fromisoformat(created_at).date().isoformat()
        except ValueError:
            snapshot_date = ""
        if snapshot_date:
            cfg.setdefault("gap_validation", {})["snapshot_date"] = snapshot_date
    cfg["paths"] = {
        "raw_data": str(run_dir / "data" / "raw"),
        "processed_data": str(run_dir / "data" / "processed"),
        "triples": str(run_dir / "data" / "triples"),
        "graph": str(run_dir / "data" / "graph"),
        "outputs": str(run_dir / "outputs"),
        "figures": str(run_dir / "outputs" / "figures"),
        "prompts": "prompts",
    }
    return cfg


def run_label(run_info: dict) -> str:
    created = run_info.get("created")
    stamp = created.strftime("%Y-%m-%d %H:%M") if created else "unknown time"
    status = "Completed" if run_info.get("completed") else str(run_info.get("stage", "unknown")).title()
    return f"{stamp} · {run_info.get('topic', 'Untitled')} · {status} · {run_info.get('total_papers', 0)} papers"
