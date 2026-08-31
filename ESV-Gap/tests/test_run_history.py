import json
import os
from pathlib import Path

from src.run_history import config_for_run, inspect_run, list_run_history, run_label


def _write(path: Path, text: str = "{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_history_lists_completed_and_resumable_runs(tmp_path):
    completed = tmp_path / "topic_a_20260830_120000"
    _write(completed / "run_metadata.json", json.dumps({
        "topic": "topic A", "max_papers": 50, "created_at": "2026-08-30T12:00:00"
    }))
    _write(completed / "data/processed/corpus_filtered.jsonl", "{}\n{}\n")
    _write(completed / "outputs/gaps_ranked_top.json", "[]")

    paused = tmp_path / "topic_b_20260830_130000"
    _write(paused / "run_metadata.json", json.dumps({
        "topic": "topic B", "max_papers": 3, "created_at": "2026-08-30T13:00:00"
    }))
    _write(paused / "data/processed/corpus_filtered.jsonl", "{}\n{}\n{}\n")
    _write(paused / "data/triples/extraction_progress.json", json.dumps({
        "completed_ids": ["p1", "p2"], "total_triples": 17
    }))
    newer = completed.stat().st_mtime + 20
    os.utime(paused / "data/triples", (newer, newer))

    history = list_run_history(tmp_path)

    assert [item["run_id"] for item in history] == [paused.name, completed.name]
    assert history[0]["stage"] == "extracting"
    assert history[0]["extraction_done"] == 2
    assert history[0]["triple_count"] == 17
    assert history[1]["completed"] is True
    assert "Completed" in run_label(history[1])


def test_existing_run_config_uses_selected_paths_and_memory_only_keys(tmp_path):
    run_dir = tmp_path / "saved_run"
    _write(run_dir / "run_metadata.json", json.dumps({"topic": "saved topic"}))
    info = inspect_run(run_dir)
    base = {
        "project": {"domain": "old"},
        "api_keys": {},
        "collection": {"max_papers": 1},
        "filtering": {"target_corpus_size": 1},
    }

    cfg = config_for_run(base, info, ["gsk_first", "gsk_second"])

    assert cfg["project"]["domain"] == "saved topic"
    assert cfg["api_keys"]["groq_keys"] == ["gsk_first", "gsk_second"]
    assert cfg["api_keys"]["groq"] == "gsk_first"
    assert Path(cfg["paths"]["outputs"]) == run_dir / "outputs"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert "api_keys" not in metadata
