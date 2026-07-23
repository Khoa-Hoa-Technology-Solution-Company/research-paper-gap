"""Synthetic signal-injection sanity checks for the temporal screen.

This is an algorithmic benchmark, not evidence of declining research activity
or a substitute for a multi-corpus evaluation. It injects known trajectories
into event-aggregated graphs, reruns the production temporal screen, and keeps
stable/increasing controls separate from declining trajectories.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
from pathlib import Path
from typing import Iterable

import networkx as nx

from src import config
from src.graph_analysis import compute_temporal_decay


YEARS = tuple(range(2018, 2026))
TRAJECTORIES = ("linear", "abrupt", "exponential", "temporary_dip", "stable", "increasing")


def _poisson(rng: random.Random, mean: float) -> int:
    """Small-mean Knuth sampler; dependency-free and deterministic by seed."""
    limit = math.exp(-mean)
    count = 0
    product = 1.0
    while product > limit:
        count += 1
        product *= rng.random()
    return max(0, count - 1)


def _trajectory(shape: str, reduction: float, base: float = 24.0) -> list[float]:
    if shape == "stable":
        return [base] * len(YEARS)
    if shape == "increasing":
        return [base * (1 + 0.5 * index / (len(YEARS) - 1)) for index in range(len(YEARS))]
    final = base * (1 - reduction)
    if shape == "linear":
        return [base + (final - base) * index / (len(YEARS) - 1) for index in range(len(YEARS))]
    if shape == "abrupt":
        return [base] * (len(YEARS) - 2) + [final, final]
    if shape == "exponential":
        ratio = (final / base) ** (1 / (len(YEARS) - 1)) if final else 0.0
        return [base * ratio**index for index in range(len(YEARS))]
    if shape == "temporary_dip":
        values = [base] * len(YEARS)
        values[len(YEARS) // 2] = final
        values[len(YEARS) // 2 + 1] = final
        return values
    raise ValueError(f"Unknown trajectory shape: {shape}")


def _graph_for_counts(counts: Iterable[int], background_events: int = 180) -> nx.DiGraph:
    """Build a graph where only ``target concept`` can become temporally eligible."""
    graph = nx.DiGraph()
    target_events = []
    background_events_by_year: dict[int, list[dict[str, int]]] = {}
    for year, count in zip(YEARS, counts):
        target_events.extend({"year": year} for _ in range(count))
        background_events_by_year[year] = [{"year": year} for _ in range(background_events)]
    graph.add_node("target concept", type="CONCEPT")
    graph.add_node("analysis", type="CONCEPT")  # excluded generic anchor
    graph.add_edge("target concept", "analysis", events=target_events)
    for year, events in background_events_by_year.items():
        graph.add_edge(f"background-{year}", "analysis", events=events)
    return graph


def run_temporal_signal_injection(
    *, trials_per_condition: int = 40, seed: int = 20260720
) -> dict:
    """Run predeclared synthetic trajectories through the production screen."""
    if trials_per_condition < 1:
        raise ValueError("trials_per_condition must be positive")
    rows = []
    reductions = (0.10, 0.25, 0.50, 0.75)
    trial_index = 0
    for shape in TRAJECTORIES:
        shape_reductions = (0.0,) if shape in {"stable", "increasing"} else reductions
        for reduction in shape_reductions:
            detections = 0
            for replicate in range(trials_per_condition):
                rng = random.Random(seed + trial_index)
                trial_index += 1
                counts = [_poisson(rng, mean) for mean in _trajectory(shape, reduction)]
                graph = _graph_for_counts(counts)
                with contextlib.redirect_stdout(io.StringIO()):
                    signals, report = compute_temporal_decay(graph, cutoff_year=2025, return_report=True)
                detected = any(item["node"] == "target concept" for item in signals)
                detections += int(detected)
                rows.append({
                    "shape": shape,
                    "nominal_reduction": reduction,
                    "replicate": replicate,
                    "counts": counts,
                    "detected": detected,
                    "eligible_nodes_tested": report["pipeline_counts"]["eligible_nodes_tested"],
                })
    summary = []
    for shape in TRAJECTORIES:
        shape_reductions = (0.0,) if shape in {"stable", "increasing"} else reductions
        for reduction in shape_reductions:
            selected = [row for row in rows if row["shape"] == shape and row["nominal_reduction"] == reduction]
            summary.append({
                "shape": shape,
                "nominal_reduction": reduction,
                "trials": len(selected),
                "detections": sum(row["detected"] for row in selected),
                "detection_rate": round(sum(row["detected"] for row in selected) / len(selected), 4),
                "interpretation": "true-positive sensitivity" if shape not in {"stable", "increasing"} else "false-positive rate control",
            })
    return {
        "schema_version": "temporal-signal-injection-v1",
        "scope": (
            "Synthetic algorithmic sanity check for the production temporal screen. "
            "It does not validate temporal research-gap detection or generalize to literature corpora."
        ),
        "design": {
            "years": list(YEARS),
            "trials_per_condition": trials_per_condition,
            "seed": seed,
            "background_events_per_year": 180,
            "noise": "independent Poisson counts for the target trajectory",
            "controls": ["stable", "increasing"],
            "decline_shapes": ["linear", "abrupt", "exponential", "temporary_dip"],
            "nominal_reductions": list(reductions),
            "production_screen": "src.graph_analysis.compute_temporal_decay",
        },
        "summary": summary,
        "trials": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic temporal signal-injection sanity checks.")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path, default=Path(config.DATA_DIR) / "temporal_signal_injection.json")
    args = parser.parse_args()
    report = run_temporal_signal_injection(trials_per_condition=args.trials, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Saved temporal signal-injection report to {args.output}")


if __name__ == "__main__":
    main()
