"""Corpus saturation analysis and new-entity discovery dynamics.

An absence claim in a knowledge graph is only credible if the corpus from
which the graph was built has reached semantic saturation in the target domain.
If each newly ingested paper continues to discover a high volume of novel entities,
the observed "absence" is likely a sampling artefact rather than a genuine
scholarly void.

This module computes:
- Cumulative entity curve E(i) and marginal discovery Delta E(i) per paper.
- Heaps' Law parameter estimation: E(n) = K * n^beta.
- Marginal discovery decay: comparing initial vs final quintiles.
- Empirical saturation verdict: SATURATED, MATURING, or UNSATURATED.
- Publication-quality saturation curve visualization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.entity_normalization import canonical_entity_key, canonical_entity_label
from src.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("corpus_saturation")


def fit_heaps_law(paper_indices: list[int], cumulative_entities: list[int]) -> dict[str, float]:
    """Fit Heaps' Law E(n) = K * n^beta via log-log ordinary least squares."""
    if len(paper_indices) < 3 or cumulative_entities[-1] == 0:
        return {"beta": 1.0, "K": 1.0, "r_squared": 0.0}

    valid_pairs = [
        (math.log(n), math.log(e))
        for n, e in zip(paper_indices, cumulative_entities)
        if n > 0 and e > 0
    ]
    if len(valid_pairs) < 3:
        return {"beta": 1.0, "K": 1.0, "r_squared": 0.0}

    x = np.array([p[0] for p in valid_pairs])
    y = np.array([p[1] for p in valid_pairs])

    # Linear regression: y = beta * x + ln(K)
    coeffs = np.polyfit(x, y, deg=1)
    beta = float(coeffs[0])
    ln_k = float(coeffs[1])
    k_param = float(math.exp(min(ln_k, 20.0)))

    # R-squared
    y_pred = beta * x + ln_k
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "beta": round(beta, 4),
        "K": round(k_param, 4),
        "r_squared": round(max(0.0, r_squared), 4),
    }


def compute_saturation_series(
    papers: list[dict[str, Any]],
    triples_by_paper: dict[str, list[dict[str, Any]]],
    entity_name_map: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate chronological entity accumulation series and summary metrics."""
    seen_entities: set[str] = set()
    seen_relations: set[tuple[str, str, str]] = set()

    series = []
    marginal_entities_list = []

    for idx, paper in enumerate(papers, start=1):
        pid = str(paper.get("paperId") or paper.get("paper_id") or paper.get("id") or f"paper_{idx}").strip()
        p_triples = triples_by_paper.get(pid) or (triples_by_paper.get(pid[:12]) if len(pid) >= 12 else None) or []

        paper_entities = set()
        paper_triples_set = set()

        for t in p_triples:
            s = canonical_entity_label(
                t.get("subject", {}).get("name") if isinstance(t.get("subject"), dict) else t.get("subject")
            )
            o = canonical_entity_label(
                t.get("object", {}).get("name") if isinstance(t.get("object"), dict) else t.get("object")
            )
            if entity_name_map:
                s = entity_name_map.get(s, s)
                o = entity_name_map.get(o, o)

            r = str(t.get("relation") or t.get("relation_type") or "").strip().upper()
            if s:
                paper_entities.add(s)
            if o:
                paper_entities.add(o)
            if s and o and r:
                paper_triples_set.add((s, r, o))

        new_entities = paper_entities - seen_entities
        seen_entities.update(paper_entities)

        new_relations = paper_triples_set - seen_relations
        seen_relations.update(paper_triples_set)

        marginal_entities_list.append(len(new_entities))

        series.append({
            "paper_index": idx,
            "paper_id": pid,
            "title": str(paper.get("title") or "")[:80],
            "year": paper.get("year"),
            "paper_entities_count": len(paper_entities),
            "new_entities": len(new_entities),
            "cumulative_entities": len(seen_entities),
            "new_relations": len(new_relations),
            "cumulative_relations": len(seen_relations),
        })

    total_papers = len(series)
    if total_papers == 0:
        return [], {"verdict": "UNSATURATED", "reason": "empty corpus"}

    indices = [s["paper_index"] for s in series]
    cum_ents = [s["cumulative_entities"] for s in series]
    heaps = fit_heaps_law(indices, cum_ents)

    # Quintile decay calculation: Delta_decay = (M_initial - M_final) / M_initial
    q_size = max(1, total_papers // 5)
    initial_window = marginal_entities_list[:q_size]
    final_window = marginal_entities_list[-q_size:]

    mean_initial = float(np.mean(initial_window)) if initial_window else 1.0
    mean_final = float(np.mean(final_window)) if final_window else 1.0

    if mean_initial > 0:
        decay_rate = round(float((mean_initial - mean_final) / mean_initial), 4)
    else:
        decay_rate = 0.0

    beta = heaps.get("beta", 1.0)
    # Determine saturation verdict
    if (decay_rate >= 0.60 and beta < 0.65) or (mean_final <= 1.0 and total_papers >= 15):
        verdict = "SATURATED"
        reason = (
            f"Marginal new entity rate dropped by {decay_rate*100:.1f}% to {mean_final:.2f}/paper "
            f"with sublinear growth (beta={beta:.2f}). Domain entity coverage is sufficient."
        )
    elif (decay_rate >= 0.30 and beta < 0.80) or mean_final <= 2.5:
        verdict = "MATURING"
        reason = (
            f"Moderate discovery decay ({decay_rate*100:.1f}%, beta={beta:.2f}, {mean_final:.2f} new entities/paper). "
            f"Absence claims are contextually grounded but warrant corpus boundary caveats."
        )
    else:
        verdict = "UNSATURATED"
        reason = (
            f"Corpus is still actively discovering new entities ({mean_final:.2f}/paper, beta={beta:.2f}, decay={decay_rate*100:.1f}%). "
            f"Absence claims may reflect corpus incompleteness rather than true scientific gaps."
        )

    summary = {
        "total_papers": total_papers,
        "total_unique_entities": len(seen_entities),
        "total_unique_relations": len(seen_relations),
        "heaps_law": heaps,
        "marginal_initial_mean": round(mean_initial, 2),
        "marginal_final_mean": round(mean_final, 2),
        "discovery_decay_rate": decay_rate,
        "verdict": verdict,
        "reason": reason,
    }

    return series, summary


def render_saturation_plot(
    series: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Render dual-axis publication-grade saturation curve PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping saturation curve plot")
        return

    if not series:
        return

    x = [s["paper_index"] for s in series]
    y_cum = [s["cumulative_entities"] for s in series]
    y_marginal = [s["new_entities"] for s in series]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    color1 = "#1f77b4"
    ax1.set_xlabel("Number of Papers Ingested (n)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Canonical Entities V(n)", color=color1, fontsize=11, fontweight="bold")
    line1 = ax1.plot(x, y_cum, color=color1, linewidth=2.5, label="Observed Canonical Entities V(n)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Plot fitted Heaps' law curve
    beta = summary.get("heaps_law", {}).get("beta", 1.0)
    k_param = summary.get("heaps_law", {}).get("K", 1.0)
    r2_val = summary.get("heaps_law", {}).get("r_squared", 0.0)
    y_heaps = [k_param * (n ** beta) for n in x]
    line_heaps = ax1.plot(
        x, y_heaps, color="#2ca02c", linestyle=":", linewidth=2,
        label=f"Heaps' Fit: V(n) = {k_param:.2f} n^{{{beta:.3f}}} (R^2 = {r2_val:.3f})"
    )

    # Secondary axis: marginal discovery per paper
    color2 = "#d62728"
    ax2 = ax1.twinx()
    ax2.set_ylabel("New Canonical Entities per Paper Delta V", color=color2, fontsize=11, fontweight="bold")
    bars = ax2.bar(x, y_marginal, color=color2, alpha=0.3, width=0.7, label="New Entities Delta V")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Rolling average of marginal discovery
    if len(y_marginal) >= 5:
        window_size = min(7, len(y_marginal) // 2 or 1)
        rolling = np.convolve(y_marginal, np.ones(window_size)/window_size, mode="valid")
        rolling_x = x[window_size - 1:]
        line_roll = ax2.plot(rolling_x, rolling, color="#ff7f0e", linewidth=1.8, label="Rolling Trend (Delta V)")

    # Title & verdict badge
    verdict = summary.get("verdict", "UNKNOWN")
    fig.suptitle(
        f"Corpus Saturation & Canonical Entity Discovery Dynamics (Verdict: {verdict})",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.title(
        f"Initial Rate: {summary.get('marginal_initial_mean')} ents/paper -> Final: {summary.get('marginal_final_mean')} ents/paper (Decay: {summary.get('discovery_decay_rate', 0)*100:.1f}%)",
        fontsize=9, color="#555555", pad=10
    )

    ensure_dir(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved saturation plot to %s", output_path)


def analyze_corpus_saturation(config: dict[str, Any]) -> dict[str, Any]:
    """Stage entry point: load corpus and triples, compute saturation, export artifacts."""
    output_dir = ensure_dir(config.get("paths", {}).get("outputs", "outputs"))
    figures_dir = ensure_dir(config.get("paths", {}).get("figures", output_dir / "figures"))

    # Load corpus papers
    processed_dir = Path(config.get("paths", {}).get("processed_data", "data/processed"))
    triples_dir = Path(config.get("paths", {}).get("triples", "data/triples"))

    # Fallback to most recent completed run in runs/ if default data dir is empty
    if (not processed_dir.exists() or not any(processed_dir.glob("*.json*"))) and Path("runs").exists():
        run_dirs = sorted([p for p in Path("runs").glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for r in run_dirs:
            candidate_p = r / "data" / "processed"
            candidate_t = r / "data" / "triples"
            if (
                candidate_p.exists()
                and any(candidate_p.glob("*.json*"))
                and candidate_t.exists()
                and any(candidate_t.glob("*.json*"))
            ):
                processed_dir = candidate_p
                triples_dir = candidate_t
                logger.info("Using corpus and triples from run: %s", r.name)
                break

    papers = []
    for candidate_name in ["filtered_papers.json", "corpus.json", "corpus_filtered.jsonl", "all_screened.jsonl"]:
        c_path = processed_dir / candidate_name
        if c_path.exists():
            if c_path.suffix == ".jsonl":
                with open(c_path, encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            try:
                                papers.append(json.loads(line_str))
                            except Exception:
                                pass
            else:
                loaded = load_json(c_path)
                if isinstance(loaded, list):
                    papers = loaded
                elif isinstance(loaded, dict) and "papers" in loaded:
                    papers = loaded["papers"]
            if papers:
                break

    # Load triples
    triples = []
    for candidate_name in ["all_triples.json", "extracted_triples.json"]:
        triples_file = triples_dir / candidate_name
        if triples_file.exists():
            triples = load_json(triples_file) or []
            if triples:
                break

    # Map triples by paper
    triples_by_paper: dict[str, list[dict[str, Any]]] = {}
    for t in triples:
        spid = str(t.get("source_paper_id") or t.get("paper_id") or "").strip()
        if spid:
            triples_by_paper.setdefault(spid, []).append(t)
            if len(spid) >= 12:
                triples_by_paper.setdefault(spid[:12], []).append(t)

    # Load entity name map if available
    graph_dir = Path(config.get("paths", {}).get("graph", "data/graph"))
    entity_name_map = None
    for emap_path in [graph_dir / "entity_name_map.json", processed_dir.parent / "graph" / "entity_name_map.json"]:
        if emap_path.exists():
            entity_name_map = load_json(emap_path)
            if entity_name_map:
                logger.info("Using entity_name_map with %d entries for canonical saturation", len(entity_name_map))
                break

    # Sort papers chronologically if year available, otherwise preserve order
    papers_sorted = sorted(papers, key=lambda p: int(p.get("year") or 9999))

    series, summary = compute_saturation_series(papers_sorted, triples_by_paper, entity_name_map=entity_name_map)

    report_path = output_dir / "corpus_saturation_report.json"
    save_json({"summary": summary, "series": series}, report_path)

    plot_path = figures_dir / "corpus_saturation_curve.png"
    render_saturation_plot(series, summary, plot_path)

    logger.info(
        "Corpus saturation analysis complete: %s (Decay=%.1f%%, Beta=%.2f)",
        summary.get("verdict"),
        summary.get("discovery_decay_rate", 0) * 100,
        summary.get("heaps_law", {}).get("beta", 1.0),
    )
    return {"summary": summary, "series": series}


if __name__ == "__main__":
    import yaml
    with open("config.yaml", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    analyze_corpus_saturation(conf)
