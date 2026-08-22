"""
ESV-Gap Evidence Workbench - Streamlit Frontend

FIX 8 — HCAI grounding:
  Added expert review panel in Tab 1 (Ranked Gaps).
  Reviewers can Accept / Reject / Modify each gap.
  Decisions are saved to expert_reviews.json which can be
  cited in the paper as the Stage 6 human-in-the-loop evidence.

Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")



import streamlit as st
import yaml
import json
import time
import threading
import queue
import sys
import os
import pickle
import logging
import datetime
from html import escape
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from src.gap_provenance import (
    build_paper_index,
    candidate_identity as provenance_candidate_identity,
    resolve_gap_provenance,
)

logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ESV-Gap · Evidence Workbench",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ───────────────────────────────────────────
token_css = Path(__file__).with_name("tokens.css").read_text(encoding="utf-8")
st.markdown(f"<style>{token_css}</style>", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────

def load_base_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_run_config(base_config, topic, num_papers, groq_key):
    import copy
    cfg = copy.deepcopy(base_config)
    groq_key = (
        groq_key
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    cfg["project"]["domain"]               = topic
    cfg["api_keys"]["groq"]                = groq_key
    cfg["collection"]["max_papers"]        = num_papers
    cfg["filtering"]["target_corpus_size"] = min(num_papers, 150)

    slug   = topic.lower().replace(" ", "_")[:30]
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{slug}_{ts}"

    cfg["paths"] = {
        "raw_data":       f"runs/{run_id}/data/raw",
        "processed_data": f"runs/{run_id}/data/processed",
        "triples":        f"runs/{run_id}/data/triples",
        "graph":          f"runs/{run_id}/data/graph",
        "outputs":        f"runs/{run_id}/outputs",
        "figures":        f"runs/{run_id}/outputs/figures",
        "prompts":        "prompts",
    }
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({
            "run_id": run_id,
            "topic": topic,
            "max_papers": num_papers,
            "created_at": datetime.datetime.now().isoformat(),
        }, indent=2),
        encoding="utf-8",
    )
    return cfg, run_id


def build_resume_config(run_info, groq_key):
    """Reconstruct a safe config that points to an existing checkpointed run."""
    cfg = load_base_config()
    run_dir = Path(run_info["run_dir"])
    topic = run_info["topic"]

    cfg["project"]["domain"] = topic
    cfg.setdefault("api_keys", {})["groq"] = groq_key
    cfg["collection"]["max_papers"] = run_info["total_papers"]
    cfg["filtering"]["target_corpus_size"] = run_info["total_papers"]
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


def generate_queries_with_llm(topic, groq_key):
    fallback_queries = [
        topic,
        f"{topic} survey",
        f"{topic} framework",
        f"{topic} deep learning",
        f"{topic} methods",
    ]

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=20.0,
            max_retries=1,
        )
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate concise academic search queries. "
                        "Return only a valid JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Generate exactly 5 academic search queries for "{topic}". '
                        "Each query must contain 2-5 words. Return them in this "
                        'format: {"queries": ["query one", "query two", '
                        '"query three", "query four", "query five"]}'
                    ),
                },
            ],
            temperature=0.5,
            max_tokens=500,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        queries = data.get("queries", []) if isinstance(data, dict) else []
        queries = [
            query.strip()
            for query in queries
            if isinstance(query, str) and query.strip()
        ]

        if len(queries) == 5:
            return queries

        logging.getLogger(__name__).warning(
            "Groq returned an invalid query list; using fallback queries. "
            "Response: %r",
            content,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not generate search queries with Groq; using fallback queries: %s",
            exc,
        )

    return fallback_queries


def run_pipeline_with_progress(cfg, progress_queue):
    try:
        sys.path.insert(0, str(Path(__file__).parent))

        def stage_progress(start, end, label):
            last_reported = {"value": -1}

            def report(completed, total):
                total = max(int(total), 1)
                completed = min(max(int(completed), 0), total)
                percent = start + round((end - start) * completed / total)
                progress_queue.put(("progress", percent))

                report_every = max(1, total // 10)
                if (
                    completed == 1
                    or completed == total
                    or completed % report_every == 0
                ) and completed != last_reported["value"]:
                    progress_queue.put((
                        "status",
                        f"{label}: {completed}/{total}",
                    ))
                    last_reported["value"] = completed

            return report

        progress_queue.put(("status", "📥 Collecting papers from Semantic Scholar..."))
        progress_queue.put(("progress", 8))
        from src.collect import CollectionAPIError, collect_papers
        try:
            collect_papers(
                cfg,
                progress_callback=stage_progress(8, 20, "Collecting queries"),
            )
        except CollectionAPIError as exc:
            progress_queue.put((
                "error",
                f"Paper collection stopped: {exc}\n\n"
                "The run was terminated after bounded retries; it is safe to retry.",
            ))
            return
        progress_queue.put(("progress", 20))

        progress_queue.put(("status", "🔎 Screening papers for relevance..."))
        progress_queue.put(("progress", 22))
        from src.filter import filter_corpus
        filtered_papers = filter_corpus(
            cfg,
            progress_callback=stage_progress(22, 40, "Screening papers"),
        )
        if not filtered_papers:
            progress_queue.put((
                "error",
                "Screening retained 0 papers, so the knowledge graph cannot be "
                "constructed. Try a more specific topic, increase Max papers, "
                "or lower filtering.relevance_threshold in config.yaml.",
            ))
            return
        progress_queue.put(("progress", 40))

        progress_queue.put(("status", "🧠 Extracting knowledge triples..."))
        progress_queue.put(("progress", 42))
        from src.extract_triples import ExtractionRateLimitError, extract_all_triples
        try:
            extracted_triples = extract_all_triples(
                cfg,
                progress_callback=stage_progress(42, 58, "Extracting papers"),
            )
        except ExtractionRateLimitError as exc:
            progress_queue.put((
                "error",
                f"Extraction paused: {exc}\n\n"
                "The completed paper files were preserved. Retry later to resume "
                "instead of recording quota failures as empty extractions.",
            ))
            return
        if not extracted_triples:
            progress_queue.put((
                "error",
                "Triple extraction produced 0 valid relations, so the knowledge "
                "graph cannot be constructed. Inspect the retained abstracts or "
                "the Groq extraction response before retrying.",
            ))
            return
        progress_queue.put(("progress", 58))

        progress_queue.put(("status", "🕸️ Building knowledge graph..."))
        progress_queue.put(("progress", 60))
        from src.build_graph import build_knowledge_graph
        graph = build_knowledge_graph(cfg)
        if graph is None or graph.number_of_nodes() == 0:
            progress_queue.put((
                "error",
                "All extracted relations were removed during graph validation; "
                "the resulting graph has 0 nodes. No gap detection was run.",
            ))
            return
        progress_queue.put(("progress", 72))

        progress_queue.put(("status", "🔬 Detecting research gaps..."))
        progress_queue.put(("progress", 74))
        from src.detect_gaps import detect_all_gaps
        detect_all_gaps(cfg)
        progress_queue.put(("progress", 84))

        progress_queue.put(("status", "📊 Scoring and ranking gaps..."))
        progress_queue.put(("progress", 86))
        from src.score_gaps import score_and_rank_gaps
        score_and_rank_gaps(cfg)
        progress_queue.put(("progress", 92))

        progress_queue.put(("status", "🎨 Generating visualisations..."))
        from src.visualise import generate_visualisations
        generate_visualisations(cfg)
        progress_queue.put(("progress", 100))

        progress_queue.put(("done", cfg))

    except Exception as e:
        import traceback
        progress_queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))


def resume_pipeline_with_progress(cfg, progress_queue):
    """Resume extraction in-place, then finish all downstream KG stages."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))

        def extraction_progress(completed, total):
            total = max(int(total), 1)
            completed = min(max(int(completed), 0), total)
            progress_queue.put(("progress", round(5 + 45 * completed / total)))
            report_every = max(1, total // 10)
            if completed == total or completed % report_every == 0:
                progress_queue.put((
                    "status",
                    f"Resuming extraction: {completed}/{total} papers",
                ))

        progress_queue.put(("status", "Resuming from the saved extraction checkpoint..."))
        progress_queue.put(("progress", 5))
        from src.extract_triples import ExtractionRateLimitError, extract_all_triples
        try:
            extracted_triples = extract_all_triples(
                cfg,
                progress_callback=extraction_progress,
            )
        except ExtractionRateLimitError as exc:
            progress_queue.put((
                "error",
                f"Extraction paused again: {exc}\n\n"
                "The same checkpoint remains intact. Wait for quota renewal, "
                "then use Resume extraction again.",
            ))
            return

        if not extracted_triples:
            progress_queue.put((
                "error",
                "Resume completed without any valid triples; the graph cannot be built.",
            ))
            return

        progress_queue.put(("status", "Building the knowledge graph..."))
        progress_queue.put(("progress", 55))
        from src.build_graph import build_knowledge_graph
        graph = build_knowledge_graph(cfg)
        if graph is None or graph.number_of_nodes() == 0:
            progress_queue.put((
                "error",
                "The resumed extraction produced an empty graph after validation.",
            ))
            return

        progress_queue.put(("status", "Detecting research-gap candidates..."))
        progress_queue.put(("progress", 70))
        from src.detect_gaps import detect_all_gaps
        detect_all_gaps(cfg)

        progress_queue.put(("status", "Scoring and ranking candidates..."))
        progress_queue.put(("progress", 84))
        from src.score_gaps import score_and_rank_gaps
        score_and_rank_gaps(cfg)

        progress_queue.put(("status", "Generating evidence visualisations..."))
        progress_queue.put(("progress", 94))
        from src.visualise import generate_visualisations
        generate_visualisations(cfg)

        progress_queue.put(("progress", 100))
        progress_queue.put(("done", cfg))

    except Exception as e:
        import traceback
        progress_queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))


def run_rag_with_progress(cfg, progress_queue):
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.rag_baseline import run_rag_baseline

        progress_queue.put(("status", "📚 Embedding abstracts for retrieval..."))
        progress_queue.put(("progress", 10))
        progress_queue.put(("status", "🤖 Running Mulla et al. RAG baseline..."))
        progress_queue.put(("progress", 20))

        results = run_rag_baseline(cfg)

        progress_queue.put(("progress", 100))
        progress_queue.put(("done", results))

    except Exception as e:
        import traceback
        progress_queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))


def load_results(cfg):
    out     = cfg["paths"]["outputs"]
    results = {}

    gaps_path = Path(out) / "gaps_ranked_top.json"
    if gaps_path.exists():
        with open(gaps_path) as f:
            results["gaps"] = json.load(f)

    raw_gaps_path = Path(out) / "detected_gaps_raw.json"
    if raw_gaps_path.exists():
        with open(raw_gaps_path, encoding="utf-8") as f:
            raw_gaps = json.load(f)
        results["raw_gap_counts"] = {
            key: len(value) for key, value in raw_gaps.items()
            if isinstance(value, list)
        }

    csv_path = Path(out) / "gaps_ranked.csv"
    if csv_path.exists():
        results["gaps_df"] = pd.read_csv(csv_path)

    graph_path = Path(cfg["paths"]["graph"]) / "knowledge_graph.pkl"
    if graph_path.exists():
        with open(graph_path, "rb") as f:
            results["graph"] = pickle.load(f)

    html_path = Path(out) / "graph_viz.html"
    if html_path.exists():
        with open(html_path) as f:
            results["graph_html"] = f.read()

    mulla_path = Path(out) / "rag_mulla_gaps.json"
    if mulla_path.exists():
        with open(mulla_path) as f:
            results["mulla_gaps"] = json.load(f)

    simple_path = Path(out) / "rag_simple_gaps.json"
    if simple_path.exists():
        with open(simple_path) as f:
            results["simple_gaps"] = json.load(f)

    metrics_path = Path(out) / "comparison_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results["comparison_metrics"] = json.load(f)

    # FIX 8: Load any previously saved expert reviews
    reviews_path = Path(out) / "expert_reviews.json"
    if reviews_path.exists():
        with open(reviews_path) as f:
            results["expert_reviews"] = json.load(f)

    post_gate_reviews_path = Path(out) / "post_gate_expert_reviews.json"
    if post_gate_reviews_path.exists():
        with open(post_gate_reviews_path, encoding="utf-8") as f:
            results["post_gate_expert_reviews"] = json.load(f)

    raw_corpus_path = Path(cfg["paths"]["raw_data"]) / "all_papers_raw.jsonl"
    filtered_corpus_path = (
        Path(cfg["paths"]["processed_data"]) / "corpus_filtered.jsonl"
    )
    corpus_papers = []
    if filtered_corpus_path.exists():
        with open(filtered_corpus_path, encoding="utf-8") as corpus_file:
            for line in corpus_file:
                if not line.strip():
                    continue
                try:
                    corpus_papers.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
    results["papers"] = corpus_papers
    results["paper_index"] = build_paper_index(corpus_papers)

    paper_outputs = list(Path(cfg["paths"]["triples"]).glob("paper_*.json"))
    run_stats = {
        "collected": 0,
        "retained": 0,
        "paper_outputs": len(paper_outputs),
        "nonempty_paper_outputs": 0,
        "triples": 0,
    }
    for key, path in (("collected", raw_corpus_path), ("retained", filtered_corpus_path)):
        if path.exists():
            with open(path, encoding="utf-8") as corpus_file:
                run_stats[key] = sum(1 for line in corpus_file if line.strip())
    for paper_path in paper_outputs:
        try:
            paper_result = json.loads(paper_path.read_text(encoding="utf-8"))
            if int(paper_result.get("num_triples", 0)) > 0:
                run_stats["nonempty_paper_outputs"] += 1
        except (OSError, ValueError, TypeError):
            continue
    all_triples_path = Path(cfg["paths"]["triples"]) / "all_triples.json"
    if all_triples_path.exists():
        try:
            run_stats["triples"] = len(json.loads(
                all_triples_path.read_text(encoding="utf-8")
            ))
        except (OSError, ValueError, TypeError):
            pass
    results["run_stats"] = run_stats

    return results


# ── FIX 8: Expert review helpers ────────────────────────────

def render_gap_source_evidence(provenance, key_prefix, max_papers=6):
    """Render paper metadata and edge-level evidence for one gap candidate."""
    papers = provenance.get("papers", [])
    label = f"Source evidence · {len(papers)} paper{'s' if len(papers) != 1 else ''}"
    with st.expander(label):
        st.caption(
            "These papers support graph relations used to infer the candidate. "
            "They do not necessarily state that the candidate is a research gap."
        )
        for index, path in enumerate(provenance.get("evidence_paths", []), start=1):
            st.caption(
                f"Evidence path {index}: "
                + " → ".join(map(str, path.get("nodes", [])))
            )

        if not papers:
            st.warning("No paper-level provenance could be resolved for this candidate.")
            return

        for paper_position, paper in enumerate(papers[:max_papers]):
            st.markdown(f"**{paper.get('title', 'Untitled paper')}**")
            metadata = []
            if paper.get("authors"):
                shown_authors = paper["authors"][:4]
                author_text = ", ".join(shown_authors)
                if len(paper["authors"]) > len(shown_authors):
                    author_text += " et al."
                metadata.append(author_text)
            if paper.get("year"):
                metadata.append(str(paper["year"]))
            if paper.get("venue"):
                metadata.append(paper["venue"])
            metadata.append(f"{paper.get('citation_count', 0)} citations")
            st.caption(" · ".join(metadata))

            link_col, id_col = st.columns([1, 3])
            with link_col:
                if paper.get("url"):
                    st.link_button(
                        "Open paper",
                        paper["url"],
                        key=f"{key_prefix}:paper-link:{paper_position}",
                        width="stretch",
                    )
            with id_col:
                identifier = f"DOI: {paper['doi']}" if paper.get("doi") else paper["paper_id"]
                st.caption(identifier)

            if paper.get("relevance_reason"):
                st.caption(f"Screening rationale: {paper['relevance_reason']}")
            for item in paper.get("evidence", [])[:3]:
                relation = (
                    f"{item.get('subject', '')} —[{item.get('relation', '')}]→ "
                    f"{item.get('object', '')}"
                )
                st.markdown(f"`{relation}`")
                if item.get("evidence"):
                    st.caption(f"Evidence: {item['evidence']}")
            if paper_position < min(len(papers), max_papers) - 1:
                st.divider()

        if len(papers) > max_papers:
            st.caption(f"{len(papers) - max_papers} more papers are listed in the Source Papers tab.")


def _run_activity_time(run_dir):
    """Return the newest artifact timestamp, not only the directory mtime."""
    timestamps = [run_dir.stat().st_mtime]
    for path in run_dir.rglob("*"):
        if path.is_file():
            if path.name == "run_metadata.json":
                continue
            try:
                timestamps.append(path.stat().st_mtime)
            except OSError:
                continue
    return max(timestamps)


def _pipeline_activity_time(run_dir):
    """Return real pipeline artifact activity, excluding metadata-only writes."""
    timestamps = []
    for path in run_dir.rglob("*"):
        if path.is_file() and path.name != "run_metadata.json":
            try:
                timestamps.append(path.stat().st_mtime)
            except OSError:
                continue
    return max(timestamps) if timestamps else run_dir.stat().st_mtime


def _topic_for_run(run_dir):
    """Read the original, untruncated topic when run metadata is available."""
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("topic"):
                return str(metadata["topic"])
        except (OSError, ValueError, TypeError):
            pass
    return run_dir.name.rsplit("_", 2)[0].replace("_", " ")


def inspect_latest_run(runs_root="runs"):
    """Describe the newest run, including an in-progress extraction."""
    runs_dir = Path(runs_root)
    if not runs_dir.exists():
        return None

    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    run_dir = max(run_dirs, key=_run_activity_time)
    activity_time = _pipeline_activity_time(run_dir)
    outputs_dir = run_dir / "outputs"
    completed = (outputs_dir / "gaps_ranked_top.json").exists()
    topic = _topic_for_run(run_dir)

    corpus_path = run_dir / "data" / "processed" / "corpus_filtered.jsonl"
    total_papers = 0
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as corpus_file:
            total_papers = sum(1 for line in corpus_file if line.strip())

    progress_path = run_dir / "data" / "triples" / "extraction_progress.json"
    extraction_done = 0
    triple_count = 0
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            extraction_done = len(set(progress.get("completed_ids", [])))
            triple_count = int(progress.get("total_triples", 0))
        except (OSError, ValueError, TypeError):
            pass

    if completed:
        stage = "completed"
    elif total_papers:
        stage = "extracting"
    elif (run_dir / "data" / "raw" / "all_papers_raw.jsonl").exists():
        stage = "screening"
    else:
        stage = "collecting"

    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "topic": topic,
        "completed": completed,
        "stage": stage,
        "activity_time": activity_time,
        "active": time.time() - activity_time < 120,
        "total_papers": total_papers,
        "extraction_done": extraction_done,
        "triple_count": triple_count,
    }


def load_latest_completed_run(runs_root="runs"):
    """Recover the newest completed run after a Streamlit refresh/rerun."""
    runs_dir = Path(runs_root)
    if not runs_dir.exists():
        return None, None

    run_dirs = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=_run_activity_time,
        reverse=True,
    )

    for run_dir in run_dirs:
        outputs_dir = run_dir / "outputs"
        if not (outputs_dir / "gaps_ranked_top.json").exists():
            continue

        cfg = load_base_config()
        cfg["paths"] = {
            "raw_data": str(run_dir / "data" / "raw"),
            "processed_data": str(run_dir / "data" / "processed"),
            "triples": str(run_dir / "data" / "triples"),
            "graph": str(run_dir / "data" / "graph"),
            "outputs": str(outputs_dir),
            "figures": str(outputs_dir / "figures"),
            "prompts": "prompts",
        }

        topic = _topic_for_run(run_dir)
        cfg["project"]["domain"] = topic
        return cfg, topic

    return None, None


def save_expert_reviews(
    cfg,
    reviews,
    reviewer_name,
    notes="",
    filename="expert_reviews.json",
    queue_name="score_ranked_top30",
    candidate_index=None,
):
    """
    Persist expert gap reviews to disk.

    This file can be cited in the paper as evidence of the Stage 6
    human-in-the-loop validation required by HCAI principles (Shneiderman 2020).
    The acceptance rate (accepted / total reviewed) is reported in Table 3.
    """
    out          = Path(cfg["paths"]["outputs"])
    reviews_path = out / filename

    summary = {k: sum(1 for v in reviews.values() if v == k)
               for k in ["Accept", "Reject", "Modify", "Pending"]}

    total_reviewed = summary["Accept"] + summary["Reject"] + summary["Modify"]
    acceptance_rate = (
        round(summary["Accept"] / total_reviewed, 3) if total_reviewed > 0 else 0.0
    )

    output = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "reviewer":        reviewer_name,
        "queue_name":      queue_name,
        "notes":           notes,
        "reviews":         reviews,
        "summary":         summary,
        "total_reviewed":  total_reviewed,
        "acceptance_rate": acceptance_rate,
        "complete":        summary["Pending"] == 0,
        "hcai_note": (
            "This file is an auditable human-in-the-loop review ledger. "
            "It does not establish independent expert validation, novelty, "
            "or precision."
        ),
    }

    if candidate_index is not None:
        updated_index = []
        for item in candidate_index:
            updated = dict(item)
            decision = reviews.get(item["review_key"], "Pending")
            updated["decision"] = decision
            if decision != "Pending" and updated.get("decision_source") == "pending_author_review":
                updated["decision_source"] = "author_ui_review"
            updated_index.append(updated)
        output["candidate_count"] = len(updated_index)
        output["candidate_index"] = updated_index

    with open(reviews_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def get_review_badge_html(decision):
    """Return a coloured HTML badge for the review decision."""
    css_class = {
        "Accept":  "review-accept",
        "Reject":  "review-reject",
        "Modify":  "review-modify",
        "Pending": "review-pending",
    }.get(decision, "review-pending")
    return f'<span class="{css_class}">{decision}</span>'


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="brand-lockup">
  <span class="brand-mark">ESV</span>
  <h1>ESV-Gap</h1>
  <p>Evidence-clear research gap discovery with temporal knowledge graphs.</p>
</div>
""", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="rail-label">Workspace access</div>', unsafe_allow_html=True)
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com",
    )
    groq_key = (
        groq_key
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()

    # FIX 8: Reviewer identity field
    st.divider()
    reviewer_name = st.text_input(
        "Reviewer name",
        value="author_internal",
        help=(
            "Saved in expert_reviews.json for HCAI accountability. "
            "Use your name or role (e.g. 'domain_expert_1')."
        ),
    )

    st.divider()
    st.markdown('<div class="rail-label">Evidence pipeline</div>', unsafe_allow_html=True)
    stages = [
        "Collect papers", "Screen corpus", "Extract triples", "Build temporal KG",
        "Detect candidates", "Score and rank", "Audit evidence", "Compare baselines",
    ]
    stage_html = "".join(
        f'<div class="pipeline-step"><span>{index:02d}</span><span>{stage}</span></div>'
        for index, stage in enumerate(stages, start=1)
    )
    st.markdown(f'<div class="pipeline-rail">{stage_html}</div>', unsafe_allow_html=True)

    # FIX 7: Incremental update option
    st.divider()
    st.markdown('<div class="rail-label">Graph update</div>', unsafe_allow_html=True)
    run_incremental = st.checkbox(
        "Incremental update",
        value=False,
        help=(
            "Only collect papers not already in the corpus. "
            "Merges new papers into the existing graph — "
            "the 'Dynamic' property of the KG (Fix 7)."
        ),
    )

    st.markdown("""
<div class="rail-colophon">
ESV-GAP V3 · FPT UNIVERSITY<br>
PROVENANCE-AWARE · EXPERT-IN-THE-LOOP
</div>
""", unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────
st.markdown("""
<header class="workbench-masthead">
  <div>
    <span class="workbench-kicker">Temporal KG · Evidence validation</span>
    <h1>ESV-Gap</h1>
  </div>
  <p class="workbench-lede">
    A provenance-aware research workbench for discovering, auditing, and comparing
    gap candidates across an academic corpus.
  </p>
</header>
<div class="command-intro">
  <h2>Start an evidence run</h2>
  <span class="command-hint">01 / DEFINE CORPUS</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. federated learning privacy, medical image segmentation...",
    )
with col2:
    num_papers = st.slider("Max papers", 20, 150, 50, 10)

run_button = st.button(
    "Run evidence discovery",
    type="primary",
    disabled=not (topic and groq_key),
)

if not groq_key:
    st.info("Add a Groq API key in the evidence rail to begin a new run.")

# ── Pipeline execution ───────────────────────────────────────
if run_button and topic and groq_key:
    groq_key = (
        groq_key
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    if not groq_key:
        st.error("Groq API key is missing. Enter a key beginning with `gsk_...`.")
        st.stop()

    # Keep the key available to CLI-style pipeline modules as well as config.
    os.environ["GROQ_API_KEY"] = groq_key
    base_cfg = load_base_config()

    with st.spinner("Generating search queries..."):
        queries = generate_queries_with_llm(topic, groq_key)

    cfg, run_id = build_run_config(base_cfg, topic, num_papers, groq_key)
    cfg["collection"]["queries"]      = queries
    cfg["collection"]["incremental"]  = run_incremental   # FIX 7: pass flag into config

    st.markdown(f"**Queries:** `{'` · `'.join(queries)}`")
    if run_incremental:
        st.info("🔄 Incremental mode: only new papers will be collected and merged into the existing graph.")

    status_box   = st.empty()
    progress_bar = st.progress(0)
    log_box      = st.empty()
    logs         = []

    pq = queue.Queue()
    t  = threading.Thread(target=run_pipeline_with_progress, args=(cfg, pq), daemon=True)
    t.start()

    done_cfg = None
    error    = None

    while t.is_alive() or not pq.empty():
        try:
            msg_type, payload = pq.get(timeout=0.5)
            if msg_type == "status":
                status_box.markdown(f"**{payload}**")
                logs.append(payload)
                log_box.markdown("\n".join(f"- {l}" for l in logs))
            elif msg_type == "progress":
                progress_bar.progress(payload)
            elif msg_type == "done":
                done_cfg = payload
            elif msg_type == "error":
                error = payload
        except queue.Empty:
            continue

    t.join()

    if error:
        st.error(f"Pipeline failed:\n```\n{error}\n```")
        st.stop()

    if done_cfg:
        st.success("Evidence pipeline complete. The latest run is ready for review.")
        st.session_state["results"]     = load_results(done_cfg)
        st.session_state["run_cfg"]     = done_cfg
        st.session_state["topic"]       = topic
        st.session_state["gap_reviews"] = {}   # FIX 8: initialise review state


# ── Results display ──────────────────────────────────────────
latest_run = inspect_latest_run()
if "results" in st.session_state and latest_run:
    displayed_outputs = (
        st.session_state.get("run_cfg", {}).get("paths", {}).get("outputs")
    )
    displayed_run_id = Path(displayed_outputs).parent.name if displayed_outputs else None
    if displayed_run_id != latest_run["run_id"]:
        for key in ("results", "run_cfg", "topic", "gap_reviews"):
            st.session_state.pop(key, None)

if "results" not in st.session_state:
    if latest_run and not latest_run["completed"]:
        if latest_run["stage"] == "extracting":
            done = latest_run["extraction_done"]
            total = latest_run["total_papers"]
            activity = "still running" if latest_run["active"] else "paused or interrupted"
            st.warning(
                f"Newest run '{latest_run['topic']}' is {activity}: extraction "
                f"{done}/{total} papers, {latest_run['triple_count']} triples "
                "checkpointed. Older results are hidden so they are not mistaken "
                "for this run. Refresh the page to update the count."
            )
            if total:
                st.progress(min(done / total, 1.0))

            resume_button = st.button(
                "Resume extraction",
                type="primary",
                disabled=not groq_key,
                key=f"resume_{latest_run['run_id']}",
                help=(
                    "Continue in the same run directory. Completed papers are "
                    "skipped; collection and screening are not repeated."
                ),
            )
            if not groq_key:
                st.caption("Add a Groq API key in the evidence rail to enable resume.")

            if resume_button:
                os.environ["GROQ_API_KEY"] = groq_key
                resume_cfg = build_resume_config(latest_run, groq_key)
                resume_status = st.empty()
                resume_progress = st.progress(min(done / max(total, 1), 1.0))
                resume_log_box = st.empty()
                resume_logs = []

                resume_queue = queue.Queue()
                resume_thread = threading.Thread(
                    target=resume_pipeline_with_progress,
                    args=(resume_cfg, resume_queue),
                    daemon=True,
                )
                resume_thread.start()

                resumed_cfg = None
                resume_error = None
                while resume_thread.is_alive() or not resume_queue.empty():
                    try:
                        msg_type, payload = resume_queue.get(timeout=0.5)
                        if msg_type == "status":
                            resume_status.markdown(f"**{payload}**")
                            resume_logs.append(payload)
                            resume_log_box.markdown(
                                "\n".join(f"- {line}" for line in resume_logs)
                            )
                        elif msg_type == "progress":
                            resume_progress.progress(payload)
                        elif msg_type == "done":
                            resumed_cfg = payload
                        elif msg_type == "error":
                            resume_error = payload
                    except queue.Empty:
                        continue

                resume_thread.join()

                if resume_error:
                    st.error(f"Resume failed:\n```\n{resume_error}\n```")
                elif resumed_cfg:
                    st.session_state["results"] = load_results(resumed_cfg)
                    st.session_state["run_cfg"] = resumed_cfg
                    st.session_state["topic"] = latest_run["topic"]
                    st.session_state["gap_reviews"] = {}
                    st.success("Resume complete. The evidence run is ready for review.")
                    st.rerun()
        else:
            st.warning(
                f"Newest run '{latest_run['topic']}' is not complete "
                f"(current stage: {latest_run['stage']}). Older results are hidden."
            )
    else:
        recovered_cfg, recovered_topic = load_latest_completed_run()
        if recovered_cfg:
            recovered_results = load_results(recovered_cfg)
            if recovered_results.get("gaps"):
                st.session_state["results"] = recovered_results
                st.session_state["run_cfg"] = recovered_cfg
                st.session_state["topic"] = recovered_topic
                st.session_state["gap_reviews"] = {}
                st.success(
                    f"Loaded the latest completed run: {recovered_topic} "
                    f"({len(recovered_results['gaps'])} gaps)."
                )


if "results" in st.session_state:
    results = st.session_state["results"]
    cfg     = st.session_state["run_cfg"]
    topic   = st.session_state.get("topic", "")

    safe_topic = escape(topic)
    st.markdown(f"""
<div class="results-head">
  <div>
    <span class="section-kicker">Evidence snapshot</span>
    <h2>{safe_topic}</h2>
  </div>
  <span class="results-status">Latest completed run</span>
</div>
""", unsafe_allow_html=True)

    gaps = results.get("gaps", [])
    G    = results.get("graph")
    run_stats = results.get("run_stats", {})
    if run_stats.get("retained"):
        st.info(
            f"Run corpus: {run_stats.get('collected', 0)} collected, "
            f"{run_stats['retained']} retained, "
            f"{run_stats.get('nonempty_paper_outputs', 0)}/"
            f"{run_stats.get('paper_outputs', 0)} papers produced triples, "
            f"{run_stats.get('triples', 0)} triples total."
        )
        empty_outputs = (
            run_stats.get("paper_outputs", 0)
            - run_stats.get("nonempty_paper_outputs", 0)
        )
        if empty_outputs:
            st.warning(
                f"Extraction coverage is incomplete: {empty_outputs} retained "
                "papers produced no valid triples, including calls affected by "
                "the Groq quota. Treat these candidates as provisional."
            )

    raw_counts = results.get("raw_gap_counts", {})
    if raw_counts:
        st.caption(
            "Raw detector signals: "
            f"{sum(raw_counts.values())} total "
            f"({raw_counts.get('missing_links', 0)} missing links, "
            f"{raw_counts.get('orphan_clusters', 0)} orphan clusters, "
            f"{raw_counts.get('temporal_decay', 0)} temporal decay)."
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Ranked candidates shown", len(gaps))
    with col2: st.metric("Missing Links",      sum(1 for g in gaps if g["type"] == "missing_link"))
    with col3: st.metric("Orphan Clusters",    sum(1 for g in gaps if g["type"] == "orphan_cluster"))
    with col4: st.metric("Decaying Concepts",  sum(1 for g in gaps if g["type"] == "temporal_decay"))

    if G:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("KG Nodes",    G.number_of_nodes())
        with col2: st.metric("KG Edges",    G.number_of_edges())
        with col3: st.metric("Components",  nx.number_weakly_connected_components(G))

    st.divider()

    paper_index = results.get("paper_index", {})

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Candidates & Review",
        "Source Papers",
        "Knowledge Graph",
        "Evidence Analytics",
        "KG vs RAG",
    ])

    # ── Tab 1 — Ranked Gaps + FIX 8 Expert Review ───────────────
    with tab1:
        st.subheader("Ranked candidates and expert review")

        # FIX 8: HCAI info banner
        st.info(
            "**Stage 6 — Human Expert Review (HCAI)**  \n"
            "Use the review controls below to Accept, Reject, or flag gaps for Modification. "
            "Decisions are saved to `expert_reviews.json` and cited in the paper as Stage 6 "
            "human-in-the-loop validation (Shneiderman 2020). "
            "This is what distinguishes responsible AI synthesis from fully automated generation.",
        )

        type_filter = st.multiselect(
            "Filter by type",
            ["missing_link", "orphan_cluster", "temporal_decay"],
            default=["missing_link", "orphan_cluster", "temporal_decay"],
        )

        TYPE_META = {
            "missing_link":   ("ML", "missing", "Missing Link"),
            "orphan_cluster": ("OC", "orphan",  "Orphan Cluster"),
            "temporal_decay": ("TD", "decay",   "Temporal Decay"),
        }

        post_gate_report = results.get("post_gate_expert_reviews")
        review_scope_options = ["Score-ranked top 30"]
        if post_gate_report:
            review_scope_options.insert(0, "Post-gate review_required")
        review_scope = st.radio(
            "Review queue",
            review_scope_options,
            horizontal=True,
            help="Complete the post-gate queue before reporting its human-review result.",
        )

        if review_scope == "Post-gate review_required":
            review_state_key = "post_gate_gap_reviews"
            saved_review = post_gate_report
            review_filename = "post_gate_expert_reviews.json"
            queue_name = "post_gate_review_required"
            candidate_index = post_gate_report.get("candidate_index", [])
            display_gaps = []
            for item in candidate_index:
                gap = dict(item["candidate"])
                gap["rank"] = f"PG-{int(item['queue_position']):02d}"
                gap["composite_score"] = item.get("ranking_score") or 0.0
                gap["_review_key"] = item["review_key"]
                gap["_review_evidence"] = (
                    f"supporting papers: {item.get('supporting_paper_count', 0)}; "
                    f"coverage hits: {item.get('closure_hit_count', 0)}; "
                    f"gate reasons: {', '.join(item.get('validation_reasons', []))}"
                )
                display_gaps.append(gap)
            st.caption(
                f"Frozen post-gate queue: {len(display_gaps)} candidates. "
                "Prior decisions are carried only by exact candidate-identity match."
            )
        else:
            review_state_key = "gap_reviews"
            saved_review = results.get("expert_reviews", {})
            review_filename = "expert_reviews.json"
            queue_name = "score_ranked_top30"
            candidate_index = None
            display_gaps = gaps

        provenance_by_gap = {
            provenance_candidate_identity(gap): resolve_gap_provenance(
                G,
                gap,
                paper_index,
            )
            for gap in display_gaps
        }

        if review_state_key not in st.session_state:
            st.session_state[review_state_key] = (
                saved_review.get("reviews", {}) if saved_review else {}
            )

        reviews = st.session_state[review_state_key]

        for g in [x for x in display_gaps if x["type"] in type_filter][:30]:
            icon, css, label = TYPE_META.get(g["type"], ("—", "", g["type"]))
            gap_key          = g.get("_review_key", f"review_{g['rank']}")
            current_decision = reviews.get(gap_key, "Pending")

            # Gap card + review controls side by side
            col_card, col_review = st.columns([3, 1])

            with col_card:
                badge_html = get_review_badge_html(current_decision)
                st.markdown(f"""
<div class="gap-card {css}">
  <strong>#{g['rank']} {icon} {label}</strong>
  &nbsp;<code>score: {g.get('composite_score', 0):.4f}</code>
  &nbsp;{badge_html}<br>
<small>{g.get('description', '')}</small><br>
  <small>{g.get('_review_evidence', '')}</small>
</div>""", unsafe_allow_html=True)
                provenance_key = provenance_candidate_identity(g)
                render_gap_source_evidence(
                    provenance_by_gap.get(provenance_key, {}),
                    key_prefix=f"{review_state_key}:{gap_key}:sources",
                )

            with col_review:
                new_decision = st.selectbox(
                    "Decision",
                    options=["Pending", "Accept", "Reject", "Modify"],
                    index=["Pending", "Accept", "Reject", "Modify"].index(current_decision),
                    key=f"{review_state_key}:{gap_key}",
                    label_visibility="collapsed",
                )
                reviews[gap_key] = new_decision

                # Show modification text box when reviewer selects Modify
                if new_decision == "Modify":
                    note_key = f"{review_state_key}:note_{g['rank']}"
                    st.text_area(
                        "Suggested modification",
                        key=note_key,
                        placeholder="Describe what should be changed...",
                        height=80,
                        label_visibility="collapsed",
                    )

        # Update session state after all widgets render
        st.session_state[review_state_key] = reviews

        st.divider()

        # FIX 8: Review summary and save controls
        st.subheader("Review ledger")
        summary_counts = {k: sum(1 for v in reviews.values() if v == k)
                          for k in ["Accept", "Reject", "Modify", "Pending"]}
        total_reviewed = (summary_counts["Accept"]
                          + summary_counts["Reject"]
                          + summary_counts["Modify"])
        acceptance_rate = (
            round(summary_counts["Accept"] / total_reviewed * 100, 1)
            if total_reviewed > 0 else 0.0
        )

        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1: st.metric("Accepted", summary_counts["Accept"])
        with rc2: st.metric("Rejected", summary_counts["Reject"])
        with rc3: st.metric("Modify",   summary_counts["Modify"])
        with rc4: st.metric("Pending",  summary_counts["Pending"])
        with rc5: st.metric("Acceptance %", f"{acceptance_rate}%")

        review_notes = st.text_area(
            "Overall reviewer notes (optional)",
            placeholder="e.g. 'Gaps 1–5 confirmed by domain expertise. Gap 8 overlaps with known work by Chen et al.'",
            height=80,
        )

        save_col, dl_col = st.columns([1, 1])
        with save_col:
            if st.button("Save expert reviews", type="primary"):
                saved_output = save_expert_reviews(
                    cfg,
                    reviews,
                    reviewer_name,
                    notes=review_notes,
                    filename=review_filename,
                    queue_name=queue_name,
                    candidate_index=candidate_index,
                )
                st.success(
                    f"Reviews saved to `{cfg['paths']['outputs']}/{review_filename}`  \n"
                    f"Acceptance rate: **{saved_output['acceptance_rate']*100:.1f}%** "
                    f"({saved_output['summary']['Accept']} / {saved_output['total_reviewed']} reviewed)  \n"
                    f"Cite this as Stage 6 HCAI evidence in Section 6.3 of the paper."
                )

        with dl_col:
            if st.button("Prepare reviewed-gap CSV"):
                reviewed_rows = []
                for g in display_gaps:
                    k = g.get("_review_key", f"review_{g['rank']}")
                    gap_provenance = provenance_by_gap.get(
                        provenance_candidate_identity(g), {}
                    )
                    reviewed_rows.append({
                        "rank":        g["rank"],
                        "type":        g["type"],
                        "score":       g.get("composite_score", 0),
                        "description": g.get("description", "")[:200],
                        "source_paper_ids": "; ".join(
                            gap_provenance.get("paper_ids", [])
                        ),
                        "source_paper_titles": "; ".join(
                            paper.get("title", "")
                            for paper in gap_provenance.get("papers", [])
                        ),
                        "decision":    reviews.get(k, "Pending"),
                    })
                df_review = pd.DataFrame(reviewed_rows)
                st.download_button(
                    "Download",
                    df_review.to_csv(index=False),
                    file_name=f"gap_reviews_{topic.replace(' ', '_')}.csv",
                    mime="text/csv",
                )

        # Show previously saved reviews if they exist
        if saved_review:
            prev = saved_review
            st.caption(
                f"Last saved: {prev.get('timestamp','?')} · "
                f"Reviewer: {prev.get('reviewer','?')} · "
                f"Acceptance rate: {prev.get('acceptance_rate',0)*100:.1f}%"
            )

        # Original CSV download (unchanged)
        if "gaps_df" in results:
            st.download_button(
                "Download full ranked gaps CSV",
                results["gaps_df"].to_csv(index=False),
                file_name=f"kg_gaps_{topic.replace(' ', '_')}.csv",
                mime="text/csv",
            )

    # ── Tab 2 — Knowledge Graph (unchanged) ─────────────────────
    with tab2:
        st.subheader("Source papers by gap candidate")
        st.caption(
            "This view traces each candidate to papers supporting its graph evidence. "
            "A source paper is not necessarily a paper that explicitly claims the gap."
        )

        paper_gap_records = {}
        for gap in display_gaps:
            gap_identity = provenance_candidate_identity(gap)
            provenance = provenance_by_gap.get(gap_identity, {})
            for paper in provenance.get("papers", []):
                paper_id = paper["paper_id"]
                if paper_id not in paper_gap_records:
                    paper_gap_records[paper_id] = {**paper, "gaps": []}
                paper_gap_records[paper_id]["gaps"].append({
                    "rank": gap.get("rank", "?"),
                    "type": gap.get("type", "unknown"),
                    "description": gap.get("description", ""),
                    "evidence": paper.get("evidence", []),
                })

        source_col1, source_col2, source_col3 = st.columns(3)
        with source_col1:
            st.metric("Source papers", len(paper_gap_records))
        with source_col2:
            st.metric(
                "Gap–paper links",
                sum(len(item["gaps"]) for item in paper_gap_records.values()),
            )
        with source_col3:
            st.metric(
                "Candidates with provenance",
                sum(1 for value in provenance_by_gap.values() if value.get("papers")),
            )

        source_search = st.text_input(
            "Search source papers",
            placeholder="Title, author, venue, or paper ID",
            key=f"source-paper-search:{queue_name}",
        ).strip().lower()
        source_types = st.multiselect(
            "Filter source papers by gap type",
            ["missing_link", "orphan_cluster", "temporal_decay"],
            default=["missing_link", "orphan_cluster", "temporal_decay"],
            key=f"source-paper-types:{queue_name}",
        )

        filtered_source_papers = []
        for paper in paper_gap_records.values():
            haystack = " ".join([
                paper.get("title", ""),
                " ".join(paper.get("authors", [])),
                paper.get("venue", ""),
                paper.get("paper_id", ""),
            ]).lower()
            paper_types = {gap["type"] for gap in paper["gaps"]}
            if source_search and source_search not in haystack:
                continue
            if not paper_types.intersection(source_types):
                continue
            filtered_source_papers.append(paper)

        filtered_source_papers.sort(key=lambda item: (
            -len(item["gaps"]),
            -(int(item.get("year") or 0)),
            item.get("title", "").lower(),
        ))

        if not filtered_source_papers:
            st.warning("No source papers match the current filters.")
        else:
            source_rows = []
            for paper in filtered_source_papers:
                gap_labels = [f"#{gap['rank']} {gap['type']}" for gap in paper["gaps"]]
                source_rows.append({
                    "Paper": paper["title"],
                    "Year": paper.get("year"),
                    "Authors": ", ".join(paper.get("authors", [])[:4]),
                    "Venue": paper.get("venue", ""),
                    "Citations": paper.get("citation_count", 0),
                    "Gap count": len(paper["gaps"]),
                    "Gap candidates": ", ".join(gap_labels),
                    "URL": paper.get("url", ""),
                })
            source_df = pd.DataFrame(source_rows)
            st.dataframe(
                source_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "URL": st.column_config.LinkColumn("Paper link", display_text="Open"),
                    "Paper": st.column_config.TextColumn("Paper", width="large"),
                    "Gap candidates": st.column_config.TextColumn(
                        "Gap candidates", width="large"
                    ),
                },
            )
            st.download_button(
                "Download gap–paper mapping CSV",
                source_df.to_csv(index=False),
                file_name=f"gap_source_papers_{topic.replace(' ', '_')}.csv",
                mime="text/csv",
            )

            selected_paper_id = st.selectbox(
                "Inspect one source paper",
                [paper["paper_id"] for paper in filtered_source_papers],
                format_func=lambda paper_id: paper_gap_records[paper_id]["title"],
                key=f"source-paper-detail:{queue_name}",
            )
            selected_paper = paper_gap_records[selected_paper_id]
            with st.container(border=True):
                st.markdown(f"### {selected_paper['title']}")
                selected_meta = []
                if selected_paper.get("authors"):
                    selected_meta.append(", ".join(selected_paper["authors"]))
                if selected_paper.get("year"):
                    selected_meta.append(str(selected_paper["year"]))
                if selected_paper.get("venue"):
                    selected_meta.append(selected_paper["venue"])
                selected_meta.append(f"{selected_paper.get('citation_count', 0)} citations")
                st.caption(" · ".join(selected_meta))
                if selected_paper.get("url"):
                    st.link_button(
                        "Open paper",
                        selected_paper["url"],
                        key=f"source-paper-detail-link:{queue_name}:{selected_paper_id}",
                    )
                if selected_paper.get("abstract"):
                    with st.expander("Abstract"):
                        st.write(selected_paper["abstract"])

                st.markdown("#### Associated gap candidates")
                for gap_reference in selected_paper["gaps"]:
                    with st.container(border=True):
                        st.markdown(
                            f"**#{gap_reference['rank']} · "
                            f"{gap_reference['type'].replace('_', ' ').title()}**"
                        )
                        st.write(gap_reference["description"])
                        for evidence_item in gap_reference.get("evidence", [])[:5]:
                            st.markdown(
                                f"`{evidence_item.get('subject', '')} "
                                f"—[{evidence_item.get('relation', '')}]→ "
                                f"{evidence_item.get('object', '')}`"
                            )
                            if evidence_item.get("evidence"):
                                st.caption(f"Evidence: {evidence_item['evidence']}")

    with tab3:
        st.subheader("Interactive evidence graph")
        st.caption("Nodes = concepts · Size = centrality · Red border = gap node · Dashed red = predicted missing link")

        if results.get("graph_html"):
            graph_path = Path(cfg["paths"]["outputs"]) / "graph_viz.html"
            if graph_path.exists():
                st.iframe(graph_path, height=650)
        else:
            st.warning("Graph visualisation not available.")

        if G:
            st.markdown("**Top 10 most connected concepts:**")
            G_simple  = nx.DiGraph(G)
            deg_cent  = nx.degree_centrality(G_simple)
            top_nodes = sorted(deg_cent.items(), key=lambda x: -x[1])[:10]
            df_nodes  = pd.DataFrame(top_nodes, columns=["Concept", "Centrality"])
            df_nodes["Type"] = df_nodes["Concept"].apply(
                lambda n: G.nodes[n].get("type", "?") if G.has_node(n) else "?"
            )
            st.table(df_nodes.astype(str))

    # ── Tab 3 — Analytics (unchanged) ───────────────────────────
    with tab4:
        st.subheader("Evidence analytics")

        for fig_name, caption in [
            ("gap_analysis.png",              "Gap distribution and scores"),
            ("temporal_decay.png",            "Temporal decay profiles"),
            ("graph_stats.png",               "Knowledge graph statistics"),
            ("figure4_kg_publication.png",    "Figure 4 — publication-ready KG figure (Fix 1)"),
        ]:
            p = Path(cfg["paths"]["figures"]) / fig_name
            if p.exists():
                st.image(str(p), caption=caption)

        if G:
            rel_counts = {}
            for _, _, d in G.edges(data=True):
                r = d.get("relation", "?")
                rel_counts[r] = rel_counts.get(r, 0) + 1
            df_rel = pd.DataFrame(
                sorted(rel_counts.items(), key=lambda x: -x[1]),
                columns=["Relation", "Count"],
            )
            st.markdown("**Edge relation distribution:**")
            st.bar_chart(df_rel.set_index("Relation"))

    # ── Tab 4: RAG Comparison (unchanged) ───────────────────────
    with tab5:
        st.subheader("Method comparison: KG vs RAG baselines")

        if not results.get("mulla_gaps"):
            corpus_path = Path(cfg["paths"]["processed_data"]) / "corpus_filtered.jsonl"
            actual_corpus_size = 0
            if corpus_path.exists():
                with open(corpus_path, encoding="utf-8") as corpus_file:
                    actual_corpus_size = sum(1 for line in corpus_file if line.strip())
            st.info(
                "RAG baselines haven't been run yet.\n\n"
                "This will run **Mulla et al. RAG** and **Simple LLM** on the same "
                f"filtered corpus ({actual_corpus_size} papers) "
                "and compare results against the KG method."
            )

            if st.button(
                "Run RAG baselines",
                type="primary",
                disabled=not groq_key,
            ):
                cfg.setdefault("api_keys", {})["groq"] = groq_key
                os.environ["GROQ_API_KEY"] = groq_key
                rag_status   = st.empty()
                rag_progress = st.progress(0)
                rag_logs_box = st.empty()
                rag_logs     = []

                rpq = queue.Queue()
                rt  = threading.Thread(
                    target=run_rag_with_progress,
                    args=(cfg, rpq),
                    daemon=True,
                )
                rt.start()

                rag_done  = None
                rag_error = None

                while rt.is_alive() or not rpq.empty():
                    try:
                        mt, pl = rpq.get(timeout=0.5)
                        if mt == "status":
                            rag_status.markdown(f"**{pl}**")
                            rag_logs.append(pl)
                            rag_logs_box.markdown("\n".join(f"- {l}" for l in rag_logs))
                        elif mt == "progress":
                            rag_progress.progress(pl)
                        elif mt == "done":
                            rag_done = pl
                        elif mt == "error":
                            rag_error = pl
                    except queue.Empty:
                        continue

                rt.join()

                if rag_error:
                    st.error(f"RAG baseline failed:\n```\n{rag_error}\n```")
                elif rag_done:
                    st.success("RAG baselines complete. Comparison data is ready.")
                    st.session_state["results"] = load_results(cfg)
                    st.rerun()

        else:
            mulla_gaps  = results.get("mulla_gaps",  [])
            simple_gaps = results.get("simple_gaps", [])
            metrics     = results.get("comparison_metrics", {})

            kg_m  = metrics.get("kg",        {})
            mu_m  = metrics.get("mulla_rag",  {})
            si_m  = metrics.get("simple_llm", {})
            ov_m  = metrics.get("overlap",    {})

            # FIX 8: Show acceptance rate from expert reviews if available
            expert = results.get("expert_reviews")
            if expert:
                st.success(
                    f"🧑‍🔬 Expert review on file: "
                    f"**{expert.get('acceptance_rate',0)*100:.1f}% acceptance rate** "
                    f"({expert['summary'].get('Accept',0)} accepted / "
                    f"{expert.get('total_reviewed',0)} reviewed) — "
                    f"cite in Section 6.3 as Stage 6 HCAI evidence."
                )

            # Summary table
            st.markdown("### Method summary")
            summary_df = pd.DataFrame({
                "Metric": [
                    "Total gaps produced",
                    "Unique gaps",
                    "Traceable evidence",
                    "Reproducible",
                    "Cross-paper retrieval",
                    "Avg gap length (words)",
                    "Expert acceptance rate",
                ],
                "🔷 KG (Ours)": [
                    kg_m.get("total_gaps", "-"),
                    kg_m.get("unique_gaps", "-"),
                    "✅ Subgraph path",
                    "✅ Deterministic",
                    "✅ Corpus-wide",
                    kg_m.get("avg_description_len", "-"),
                    f"{expert.get('acceptance_rate',0)*100:.1f}%" if expert else "Pending review",
                ],
                "🟡 Mulla RAG": [
                    mu_m.get("total_gaps", "-"),
                    mu_m.get("unique_gaps", "-"),
                    "❌ Free text",
                    "❌ Stochastic",
                    "✅ Top-3 similar",
                    mu_m.get("avg_gap_length", "-"),
                    "Not evaluated",
                ],
                "⚪ Simple LLM": [
                    si_m.get("total_gaps", "-"),
                    si_m.get("unique_gaps", "-"),
                    "❌ Free text",
                    "❌ Stochastic",
                    "❌ Per-paper only",
                    si_m.get("avg_gap_length", "-"),
                    "Not evaluated",
                ],
            })
            st.table(summary_df.set_index("Metric"))

            # Overlap
            st.markdown("### Lexical overlap between methods")
            st.caption("Jaccard similarity — lower means methods find more complementary gaps")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("KG vs Mulla RAG",     f"{ov_m.get('kg_vs_mulla',  0):.3f}")
            with col2: st.metric("KG vs Simple LLM",    f"{ov_m.get('kg_vs_simple', 0):.3f}")
            with col3: st.metric("Mulla vs Simple LLM", f"{ov_m.get('mulla_vs_simple', 0):.3f}")

            st.divider()

            # Side-by-side sample
            st.markdown("### Gap samples — side by side")
            paper_titles = [g.get("title", f"Paper {i}") for i, g in enumerate(mulla_gaps[:20])]
            sel = st.selectbox(
                "Select paper",
                range(len(paper_titles)),
                format_func=lambda i: paper_titles[i],
            )

            if sel < len(mulla_gaps):
                mulla_gap  = mulla_gaps[sel]
                pid        = mulla_gap.get("paper_id", "")
                simple_gap = next(
                    (g for g in simple_gaps if g.get("paper_id") == pid),
                    simple_gaps[sel] if sel < len(simple_gaps) else {},
                )
                kg_sample = gaps[:3] if gaps else []

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="method-header">KG Method (Ours)</div>',
                                unsafe_allow_html=True)
                    for g in kg_sample:
                        css = g["type"].split("_")[0]
                        k   = f"review_{g['rank']}"
                        badge = get_review_badge_html(reviews.get(k, "Pending"))
                        st.markdown(f"""
<div class="gap-card {css}">
  <strong>{g.get('type','').replace('_',' ').title()}</strong>
  — score {g.get('composite_score',0):.3f} {badge}<br>
  <small>{g.get('description','')[:220]}</small>
</div>""", unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="method-header">Mulla et al. RAG</div>',
                                unsafe_allow_html=True)
                    for field, label in [
                        ("research_gaps",      "Research Gaps"),
                        ("remaining_gaps",     "Remaining Gaps"),
                        ("research_direction", "Direction"),
                    ]:
                        val = mulla_gap.get(field, "")
                        if val:
                            st.markdown(f"""
<div class="gap-card mulla">
  <strong>{label}</strong><br>
  <small>{val[:220]}</small>
</div>""", unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="method-header">Simple LLM</div>',
                                unsafe_allow_html=True)
                    for j in range(1, 4):
                        val = simple_gap.get(f"gap_{j}", "")
                        if val:
                            st.markdown(f"""
<div class="gap-card simple">
  <strong>Gap {j}</strong><br>
  <small>{val[:220]}</small>
</div>""", unsafe_allow_html=True)

            st.divider()
            if metrics:
                st.download_button(
                    "Download comparison metrics (JSON)",
                    json.dumps(metrics, indent=2),
                    file_name="comparison_metrics.json",
                    mime="application/json",
                )
