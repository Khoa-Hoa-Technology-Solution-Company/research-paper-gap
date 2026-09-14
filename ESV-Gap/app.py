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
from src.groq_key_pool import create_groq_client, normalise_groq_keys
from src.run_history import (
    config_for_run,
    list_run_history,
    run_label,
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


def build_run_config(base_config, topic, num_papers, groq_keys):
    import copy
    cfg = copy.deepcopy(base_config)
    groq_keys = normalise_groq_keys(groq_keys, include_environment=True)
    cfg["project"]["domain"]               = topic
    cfg["api_keys"]["groq_keys"]           = groq_keys
    cfg["api_keys"]["groq"]                = groq_keys[0] if groq_keys else ""
    # The slider is the requested screened-corpus size. Oversample the raw
    # pool because topic screening will reject some collected records.
    # Relevant-paper yield is often only 20-40%. A 4x pool gives the screener a
    # realistic chance of reaching the requested retained-corpus target.
    collection_pool_size = min(max(num_papers * 4, num_papers + 100), 600)
    cfg["collection"]["max_papers"]        = collection_pool_size
    cfg["filtering"]["target_corpus_size"] = min(num_papers, 150)
    cfg.setdefault("gap_validation", {})["snapshot_date"] = (
        datetime.datetime.now().date().isoformat()
    )

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
            "collection_pool_size": collection_pool_size,
            "created_at": datetime.datetime.now().isoformat(),
        }, indent=2),
        encoding="utf-8",
    )
    return cfg, run_id


def build_resume_config(run_info, groq_keys):
    """Reconstruct a safe config that points to an existing checkpointed run."""
    return config_for_run(load_base_config(), run_info, groq_keys)


def generate_queries_with_llm(topic, groq_keys):
    from src.query_planning import (
        fallback_search_queries,
        validate_generated_queries,
    )
    topic_lower = str(topic).casefold()
    search_topic = str(topic).strip()
    if "monolith" in topic_lower and any(
        token in topic_lower for token in ("security", "secure", "vulnerability")
    ):
        search_topic = "security of monolithic software architectures"
    fallback_queries = fallback_search_queries(search_topic)

    try:
        client = create_groq_client(
            explicit_keys=groq_keys,
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
                        "Disambiguate overloaded scientific terms, preserve the "
                        "essential topic anchors in every query, cover distinct "
                        "evidence strata, and return only a valid JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Generate exactly 5 academic search queries for "{topic}". '
                        f'The intended search scope is "{search_topic}". '
                        "The five queries must respectively target: (1) core topic, "
                        "(2) systematic reviews, (3) limitations/challenges, "
                        "(4) empirical comparisons, and (5) open problems/future "
                        "research. Use 2-7 plain keywords per query. Do not use "
                        "quotation marks, parentheses, AND, OR, NOT, or field syntax. "
                        "Each query should contain only the minimum useful anchors "
                        "and must not drift to another meaning. Return them in this "
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
        queries = validate_generated_queries(queries, search_topic)

        if len(queries) == 5:
            return queries

        logging.getLogger(__name__).warning(
            "Groq returned an invalid query list; using fallback queries. "
            "Response: %r",
            content,
        )
    except Exception as exc:
        from src.llm_errors import (
            LLMAuthenticationError,
            authentication_error_message,
            is_authentication_error,
        )
        if is_authentication_error(exc):
            raise LLMAuthenticationError(authentication_error_message()) from exc
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
            collected_papers = collect_papers(
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
        minimum_screenable = int(
            cfg.get("gap_synthesis", {}).get("min_screened_corpus_size", 30)
        )
        if len(collected_papers or []) < minimum_screenable:
            progress_queue.put((
                "error",
                f"Collection retained only {len(collected_papers or [])} papers "
                f"after adaptive query broadening; at least {minimum_screenable} "
                "raw papers are required before screening. No Groq screening "
                "quota was consumed. Retry the topic or use a broader formulation.",
            ))
            return
        progress_queue.put(("progress", 20))

        progress_queue.put(("status", "🔎 Screening papers for relevance..."))
        progress_queue.put(("progress", 22))
        from src.filter import filter_corpus
        from src.llm_errors import LLMAuthenticationError, LLMRateLimitError
        try:
            filtered_papers = filter_corpus(
                cfg,
                progress_callback=stage_progress(22, 40, "Screening papers"),
            )
        except (LLMAuthenticationError, LLMRateLimitError) as exc:
            progress_queue.put(("error", str(exc)))
            return
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
        progress_queue.put((
            "status",
            "Retrieving open-access full text for the strongest evidence cells...",
        ))
        from src.full_text import enrich_candidate_source_full_text
        full_text_report = enrich_candidate_source_full_text(cfg)
        if full_text_report.get("enriched", 0):
            detect_all_gaps(cfg)
        progress_queue.put(("progress", 82))

        progress_queue.put((
            "status",
            "Auditing evidence and searching for potentially closing literature...",
        ))
        progress_queue.put(("progress", 83))
        from src.validate_gaps import validate_all_gaps
        validate_all_gaps(cfg)
        progress_queue.put(("progress", 88))

        progress_queue.put((
            "status",
            "Synthesising convergent signals into an answerable research gap...",
        ))
        from src.synthesise_research_gap import synthesise_research_gap
        synthesise_research_gap(cfg)
        progress_queue.put(("progress", 92))

        progress_queue.put(("status", "📊 Scoring and ranking gaps..."))
        progress_queue.put(("progress", 91))
        from src.score_gaps import score_and_rank_gaps
        score_and_rank_gaps(cfg)
        progress_queue.put(("progress", 95))

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
        progress_queue.put((
            "status",
            "Retrieving open-access full text for the strongest evidence cells...",
        ))
        from src.full_text import enrich_candidate_source_full_text
        full_text_report = enrich_candidate_source_full_text(cfg)
        if full_text_report.get("enriched", 0):
            detect_all_gaps(cfg)

        progress_queue.put((
            "status",
            "Auditing evidence and searching for potentially closing literature...",
        ))
        progress_queue.put(("progress", 78))
        from src.validate_gaps import validate_all_gaps
        validate_all_gaps(cfg)

        progress_queue.put((
            "status",
            "Synthesising convergent signals into an answerable research gap...",
        ))
        progress_queue.put(("progress", 84))
        from src.synthesise_research_gap import synthesise_research_gap
        synthesise_research_gap(cfg)

        progress_queue.put(("status", "Scoring evidence-cleared candidates..."))
        progress_queue.put(("progress", 88))
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

    claims_path = Path(out) / "research_gap_claims.json"
    if claims_path.exists():
        with open(claims_path, encoding="utf-8") as f:
            results["research_gap_claims"] = json.load(f)

    synthesis_path = Path(out) / "research_gap_synthesis.json"
    if synthesis_path.exists():
        with open(synthesis_path, encoding="utf-8") as f:
            results["research_gap_synthesis"] = json.load(f)

    primary_gap_path = Path(out) / "primary_research_gap.json"
    if primary_gap_path.exists():
        with open(primary_gap_path, encoding="utf-8") as f:
            results["primary_research_gap"] = json.load(f)

    confirmed_claims_path = Path(out) / "confirmed_research_gaps.json"
    if confirmed_claims_path.exists():
        with open(confirmed_claims_path, encoding="utf-8") as f:
            results["confirmed_research_gaps"] = json.load(f)

    validation_audit_path = Path(out) / "gap_validation_audit.json"
    if validation_audit_path.exists():
        with open(validation_audit_path, encoding="utf-8") as f:
            results["validation_audit"] = json.load(f)

    review_required_path = Path(out) / "review_required_gaps.json"
    if review_required_path.exists():
        with open(review_required_path, encoding="utf-8") as f:
            results["review_required_gaps"] = json.load(f)

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

    validation_reviews_path = Path(out) / "validation_expert_reviews.json"
    if validation_reviews_path.exists():
        with open(validation_reviews_path, encoding="utf-8") as f:
            results["validation_expert_reviews"] = json.load(f)

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


def inspect_latest_run(runs_root="runs"):
    """Describe the newest run, including an in-progress extraction."""
    history = list_run_history(runs_root)
    return history[0] if history else None


def load_latest_completed_run(runs_root="runs"):
    """Recover the newest completed run after a Streamlit refresh/rerun."""
    for run_info in list_run_history(runs_root):
        if run_info["completed"]:
            return config_for_run(load_base_config(), run_info), run_info["topic"]

    return None, None


def save_expert_reviews(
    cfg,
    reviews,
    reviewer_name,
    notes="",
    filename="expert_reviews.json",
    queue_name="score_ranked_top30",
    candidate_index=None,
    candidates=None,
    review_rationales=None,
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

    review_rationales = review_rationales or {}
    output = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "reviewer":        reviewer_name,
        "queue_name":      queue_name,
        "notes":           notes,
        "reviews":         reviews,
        "rationales":      review_rationales,
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

    if queue_name == "evidence_cleared_claims" and candidates is not None:
        confirmed = []
        internal_reviewer = reviewer_name.strip().lower() in {
            "author_internal", "author", "internal"
        }
        for candidate in candidates:
            review_key = candidate.get("_review_key", f"review_{candidate['rank']}")
            if reviews.get(review_key) != "Accept":
                continue
            rationale = str(review_rationales.get(review_key, "")).strip()
            if not rationale:
                continue
            validation = candidate.get("validation", {}) or {}
            confirmed.append({
                "claim_id": provenance_candidate_identity(candidate),
                "claim": validation.get("scoped_claim", candidate.get("description", "")),
                "claim_status": (
                    "author_confirmed_scoped_gap"
                    if internal_reviewer
                    else "expert_confirmed_scoped_gap"
                ),
                "reviewer": reviewer_name,
                "review_rationale": rationale,
                "reviewed_at": output["timestamp"],
                "supporting_paper_ids": validation.get("supporting_paper_ids", []),
                "closure_hits": validation.get("closure_hits", []),
                "validation": validation,
                "candidate": candidate,
            })
        with open(out / "confirmed_research_gaps.json", "w", encoding="utf-8") as stream:
            json.dump(confirmed, stream, indent=2, ensure_ascii=False)

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
run_history = list_run_history()
run_history_by_id = {item["run_id"]: item for item in run_history}

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
    groq_key_count = int(st.number_input(
        "Number of Groq keys",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Requests rotate to the next key automatically after 429/quota errors.",
    ))
    entered_groq_keys = [
        st.text_input(
            f"Groq API Key {index + 1}",
            type="password",
            placeholder="gsk_...",
            help="Keys remain in memory and are not written to run metadata.",
            key=f"groq_api_key_{index + 1}",
        )
        for index in range(groq_key_count)
    ]
    groq_keys = normalise_groq_keys(entered_groq_keys, include_environment=True)
    if groq_keys:
        st.caption(
            f"{len(groq_keys)} unique key(s) ready · automatic quota failover enabled"
        )
    else:
        st.caption("No Groq key configured yet.")

    st.divider()
    st.markdown('<div class="rail-label">Run history</div>', unsafe_allow_html=True)
    history_options = [""] + [item["run_id"] for item in run_history]
    selected_history_id = st.selectbox(
        "Saved evidence runs",
        history_options,
        format_func=lambda run_id: (
            "Select a previous run..."
            if not run_id else run_label(run_history_by_id[run_id])
        ),
        help="Open completed results or select an interrupted checkpoint to resume it.",
    )
    open_saved_run = st.button(
        "Open selected run",
        disabled=not selected_history_id,
        width="stretch",
    )

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
        "Detect signals", "Validate evidence", "Synthesise research gap",
        "Score candidates", "Compare baselines",
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
if open_saved_run and selected_history_id:
    selected_run = run_history_by_id[selected_history_id]
    st.session_state["history_focus_run_id"] = selected_history_id
    if selected_run["completed"]:
        selected_cfg = config_for_run(load_base_config(), selected_run, groq_keys)
        st.session_state["results"] = load_results(selected_cfg)
        st.session_state["run_cfg"] = selected_cfg
        st.session_state["topic"] = selected_run["topic"]
        st.session_state["displayed_run_id"] = selected_history_id
        st.session_state["gap_reviews"] = {}
    else:
        for state_key in (
            "results", "run_cfg", "topic", "displayed_run_id", "gap_reviews"
        ):
            st.session_state.pop(state_key, None)

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
    num_papers = st.slider(
        "Target screened papers",
        20,
        150,
        50,
        10,
        help=(
            "The collector automatically retrieves a larger raw pool so the "
            "requested number can still remain after relevance screening."
        ),
    )

minimum_gap_corpus = int(
    load_base_config().get("gap_synthesis", {}).get("min_screened_corpus_size", 30)
)
if num_papers < minimum_gap_corpus:
    st.warning(
        f"Automatic research-gap synthesis requires at least "
        f"{minimum_gap_corpus} screened papers. A {num_papers}-paper run remains "
        "exploratory and will fail the corpus-size hard gate."
    )

run_button = st.button(
    "Run evidence discovery",
    type="primary",
    disabled=not (topic and groq_keys),
)

if not groq_keys:
    st.info("Add at least one Groq API key in the evidence rail to begin a new run.")

# ── Pipeline execution ───────────────────────────────────────
if run_button and topic and groq_keys:
    if not groq_keys:
        st.error("Groq API keys are missing. Enter at least one `gsk_...` key.")
        st.stop()

    # Keep the key available to CLI-style pipeline modules as well as config.
    os.environ["GROQ_API_KEY"] = groq_keys[0]
    base_cfg = load_base_config()

    from src.llm_errors import LLMAuthenticationError
    try:
        with st.spinner("Generating search queries..."):
            queries = generate_queries_with_llm(topic, groq_keys)
    except LLMAuthenticationError as exc:
        st.error(str(exc))
        st.stop()

    cfg, run_id = build_run_config(base_cfg, topic, num_papers, groq_keys)
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
        st.session_state["displayed_run_id"] = run_id
        st.session_state["history_focus_run_id"] = run_id
        st.session_state["gap_reviews"] = {}   # FIX 8: initialise review state


# ── Results display ──────────────────────────────────────────
focused_run_id = st.session_state.get("history_focus_run_id")
latest_run = run_history_by_id.get(focused_run_id) or inspect_latest_run()

if "results" not in st.session_state:
    if latest_run and not latest_run["completed"]:
        if latest_run["stage"] == "extracting":
            done = latest_run["extraction_done"]
            total = latest_run["total_papers"]
            activity = "still running" if latest_run["active"] else "paused or interrupted"
            st.warning(
                f"Selected run '{latest_run['topic']}' is {activity}: extraction "
                f"{done}/{total} papers, {latest_run['triple_count']} triples "
                "checkpointed. Resume continues in this same run directory."
            )
            if total:
                st.progress(min(done / total, 1.0))

            resume_button = st.button(
                "Resume extraction",
                type="primary",
                disabled=not groq_keys,
                key=f"resume_{latest_run['run_id']}",
                help=(
                    "Continue in the same run directory. Completed papers are "
                    "skipped; collection and screening are not repeated."
                ),
            )
            if not groq_keys:
                st.caption("Add at least one Groq API key to enable resume.")

            if resume_button:
                os.environ["GROQ_API_KEY"] = groq_keys[0]
                resume_cfg = build_resume_config(latest_run, groq_keys)
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
                    st.session_state["displayed_run_id"] = latest_run["run_id"]
                    st.session_state["history_focus_run_id"] = latest_run["run_id"]
                    st.session_state["gap_reviews"] = {}
                    st.success("Resume complete. The evidence run is ready for review.")
                    st.rerun()
        else:
            st.warning(
                f"Selected run '{latest_run['topic']}' is not complete "
                f"(current stage: {latest_run['stage']})."
            )
    else:
        recovered_cfg, recovered_topic = load_latest_completed_run()
        if recovered_cfg:
            recovered_results = load_results(recovered_cfg)
            if recovered_results.get("graph") or recovered_results.get("validation_audit"):
                st.session_state["results"] = recovered_results
                st.session_state["run_cfg"] = recovered_cfg
                st.session_state["topic"] = recovered_topic
                recovered_outputs = recovered_cfg.get("paths", {}).get("outputs", "")
                recovered_run_id = Path(recovered_outputs).parent.name if recovered_outputs else ""
                st.session_state["displayed_run_id"] = recovered_run_id
                st.session_state["history_focus_run_id"] = recovered_run_id
                st.session_state["gap_reviews"] = {}
                st.success(
                    f"Loaded the latest completed run: {recovered_topic} "
                    f"({len(recovered_results.get('gaps', []))} evidence-cleared claims)."
                )


if "results" in st.session_state:
    results = st.session_state["results"]
    cfg     = st.session_state["run_cfg"]
    topic   = st.session_state.get("topic", "")
    displayed_run_id = st.session_state.get("displayed_run_id", "")

    safe_topic = escape(topic)
    st.markdown(f"""
<div class="results-head">
  <div>
    <span class="section-kicker">Evidence snapshot</span>
    <h2>{safe_topic}</h2>
  </div>
  <span class="results-status">Saved run · {escape(displayed_run_id)}</span>
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
            f"({raw_counts.get('evidence_gaps', 0)} explicit limitations, "
            f"{raw_counts.get('missing_links', 0)} missing links, "
            f"{raw_counts.get('orphan_clusters', 0)} orphan clusters, "
            f"{raw_counts.get('temporal_decay', 0)} temporal decay)."
        )

    validation_summary = results.get("validation_audit", {}).get("summary", {})
    if validation_summary:
        st.success(
            f"Evidence gate: {validation_summary.get('automatically_eligible', 0)} "
            "evidence-cleared claim(s), "
            f"{validation_summary.get('review_required', 0)} require expert review, "
            f"{validation_summary.get('rejected', 0)} rejected. "
            f"Candidate-specific closure searches completed: "
            f"{validation_summary.get('external_closure_searches_completed', 0)}."
        )
        closure_attempts = validation_summary.get(
            "external_closure_searches_attempted", 0
        )
        closure_completed = validation_summary.get(
            "external_closure_searches_completed", 0
        )
        if closure_attempts and closure_completed < closure_attempts:
            st.error(
                "Automatic novelty verification is incomplete: "
                f"{closure_completed}/{closure_attempts} candidate searches "
                "completed. A null research-gap result from this run means the "
                "evidence contract could not finish; it does not mean that the "
                "literature contains no gap. Retry validation when a scholarly "
                "search provider is available."
            )
            if st.button(
                "Retry novelty verification",
                key="retry-novelty-verification",
                help=(
                    "Rerun only evidence validation, external closure search, "
                    "research-gap synthesis, and scoring for this saved run."
                ),
            ):
                try:
                    with st.spinner(
                        "Retrying scholarly closure search and gap synthesis..."
                    ):
                        from src.validate_gaps import validate_all_gaps
                        from src.synthesise_research_gap import synthesise_research_gap
                        from src.score_gaps import score_and_rank_gaps

                        validate_all_gaps(cfg)
                        synthesise_research_gap(cfg)
                        score_and_rank_gaps(cfg)
                    st.session_state["results"] = load_results(cfg)
                    st.success("Novelty verification completed.")
                    st.rerun()
                except Exception as exc:
                    st.error(
                        "Novelty verification could not complete. The saved "
                        f"evidence remains intact. Details: {exc}"
                    )
    if st.button(
        "Rebuild evidence cells and verify",
        key="rebuild-evidence-cells",
        help=(
            "Regenerate high-quality paper-centred candidates, retrieve bounded "
            "open-access full text, and rerun every unchanged certification gate."
        ),
    ):
        try:
            with st.spinner(
                "Rebuilding evidence cells, retrieving full text, and checking closure..."
            ):
                from src.detect_gaps import detect_all_gaps
                from src.full_text import enrich_candidate_source_full_text
                from src.validate_gaps import validate_all_gaps
                from src.synthesise_research_gap import synthesise_research_gap
                from src.score_gaps import score_and_rank_gaps
                from src.visualise import generate_visualisations

                detect_all_gaps(cfg)
                enrichment = enrich_candidate_source_full_text(cfg)
                if enrichment.get("enriched", 0):
                    detect_all_gaps(cfg)
                validate_all_gaps(cfg)
                synthesise_research_gap(cfg)
                score_and_rank_gaps(cfg)
                generate_visualisations(cfg)
            st.session_state["results"] = load_results(cfg)
            st.success(
                "Evidence cells rebuilt and certification rerun without changing "
                "the certificate thresholds."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Candidate rebuild could not complete: {exc}")

    if not gaps:
        st.warning(
            "No candidate passed the fail-closed evidence contract in this run. "
            "This is a valid null result—not evidence that no research gaps exist. "
            "Review the validation queue or expand the corpus."
        )

    synthesis_report = results.get("research_gap_synthesis", {}) or {}
    primary_research_gap = synthesis_report.get("primary_gap") or {}
    primary_certificate = primary_research_gap.get("gap_certificate", {}) or {}
    has_certified_primary = bool(primary_certificate.get("passed"))
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Certified corpus-bounded gaps",
            synthesis_report.get("certified_gap_count", 0),
        )
    with col2: st.metric("Evidence-cleared claims", len(gaps))
    with col3: st.metric("Human reviews", len(results.get("confirmed_research_gaps", [])))
    with col4: st.metric("Explicit evidence", sum(1 for g in gaps if g["type"] == "evidence_gap"))
    with col5: st.metric("Review required", validation_summary.get("review_required", 0))

    if G:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("KG Nodes",    G.number_of_nodes())
        with col2: st.metric("KG Edges",    G.number_of_edges())
        with col3: st.metric("Components",  nx.number_weakly_connected_components(G))

    st.divider()

    paper_index = results.get("paper_index", {})

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gap certificate",
        "Candidates & Review",
        "Source Papers",
        "Knowledge Graph",
        "Evidence Analytics",
        "KG vs RAG",
    ])

    with tab0:
        st.subheader("Certified corpus-bounded research gap")
        st.caption(
            "The certificate applies only to the frozen screened corpus and documented "
            "counterevidence searches. It never proves that no relevant study exists globally."
        )
        if has_certified_primary:
            st.success(primary_research_gap.get("claim", ""))
            st.caption(
                f"Certificate: `{primary_certificate.get('certificate_id', '')}` · "
                f"Snapshot: {primary_certificate.get('scope', {}).get('snapshot_date', '')}"
            )
            st.markdown("**Research question**")
            st.write(primary_research_gap.get("research_question", ""))

            framework = primary_research_gap.get("PMCOST", {})
            framework_rows = [
                {"Element": key, "Operational value": value}
                for key, value in framework.items()
            ]
            st.dataframe(pd.DataFrame(framework_rows), hide_index=True, width="stretch")

            decision = primary_research_gap.get("decision", {})
            dg1, dg2, dg3 = st.columns(3)
            with dg1:
                st.metric("Signal families", decision.get("signal_family_count", 0))
            with dg2:
                st.metric("Graph signal families", decision.get("graph_signal_family_count", 0))
            with dg3:
                st.metric("Automatic strength", decision.get("automatic_strength_score", 0.0))

            with st.expander("Synthesis contract, certificate, and evidence chain"):
                gate_rows = [
                    {"Hard gate": gate, "Passed": passed}
                    for gate, passed in decision.get("hard_gates", {}).items()
                ]
                st.dataframe(pd.DataFrame(gate_rows), hide_index=True, width="stretch")
                certificate_rows = [
                    {"Certificate gate": gate, "Passed": passed}
                    for gate, passed in primary_certificate.get("gates", {}).items()
                ]
                st.markdown("**Certificate gates**")
                st.dataframe(
                    pd.DataFrame(certificate_rows), hide_index=True, width="stretch"
                )
                scope = primary_certificate.get("scope", {})
                external = primary_certificate.get("external_search", {})
                counterevidence = primary_certificate.get("counterevidence", {})
                st.json({
                    "scope": scope,
                    "source_evidence": primary_certificate.get("source_evidence", []),
                    "external_search": external,
                    "counterevidence": counterevidence,
                })
                st.markdown("**Convergent signal families**")
                st.json(primary_research_gap.get("convergent_signals", {}))
                st.markdown("**Source papers**")
                st.dataframe(
                    pd.DataFrame(primary_research_gap.get("source_papers", [])),
                    hide_index=True,
                    width="stretch",
                )
                corroborating = primary_research_gap.get("corroborating_papers", [])
                if corroborating:
                    st.markdown("**Source-disjoint corroborating papers**")
                    st.dataframe(
                        pd.DataFrame(corroborating),
                        hide_index=True,
                        width="stretch",
                    )

            study = primary_research_gap.get("suggested_study", {})
            with st.expander("Suggested empirical study"):
                st.json(study)
        elif synthesis_report:
            st.warning(
                synthesis_report.get(
                    "null_result_note",
                    "No certified corpus-bounded research gap was produced.",
                )
            )
            if primary_research_gap:
                st.info(
                    "A legacy or evidence-cleared candidate exists, but it is not shown "
                    "as a certified research gap because it has no passing certificate."
                )
            next_actions = synthesis_report.get("next_actions", [])
            if next_actions:
                st.markdown("**Recommended next actions**")
                for action in next_actions:
                    st.write(f"- {action}")
            audit = synthesis_report.get("candidate_audit", [])
            if audit:
                quality_rows = []
                for item in sorted(
                    audit,
                    key=lambda value: value.get("source_candidate", {}).get(
                        "candidate_quality", {}
                    ).get("score", 0.0),
                    reverse=True,
                )[:10]:
                    source = item.get("source_candidate", {})
                    quality = source.get("candidate_quality", {}) or {}
                    readiness = quality.get("certificate_readiness", {}) or {}
                    quality_rows.append({
                        "Evidence cell": source.get(
                            "missing_capability", source.get("tail", "")
                        ),
                        "Candidate quality": quality.get("score", 0.0),
                        "Independent sources": quality.get(
                            "independent_source_count",
                            len(source.get("supporting_paper_ids", [])),
                        ),
                        "Frame completeness": quality.get(
                            "frame_completeness", 0.0
                        ),
                        "Full-text coverage": quality.get(
                            "full_text_coverage", 0.0
                        ),
                        "Independent-source ready": readiness.get(
                            "independent_sources_ready", False
                        ),
                    })
                if quality_rows:
                    st.markdown("**Highest-quality candidates closest to certification**")
                    st.dataframe(
                        pd.DataFrame(quality_rows), hide_index=True, width="stretch"
                    )
                failed_gate_counts = {}
                certificate_failed_gate_counts = {}
                audit_rows = []
                for item in audit:
                    failed = [
                        gate for gate, passed in item.get("decision", {}).get(
                            "hard_gates", {}
                        ).items() if not passed
                    ]
                    for gate in failed:
                        failed_gate_counts[gate] = failed_gate_counts.get(gate, 0) + 1
                    certificate_failed = item.get("gap_certificate", {}).get(
                        "failed_gates", []
                    ) or ["missing_certificate"]
                    for gate in certificate_failed:
                        certificate_failed_gate_counts[gate] = (
                            certificate_failed_gate_counts.get(gate, 0) + 1
                        )
                    source = item.get("source_candidate", {})
                    audit_rows.append({
                        "Candidate": (
                            f"{source.get('subject', '')} → "
                            f"{source.get('missing_capability', '')}"
                        ),
                        "Failed hard gates": ", ".join(failed),
                        "Failed certificate gates": ", ".join(certificate_failed),
                    })
                st.markdown("**Why no certified gap was emitted**")
                certificate_counts = synthesis_report.get(
                    "certificate_rejection_gate_counts", certificate_failed_gate_counts
                )
                st.dataframe(
                    pd.DataFrame([
                        {"Failed certificate gate": gate, "Candidates affected": count}
                        for gate, count in sorted(
                            certificate_counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    ]),
                    hide_index=True,
                    width="stretch",
                )
                st.markdown("**Candidate-level audit**")
                st.dataframe(pd.DataFrame(audit_rows), hide_index=True, width="stretch")
        else:
            st.info("Run the updated pipeline to produce `research_gap_synthesis.json`.")

    # ── Tab 1 — Ranked Gaps + FIX 8 Expert Review ───────────────
    with tab1:
        st.subheader("Evidence-cleared gap claims and expert review")

        # FIX 8: HCAI info banner
        st.info(
            "**Stage 6 — Human Expert Review (HCAI)**  \n"
            "Only candidates that passed the fail-closed evidence gate appear in the "
            "evidence-cleared queue. Use the controls below to Accept, Reject, or flag "
            "claims for Modification. "
            "Decisions are saved to `expert_reviews.json` and cited in the paper as Stage 6 "
            "human-in-the-loop validation (Shneiderman 2020). "
            "These decisions provide an additional audit layer; they do not change "
            "the automatic research-gap verdict shown in the first tab.",
        )

        type_filter = st.multiselect(
            "Filter by type",
            ["evidence_gap", "missing_link", "orphan_cluster", "temporal_decay"],
            default=["evidence_gap", "missing_link", "orphan_cluster", "temporal_decay"],
        )

        TYPE_META = {
            "evidence_gap":   ("EG", "evidence", "Evidence-backed Gap"),
            "missing_link":   ("ML", "missing", "Missing Link"),
            "orphan_cluster": ("OC", "orphan",  "Orphan Cluster"),
            "temporal_decay": ("TD", "decay",   "Temporal Decay"),
        }

        post_gate_report = results.get("post_gate_expert_reviews")
        review_required_data = results.get("review_required_gaps", {})
        has_validation_queue = any(
            review_required_data.get(category, [])
            for category in (
                "evidence_gaps", "missing_links", "orphan_clusters", "temporal_decay"
            )
        )
        review_scope_options = ["Evidence-cleared claims"]
        if has_validation_queue:
            review_scope_options.append("Validation review_required")
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
        elif review_scope == "Validation review_required":
            review_state_key = "validation_gap_reviews"
            saved_review = results.get("validation_expert_reviews", {})
            review_filename = "validation_expert_reviews.json"
            queue_name = "validation_review_required"
            candidate_index = None
            display_gaps = []
            position = 0
            for category in (
                "evidence_gaps", "missing_links", "orphan_clusters", "temporal_decay"
            ):
                for candidate in review_required_data.get(category, []):
                    position += 1
                    gap = dict(candidate)
                    gap["rank"] = f"VR-{position:02d}"
                    gap["composite_score"] = gap.get("validation", {}).get(
                        "ranking_score", 0.0
                    )
                    gap["_review_key"] = f"validation_review_{position:03d}"
                    gap["_review_evidence"] = (
                        f"supporting papers: "
                        f"{gap.get('validation', {}).get('supporting_paper_count', 0)}; "
                        f"closure hits: "
                        f"{gap.get('validation', {}).get('closure_hit_count', 0)}; "
                        f"gate reasons: "
                        f"{', '.join(gap.get('validation', {}).get('reasons', []))}"
                    )
                    display_gaps.append(gap)
            st.caption(
                f"Validation queue: {len(display_gaps)} candidates need a qualified "
                "human decision and are not yet research-gap claims."
            )
        else:
            review_state_key = "gap_reviews"
            saved_review = results.get("expert_reviews", {})
            review_filename = "expert_reviews.json"
            queue_name = "evidence_cleared_claims"
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
        rationale_state_key = f"{review_state_key}_rationales"
        if rationale_state_key not in st.session_state:
            st.session_state[rationale_state_key] = (
                saved_review.get("rationales", {}) if saved_review else {}
            )

        reviews = st.session_state[review_state_key]
        review_rationales = st.session_state[rationale_state_key]

        for g in [x for x in display_gaps if x["type"] in type_filter][:30]:
            icon, css, label = TYPE_META.get(g["type"], ("—", "", g["type"]))
            gap_key          = g.get("_review_key", f"review_{g['rank']}")
            current_decision = reviews.get(gap_key, "Pending")
            validation = g.get("validation", {}) or {}
            claim_text = escape(str(
                validation.get("scoped_claim") or g.get("description", "")
            ))
            claim_status = escape(str(
                validation.get("claim_status", "unvalidated_candidate")
            ))

            # Gap card + review controls side by side
            col_card, col_review = st.columns([3, 1])

            with col_card:
                badge_html = get_review_badge_html(current_decision)
                st.markdown(f"""
<div class="gap-card {css}">
  <strong>#{g['rank']} {icon} {label}</strong>
  &nbsp;<code>score: {g.get('composite_score', 0):.4f}</code>
  &nbsp;{badge_html}<br>
  <small>{claim_text}</small><br>
  <small>claim status: {claim_status}</small><br>
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

                if new_decision != "Pending":
                    rationale_key = f"{review_state_key}:rationale:{gap_key}"
                    rationale = st.text_area(
                        "Review rationale",
                        value=review_rationales.get(gap_key, ""),
                        key=rationale_key,
                        placeholder=(
                            "State why the evidence supports this decision; "
                            "for Modify, include the revised scope."
                        ),
                        height=80,
                    )
                    review_rationales[gap_key] = rationale.strip()

        # Update session state after all widgets render
        st.session_state[review_state_key] = reviews
        st.session_state[rationale_state_key] = review_rationales

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
                missing_rationales = [
                    key for key, decision in reviews.items()
                    if decision == "Accept" and not review_rationales.get(key, "").strip()
                ]
                if queue_name == "evidence_cleared_claims" and missing_rationales:
                    st.error(
                        "An accepted gap requires a written review rationale before "
                        "it can be recorded as confirmed."
                    )
                else:
                    saved_output = save_expert_reviews(
                        cfg,
                        reviews,
                        reviewer_name,
                        notes=review_notes,
                        filename=review_filename,
                        queue_name=queue_name,
                        candidate_index=candidate_index,
                        candidates=display_gaps,
                        review_rationales=review_rationales,
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
            ["evidence_gap", "missing_link", "orphan_cluster", "temporal_decay"],
            default=["evidence_gap", "missing_link", "orphan_cluster", "temporal_decay"],
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
        graph_view = st.radio(
            "Graph view",
            ["Gap evidence map", "Entity knowledge graph"],
            horizontal=True,
            key="graph-view-selector",
        )
        if graph_view == "Gap evidence map":
            st.caption(
                "Layout: papers → consolidated limitation cells → methods, datasets, "
                "metrics and concepts. Larger red diamonds are higher-quality candidates."
            )
            evidence_map_path = (
                Path(cfg["paths"]["outputs"]) / "gap_evidence_graph.html"
            )
            if evidence_map_path.exists():
                st.iframe(evidence_map_path, height=650)
            else:
                st.warning(
                    "Gap evidence map is unavailable for this legacy run. Rerun "
                    "detection and visualisation with the updated pipeline."
                )
        st.caption("Nodes = concepts · Size = centrality · Red border = gap node · Dashed red = predicted missing link")

        if graph_view == "Entity knowledge graph" and results.get("graph_html"):
            graph_path = Path(cfg["paths"]["outputs"]) / "graph_viz.html"
            if graph_path.exists():
                st.iframe(graph_path, height=650)
        elif graph_view == "Entity knowledge graph":
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
                disabled=not groq_keys,
            ):
                cfg.setdefault("api_keys", {})["groq_keys"] = groq_keys
                cfg["api_keys"]["groq"] = groq_keys[0]
                os.environ["GROQ_API_KEY"] = groq_keys[0]
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
