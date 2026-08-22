# ESV-Gap Evidence Workbench

This directory contains the executable ESV-Gap pipeline, Streamlit interface,
tests, experiment scripts, and LaTeX manuscript. For the research motivation,
architecture, reported results, authorship, and interpretation limits, see the
[repository README](../README.md).

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
$env:GROQ_API_KEY="your-key-here"
streamlit run app.py
```

The interface is available at `http://127.0.0.1:8501`. You can alternatively
paste the Groq key into the password field in the sidebar.

## Pipeline stages

| Stage | Implementation | Purpose |
|---|---|---|
| `collect` | `src/collect.py` | Retrieve scholarly metadata from Semantic Scholar |
| `filter` | `src/filter.py` | Apply local checks and topic-conditioned relevance screening |
| `extract` | `src/extract_triples.py` | Extract checkpointed, paper-linked relation events |
| `build` | `src/build_graph.py` | Normalize entities and construct the temporal knowledge graph |
| `detect` | `src/detect_gaps.py` | Generate missing-link, orphan, and temporal candidates |
| `validate` | `src/validate_gaps.py` | Apply the fail-closed evidence and stability contract |
| `score` | `src/score_gaps.py` | Rank candidates after validation |
| `visualise` | `src/visualise.py` | Produce graph and analytical visualizations |

Run all stages:

```powershell
python run_pipeline.py --stage all
```

Run one stage:

```powershell
python run_pipeline.py --stage validate
```

## Resume after quota exhaustion

Extraction writes one completed artifact per paper plus progress metadata. If a
Groq daily or temporary quota is exhausted, the interface pauses the run without
recording unprocessed papers as empty extractions. Later, reopen the interface,
provide an API key, and select **Resume extraction**. Completed papers are
skipped and downstream stages continue after extraction finishes.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The current suite contains 30 offline tests. It does not call Groq or Semantic
Scholar.

## Research artifacts

- `paper_v2/main_ieee.tex`: primary IEEE manuscript source.
- `paper_v2/main_ieee.pdf`: compiled IEEE manuscript.
- `paper_v2/results_summary.json`: frozen reported values.
- `paper_v2/REPRODUCIBILITY.md`: commands, hashes, and interpretation limits.
- `experiments/`: benchmark, audit, review-packet, and manifest utilities.

Generated run data are intentionally excluded from Git. Do not commit `.env`,
API keys, or reviewer data that have not been cleared for release.
