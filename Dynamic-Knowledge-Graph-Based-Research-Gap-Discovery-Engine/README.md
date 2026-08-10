# ESV-Gap

**Evidence- and Stability-Validated Knowledge-Graph Research-Gap Triage**

This project constructs a temporal knowledge graph from academic papers and uses topological analysis to identify research gaps — replacing traditional prompt-based (RAG) gap identification with an explainable, reproducible, graph-theoretic approach.

## Quick Start

```bash
# 1. Clone/download the project
# 2. Run the setup script to create folder structure
python setup_project.py

# 3. Navigate into the project
cd kg-gap-discovery

# 4. Create virtual environment
python -m venv venv

# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Download spaCy model (one-time)
python -m spacy download en_core_web_sm

# 7. Set GROQ_API_KEY or enter a Groq API key in the Streamlit interface
# 8. Run the pipeline stage by stage
python run_pipeline.py --stage collect
python run_pipeline.py --stage filter
python run_pipeline.py --stage extract
python run_pipeline.py --stage build
python run_pipeline.py --stage detect
python run_pipeline.py --stage validate
python run_pipeline.py --stage score
python run_pipeline.py --stage visualise

# Or run everything at once
python run_pipeline.py --stage all
```

## Pipeline Stages

| Stage | Script | What It Does |
|-------|--------|-------------|
| **collect** | `src/collect.py` | Fetches ~1000 papers from Semantic Scholar API |
| **filter** | `src/filter.py` | Filters to ~150 relevant papers (PRISMA-style) |
| **extract** | `src/extract_triples.py` | Extracts (entity, relation, entity) triples via LLM |
| **build** | `src/build_graph.py` | Constructs temporal knowledge graph + deduplication |
| **detect** | `src/detect_gaps.py` | Runs 3 gap detection algorithms on the graph |
| **validate** | `src/validate_gaps.py` | Audits path-specific provenance, multi-mode perturbation stability, and candidate-coverage retrieval |
| **score** | `src/score_gaps.py` | Scores and ranks candidates that passed validation |
| **visualise** | `src/visualise.py` | Generates interactive graph + gap visualisations |

## Requirements

- Python 3.9+
- Groq API key (for relevance screening, triple extraction, and LLM baselines)
- ~8GB RAM (no GPU needed)
- Internet connection (for API calls)

## Domain

Default domain: **Explainable AI (XAI)** — configurable in `config.yaml`.

## Project Structure

```
kg-gap-discovery/
├── config.yaml              # All tuneable parameters
├── run_pipeline.py          # Main entry point
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                 # Downloaded paper metadata
│   ├── processed/           # Filtered corpus
│   ├── triples/             # Extracted triples (JSON)
│   └── graph/               # Serialised knowledge graph
├── src/
│   ├── collect.py           # Data collection
│   ├── filter.py            # Corpus filtering
│   ├── extract_triples.py   # LLM triple extraction
│   ├── build_graph.py       # KG construction
│   ├── detect_gaps.py       # Gap detection algorithms
│   ├── validate_gaps.py     # Evidence and stability validation gate
│   ├── score_gaps.py        # Gap confidence scoring
│   ├── visualise.py         # Visualisation
│   └── utils.py             # Shared helpers
├── prompts/                 # LLM prompt templates
├── notebooks/               # Jupyter exploration notebooks
└── outputs/                 # Final results and figures
```
