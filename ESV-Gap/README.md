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
$env:GROQ_API_KEYS="key-one;key-two;key-three"
streamlit run app.py
```

The interface is available at `http://127.0.0.1:8501`. You can alternatively
choose the number of Groq keys in the sidebar and paste each key into a masked
password field. `GROQ_API_KEY` remains supported for single-key CLI use.

Keys are de-duplicated in priority order and kept in memory; they are never
written to `run_metadata.json`. Query generation, screening, triple extraction,
and RAG baselines automatically retry the same request with the next key after
a `429`, daily quota exhaustion, or a rejected credential. If every key is
unavailable, the pipeline preserves its checkpoint and reports that another key
or a later resume is required.

The corpus-size slider denotes papers retained after relevance screening. The
collector automatically builds a larger raw pool and stops screening when the
target is reached or the pool is exhausted. Candidate-specific novelty checks
prefer OpenAlex when no Semantic Scholar API key is configured and can be
retried from the dashboard without repeating collection or extraction.

## Pipeline stages

| Stage | Implementation | Purpose |
|---|---|---|
| `collect` | `src/collect.py` | Retrieve scholarly metadata from Semantic Scholar |
| `filter` | `src/filter.py` | Apply local checks and topic-conditioned relevance screening |
| `extract` | `src/extract_triples.py` | Extract checkpointed, paper-linked relation events |
| `build` | `src/build_graph.py` | Normalize entities and construct the temporal knowledge graph |
| `detect` | `src/detect_gaps.py`, `src/evidence_graph.py` | Consolidate limitation paraphrases and generate typed, source-disjoint empty evidence cells |
| `fulltext` | `src/full_text.py` | Retrieve bounded direct open-access PDFs for the strongest certifiable candidates |
| `validate` | `src/validate_gaps.py` | Apply the fail-closed evidence, stability, and closure-search contract |
| `synthesise` | `src/synthesise_research_gap.py` | Combine convergent signals into one answerable gap using predeclared hard gates |
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
Groq daily or temporary quota is exhausted, the client first rotates through all
configured keys. If the whole pool is exhausted, the interface pauses the run
without recording unprocessed papers as empty extractions. Later, reopen the
interface, add one or more keys, select the interrupted item under **Run
history**, choose **Open selected run**, and then select **Resume extraction**.
Completed papers are skipped and downstream stages continue after extraction.

## View previous runs

Every directory under `runs/` appears in the sidebar under **Run history** with
its timestamp, topic, completion state, and retained-paper count. Select any
completed run and choose **Open selected run** to load its graph, candidates,
source papers, validation audit, reviews, and baseline results without running
the pipeline again. Selecting an incomplete run opens its checkpoint and exposes
the resume action. Opening an older run no longer gets overridden merely because
a newer run exists.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The current suite contains 99 offline tests. It does not call Groq, Semantic
Scholar, or OpenAlex.

Graph construction and the TF-IDF RAG fallback use NumPy-only vector
operations. They do not import scikit-learn, so the pipeline remains executable
on managed Windows devices where Application Control blocks scikit-learn's
compiled neighbor-search DLLs. Sentence-Transformer import failures fall back
to lexical entity matching and NumPy TF-IDF retrieval.

## Troubleshooting Groq authentication

If Groq returns `401 invalid_api_key`, the client skips that credential and tries
the next configured key. Replace invalid keys in the Streamlit sidebar or set
valid `GROQ_API_KEYS`/`GROQ_API_KEY` values, then resume the selected run. The
app stops at the initial query-generation check (and also fails fast during
screening if a key is revoked mid-run), rather than classifying API failures as
irrelevant papers.
Lowering `filtering.relevance_threshold` does not fix an authentication error.

## Research-gap outputs

The dashboard deliberately distinguishes topology-only triage signals from a
research gap. `research_gap_synthesis.json` applies a deterministic contract
grounded in the Robinson research-gap framework and EPICOT. A candidate must
establish either a semantically entailed unresolved limitation or a
source-disjoint empty evidence-map cell. Source-study non-resolution, minimum
corpus size, perturbation stability, specificity, local and external closure
clearance, multi-signal convergence, and an answerable PMCOST question must also
all pass. Reviewer choices are not an input to this verdict.
`primary_research_gap.json` contains the one highest-strength passing gap, or an
explicit null result.

Candidate generation uses a separate paper-centred evidence graph. It connects
papers to consolidated limitation cells and to method, dataset, metric, and
concept context rather than treating every arbitrary entity pair as a possible
gap. The typed empty-cell detector requires meaningful research-role pairs,
zero abstract-level co-mention, adequate domain alignment, and at least two
source-disjoint paths. Candidate quality controls retrieval priority only; it
never relaxes a certification gate. `gap_evidence_graph.html` exposes this
provenance layout in the dashboard.

`gap_certificates.json` records the final corpus-bounded certificate gates. A
positive certificate additionally requires independent sources, sufficient
full text, completed counterevidence queries, no local/source/external closure,
an answerable PMCOST frame, and a frozen corpus fingerprint. Failure produces a
candidate or null result, never an automatic global novelty claim.

`research_gap_claims.json` remains the broader set of evidence-cleared
candidates. Optional reviewer decisions are stored separately in
`confirmed_research_gaps.json` and do not change the automatic synthesis.

The external novelty check and its retrieved paper metadata are recorded in
`gap_closure_search.json`. Because this is a bounded search of configured
sources, all generated claims remain scoped to the searched corpus and snapshot
date rather than asserting global novelty.

## Temporal backtesting

Temporal backtesting tests whether candidates generated before a cutoff can
anticipate observable outcomes in papers published after that cutoff. It uses a
leakage check over supporting paper IDs and an outcome-based silver standard:
post-cutoff limitation-plus-resolution events and newly observed typed research
relations are positive controls; pre-cutoff source-resolved limitations are
negative controls. Candidates matching neither control remain unlabelled.

```powershell
python experiments/run_temporal_backtest.py `
  runs/deep_learning_iot_intrusion_de_20260830_213021 `
  --cutoffs 2022 2023 2024
```

The report is stored under
`outputs/temporal_backtest/temporal_backtest_summary.json`. It reports
candidate recall@k, precision after local closure only when labelled outcomes
exist, known-negative selection rate, abstention, and certificate-metric
availability. It does not prove global novelty or replace expert evaluation.

To test sensitivity of the non-compensatory problem-alignment policy without
altering a frozen run, write a separate child re-evaluation directory:

```powershell
python experiments/reevaluate_frozen_run.py `
  runs/deep_learning_iot_intrusion_de_20260830_213021 `
  --output-name threshold_sensitivity_060 `
  --min-problem-relevance 0.60
```

The run reports 0.40--0.70 sensitivity separately from the temporal backtest.
The threshold is a domain-anchor coverage policy, not a learned probability;
this analysis tests robustness of the final disposition rather than tuning for
the preferred result.

Collection uses five query strata: core topic, systematic reviews,
limitations/challenges, empirical comparisons, and open problems/future
research. Generated queries are restricted to 2–7 plain keywords; Boolean
operators, quotation marks, parentheses, and exact-match expressions are
rejected. If the initial plan produces a sparse raw corpus, the collector
automatically retries a broader deterministic query plan. When more relevant
papers are available than requested, the final
corpus is balanced across retrieval query and publication year. Explicit
`lacks`, `limitations due to`, and `remains an open challenge` statements also
have a deterministic, quote-preserving extraction fallback; this improves
recall without bypassing the downstream evidence contract.

## Research artifacts

- `paper_v2/main_ieee.tex`: primary IEEE manuscript source.
- `paper_v2/main_ieee.pdf`: compiled IEEE manuscript.
- `paper_v2/results_summary.json`: frozen reported values.
- `paper_v2/REPRODUCIBILITY.md`: commands, hashes, and interpretation limits.
- `experiments/`: benchmark, audit, temporal-backtest, review-packet, and manifest utilities.

Generated run data are intentionally excluded from Git. Do not commit `.env`,
API keys, or reviewer data that have not been cleared for release.
