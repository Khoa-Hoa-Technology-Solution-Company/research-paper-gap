# GapClose: certified research-gap claims under corpus incompleteness

Code and data pipeline for the ICAI-FAI 2026 paper
*Absence of Evidence Is Not a Research Gap: Certified Gap Claims under Corpus Incompleteness*
(paper sources in [`../paper_gapclose_icai2026/`](../paper_gapclose_icai2026/)).

A gap finder that calls a question open whenever its corpus holds no answer makes *false novelty*
errors as soon as the corpus is incomplete. This repository measures that error and controls it:

* **GapClose-SciFact** (1,109 questions) and **GapClose-ClimateFEVER** (1,381 questions), built from
  expert evidence labels; a question is CLOSED if a witness document answers it, OPEN otherwise.
* **Deletion protocol**: witness documents are removed at rate `p` (independently, or targeted at
  strong evidence) to simulate missing literature.
* **ESV-Scope**: sentence-level NLI x facet/numeric scope, quoting the witness sentence.
* **Certification** (see [`THEORY.md`](THEORY.md)): conformal thresholds calibrated in situ
  (Theorem 1) or transferred with a Clopper-Pearson bound on the missing rate (Theorem 2) keep
  false novelty <= alpha for monotone scores.
* **External recovery**: OpenAlex is queried only for questions about to be declared open.
* **Retrospective validation**: literature frozen at 2008/2010/2012; gaps are checked against the
  papers that answered them later, and each certified gap gets an evidence dossier.

## Main results (alpha = 0.10)

| Finding | Number |
|---|---|
| F1-optimal threshold, false novelty with all evidence missing | up to 0.73 (BGE reranker) |
| Split conformal calibrated on a complete corpus, worst false novelty | 0.30 |
| Certified rules (in situ / transfer), worst mean false novelty, 12 scores | 0.101 / 0.100 |
| AUC separating closed from open after deleting all witnesses (SciFact) | 0.45 to 0.55 |
| Retrospective 2010 cutoff: later-filled gaps certified / precision | 68% / 96% |

## Layout

```
src/
  common.py, methods.py, onnx_models.py    shared code, ESV-Scope and baselines, ONNX backend
  build_benchmark.py                        GapClose-SciFact from the SciFact release
  build_climate.py                          GapClose-ClimateFEVER
  score_candidates.py                       retrieval (BM25 + MiniLM, RRF) and sentence-level NLI
  cv_dev.py, evaluate.py                    train-only CV, uncertified test evaluation (Table II)
  bge_scores.py                             BGE-large dense and BGE reranker baselines
  llm_judge.py                              local LLM judge (Qwen2.5-1.5B-Instruct, ONNX)
  extra_uncertified.py                      uncertified metrics of BGE and LLM baselines
  fetch_external.py, external.py            OpenAlex evidence recovery (cached)
  certify.py                                deletion experiments, all certification rules (Tables IV-VII)
  fetch_years.py, retro.py                  publication years, retrospective validation, gap dossiers (Tables VIII-IX)
  make_audit.py, audit_stats.py             optional manual audit sheets (not used in the paper)
  summarize.py                              quick text summary of evaluate.py results
THEORY.md                                   theorems, proofs, corollary
overnight_refresh.sh                        example unattended chain (keep the machine awake)
```

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
```

Models run through ONNX Runtime (DirectML on Windows, CPU elsewhere) with the published weights
from Hugging Face; PyTorch is not needed (`score_candidates.py --backend torch` exists but is optional).
On Windows set `PYTHONIOENCODING=utf-8`.

Data (not redistributed, see `.gitignore`):

* SciFact: download `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` and
  extract to `data/scifact/` (so that `data/scifact/data/corpus.jsonl` exists).
* Climate-FEVER: `https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/main/dataset/climate-fever.jsonl`
  saved as `data/climatefever/climate-fever.jsonl`.

Keys: `OPENALEX_API_KEY` (free OpenAlex key) for `fetch_external.py`. Semantic Scholar
(`fetch_years.py`) needs no key.

## Reproducing the paper

Run from `src/`. Times are for an RTX 3060 laptop GPU.

```
# 1. SciFact benchmark, retrieval and NLI scores (about 1 h)
python build_benchmark.py
python score_candidates.py --nli MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --batch 32 --tag _large

# 2. uncertified accuracy (Table II): CV on the training split selects the ESV variant, test is used once
python cv_dev.py --tag _large
python evaluate.py --tag _large

# 3. extra baselines (BGE about 15 min, LLM judge about 75 min)
python bge_scores.py
python llm_judge.py
python extra_uncertified.py

# 4. external evidence from OpenAlex (about 1,100 searches; resumable across days)
python fetch_external.py

# 5. certification under deletion, four settings (about 50 min in parallel)
python certify.py --protocol pooled  --deletion mcar
python certify.py --protocol pooled  --deletion targeted
python certify.py --protocol holdout --deletion mcar
python certify.py --protocol holdout --deletion targeted

# 6. retrospective validation and gap dossiers
python fetch_years.py
python retro.py

# 7. second domain (outputs kept separate)
GAPCLOSE_OUT=../outputs_climate python build_climate.py
GAPCLOSE_OUT=../outputs_climate python score_candidates.py --nli MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --batch 32 --tag _large
GAPCLOSE_OUT=../outputs_climate python certify.py --protocol pooled --deletion mcar
GAPCLOSE_OUT=../outputs_climate python certify.py --protocol pooled --deletion targeted

# 8. tables, figure data and the PDF
cd ../../paper_gapclose_icai2026 && python make_tables.py && pdflatex main.tex && pdflatex main.tex
```

Results are written to `outputs/` (`bench/`, `scores/`, `external/`, `results/`); the paper's
tables and every number quoted in its text are generated from `outputs/results/` by `make_tables.py`.

## Protocol notes

* All ESV design choices come from five-fold CV on the SciFact training split. The test split was
  evaluated twice for uncertified accuracy: first with DeBERTa-v3-small and untuned defaults
  (`results/main_run1_small_untuned.json`, kept for disclosure), then with the large verifier.
* The train-to-test certification protocol calibrates on the training split and evaluates on the
  test split, which was never used to choose the certified scores.
* Certification requires monotone scores (maximum over available documents); top-k truncated or
  learned aggregates such as ESV-Learned are reported only for uncertified accuracy.
* `retro.py` discards OpenAlex results published after the cutoff, so the papers that later answer
  a question cannot leak into the decision.
