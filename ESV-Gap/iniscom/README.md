# GapCast-6G: forecasting structural-gap closure in 6G and IoT

Paper and analysis for the EAI INISCOM 2027 submission, *Which Research Gaps Close?
Forecasting Gap Closure in the 6G and IoT Literature*. The anonymous entry point is
`paper/main_submission.tex`; `paper/main.tex` also contains the named-author branch.

A structural gap is an established concept pair with no detected joint title/abstract
mentions in the selected corpus at cutoff T. Closure means at least k later articles
mention both within H years. These are automatic proxy labels, without independent
human annotation. Predicting persistence is the complement of future-link prediction.

## Reproduce from the fixed snapshot

Run from this directory. Tested with Python 3.14.6, NumPy 2.5.3 and scikit-learn 1.9.1.
`requirements.txt` pins the analysis packages. `data/index.npz` and
`data/semantic.npy` are required; the cached semantic matrix avoids downloading or
re-encoding a model. Features and training labels are separated by publication year,
with a fixed retrospective vocabulary and metadata snapshot collected 30 October 2026.

```bash
python -m pip install -r requirements.txt
python src/gaps.py --tag main
python src/subsets.py --share 0.6666666666666666 --n 20 --seed 2027
python src/verify_revision.py
```

On the original Windows checkout, the existing analysis interpreter is
`../gapclose/.venv/Scripts/python.exe`. Setting `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` avoids unnecessary thread overhead on these small tables.

The other definition settings, ablation and calibration are run by `run_all.sh`.
It also rebuilds the lexical index from the local raw corpus, which is the slow step.
For an already built index, run the individual analysis scripts instead.

```bash
python src/gaps.py --tag g2 --g 2
python src/gaps.py --tag k1 --k 1
python src/gaps.py --tag k5 --k 5
python src/gaps.py --tag h2 --h 2
python src/gaps.py --tag occ20 --min-occ 20
python src/gaps.py --tag occ100 --min-occ 100
python src/gaps.py --tag sent --label sentence
python src/gaps.py --tag vocab2013 --vocab-year 2013
python src/ablation.py
python src/extra.py
```

`src/audit_examples.py` scans `data/corpus_T*.jsonl` and checks the three example
pairs against the index. Its output, `outputs/example_evidence.json`, preserves all
matched records; six selected papers have DOI metadata in `outputs/reference_metadata.json`.
Raw corpus files are local and ignored by Git. On a fresh checkout, the saved evidence
trace suffices for `verify_revision.py`; rebuilding the trace requires the raw files.

To fetch a new corpus, set `OPENALEX_API_KEY` and run `python src/fetch_corpus.py`.
This is a new API snapshot and need not reproduce the 2026 count tensors. Work IDs
alone do not freeze changing abstracts, publication dates or topic assignments.

## Build and verify the anonymous PDF

Run these commands sequentially from `paper/`:

```bash
python make_tables.py
pdflatex -interaction=nonstopmode -halt-on-error main_submission.tex
bibtex main_submission
pdflatex -interaction=nonstopmode -halt-on-error main_submission.tex
pdflatex -interaction=nonstopmode -halt-on-error main_submission.tex
python check_submission.py
```

Alternatively, compile directly from `main.tex` with the anonymous flag:

```bash
pdflatex -interaction=nonstopmode -jobname=main_submission "\def\anonymous{1}\input{main.tex}"
```

The compiled anonymous submission (`paper/main_submission.pdf`, copied to
`gapcast_iniscom2027_submission.pdf`) is 15 pages (13 pages before References, 22
bibliography entries, all fonts embedded, zero overfull/underfull boxes). The tables and
number macros are generated from analysis JSON files. Build logs and mechanical checks
are saved in `outputs/submission_pdf_checks.json` and `outputs/revision_analysis_checks.json`.
Descriptions of figures and tables are in `paper/ALT_TEXT.md`.

## Results in the revised submission

Main setting: m=50, g=0, k=3, H=3; test cutoffs 2019--2022; 194,221 works, 90 concepts.

| Method | Mean AUC |
|---|---:|
| Expected co-occurrence | 0.635 |
| Adamic--Adar | 0.659 |
| Logistic, popularity + growth | 0.776 |
| Gradient boosting | 0.773 |
| Full logistic forecaster | 0.809 |

The concept-bootstrap interval for full minus Adamic--Adar excludes zero at three of
four cutoffs; the 2021 interval crosses zero and the 2019 lower bound is near zero.
Intervals are conditional on fixed fitted models, pointwise and unadjusted.

Twenty fixed-seed subsets of 60 concepts give median mean AUC 0.797 (range
0.737--0.845); the strongest single score in each subset has median 0.651. These
overlapping subsets assess vocabulary sensitivity within this corpus, not external
generalisation or independent statistical replication.

At cutoff T=2022, Section 5.7 traces three closed top-ten gaps to six matched corpus
records (with DOIs, OpenAlex IDs and primary topics `T11458`/`T10273`), and decomposes
the standardised feature-to-logit contributions $c_j = w_j(x_j - \mu_j)/\sigma_j$ of the
two top-ten false positives (`intrusion detection × semantic communication` at rank 5:
PPMI profile cosine +1.93, faster-concept growth +1.57, expected co-occurrence −1.07;
`RIS × wearable` at rank 8: faster-concept growth +1.85, expected co-occurrence +1.20).
All counts and logit decompositions are verified by `src/verify_revision.py`.

See `REVISION_REPORT_VI.md` for the changes, six concrete paper references and the
submission checks. A public artifact release is planned upon publication; this local
workspace is not presented as an already released anonymous repository.
