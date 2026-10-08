# ESV-Gap ICAI 2026: local artifact review candidate

This bundle accompanies *Auditing AI-Assisted Research-Gap Retrieval: A Four-Corpus
Time-Split Evaluation*. Author: Lê Anh Hòa, FPT University, Vietnam.

**Unpublished local review copy.** No public URL, archival identifier, independent
reproduction, or distribution license is claimed. Publication requires author
approval, selection of a destination and license, and rights review of the code
and configuration. No license has been added on the author's behalf.

## What is included

- Byte-preserved offline replay code, four analysis/runner scripts, configuration,
  dependency lock, and the four existing temporal unit tests.
- Aggregate and per-fold numeric reports, input/code/configuration fingerprints.
- A standard-library checker for manifest integrity and aggregate arithmetic.
- A manifest explaining every copied or projected report and its original hash.

No manuscript PDF is included; the revised manuscript is delivered separately.
No source corpus, abstract text, raw triples, graph serialization, raw fold replay,
candidate/control strings, future-paper identifiers, or internal inspection notes
are included. This is a minimal offline subset, not the complete ESV-Gap application.

## Check the retained arithmetic without data or dependencies

From the extracted bundle root:

```text
python -B verify_aggregate_bundle.py
```

The checker recalculates macro recall from retained per-fold values, checks row
totals, query sensitivity, Q1 consistency, and all payload SHA-256 digests. It does
not independently regenerate matches, validate the abstracts, or reproduce the
pipeline. A hash establishes file identity, not scientific validity.

## Existing synthetic temporal tests

Recorded replay environment: CPython 3.14.6; NetworkX 3.6.1; PyYAML 6.0.3.
The local packaging check also uses NumPy 2.5.1 and python-louvain 0.16.
For the dependency-based tests, use an appropriately isolated environment:

```text
python -m pip install -r requirements-experiments.lock
python -B -m unittest discover -s tests -p test_temporal_backtest.py -v
```

These are implementation tests, not expert annotation or a discovery evaluation.

## Full offline replay: requires omitted inputs

The original runner reads `config.yaml` and the following files for each run:
`runs/<run>/data/processed/corpus_filtered.jsonl` and
`runs/<run>/data/triples/all_triples.json`.

| Domain | Preserved run directory |
| --- | --- |
| IoT intrusion detection | deep_learning_iot_intrusion_de_20260831_114802 |
| Microservice security | security_of_microservices_20260806_114632 |
| MongoDB security | security_of_mongodb_20260806_135447 |
| Handwritten math recognition | handwritten_mathematical_expre_20260823_215228 |

Inputs are deliberately absent. An external reader cannot perform the full replay
from this bundle alone. Only restore lawfully held matching inputs, checking their
SHA-256 values against `reports/multidomain_summary.json`. Then, from the bundle
root, run in order:

```text
python -B paper_icai2026/expanded_run/run_multidomain_backtest.py
python -B paper_icai2026/expanded_run/analyze_candidate_sources.py
python -B paper_icai2026/expanded_run/run_bm25_lacks_baseline.py
python -B paper_icai2026/expanded_run/run_bm25_query_sensitivity.py
```

The runner creates raw fold outputs under `paper_icai2026/expanded_run`; the other
scripts require these outputs and/or the retained inputs. They may contain text
and identifiers excluded from this review copy. Run in a separate extraction and
do not redistribute generated outputs without reviewing their rights/content.
The online settings in the byte-preserved config are not executed by this replay.

## Interpretation boundaries

The four corpora contain 378 keyed records; ten folds execute, nine have positive
heuristic controls. The 64 control rows and 39 ESV candidate rows reuse papers and
are not independent. BM25 reranks the same extracted LACKS pool, not raw abstracts.
Aggregate checking cannot establish within-scope closure: the five-case inspection
remains AI-assisted and abstract-only, without independent human adjudication.
Zero independently confirmed closure witnesses does not mean five proven false
positives. Six research-gap types have not been assigned or evaluated.

## Redactions/projections

`multidomain_summary.json` and `candidate_source_ablation.json` are byte-preserved.
In `bm25_lacks_baseline.json`, the original `matched_pairs_at_20` and
`matched_control_ids_at_20` fields are removed; each row retains only the number
of distinct matched IDs. In `bm25_query_sensitivity.json`, query-specific matched
IDs are likewise replaced by counts. Query wording, numeric values, and input
fingerprints are retained. Original report SHA-256 digests and projection rules
are recorded in `MANIFEST.json`; the projected reports have their own payload
digests. No original result files are modified.
