# Reproducing ESV-Gap results

Run commands from the project root. The controlled experiment and unit tests
do not require an API or network connection.

## Runtime

- CPython 3.14.6 was used for the reported run.
- Install the minimal experiment environment with:

```powershell
python -m pip install -r requirements-experiments.lock
```

## Tests and controlled benchmark

```powershell
python -m unittest discover -s tests -v
python experiments\run_validation_benchmark.py `
  --development-repetitions 10 `
  --test-repetitions 50 `
  --output outputs\validation_benchmark_v3.json
```

The controlled binary mapping is frozen as follows: planted
evidence-sufficient candidates are ground-truth positive; planted artefacts or
covered/qualified candidates are ground-truth negative; only
`automatically_eligible` is prediction-positive, while `review_required` and
`rejected` are prediction-negative. The benchmark report SHA-256 is
`9796cbcdd08b7a13e60b8a37eea8cbd5a8418feb2c845a5e852fe5e6ecb2e4fe`.

## Retrospective corpus diagnostic

```powershell
python experiments\run_corpus_validation.py `
  --triples path\to\resolved_triples.json `
  --corpus path\to\screened_papers.json `
  --output outputs\microservices_corpus_validation_v3.json
```

The reported snapshot has these SHA-256 checksums:

- `resolved_triples.json`: `2dc2ba9adba91f1f5ba8d49c0efe3ef56e8a906289e606242b10dcf866760253`
- `screened_papers.json`: `413d757a467f0e67166b4a7d83a9ed296c4414406f3ccf0aa6ddc323a4bc13a2`

The inherited snapshot does not preserve its original database query,
retrieval ledger, deduplication counts, or independent screening agreement.
It is therefore a retrospective diagnostic, not a reproducible PRISMA corpus
construction.

## Corrected live MongoDB-security run (August 6, 2026)

The completed run is preserved under:

```text
runs/security_of_mongodb_20260806_135447
```

The Streamlit controls used topic `security of mongodb` and collection budget
100. Semantic Scholar returned 74 unique records across five queries, and the
topic-conditioned screener retained 53. All 53 paper outputs contained triples:
643 relation events were extracted, and graph-construction filtering produced
578 nodes, 636 edges, and 61 weak components. The detectors generated 128 raw
signals: 50 missing links, 71 orphan communities, and seven temporal signals.
The ranked top 30 contain four missing links, 25 orphan communities, and one
temporal signal.

The same 53-paper corpus was used for both baselines. B1 is the Mulla-style
top-3 retrieval prompt; because `sentence_transformers` was unavailable, all
53 cases used the TF-IDF fallback. B1 produced 53 unique remaining-gap fields.
B2 is the no-retrieval per-abstract prompt and produced 159 unique outputs.
Corpus-level token-set Jaccard overlap was 0.1239 for KG--B1, 0.1113 for
KG--B2, and 0.3333 for B1--B2. These are descriptive overlap values, not
accuracy measurements.

The updated UI review file contains one unblinded internal reviewer and a
decision for every ranked item: 24 Accept and six Reject (80.0% acceptance),
with no Modify or Pending items. By family, the reviewer rejected all four
missing links, accepted 23 of 25 orphan communities, and accepted the single
temporal item. B1/B2 outputs were not reviewed. The deployed live runner did
not invoke the separate
`validate_all_gaps` stage, so the ranked 30 are detector-ranked candidates and
must not be reported as automatically eligible under the frozen gate.

The same internal reviewer also completed the separately prepared post-gate
queue: 8 Accept and 1 Reject (88.9% acceptance), with no Modify or Pending
items. All three orphan communities were accepted; five of six temporal items
were accepted, and the `Cloud infrastructure` temporal signal was rejected.
One candidate overlaps the pre-gate top-30 cohort. Neither ledger is blinded
or independent, and neither contains item-level rationales; these rates are
reviewer dispositions, not estimates of precision or novelty.

The frozen gate was subsequently applied offline to the preserved run. It
routed nine of 128 signals to review and rejected 119; none was automatically
eligible. By family, review/reject counts were 0/50 for missing links, 3/68 for
orphan communities, and 6/1 for temporal signals. All 128 detector candidates
predate the explicit plausible-edge-search completion marker. Their addition
mode is therefore serialized as unavailable (`null`) and cannot count as
survived; the nine candidates that pass all hard rules remain review-required.

The checksum-addressed submission archive is:

```text
output/releases/esv-gap-final-20260810-r1.zip
SHA-256: c6e3d3daf7959598b68b386d39fb8af4126a1d98e7d314e64b239cd07483c835
```

It freezes the validator, exact YAML configuration, graph, detector output,
screened corpus, reproduced output, controlled benchmark, both review ledgers, tests, and
`MANIFEST.sha256`. From the extracted archive root, the exact offline command
is:

```powershell
python reproduce_frozen_validation.py --config frozen_validation_config.yaml
```

The rerun produces 128 raw candidates, zero automatically eligible, nine
review-required, and 119 rejected. It needs no API key or network connection.

To repeat through the UI, run `streamlit run app.py`, enter the same topic,
select a maximum of 100 papers, and run both baseline methods after the KG
pipeline. Live API results are not guaranteed to be identical because the
search index, hosted model, and generation are time-dependent. The preserved
directory is the run artifact used by the manuscript; its hashes should be
checked before reuse.

Live-run SHA-256 checksums:

- `data/processed/corpus_filtered.jsonl`: `1f47c496466ce241b72330c9a80ceaf384de683f8af49b98bc30b75e7d244853`
- `data/triples/all_triples.json`: `bf17038593413d26c60b1a8d781ed1d5ad580722c8d93542be2cba4b176a0386`
- `outputs/detected_gaps_raw.json`: `3af4d23f82ab1e3c7200825743ecce9b2f418bfcdc4ff6c6591997ce4cc51d12`
- `outputs/gaps_ranked_top.json`: `9cd92844fb48d24534b484838c3b1c4946eb6bb737b7edeb48d3657c58d506e2`
- `outputs/gap_validation_audit.json`: `06eadc23301857b636378cdd8a0e236b08df94937a688a9f91e44b3564e2f46d`
- `outputs/expert_reviews.json`: `02a355a757f67f7af1aef9b08c13363d7a73c8d5f285f7d1f1252f5671cbec64`
- `outputs/post_gate_expert_reviews.json`: `390b91827507a27383a3c972b90e81425cbd44f921b2ac5cdc87e791b84e18aa`
- `outputs/comparison_metrics.json`: `ecee805edf656c6bb043be39846e50a7eb5e729ed359a03908c2a712cadfe25e`
- `outputs/rag_mulla_gaps.json`: `582e860e1dd7ce17bd1884c060fa2215027e5bd4602173d49fdf48d4a00990ba`
- `outputs/rag_simple_gaps.json`: `e3047e3d160e138353f8ff65deed7c66d52118b18a0d02abd45e299049f3a862`

The earlier `runs/reseach_gap_20260806_092230` directory is retained only as a
configuration-leakage diagnostic and is not the live result summarized in the
manuscript.

## Corpus integrity audit

```powershell
python experiments\audit_corpus_integrity.py `
  --triples path\to\resolved_triples.json `
  --corpus path\to\screened_papers.json `
  --output outputs\corpus_integrity_audit.json `
  --resolve-dois
```

DOI resolution is a time-stamped online availability check. The remaining
identity, provenance, and verbatim-evidence checks are deterministic and
offline.

## Blinded expert-review packet

```powershell
python experiments\prepare_expert_review.py `
  --report outputs\microservices_corpus_validation_v3.json `
  --review-csv outputs\expert_review_packet.csv `
  --manifest outputs\expert_review_manifest.json `
  --rejected-sample 30 `
  --seed 20260805
```

The packet intentionally contains blank rating columns. No expert result is
reported until at least three independent reviewers complete the protocol.

## Artifact hashes

```powershell
python experiments\create_artifact_manifest.py
```

The final submission ZIP contains a self-verifying `MANIFEST.sha256` covering all 43
archived files. The outer ZIP checksum is stored in the adjacent
`esv-gap-final-20260810-r1.zip.sha256` file and printed above. The archive was
extracted, every manifest entry was verified, and the offline validation command
was rerun successfully before the manuscript was compiled.

## Compile the IEEE single-column manuscript

```powershell
cd paper_v2
pdflatex -interaction=nonstopmode -halt-on-error main_ieee.tex
bibtex main_ieee
pdflatex -interaction=nonstopmode -halt-on-error main_ieee.tex
pdflatex -interaction=nonstopmode -halt-on-error main_ieee.tex
```

`main_ieee.tex` is the submission manuscript. The older `main.tex` file is an
LNCS-format working draft and should not be submitted in place of the IEEE file.

## AMI 2027 triage results (`canonical_results.json`)

The numbers in Tables 3 and 5 of `main_springer_ami.tex` are stored in
`paper_v2/canonical_results.json`. The file was assembled by hand from
the frozen run `runs/deep_learning_iot_intrusion_de_20260831_114802`, plus scratch
scripts in `scratch/` and manual edits. `tests/test_result_consistency.py`
checks only that file. It does not regenerate it.

To rebuild the file offline from cached data and diff it against the committed
copy:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python experiments\build_canonical_results.py
```

Outputs: `outputs/canonical_results_rebuilt.json` (rebuilt values, plus
`_`-prefixed audit fields) and `outputs/canonical_results_diff.json`. Each
committed field gets one of these statuses: `MATCH`, `MISMATCH`, or
`NOT_REPRODUCIBLE`. The comparison run on 2026-10-27 gave 157 MATCH,
17 MISMATCH and 35 NOT_REPRODUCIBLE.

**Reproduced from cached run data:**

- Corpus: 150 retained papers, years 2019–2026.
- Multigraph: 1,185 canonical entities, 1,404 relation events, 1,396/1,403
  cumulative raw entities/relations.
- Raw and canonical saturation statistics (beta, K, R², marginal means, decay).
- Stage A1 for C1–C5: raw index counts 864/400/146/189/1095, 5 inspected
  records each, co-mention counts 5/4/4/0/5 (`outputs/external_verification.json`).
- Pillar B: C4 has 3 independent corroborations, author-disjoint from the
  origin paper. C2 has 1 circular corroboration traced to paper `04b3ca64`
  (`outputs/author_stated_gaps.json`).
- C4 disposition `EVIDENCE_SUPPORTED_BUT_OPEN` / `RETAIN_WITH_WARNING`. With
  zero co-mentions, this does not depend on Stage A2.
- For C1, C3 and C5, the primary counterevidence DOIs are among the run's
  Stage A1 hits, and the quoted spans appear verbatim, as segments split by
  "...", in abstracts cached in `scratch/`.
- K_fetch lexical co-mention counts, strict (0/0/1/2) and expanded (4/7/13/33),
  recomputed from `scratch/c4_s2_50.json`.

**Not reproducible:**

- `semantic_verified_counterevidence_count` (2/0/2/0/1) and the dispositions
  of C1, C2, C3 and C5. No Stage A2 classifier exists in `src/`, `experiments/`
  or `scratch/`, and the run stores no Stage A2 labels. Applying the current
  code to the run gives C2 = `REFUTED`, because lexical co-mention is treated
  as refutation. If the committed A2 counts are fed into
  `decide_hypothesis`, it returns the committed dispositions for all five
  candidates. For C2, though, the code's reason is "no independent
  corroboration", not circularity.
- The K_fetch `verified_counterevidence` and `disposition` values are
  hard-coded constants in `scratch/eval_k_fetch.py`.
- Free-text fields (`target_domain`, `rationale`, `abstention_reason`) and
  some bibliographic fields with no cached source.

**Mismatches:**

- `screened_records`: 192 in the file, 191 in the run.
- The precision Wilson CI is [0.7194, 0.9501] in the file. With z = 1.96 it is
  [0.7193, 0.9503]. The paper's rounded [71.9%, 95.0%] is correct.
- The primary-counterevidence author names in the JSON ("T. R. Gadekallu",
  "S. Ullah") do not match the cached metadata or the paper (Nitrat,
  Ogundokun).
- Candidate subject and capability labels are paraphrased from the run's
  strings.

**Caveat for Pillar C:** `benchmarks/gold_benchmark_extracted.json` is not
extractor output. `scratch/setup_exact_benchmark.py` builds it by deleting
eight chosen gold triples and adding four hand-written false positives. The
28/8/4 counts and the derived precision, recall and CIs are reproducible
arithmetic over that constructed file. They are not a measurement of the
extractor.
