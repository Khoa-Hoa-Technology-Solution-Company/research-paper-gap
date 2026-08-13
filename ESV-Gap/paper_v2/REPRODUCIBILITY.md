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
