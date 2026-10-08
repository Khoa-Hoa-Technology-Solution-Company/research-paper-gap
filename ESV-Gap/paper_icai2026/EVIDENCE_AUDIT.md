# ESV-Gap evidence audit for the ICAI 2026 revision

Date: 2026-09-29. Scope: audit of the preserved IoT run, the submitted PDF/source, existing offline reports, and the reviewer screenshots. A local, no-network temporal replay used the same hash-verified frozen inputs and wrote only to `paper_icai2026/audit/replay/`; no model/API run or human assessment was performed. Counts below describe specific artifacts; they must not be mixed across experiments.

## 1. Which manuscript and run do the reviews describe?

The root `ESV-gap submit AMI.pdf` and `ESV-Gap/paper_v2/main_springer_ami.pdf` have the identical SHA-256 `4770fb65597dfa28b0d7c368ac053ad9ed1eb7efa5145e6e287c4e392a9f862e`. The corresponding editable source is `ESV-Gap/paper_v2/main_springer_ami.tex`. It says 150 retained papers and five post-gate, hand-triaged candidates (C1--C5): three externally refuted, one review-required, one qualified as evidence-supported/open. This is a *case study of five candidates*, not an end-to-end positive-yield result. Its only surviving C4 is explicitly an open hypothesis, not a validated discovery.

The supplied reviewer screenshots instead state 153 initial signals, zero automatically supported hypotheses, and zero recall on 22 positive controls. I found no preserved report in the inspected primary run supporting the denominators 153 or 22. Treat these as *reviewer-reported numbers for an unidentified version/evaluation*, not numbers to copy into a revised paper. Ask the authors for the exact reviewed submission/revision and report files before attributing them to a frozen experiment. The preserved primary run and its temporal backtest have different, verifiable counts below.

## 2. Primary frozen IoT run: directly verifiable numbers

Run ID: `ESV-Gap/runs/deep_learning_iot_intrusion_de_20260831_114802` (metadata timestamp 2026-08-31 11:48:02). Raw artifacts are left unchanged.

| Measure | Recorded value | Source and interpretation |
|---|---:|---|
| Collection pool **requested** | 600 | `run_metadata.json`, `collection_pool_size`; a budget, not records harvested |
| Raw records **observed** | 486 | `data/processed/screening_diagnostics.json`, `raw_paper_count` |
| Screened records | 191 | same diagnostic report, `screened_paper_count` |
| Retained papers | 150 | same report; independently, 150 JSONL lines in `data/processed/corpus_filtered.jsonl` |
| Extracted relation events | 1,404 | 1,404 entries in `data/triples/all_triples.json` |
| Canonical entities | 1,185 | `paper_v2/canonical_results.json` and canonical saturation report; this is a derived graph count |
| Raw candidate signals | 133 | `outputs/detected_gaps_raw.json`: 16 evidence gaps, 8 missing links, 91 orphan clusters, 18 temporal decay signals |
| Final validation disposition | 0 automatic; 5 review; 128 reject | `outputs/gap_validation_audit.json`, `summary`; total 133 |
| Certified research gaps | 0 | `outputs/validated_gaps.json` has empty arrays; `outputs/primary_research_gap.json` status `no_certified_corpus_bounded_research_gap` |
| Full-text enrichment | 0 successful | `outputs/full_text_enrichment.json`: target 20, 14 attempted, six lacked direct OA PDFs, zero enriched |
| External closure searches | 8 completed | `outputs/gap_validation_audit.json`; 12 candidates skipped by preflight, including 11 domain-preflight skips |

The submitted source's statement that 600 records were harvested, 192 screened, and full-text accessibility was verified does not match this run (486, 191, and zero full-text enrichments). If those numbers came from a different snapshot, that snapshot needs its own immutable input and log. For the frozen run, describe an abstract-based corpus and reserve 600 for the requested retrieval budget.

The eight-row detector ablation in `outputs/ablation_table.csv` reports **zero final gaps** for TransE alone, Louvain alone, temporal decay alone, explicit evidence alone, every pair, and all methods. The table does not establish detector-level positive utility, nor an incremental benefit for topology. The zero automatic output also does not prove there are no real-world gaps; it records what this evidence contract could certify from the available corpus and search traces.

For byte-level provenance, SHA-256 of the retained corpus is `852b63bf0b7f3fb641046b416c45723156d5663b56f87352b9145efb22282583`, triples `a352427bffb0d6d63bce589ec13df54e0f71b615bd6f2f9470d694fee4e21a31`, raw candidate report `92faced7193e3b6a479dce52a84eff8b047ce741facc1b3f5b87e649db9eb6ff`, validation audit `773b8aab1a378ab28e9d79621016b3e7915dc0f40e4125ffbdb191ea4ff52aca`, and temporal summary `9806549a3bbc8e88803c5ab7b1eb496910498a2e02ae088bde02c0b772ec765a`.

## 3. Temporal backtest: archived report versus current-code replay

Archived source: `runs/deep_learning_iot_intrusion_de_20260831_114802/outputs/temporal_backtest/temporal_backtest_summary.json`, with full per-cutoff records in sibling `cutoff_YYYY/temporal_backtest.json`. Its controls are an automated **silver standard**: future papers with limitation-plus-action triples or previously absent typed relations are positive controls; pre-cutoff source-resolved limitations are negative controls. The backtest does not establish global novelty or independently adjudicated solution quality.

**Important reproducibility discrepancy:** Running current `src/temporal_backtest.py` and current `config.yaml` on the identical 150-paper corpus and 1,404 triples produces different control counts and recall from the archived report. The replay rebuilt each pre-cutoff graph and wrote results only to `paper_icai2026/audit/replay/audit_summary.json`. The script is `paper_icai2026/audit/replay_backtest.py`; run it from `ESV-Gap` with `python -B paper_icai2026/audit/replay_backtest.py`. It checks both input SHA-256 hashes before running. This appears to be historical code or configuration drift, but the exact cause is not established. Do not call the archived 108-control result reproducible with the current tree.

Archived report:

| Cutoff | Pre/future papers | Silver positives / negatives | Generated candidates | Recall@5 / @10 / @20 | Unlabelled candidates | Certified |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 46 / 104 | 41 / 13 | 0 | 0 / 0 / 0 | 0 | 0 |
| 2023 | 65 / 85 | 39 / 15 | 0 | 0 / 0 / 0 | 0 | 0 |
| 2024 | 91 / 59 | 28 / 26 | 16 | 0.0357 / 0.0357 / 0.0714 | 10 | 0 |

Across cutoffs there are 108 positive and 54 negative silver controls. Macro recall@20 is 0.0238, the mean of 0, 0, and 0.0714; only two of 28 controls are reached at the 2024 cutoff's top 20. All recorded leakage checks pass **for supporting paper IDs**, as implemented. Ten of sixteen 2024 candidates (62.5%) have neither positive nor negative silver label. Precision@20 on the *labelled* controls is 0.2857, with unlabelled candidates excluded. Do not call the report's 1.0 `precision_after_local_closure_on_labelled_controls` an end-to-end or human-verified precision: only the 2024 slice contributes, and `src/temporal_backtest.py` explicitly removes matches to known negative controls before computing it.

Current-code replay, same frozen input hashes:

| Cutoff | Silver positives / negatives | Generated candidates | Recall@20 | Unlabelled candidates | Certified |
|---|---:|---:|---:|---:|---:|
| 2022 | 14 / 0 | 0 | 0 | 0 | 0 |
| 2023 | 14 / 0 | 0 | 0 | 0 | 0 |
| 2024 | 8 / 6 | 16 | 0.1250 | 11 | 0 |

The current-code total is 36 positive and six negative controls, macro recall@20 0.0417. At 2024 the replay matches one of eight positive controls, while eleven of sixteen candidates (68.75%) are unlabelled. The stable part of both reports is the count of generated candidates (0/0/16) and no replayed certificate. These different control denominators materially change any effectiveness statement; the manuscript must present the two as separate artifact versions or use only the current replay after explaining the historical discrepancy. The reviewer-reported 0/22 remains untraceable to either artifact.

The backtest source hard-codes `certified_candidate_count = 0` and `certificate_precision = None` because historical full-text and external-search snapshots were not replayed. Thus `certified=0` is an *unevaluated certification path*, not measured zero precision or measured failure of all possible certificates. The pre-cutoff graph is correctly built from pre-cutoff triples, but the leakage check only traverses each candidate's `supporting_paper_ids`; the paper should not claim a comprehensive audit of every feature or external retrieval timestamp.

The archived silver labels require adjudication. For example, archived 2024 positive control `tc_87c1a142a127c154` connects a limitation called “high computational overhead” to evidence merely saying the approach was evaluated on CIC IoT 2022 data; that quote does not directly show overhead was resolved. Archived control `tc_1385f375f45e9549` similarly uses an `EXTENDS` relation about an evaluation framework for a “computational efficiency” limitation. These are possible label errors, not proven corrections. A blinded stratified sample should be inspected before either archived or replayed controls are treated as scientific ground truth. The current `src/temporal_backtest.py` allows broad action relations (`EXTENDS`, `APPLIED_TO`, `EVALUATES_ON`, etc.) and a token-similarity threshold of 0.28, explaining the risk.

## 4. Other evidence quality and attribution limits

* The extractor diagnostic is 28 true positives, four false positives, and eight false negatives over 36 labelled triples in ten short texts: recall 77.8%, precision 87.5%, miss rate 22.2%. Source: `runs/.../outputs/extractor_recall_report.json` and `benchmarks/gold_triples_benchmark.json`. The benchmark metadata calls its domain “Explainable AI (XAI) & ML Security,” and its texts are compact, constructed/paraphrased descriptions of named papers, not a sample of the frozen IoT abstracts. The file names no annotators, adjudication procedure, or agreement. The numbers are an out-of-domain diagnostic and do not bound IoT extraction error. Its report sentence interpreting 22.2% as `P(extractor omission | an absent graph edge)` is statistically invalid; the denominator is *present gold relations*, so it estimates `P(missed | a gold relation is present)` on that small benchmark only.
* `outputs/fault_injection_results.json` reports six of six programmed abstentions, but scenarios 5 (circular-only corroboration) and 6 (missing provenance) both abstain with the reason “Unrecognized external verdict state 'absence_corroborated'.” This demonstrates a conservative fallback for those inputs, not successful isolation of the intended circularity/provenance mechanism. It is a deterministic test suite, not a 100% field reliability estimate.
* The submitted five-candidate table and `paper_v2/canonical_results.json` are a curated verification cohort. Their 3/5 refuted and 1/5 retained-open figures are descriptive within that cohort. External semantic relation labels have no independent human benchmark in the source manuscript; a verbatim substring check verifies *quotation presence*, not correctness of the asserted resolution relation. C4's three supporting 2026 citations indicate a stated limitation; they cannot establish unsolved global novelty.
* `paper_v2/results_summary.json` includes an older controlled benchmark with 800 synthetic/planted candidates and perfect full-contract precision/recall. This is implementation validation under designed labels, not measured real-corpus effectiveness. The microservices snapshot has 185 papers and zero automatically eligible. The separate MongoDB run has 53 retained papers and 128 raw candidates, yielding 0 automatic / 9 review / 119 rejected offline. Its author-internal unblinded Accept decisions must not be called independent expert precision. Keep these cohorts separate from the 150-paper IoT experiment.

## 5. Defensible revision and experiments needed for a discovery claim

**Claims possible now.** A compact method or audit paper may report the 150-paper abstract-based IoT case, 133 raw signals, zero certificates, five review items, 128 rejections, five-candidate evidence-trace illustration, and weak silver-standard temporal recall. Its contribution is an auditable abstention/triage mechanism with explicit limits. State clearly that positive research-gap discovery and graph-topology utility are not demonstrated by the current run. Do not promise acceptance.

**Smallest high-value new study.** Freeze a larger, timestamped, multi-domain corpus with query strings, retrieval dates, DOI/title deduplication, inclusion/exclusion logs, text availability, and immutable hashes. Construct an expert-adjudicated historical set of *pre-cutoff unresolved questions* and *post-cutoff resolutions*, including hard negative controls where papers only mention a limitation. Use at least two or three independent domain reviewers, blind them to system arm and future outcome, preserve reasons and disagreement adjudication, and publish the annotation guide. Power the sample size against confidence intervals rather than targeting a convenient percentage.

**Method candidate.** Add a typed, provenance-preserving evidence map and a two-stage retrieval/verifier with source-linked quotations and explicit resolution criteria. Generate candidates from direct limitation statements plus feasible relation proposals, and reserve graph topology as an optional ranking feature. Require a machine-readable certificate containing the pre-cutoff claim, disjoint supporting sources, external search ledger, counterevidence disposition, and coverage diagnostics. Keep `review_required` separate from `certified`. The architectural novelty must be tested by measured outcome, not by naming additional gates.

**Ablation that can settle the topology question.** On the *same* frozen corpus and candidate budget, compare (i) direct author-stated limitations, (ii) retrieval-only gap proposals, (iii) graph typed relations without TransE/Louvain, (iv) topology additions, and (v) each of these with the identical verifier. Report unique expert-confirmed discoveries, recall@k against a fixed adjudicated set, precision@k with unlabelled outcomes disclosed, review workload, latency/cost, and paired uncertainty intervals. If topology adds no confirmed positives, remove TransE/Louvain claims from the paper rather than defending complexity with graph size.

**Temporal protocol.** Lock model, rules, prompts, and corpus before test years; exclude all post-cutoff metadata from generation and retrieval; compare across at least three domain/year folds. Maintain point-in-time search snapshots or clearly mark retrospective search as leakage-prone. Manually review a stratified sample of future silver controls, including the two suspect examples above. Report candidate generation, verification, and certificate metrics separately so a zero at the first stage cannot be hidden by downstream abstention metrics.

## Source index

Primary run: `ESV-Gap/runs/deep_learning_iot_intrusion_de_20260831_114802/`.

Submitted source/PDF: `ESV-Gap/paper_v2/main_springer_ami.tex`, `ESV-gap submit AMI.pdf`.

Main machine reports: `data/processed/screening_diagnostics.json`, `data/processed/corpus_filtered.jsonl`, `data/triples/all_triples.json`, `outputs/detected_gaps_raw.json`, `outputs/gap_validation_audit.json`, `outputs/validated_gaps.json`, `outputs/primary_research_gap.json`, `outputs/full_text_enrichment.json`, `outputs/ablation_table.csv`, `outputs/temporal_backtest/temporal_backtest_summary.json`, `outputs/temporal_backtest/cutoff_2024/temporal_backtest.json`, `outputs/extractor_recall_report.json` under that run. Additional source files: `ESV-Gap/src/temporal_backtest.py`, `ESV-Gap/benchmarks/gold_triples_benchmark.json`, `ESV-Gap/outputs/fault_injection_results.json`, `ESV-Gap/paper_v2/canonical_results.json`, `ESV-Gap/paper_v2/results_summary.json`.

For a standard-library-only recount of historical files, run `python -B paper_icai2026/audit/reproduce_counts.py` from `ESV-Gap`; it writes `paper_icai2026/audit/reproduced_counts.json` without touching the run. For a current-code temporal replay, run `python -B paper_icai2026/audit/replay_backtest.py`; output is `paper_icai2026/audit/replay/audit_summary.json`. On 2026-09-29, `python -B -m unittest discover -s tests -p 'test_temporal_backtest.py' -v` passed 4/4 tests. These tests check small constructed examples; they do not resolve the archived/replay label discrepancy.

The replay JSON records the exact execution environment and SHA-256 of every code/config input, without publishing configuration contents. Environment: CPython 3.14.6, NetworkX 3.6.1, PyYAML 6.0.3, NumPy 2.5.1, SciPy 1.18.0, python-louvain 0.16. SHA-256: `src/temporal_backtest.py` `79b7072315bf394340d5343c574b18c5f706a2e99d2689314cfd491da7d34fc2`; `src/detect_gaps.py` `be8f6bdd4259ed324ae95b7d02a9745ba71c65305b7a0dd57140f9a567bffa6f`; `src/evidence_graph.py` `7f13b2a4cac56bf7824659fc1a3d2e956865b436a7d10dad1ca6fd0937322f7e`; `config.yaml` `0024f3c746c62d958d96a55d07c41dbfb9b39e21097bfeef9a8da1a750619511`; replay script `620601a1e807e6e2d544f2d1e6c7c8121a9389d2769d10635196a6874ba411bc`. The rerun exited 0 and printed 2022/2023/2024 candidate counts 0/0/16, silver positives 14/14/8, and macro Recall@20 0.0417. This fingerprint pins the current replay, but the archived run has no corresponding source/config fingerprint, so the drift's exact cause remains unresolved.
