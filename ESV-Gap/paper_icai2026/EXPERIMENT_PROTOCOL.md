# ESV-Gap Follow-up Evaluation Protocol

**Status:** Proposed executable protocol. No result below is claimed as completed. Freeze this document, code commit, data manifest, prompts, thresholds, and analysis plan before looking at held-out test outcomes.

**Existing prototype boundary:** the current `prototype/scoped_contract.py` validates same-experiment scope/condition/metric bounds, cutoff chronology, and trusted human evidence-span metadata; the project owner reports 10 passing synthetic tests. Treat this as a data-contract preflight component only. It does not validate semantic entailment, research novelty, candidate-recall, or KG utility, and synthetic unit tests are not the proposed empirical evaluation below.

## 1. Study question and estimands

The study evaluates whether ESV-Gap helps researchers retrieve and verify scoped, potentially unresolved research questions under a fixed review budget. It does not attempt to prove global absence of research.

Treat the system as two linked but separately scored tasks:

1. **Candidate retrieval/ranking:** Given literature available by cutoff `t`, rank scoped gap candidates so that expert-judged useful candidates and future problem–solution events appear near the top.
2. **Evidence verification:** Given a fixed candidate and a retrieved evidence pool, annotate separate orthogonal axes for each document: mutually exclusive `closure_state`, mutually exclusive `claim_stance`, and `evidence_adequacy`; return sentence spans and scope qualifiers. `INSUFFICIENT` must not be treated as proof of absence.

Primary estimands: candidate `Recall@k` under a fixed review budget; nDCG@k against expert utility labels; verifier macro-F1 and classwise recall on pooled evidence; and the incremental difference between text-only and KG-assisted variants. Report risk–coverage and abstention separately. None is a direct measure of world-level novelty.

### Confirmatory hypotheses

- **H1 (retrieval):** Hybrid lexical+dense retrieval improves expert-relevant candidate Recall@k over BM25 at equal query/document budget.
- **H2 (verification):** Scope-aware verification distinguishes full from partial closure better than a co-mention/keyword rule on identical pooled documents.
- **H3 (graph incremental value):** Adding graph structure/provenance to the best text-only system improves nDCG@k or expert-rated utility at fixed review budget. If the CI includes zero or the gain is negligible, conclude no demonstrated incremental discovery value for KG topology.
- **H4 (selective reliability):** Calibrated deferral reduces verification error among automatically decided items as coverage decreases. The operating point is selected on the calibration split only; abstention rate and errors remain visible.

## 2. Corpus and temporal split

### Domains and sources

Run on at least three domains selected before labels are inspected: one cybersecurity/IoT domain for continuity, one non-security computing/engineering domain, and one domain with suitable publication metadata and expert access outside computer science. Selection criteria: machine-readable year/DOI/abstract, stable retrieval API or archive, terms definable without reference to outcomes, and availability of domain annotators. Record database/API, query strings, filters, retrieval time, pagination/limits, raw IDs, title/abstract, DOI, year, venue, deduplication and exclusions. Preserve raw query response or its permitted checksum/manifest.

The existing 150-paper snapshot and 53-paper MongoDB run are pilot artifacts, not substitutes for these cohorts. They may be used only if their provenance and time cutoffs can be independently established; mark missing retrieval logs as a reproducibility limitation.

### Time slicing

For each domain, pre-register at least three cutoffs selected to leave a useful future window (e.g., yearly cutoffs with 2–3 years of future observation, subject to corpus availability). At cutoff `t`, candidate generation may access only papers with publication date `≤t`, and only metadata/text that could actually have been indexed on the snapshot date. Freeze external retrieval index snapshot/date as well. Exclude uncertain/incomplete year dates from confirmatory slices; report them separately.

Use paper-level splits so no DOI, preprint/published duplicate, or same source document appears in more than one split. Use an older period for training/development, a later pre-test period for calibration, and the latest eligible temporal slice as locked test. Do not tune aliases, thresholds, prompts, matching rules, query budget, or labels after inspecting test results. A leave-one-domain-out evaluation is required as the cross-domain stress test.

## 3. Scoped hypothesis and label standard

Represent each candidate as a structured proposition, at minimum:

`<method/approach, capability or limitation, dataset/system/context, operating conditions, metric/outcome, population/threat model, time scope>`.

Annotators may mark a field `unspecified`; the system must not silently broaden it. A broad statement is not considered resolved by a paper that only addresses one dataset, one threat model, or one metric. Record whether the paper explicitly proposes/evaluates the relevant action, reports a measurable outcome, and whether that outcome meets a predeclared criterion (when a criterion exists).

### Evidence-document labels

Use three independent axes. Labels within each axis are mutually exclusive; labels across axes can co-occur. Thus a paper can partially close one scoped condition and still support a broader limitation.

- **Closure state:** `FULL` (evaluates an action addressing the full stated scope with adequate evidence); `PARTIAL` (addresses only a subset of conditions/datasets/threat models/metrics/populations/settings; enumerate covered and uncovered qualifiers); `NO_RESOLUTION_EVIDENCE` (no demonstrated solution/evaluation for the scoped capability, including limitation-only or related mention); `INDETERMINATE` (source/evidence unavailable or insufficient to decide).
- **Claim stance:** `SUPPORTS_LIMITATION`, `REFUTES_LIMITATION`, `MIXED`, or `NEUTRAL_MENTION`, always relative to the exact scoped hypothesis. A partial closure may be `MIXED` relative to a broader claim.
- **Evidence adequacy:** `ADEQUATE` (source inspectable and cited span supports the assigned judgments); `INADEQUATE` (inspected context does not substantiate them); `UNAVAILABLE` (retrieval/API/full-text failure or incomplete record). `UNAVAILABLE` cannot establish closure and cannot establish that the gap remains open.

### Candidate-level labels

Experts label separately: (a) retrieval relevance; (b) current evidence state `closed / partially addressed / plausibly open in stated scope / unclear`; (c) novelty within the indexed corpus at cutoff, not global novelty; (d) scientific/actionability value; (e) rationale spans and references. Candidate labels use four-state logic: `supported-open-in-scope`, `partially-addressed`, `closed/refuted`, `uncertain`. “No retrieved evidence” maps only to `uncertain` unless the search coverage standard has been met; even then it is bounded to named sources, queries and date.

## 4. Pooled benchmark and expert blinding

Construct candidate and evidence pools by pooling top outputs from every retrieval system (BM25, dense, hybrid, KG, and relevant human/library search if feasible), plus future event controls. Deduplicate and randomize candidates before annotation. This avoids annotating only the proposed method’s favorites. Keep a hidden mapping from randomized ID to system/rank.

At least two domain experts independently annotate every test item. A third adjudicates disagreement without seeing model identity or system rank. Experts see claim scope, bibliographic record, source text/evidence, and a neutral candidate ID, but not whether it came from ESV-Gap or a baseline. Randomize order and mix controls. Pilot the rubric on development items, revise it, then freeze before test. Collect rationale spans and confidence. Report raw agreement plus Krippendorff’s alpha (nominal/ordinal as appropriate) or a justified alternative; report adjudication rates. If expertise or blind status cannot be achieved, state this and downgrade claims; internal authors’ ratings are not an independent gold standard.

For each expert decision, record whether identity/title reveals a likely source system. Ask experts to disclose conflicts and whether they recognize the paper. Analyze sensitivity excluding recognized items if non-trivial.

## 5. Systems and matched-budget baselines

Run every method on the same cutoff corpus and candidate universe, with the same maximum retrieved documents per candidate, same query count/token budget, same verifier model/version where applicable, and identical full-text access. Freeze API/model version and prompts; log request/response IDs, timestamps, failures, cost and latency. Cache responses to make reruns stable.

Minimum systems:

| ID | System | Purpose |
|---|---|---|
| B0 | Random ranking within eligible candidate pool | Chance reference; use multiple fixed seeds |
| B1 | BM25/TF-IDF over paper title+abstract | Sparse lexical retrieval |
| B2 | Dense embedding retrieval | Semantic text-only retrieval |
| B3 | Hybrid B1+B2, fixed fusion rule | Strong text-only retrieval baseline |
| B4 | KG structure/link prediction only (e.g. TransE if implemented reproducibly) | Topology baseline; exclude if code/artifact cannot be frozen |
| B5 | ESV-Gap full candidate retriever | Proposed retrieval system |
| V0 | Co-mention/keyword closure heuristic on shared pooled docs | Simple verification baseline |
| V1 | Same LLM verifier without KG/provenance context | Isolate text verification |
| V2 | ESV-Gap verifier with provenance/KG evidence context | Proposed verification stage |
| V3 | Human evidence packet without model verdict | Estimate expert task performance/time under same evidence budget |

Also run ablations: remove KG structure but retain text; remove source-disjoint provenance; remove temporal features; remove external search; remove scope qualifiers/partial-closure labels; disable calibration/defer policy. Do not label algorithm components “essential” absent controlled ablation.

## 6. Temporal controls and leakage prevention

Create two control families, with publication-level provenance:

1. **Future problem–solution controls (silver positives):** a paper after cutoff `t` states a scoped limitation and presents/evaluates an action on that capability, or a future paper realizes a typed relation absent pre-cutoff. These show observable later activity, not a globally first discovery. Require capability-specific matching; subject-only match is insufficient.
2. **Pre-cutoff closure controls (silver negatives):** prior papers that demonstrate full or partial resolution before `t`. A candidate should not be labeled “open” for a scope already resolved. Partial closure remains its own label and cannot be collapsed into fully solved/unsolved.

Unmatched cases remain unlabeled, not false positives. Use one independent process to construct controls and another blind process for candidate annotation. Deduplicate future events by normalized scoped capability and paper ID; ensure the future source does not leak through citations, embeddings, graph edges, cached retrieval, author-stated-gap index, or candidate-generation prompts. Run automated assertions that every support paper used at cutoff is `≤t` and every positive outcome is `>t`. Keep test-year controls inaccessible until output rankings are frozen.

The current `src/temporal_backtest.py` is a useful starting point: it separates pre/post papers and reports silver-standard caveats, but lexical `token_similarity` and control construction need expert-validated scope matching. Keep three temporal artifacts separate: (1) reviewer screenshot: 0 recall on 22 positive controls and 153 signals; (2) archived summary on primary frozen inputs (corpus SHA-256 `852b63bf0b7f3fb641046b416c45723156d5663b56f87352b9145efb22282583`, triples SHA-256 `a352427bffb0d6d63bce589ec13df54e0f71b615bd6f2f9470d694fee4e21a31`): 108 positives (41/39/28) and macro candidate Recall@20 = 0.0238 (0/0/0.0714); (3) current-code replay on those same hash-verified inputs: 36 positives (14/14/8), 6 negatives and macro Recall@20 = 0.0417 (0/0/0.125). Candidate counts are 0/0/16 for archived summary and replay. The difference is not evidence of improved performance: the control/label denominator changed due to unreconciled code/config drift in control generation; its cause is not yet established. The screenshot values do not currently reconcile with either artifact. Both archived and replay artifacts have zero certified candidates; archived certificate precision is null/not estimable, and replay's `certified=0` is a count, not a precision estimate. Before citing one result, pin code/config and diff the per-control IDs and labels. Do not merge candidate retrieval and certification metrics.

## 7. Metrics and statistical analysis

### Candidate retrieval/ranking

- `Recall@k` of expert-relevant candidates and future silver controls; choose `k` as a realistic fixed review budget before test (e.g. 5/10/20 only if those match workflow capacity).
- `Precision@k`, `nDCG@k`, and average precision for expert relevance/utility; label which annotation target each score uses.
- Coverage of positive controls by cutoff; report unreachable controls and unmatched label rate.
- Candidate diversity/redundancy (semantic duplicate rate) and per-domain/candidate-family breakdown.

### Verification

- Compute macro-F1 and per-class precision/recall separately for mutually exclusive `closure_state` labels and, independently, mutually exclusive `claim_stance` labels; report evidence-adequacy classification separately. Do not collapse orthogonal axes into one macro-F1.
- Evidence sentence span precision/recall or token F1 under a frozen overlap protocol, plus expert evidence sufficiency rating.
- Confusion between closure-state labels `FULL` and `PARTIAL` is reported explicitly.
- Calibration: do not treat an LLM's verbal self-reported confidence as probability. Use frozen numeric class scores/logits from a verifier that exposes them, or deterministic scores calibrated against human labels on a disjoint calibration split; freeze the calibration mapping before test. If no valid numeric score is available, report confidence as ordinal only and omit Brier/log loss/ECE. When probabilities are available, report multiclass Brier score, log loss, reliability diagram and ECE with stated binning plus uncertainty; interpret small samples descriptively.
- Selective behavior: risk–coverage curve, error among decided cases, abstention rate and reasons; report coverage at predeclared risk targets only if sample size supports them.

### KG incremental value and cost

All key comparisons are paired at query/topic, cutoff and domain. Report `Δ` from best text-only baseline to KG-assisted system, with 95% cluster bootstrap confidence intervals resampling topics (and domains as top-level clusters where feasible). Use paired randomization/permutation for predeclared primary comparisons. Correct secondary hypothesis tests for multiplicity (e.g. Holm); mark exploratory analyses. Also report added latency, API calls, tokens, corpus construction overhead and human minutes per accepted useful item. A positive score on one metric cannot offset a failure on the primary retrieval metric.

### Sample size and uncertainty

Select topic/candidate counts before annotation using a precision/CI-width target and expected prevalence from development only, accounting for clustering by topic/domain. Do not use an arbitrary paper count as a substitute for effective positive labels. If the number of independent domains/topics is too small for reliable generalization, present estimates descriptively with wide uncertainty and explicitly limit the inference.

## 8. Thresholds, calibration and fail-closed policy

Fit/tune on train/dev; select operating threshold on a separate calibration set; lock it before test. Never choose thresholds from test performance. Calibrate only frozen numeric class scores or deterministic verifier scores against held-out human labels; state that calibration may drift across domain/time. A sufficiently reliable, inspectable, provenance-valid, direct piece of evidence can establish `FULL` closure for its stated scope and refute that same scoped gap even if other search queries fail. `PARTIAL` closure preserves unresolved qualifiers. Promoting a candidate as `SUPPORTED-OPEN` requires a completed predeclared search, no full-closure evidence, adequate source coverage, and required independent support. Any failed query, unavailable evidence, missing provenance, or incomplete search prevents open promotion and routes to `UNCERTAIN/REVIEW`. Search failure is not evidence for either side.

Report thresholds and rationale, all policy branches, sensitivity over a predeclared threshold grid, and risk–coverage. Do not advertise formal risk guarantees unless a guarantee is derived from the actual sampling assumptions and validated protocol.

## 9. Reproducibility and audit output

For each run release (subject to API/data licenses):

- code commit, environment lock file, config, prompts, model and API version, random seeds;
- corpus query/retrieval ledger, timestamp, IDs/DOIs, dedup and screening flow, exclusions;
- cutoff snapshots or checksums, external retrieval query logs and raw result IDs;
- candidate pool before/after each stage, per-candidate scores, evidence sentences, disposition and abstention reason;
- annotation handbook, randomized packets, anonymized rating ledger, adjudications and agreement calculations;
- all baseline, ablation and metric outputs, confidence interval script, failure logs, runtime/cost;
- machine-readable manifest with SHA-256 hashes and exact reproduction commands; license/consent for each artifact.

Never claim source-complete retrieval if the index snapshot is incomplete. Never infer “global absence” from an empty bounded search.

## 10. Stop rules and interpretation

- If future positive controls cannot be matched at useful rates, state that ESV-Gap has not demonstrated discovery utility; analyze why and report observed recall unchanged, including low non-zero values. Preserve zero recall where it occurred at individual cutoffs or in the screenshot-reviewed run, with the exact run denominator.
- If graph-assisted results fail to exceed text-only baselines within uncertainty, retain KG only as a provenance/audit layer and drop claims that graph topology drives discovery.
- If expert agreement is poor, refine the construct/rubric on development, not test; report the construct as ambiguous if disagreement persists.
- If calibrated high-confidence decisions have unacceptable error or calibration shifts across domains, lower coverage or abstain; do not move threshold after unblinding the final test.
- A completed protocol can support an empirical paper; it cannot assure conference acceptance.

## 11. Four-to-six-week execution schedule

| Period | Deliverable |
|---|---|
| Week 1 | Freeze preregistration, domains, queries, temporal cutoffs, label manual, data manifest schema |
| Week 2 | Build and audit corpora, establish aliases/scope ontology, generate train/dev/calibration/test locks |
| Week 3 | Pool baseline retrievals, expert pilot on development set, freeze annotation rubric and prompts |
| Week 4 | Annotate blinded test, run retrieval/verifier/ablation with cached API outputs |
| Week 5 | Adjudicate, compute paired metrics/CIs/calibration/cost, reproduce from frozen artifacts |
| Week 6 | Error analysis, rewrite claims and limitations, release permitted artifacts and final paper |

If any phase slips, reduce the number of confirmatory questions rather than silently shrinking the test set or dropping failed domains after observing results.
