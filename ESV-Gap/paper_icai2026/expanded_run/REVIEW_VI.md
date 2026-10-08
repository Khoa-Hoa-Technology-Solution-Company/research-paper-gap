# Đánh giá khoa học độc lập: replay đa miền

## Kết luận ngắn

Replay mở rộng là một cải thiện về **độ rộng chẩn đoán**: cùng code hiện tại đã được chạy offline trên bốn corpus có sẵn, 378 record bài báo phân biệt theo khóa DOI-ưu-tiên-rồi-paper-ID, trong 10 domain–cutoff fold đủ điều kiện. Nó chưa phải đánh giá chứng minh khả năng khái quát hoặc hiệu quả discovery. Macro candidate Recall@20 = 0.0139 trên chín fold có positive control; tổng 64 positive controls và 39 candidate rows đều không độc lập giữa các fold. Nhãn positives là silver controls sinh bằng heuristic, không phải gap do chuyên gia xác nhận. Không có live ingestion/retrieval, human baseline, expert annotation, hay chứng nhận gap. Nên gọi là **offline retrospective multi-corpus replay**, và trình bày như robustness check thăm dò, không phải xác nhận đa miền.

## Những gì phép chạy thực sự cho thấy

| Domain | Papers | Completed cutoffs | Positive controls theo cutoff | Candidates | Recall@20 |
|---|---:|---:|---:|---:|---|
| IoT intrusion detection | 150 | 2022, 2023, 2024 | 14, 14, 8 | 0, 0, 16 | 0, 0, 0.125 |
| Microservice security | 143 | 2022, 2023, 2024 | 14, 8, 2 | 5, 6, 7 | 0, 0, 0 |
| MongoDB security | 53 | 2023, 2024 | 2, 1 | 0, 1 | 0, 0 |
| Handwritten math recognition | 32 | 2022, 2023 | 1, 0 | 0, 4 | 0, not scored |

MongoDB 2022 had only 9 pre-cutoff papers, and math 2024 had only 8 future papers, so each was excluded by the 10-paper minimum. Math 2023 had no positives and is excluded from the recall macro. The macro 0.0139 is the equal-weight mean of the nine remaining fold recalls, not a pooled estimate weighted by number of controls. Since cutoffs overlap in their train/future periods and may reuse papers and controls, the 64 controls must not be treated as 64 independent observations. The 39 candidate rows likewise count repeated cutoff outputs, not 39 unique ideas.

Across all scored folds, recall@5 and recall@10 are both zero; retrieval of any silver positive occurs only among top 20 in the IoT 2024 fold (one of eight, recall 0.125). The generator produces zero candidates in IoT 2022/2023 and MongoDB 2023; this is a retrieval-coverage failure for those folds, not a successful selective verifier. All folds report zero certificates or certification as unestimable because historical full text and search snapshots are unavailable. Candidate recall and certificate validity answer different questions.

The runner reports all implemented supporting-paper-ID checks passed. This is a useful leakage diagnostic, but it only checks the candidate `supporting_paper_ids` against pre-cutoff IDs in code. It does not establish that historical search/index availability, citation metadata, aliases, extraction results, or other features were available at the cutoff. Publication year is not an index-ingestion timestamp. The run is explicitly retrospective, not a point-in-time replay of what a researcher could have retrieved at that date.

## Valid inferences

- The same current detector/backtest implementation can execute on four cached corpora and emit candidates in some folds. This demonstrates operational portability across these files, not statistical or scientific generalization.
- The run preserves a reproducible path: corpus/triples hashes, config and code fingerprints, environment, cutoffs, per-domain outputs, and leakage assertions. Preserve this artifact if results are cited.
- The low recall and empty candidate sets reveal a concrete limitation: the candidate generation stage has not shown useful anticipatory coverage of these heuristic future controls. This result should remain visible, including folds with zero candidates.
- Broader-domain outcomes vary: no positive control is matched in microservice security, MongoDB security, or math recognition; only one is matched in the IoT 2024 fold. This is not evidence for broad discovery efficacy.

## What the run does not establish

### Corpus and indexing

The inputs are four pre-existing run folders, not newly collected corpora under a shared retrieval and screening protocol. The runner does not query live OpenAlex/Semantic Scholar or establish index coverage at each historical cutoff. Historical full-text snapshots and external search traces are absent. Abstract availability and screening procedures may differ across source runs, so domain contrasts are confounded by corpus construction, size, topic query, and time coverage.

The script's identity key uses DOI where present, otherwise paper ID, otherwise normalized title. It reports 378/378 distinct across these keys; that does not prove 378 bibliographically unique works. DOI punctuation/canonicalization, DOI-vs-paper-ID variants across corpora, preprint/journal versions, and title variants can evade or distort duplicate detection. State this as “378 distinct under the runner’s DOI/ID key,” not as a verified deduplication audit.

### Imbalanced/small folds

The 53-paper MongoDB and 32-paper math corpora yield only one or two positives in several scored folds; macro recall at those denominators is highly unstable and has no confidence intervals. A zero can arise from sparse candidate generation, weak matching, a poor proxy label, or a small corpus. Conversely, 0.125 in the IoT 2024 fold is just one matched control out of eight. Four corpora are not four well-powered domains when two are small and most folds have few labels.

The macro average gives each fold equal weight even though positive denominators range from one to fourteen and the folds/domains are dependent. It is a descriptive mean, not a well-calibrated pooled discovery estimate. No bootstrap or uncertainty interval is in the summary. If reporting, give per-fold counts and the exact macro definition; avoid p-values or population-level claims.

### Silver control validity and matching

Future controls are created heuristically from extracted `LACKS` plus action relations, or future typed links between nodes already in the graph. Matching applies lexical token similarity. Code allows broad action relations such as `EXTENDS`, `APPLIED_TO`, and `EVALUATES_ON`; these may describe a related method or evaluation without resolving the scoped limitation. Prior audit found examples where a computational-overhead limitation was paired with a paper merely evaluated on a named dataset. Such records require independent semantic adjudication.

Controls are not gold labels for open gaps or novelty. A later paper can appear after a cutoff without being the first or only resolution; extraction can miss true controls; entity strings can mismatch; and relation/subject/capability similarities can be misleading. Unmatched candidates are explicitly unlabeled, and precision over only labelled controls cannot be interpreted as end-to-end precision. The 64 positives are across overlapping folds, so some future events may contribute to multiple cutoff evaluations.

### Baselines, experts, and system utility

The replay compares the system with its own heuristic outcome controls only. It has no matched-budget BM25, dense/hybrid retrieval, TransE/Louvain, random, or human-search baseline; no component ablation; no blind expert assessment; no inter-rater agreement; no review-time or practical utility measure. Thus it cannot attribute behavior to KG topology or demonstrate that KG adds value over ordinary literature retrieval.

The archived replay is offline over cached triples. It does not test real-new-paper ingestion, changing external index coverage, fresh relation extraction, or stability under current retrieval APIs. It therefore does not answer whether the system works on new research appearing after the frozen snapshots.

## Reviewer feedback → what this new evidence changes

| Reviewer concern | Effect of multi-domain replay | Remaining work / defensible response |
|---|---|---|
| W1: temporal recall zero; no future gaps anticipated | Adds three additional corpora and ten completed folds; macro Recall@20 is still only 0.0139, and eight of nine scored folds have recall 0. Only one IoT 2024 control is matched. | Acknowledge discovery performance remains unproven/weak. Label future controls as heuristic silver proxy, not true discoveries. Fix and adjudicate scope matching, then evaluate against independently labeled temporal cases and matched baselines. |
| W2: KG topology methods did not contribute to validated hypotheses | Multi-domain replay shows candidate output in some folds but performs no no-KG/with-KG ablation and no expert verification. | Do not claim topology adds value. Compare text-only baselines to KG-enhanced retrieval under matched budgets, with paired uncertainty and expert outcome labels. |
| W3: only 150 papers in one narrow subfield | Expands cached inputs to 378 papers across four named topics, but only two domains have >100 papers; MongoDB and math are 53 and 32. The cohorts were not newly retrieved under one harmonized protocol. | This answers “single corpus only” as a preliminary breadth check, not as adequate powered multi-domain validation. Build source-complete, timestamped corpora, with enough independent topics/domains and expert-labeled items. |
| W4: no practical discovery effectiveness or incremental yield | Still no human-validated useful gap, no expert study, no human baseline, and certification remains unavailable. | Keep contribution at auditability/failure analysis and scoped-contract design proposal. Measure expert-rated usefulness, false-open risk, candidate recall under fixed review budget, and time saved before efficacy claims. |
| Reviewer B: clarify KG role and avoid overstatement | More folds reveal portability of execution; no causal evidence on the role of KG. | Describe KG as current candidate/evidence structure only; separate data organization/provenance from topology-driven discovery. |
| Reviewer B: across multiple domains | Four cached topics now present, but not cross-domain expert generalization. | Say “four retrospective corpora” or “multi-corpus exploratory replay,” not validated cross-domain performance. |
| Reviewer B: expert validation and threshold justification | Neither provided by this offline replay. | Two blinded experts plus adjudicator, frozen thresholds, calibration split, risk–coverage and inter-rater agreement remain needed. |

## Suggested manuscript wording

> “We ran a current-code, offline retrospective replay on four pre-existing domain corpora (378 records distinct under a DOI-first/ID-fallback key). Ten domain–cutoff folds met the minimum corpus-size rule, and nine contained at least one heuristic future control. Macro Recall@20 over those nine folds was 0.0139; recall was non-zero only in the IoT 2024 fold, where one of eight heuristic controls was matched. The 64 positive controls are silver-standard and not independent across overlapping cutoffs. No gap was certified: historical full-text and external-search snapshots were unavailable. The result is a multi-corpus diagnostic, not evidence of expert-validated research-gap discovery or a controlled estimate of graph value.”

If the archived 108-control result is also included, show it alongside the fresh 64-control result and explain that the same frozen literature inputs yield different heuristic label sets under archived versus current code/config. Do not call 0.0139 an improvement over earlier figures. Reconcile or explicitly document the code/config drift; show matched control IDs/labels and fingerprints. Until then, frame the current result as exploratory sensitivity analysis.

## Recommended follow-up before any discovery claim

1. Freeze the code/config and manually adjudicate a stratified sample of positive controls and candidate matches, especially broad `EXTENDS`, `EVALUATES_ON`, and `APPLIED_TO` matches; publish per-control label changes and reasons.
2. Add point-in-time index/search snapshots or narrow the estimand to a retrospective corpus-time exercise; publication dates alone do not recreate index availability.
3. Harmonize collection/screening criteria and normalize bibliographic identity across DOI, preprint, venue record, and paper ID; report unresolved duplicates.
4. Add matched-budget sparse, dense, hybrid and graph baselines, plus no-KG ablations; report per-fold candidate counts, Recall@k/nDCG, labelled and unlabelled denominators, paired CIs, and costs.
5. Obtain blinded independent domain-expert labels for retrieval relevance, full/partial/absent resolution, gap-in-scope status, and usefulness. Keep future publication as one silver signal rather than ground truth.
6. Evaluate newly ingested post-snapshot literature prospectively or with a properly archived point-in-time benchmark. Report deployment failures, coverage, abstention and human time saved.
