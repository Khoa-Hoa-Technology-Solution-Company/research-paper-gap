# Response to Major-Revision Review

We thank the reviewer for identifying two central methodological defects and
for proposing a concrete revision agenda. The manuscript and implementation
have been revised as follows.

| Review point | Revision | Evidence/artifact |
|---|---|---|
| Weighted validation threshold is redundant | Removed the threshold from the three-way decision. The weighted value is now `ranking_score` only. | `src/validate_gaps.py`; Method: Decision and ranking |
| A fragile two-edge path survives edge dropout | Missing links now require at least two edge-disjoint and source-paper-disjoint paths. | Unit test `test_single_path_is_rejected_even_when_edge_survival_is_high` |
| Edge deletion misses KG false negatives | Added plausible-edge addition alongside relation-event deletion. | Per-mode survival in `gap_validation_audit.json` |
| Edges from one paper are correlated | Added paper-level dropout that removes all events from a sampled-out paper. | `perturbation_stability`; sensitivity grid |
| Benchmark is circular and single-signal | Added qualified relations, source-dependent paths, synonym and typed-suffix coverage, canonical self-links, plausible bridges, normalized temporal cases, short valid entities, orphan and temporal signals. Development and test seeds are disjoint. | `outputs/validation_benchmark_v3.json` |
| Baseline is weak; no ablation/ranking metrics | Added seven ablations plus Precision@k, Recall@k, nDCG@k, and average precision under a matched budget. | Tables 2 and ranking subsection |
| Incorrect “halves false positives” wording | Removed. The revised paper reports exact counts and percentage reductions. | Results section |
| Entity labels can create artificial gaps | Entity type suffixes are stripped before graph construction and retrieval, case-only variants share an identity key, numeric protocol variants such as OAuth2 also emit a versionless token, and canonical self-links are rejected. The previous five automatically eligible outputs all moved to review or disappeared; no real-corpus item is automatically eligible. | Method, Results, `src/entity_normalization.py` |
| Real corpus does not prove gap quality | Claims are limited to triage workload and auditability. All 40 retained outputs require review; none is called a validated gap. | Abstract, Results, Limitations |
| Missing expert evaluation | Added complete, separately reported author-internal reviews of all nine post-gate candidates (8 Accept, 1 Reject) and the pre-gate top 30 (24 Accept, 6 Reject). The queues overlap by one item; neither review is blinded or independent, and neither acceptance rate is equated with expert precision. The 70-item blinded packet remains the protocol for a future independent study. | `post_gate_expert_reviews.json`; `expert_reviews.json`; `outputs/expert_review_packet.csv` |
| Perturbation probability was ambiguous | Clarified that $\rho=0.95$ is the event/paper retention probability, corresponding to 5% expected deletion/dropout. | Method: Multi-mode perturbation stability |
| Empty plausible-edge pools were implicitly counted as survival | Added an explicit completed-search marker. Without it, the addition mode is `null`, automatic eligibility is blocked, and otherwise hard-passing candidates route to review. Added regression tests for absent, irrelevant, completed-empty, and non-empty pools. | `src/validate_gaps.py`; `tests/test_validate_gaps.py`; archived live audit |
| Reproduction command/config was not frozen | Created a 43-file checksum-addressed final archive containing the exact YAML, executable validator, inputs, both review ledgers, synchronized review CSV, outputs, tests, and manifest. An extraction-and-rerun check reproduces 0 eligible, 9 review-required, and 119 rejected candidates offline. | `output/releases/esv-gap-final-20260810-r1.zip` |
| Zero historical temporal activity was undefined | Documented the detector's piecewise zero-denominator rule and the validator's fail-closed recomputation. | Method: Normalized temporal activity |
| Plausible-edge addition was underspecified | Documented the upstream candidate-edge pool, 0.50 independent sampling probability, retained metadata, endpoint treatment, and empty-pool behavior. | Method: Multi-mode perturbation stability |
| No live B1/B2 comparison | Ran a Mulla-style TF-IDF retrieval baseline (B1) and a no-retrieval per-abstract LLM baseline (B2) on the same 53-paper corpus. Only descriptive yield, length, uniqueness, and lexical overlap are reported because the baselines lack shared expert labels. | Table on descriptive baselines; `comparison_metrics.json` |
| Ranking contribution was unevaluated | Added matched-budget ranking metrics and five ranking-weight sensitivity schemes. | `ranking` and `ranking_weight_sensitivity` report sections |
| Only one signal family had controlled labels | Controlled test now covers missing-link, orphan-community, and temporal signals; all three also run on the real snapshot. | Per-signal confusion matrices |
| Provenance was node-incident rather than path-specific | Missing-link support now comes only from selected source-disjoint evidence paths; orphan provenance uses internal edges only. | `independent_evidence_paths`, `community_internal_papers` |
| Direct edge does not necessarily close a research gap | Observed relations now route to review, with qualifiers explicitly left to experts. | Decision logic and Method |
| “Evidence closure” is too strong | Renamed to candidate-coverage retrieval and documented its lexical limitations. | Method and Limitations |
| Temporal counts and partial 2026 are biased | Uses distinct-paper annual rates and excludes 2026 using the July 20, 2026 snapshot date. | Temporal equation and corpus report |
| Corpus protocol is incomplete | The paper now discloses that the inherited snapshot lacks its retrieval ledger and cannot support a PRISMA replication claim. | Experimental Design and Artifact Availability |
| Reproducibility statement lacked concrete identifiers | Added repository, base commit, lock file, exact commands, data/report hashes, and a patchset manifest. Missing DOI and license are disclosed rather than invented. | `REPRODUCIBILITY.md`; `outputs/artifact_manifest.json` |
| Related work was thin | Added systematic-review automation, literature-based discovery, scientific claim verification, uncertain KG embeddings, provenance standards, and stability selection. | Expanded Related Work and bibliography |

## Remaining required work

The independent blinded expert study (including B1/B2), source-complete
retrieval, multi-domain replication, archive DOI, and explicit software
license remain incomplete. The current 30 decisions come from one unblinded
internal reviewer and do not close this requirement.
The revised paper labels these as prerequisites for external effectiveness
claims rather than presenting planned work as completed evidence.
