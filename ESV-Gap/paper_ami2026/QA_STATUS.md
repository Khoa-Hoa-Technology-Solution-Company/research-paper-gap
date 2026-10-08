# Verification status — current revision, 1 October 2026

## Current manuscript

- Edited the existing open main.tex in place; kept its document editor.
- Added methodological-context retrieval, six baseline/ablation arms, a public expert-labelled evaluation, matched budgets, paper-level descriptive uncertainty, budget sensitivity, and negative cross-lab transfer results.
- Applied the pasted review to the current source: control-first positioning, a native LaTeX system figure, conditional control properties, and primary/secondary audit separation. The pasted review's 11-page PDF is an older export.
- Native compile_latex_document returned compile-failed: Unable to find standard directories for platform.
- Current compilation, page count, and rendered layout are UNVERIFIED. No separate PDF compilation/export was performed in this revision.
- Existing main.pdf and output/pdf/ESV-Gap_AMI2026_Scoped_Draft.pdf are the earlier 11-page draft and STALE relative to current source. Previous visual QA applies only to that version.
- Static citation/reference, environment-nesting and rounded-result checks pass. This is not a substitute for LaTeX compilation or rendered QA.

## New evidence

- Context24 author dataset at revision 457d3b5cb4bb8ade34e37458f4900c6eae0959bb, CC BY 4.0; six downloaded files fingerprinted.
- All 42 expert-labelled Task-2 examples/31 papers; no missing full texts. Held-paper claims and full text excluded from prior training.
- Same windows/512-word limit in all six arms; 252 predictions separately verified for source offsets, quote/budget integrity and aggregate arithmetic.
- Claim-macro ROUGE-L: BM25 0.1805, TF-IDF 0.1857, prior-only 0.2045, BM25-diverse 0.1805, hybrid 0.2102, hybrid-diverse 0.2122.
- Proposed-minus-BM25 paper-macro difference 0.0393, descriptive paired-paper interval [0.0131, 0.0666]; different from claim-macro difference 0.0317.
- Paired-paper intervals against prior-only/nondiversified hybrid include zero. No established incremental diversification advantage.
- Cross-lab transfer to MegaCog: 0.1504 versus BM25 0.1714. No domain-portability claim.
- Post-hoc 256/768-word sensitivities retain higher aggregate ROUGE-L than BM25; original weights and budget retained.
- Optimized LCS agrees with conventional DP on ten bounded prefixes. Six retrieval unit tests pass.
- Official test has 109 claims without gold contexts, and is not scored. Metrics differ from official stemmed snippet-wise ROUGE/BERTScore. No leaderboard/SOTA claim.

## Preserved evidence and boundaries

- Original source PDF and historical material preserved.
- Four audit corpora: 378 keyed records/3,188 triples; earlier recount and fingerprint checks retained.
- Twenty-five constructed controller tests pass; they check policy behavior, not scientific efficacy.
- Bounded property verification passes on 8,192 constructed two-record/protocol inputs, including 180 cross-record union states lacking an individual witness. Unique-key insertion, semantic permutation, repeatability, threshold strengthening and duplicate-key counterexample checks pass. This is not full state-space model checking or human efficacy evaluation.
- Historical ESV candidate yield 39 rows, all explicit limitations; typed branch zero. Silver-control R20 0.0139 versus BM25-LACKS 0.0933; non-independent controls and MongoDB provenance issue disclosed.
- The primary diagnostic analysis now regenerates candidates, controls and rankings in three domains with MongoDB excluded: 325 records/2,545 triples, eight completed/seven scored folds, 61 controls/29 distinct IDs, 38 candidate rows. Regenerated records equal archived counterparts; ESV proxy R20 0.0179 versus BM25 0.0485. Exclusion is post hoc, not provenance repair or expert validation of remaining domains.
- Review preparation remains 188 pairs/39 candidate rows/97 evidence-paper IDs. Human fields remain blank.
- A frozen pilot selects 12 candidate rows/53 pairs/40 evidence-paper IDs without future controls. Both reviewer packets remain blank. Scorer with the pilot manifest was checked to reject these unlabelled inputs; it created no agreement score. The user confirmed no reviewers/labels are currently available.
- No controller closure precision, witness-document recall, new independent annotation, gap-discovery efficacy, public artifact release, external submission or independent reproduction claimed.
- Read SELF_REVIEW_VI.md for the current readiness assessment and remaining evidence requirements.
