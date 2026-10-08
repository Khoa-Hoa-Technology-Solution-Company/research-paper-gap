# Context24 grounding-context experiment

Protocol recorded before running the comparative experiment in this session.
This is an exploratory extension, not a preregistered study.

## Data and scope

- Author dataset: `joelchan/contextualizing-scientific-claims`, revision `457d3b5cb4bb8ade34e37458f4900c6eae0959bb`, CC BY 4.0.
- Task 2 supplies 42 example claims with expert grounding-context quotations. Its 109 official test claims in the downloaded file omit gold contexts; do not report official test accuracy or invent those labels.
- Evaluate the 42 labelled examples using leave-one-paper-out prediction. Every claim from the held-out `citekey` is excluded from training the context prior. The official test set is not used for tuning or scoring.
- Exclude no case based on score. Report missing full-text parses explicitly and use empty predictions for unavailable inputs.
- Task is retrieval of methodological context within the supplied source paper, not scientific-gap novelty or complete closure.

## Frozen retrieval methods

- Candidate windows: up to 96 whitespace-delimited words, stride 80; preserve original character offsets. Strip no document sections with gold-aware rules.
- Output budget: at most 512 whitespace words of quoted source text for every method; no generated paraphrases.
- Baselines: BM25, TF-IDF cosine; optionally a context-prior-only ablation.
- Proposed context retrieval: 0.5 normalized BM25 claim relevance plus 0.5 normalized context-prior score; then subtract 0.2 maximum token-Jaccard similarity to previously selected windows. Context prior is unigram log-ratio learned from expert method quotes vs source full-texts of other papers. Positive log-ratios only; minimum positive-quote frequency 2.
- Ablations: remove context prior; remove diversification; prior only. Every arm has the same windows and output budget.
- All weights are fixed here; no test-driven parameter search. This is a lightweight retrieval method, not a new language model.

## Scoring and uncertainty

- Concatenate predicted quotes and gold quotations in their stored order. Compute non-stemmed word ROUGE-1, ROUGE-2 and ROUGE-L F1, plus unigram recall/precision, with deterministic normalization.
- These are transparent lexical grounding measures, not the complete Context24 leaderboard evaluation, which additionally uses BERTScore and its own preprocessing.
- Report per-claim scores, macro means, domain-level means, exact source-substring checks and word-budget checks.
- For the proposed-minus-BM25 ROUGE-L difference, average claim differences within paper, then use 2,000 paired paper bootstrap replicates (seed 20261001). Report the percentile interval as descriptive uncertainty; source papers and benchmark examples are not a representative population.
- Report all baseline/ablation results even if the proposed method does not improve them. Do not relabel failed retrieval as successful verification.
