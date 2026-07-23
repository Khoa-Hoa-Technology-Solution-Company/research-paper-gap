# Blinded candidate evaluation: operational procedure

For a Vietnamese step-by-step PowerShell runbook, see
[`BLINDED_EXPERT_REVIEW_RUNBOOK_VI.md`](BLINDED_EXPERT_REVIEW_RUNBOOK_VI.md).

Human labels cannot be substituted or generated. This procedure evaluates only
candidates actually produced by a locked forward run and pre-specified
comparators. This protocol is not preregistered yet; archive a dated, immutable
protocol before recruiting reviewers or inspecting comparative outcomes.

## Before recruitment

1. Lock at least two independent domain corpora, prompts, model identifiers,
   graph settings, candidate rule, closure procedure, comparators, and analysis
   plan. Do not select a domain or relax a gate because it happens to yield
   candidates.
2. Run every system and retain one JSON output per system.
3. Build a reviewer packet and keep the mapping private:

```powershell
py -3.14 -m src.prepare_candidate_blind_review `
  --input kgtabi=data/runs/<run>/gaps/kgtabi_gaps.json `
  --input direct=data/runs/<run>/gaps/baseline_direct.json `
  --packet data/runs/<run>/audits/candidate_blind_packet.csv `
  --key data/runs/<run>/audits/candidate_unblinding_key.csv
```

Do not send the key, source-system names, topology scores, prompt wording, or
generation order to reviewers. Do not alter candidate text after packet
creation other than removing system identifiers.

## Reviewers and ratings

Recruit at least three independent domain experts. Each reviewer labels every
row without communicating with other reviewers:

- source support, claim clarity, novelty after closure search, importance,
  actionability, feasibility: integer 1--5;
- already addressed: yes / no / uncertain;
- unsupported or hallucinated evidence: yes / no / uncertain;
- a short rationale for every 1, 5, or uncertain label.

Run closure search before novelty ratings and provide the same retrieved-source
bundle for every candidate, without system labels. Preserve completed reviewer
files unchanged. Calculate agreement before adjudication, then adjudicate only
after all independent ratings are locked. Add pre-specified generation ablations:
full graph Grounds, concept-only, abstract-bundle-only, and shuffled Grounds.
The direct-LLM baseline and every ablation must use the same corpus, source
bundle, output schema, length limit, and balanced candidate budget.

## Analysis and reporting

Report per system: median and IQR for ordinal outcomes, acceptance rate,
bootstrap confidence intervals, effect sizes against pre-specified baselines,
and reviewer agreement (Krippendorff's alpha or an appropriate alternative).
Report candidate counts, exclusions, missing ratings, and the unblinding key
after the analysis is frozen. Report each domain separately before any pooled
analysis. A small or zero candidate set is a result; do not relax topology
thresholds after observing it.

## What is not a replacement for experts

Evidence-quote offsets, source retrieval rank, NLI scores, LLM-as-a-judge
ratings, masking benchmarks, and synthetic temporal injections are useful
engineering diagnostics. None is a blind human assessment of novelty,
importance, feasibility, or hallucination, and none may be presented as one.
