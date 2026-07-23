# Blinded annotation protocol for KG-TABI audit packets

This protocol is for the unlabelled packets generated under
`data/runs/<run-id>/audits/`. It must be completed before reporting corpus
screening accuracy, triple quality, or entity-resolution quality. Do not infer
or fill labels from model scores, lexical overlap, or the historical pipeline.

## General procedure

1. Give each blinded packet to two independent reviewers without its key.
2. Keep the reviewer files separate and preserve all original rows.
3. Calculate agreement before adjudication: Cohen's kappa for two applicable
   categorical raters; use Krippendorff's alpha if the study design requires
   more raters or missing-label handling.
4. Give disagreements and `uncertain` labels to an adjudicator. Preserve the
   reviewer-level labels, adjudicated label, protocol version, and reviewer
   expertise description.
5. Do not open the corresponding key until reviewer labels are locked.

## Screening packet

Use `screening_audit_packet_blinded.csv` for all retrieved records.

- Labels: `include`, `exclude`, or `uncertain`.
- Record an exclusion reason and comments for every non-include decision.
- The adjudicated inclusion list becomes the corpus lock for a future rerun.
- Report counts by final decision, reviewer agreement, adjudication rate, and
  agreement between a deterministic screen and adjudicated labels if a lexical
  baseline is evaluated.

The historical 149-record selection is not a gold standard and must not be
used as a reference label.

## Triple packet

Use `triple_extraction_audit_packet.csv` (the current packet is a
coverage-first sample over relation, model-reported-score bin, and year).

For each item, label:

- subject correct: yes / no / partial;
- object correct: yes / no / partial;
- relation correct: yes / no;
- entity types correct: yes / no;
- evidence quote supports the triple: yes / no / unclear;
- source metadata correct: yes / no / unclear;
- triple specific enough: yes / no / unclear.

Define a *fully correct triple* before analysis as an item with correct subject,
relation, object, and evidence support. Report full-triple precision,
relation/type accuracy, evidence-support rate, results by relation and score
stratum, agreement, and adjudicated counts. A quote appearing in a chunk is a
location check only; it is not evidence support until reviewers label it.

## Entity-resolution packet

Use `entity_resolution_audit_packet_blinded.csv`, which contains both pipeline
merges and high-lexical-similarity non-merges. In addition to the coverage
sample, deliberately oversample large components, low-cohesion components,
type-conflict labels, lexical-only decisions, and semantic-only decisions.

For each pair, choose exactly one primary judgment:

- `same_entity` — the pair should resolve to one entity;
- `related_but_distinct` — semantically related but should remain distinct;
- `unrelated` — should remain distinct;
- `uncertain` — require adjudication.

After joining the key, report merge precision for sampled pipeline merges and
the observed missed-merge control rate for sampled non-merges. Stratify merge
precision by lexical-only versus semantic-only path, component size (2 versus
3+), low-cohesion status, and type-conflict status. The non-merge sample is
intentionally high-similarity and is not a random sample of all non-merges, so
do not call its rate a population recall estimate without an additional
sampling design. Compare strict, compatible-type, and type-agnostic graph
variants only after this development audit; choose a primary variant by merge
accuracy, never by whether it creates candidates.

## Future TABI comparison

Before generating candidates, write and archive a dated protocol containing the
corpus lock, graph configuration, model/prompt versions, seeds,
candidate-selection rule, closure procedure, baseline prompts, primary
outcomes, and analysis plan. This is a pre-specified protocol, not a completed
preregistration, unless it is deposited in an immutable registry before data
collection. Use the frozen gates;
if fewer than 30 candidates exist, report an underpowered study rather than
relaxing a threshold. Blind at least three domain reviewers to system origin.
Collect source support, clarity, novelty after closure search, importance,
actionability, feasibility, already-addressed status, and hallucinated-evidence
labels. Release reviewer-level and adjudicated data with the selection key.
