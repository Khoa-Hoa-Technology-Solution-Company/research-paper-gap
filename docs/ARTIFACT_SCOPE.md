# Artifact scope for the current KG-TABI manuscript

The current manuscript is a protocol and diagnostic paper. Its paper-backed
claims are limited to implementation behavior, traceability, sensitivity, and
the null decision under the frozen configuration.

## Paper-backed artifacts

- `data/runs/microservices-security-v1/`: historical diagnostic and its
  explicitly documented provenance limitations.
- `data/runs/microservices-security-lexical-v1/`: deterministic lexical screen
  replay over the frozen 314-record snapshot.
- `data/runs/microservices-security-e2e-v1/`: provider-field-recorded forward
  run, complete only to the granularity exposed by the provider.
- `paper.tex`, `supplementary.tex`, and the content-addressed manifests linked
  by those runs.

## Legacy development outputs

Root-level outputs such as `data/gaps/`, `data/evaluation_results.md`,
`data/expert_reviews.json`, root-level baseline outputs, and the 310-item dry
packet were created during earlier development or packet testing. They may use
older prompts, probability-sounding bucket labels, relaxed gates, LLM-judge
metrics, or unverified dashboard entries. They are retained for historical
inspection only and must not be cited as:

- expert validation;
- evidence of novelty, source support, or usefulness;
- a balanced baseline comparison;
- a result produced by the frozen primary configuration; or
- evidence that KG-TABI automatically validates research gaps.

Future confirmatory artifacts must be written under a new immutable run ID,
use the current TABI schema, include source closure and reviewer-level labels,
and be archived under a reviewed tag/DOI before publication.
