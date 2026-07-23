# KG-TABI submission-readiness gate

This checklist distinguishes manuscript correctness from venue administration
and from evidence that the current study does not yet provide.

## Must pass before uploading any protocol/tool submission

- [ ] Freeze the final `paper.tex`, `supplementary.tex`, bibliography, code, and
  paper-backed run artifacts.
- [ ] Build both PDFs with no LaTeX errors, undefined citations/references, or
  overfull boxes.
- [ ] Run all tests in `tests/` with the Python version recorded for the release.
- [ ] Regenerate both paper-backed reproducibility manifests after the final PDF
  build and verify every listed SHA-256.
- [ ] Remove bytecode, local environments, secrets, logs, and unrelated legacy
  outputs from the submission archive.
- [ ] Obtain rights-holder approval for a software license and add the license
  identifier to `CITATION.cff`.
- [ ] Commit the reviewed snapshot, create an immutable tag/release, and archive
  it with a DOI if the venue permits a public artifact before review.
- [ ] Apply the target venue's page limit, formatting, anonymity, supplementary
  material, ethics, AI-use, and artifact policies.

## Claims allowed by the current evidence

- The frozen implementation returns the recorded null decisions.
- The trace chain and provider-exposed execution metadata are inspectable.
- Partition, traceability, masking, and synthetic diagnostics characterize
  limited implementation behavior and configuration dependence.

## Claims not allowed without a new confirmatory study

- Candidate novelty, usefulness, source support, or scientific validity.
- Superiority over a direct-LLM or other baseline.
- Generalization beyond the overlapping abstract-only microservices-security
  corpus.
- Calibrated screening, extraction, entity-resolution, or gate accuracy.

For any research-track submission making candidate-quality claims, complete the
pre-specified multi-domain source-closure and blinded expert study described in
the paper and `docs/BLINDED_CANDIDATE_EVALUATION.md`.
