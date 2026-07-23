# Response to the major-revision report

## Implemented in this revision

- Repositioned the manuscript as an **auditable protocol** rather than an
  effectiveness evaluation of research-gap hypothesis generation.
- Replaced probability-sounding feasibility values with
  `near_term_feasible` and `long_term_or_speculative` in future TABI output.
- Stated that the overlapping forward run is a reproducibility check, not an
  independent replication or a generalizability result.
- Added an offline, versioned forward sensitivity diagnostic:
  `src.forward_sensitivity_diagnostics`. It reports the frozen global gate,
  absolute and component-relative exploratory size profiles, and an
  paper-normalized temporal sensitivity where forward paper IDs exist.
- Reported the forward diagnostic in the paper and supplement. The frozen
  global gate yields 0 communities; exploratory absolute-10 and
  component-relative profiles yield 6 and 5 respectively. These results were
  not used to generate candidates or choose a replacement gate.
- Added the forward paper-normalized screen. With the unchanged temporal
  thresholds it tests 6 nodes and yields 0 signals; the historical graph cannot
  support this analysis because its event-level paper IDs are incomplete.
- Clarified projection loss, compatibility-pair screening, terminology,
  provider-model variability, and the required direct-LLM/graph-ground
  ablations for a future confirmatory study.

## Protocol-review additions

- Added a scoped comparison with scientific information-extraction/knowledge
  graph, graph-retrieval, link-prediction, and LLM hypothesis-discovery
  workflows. It makes clear that the contribution is an integration and audit
  contract, not a new graph algorithm or a demonstrated effectiveness result.
- Added a concise audit-contract table, a full source-trace-chain document,
  and explicit packet types and reproduction tasks. The trace preserves the
  record-to-review joins while distinguishing auditable fields from information
  a provider does not expose.
- Added a provider-identity audit for the completed forward run. All 185 calls
  succeeded without retry, but the endpoint returned five identifiers; the
  resulting per-identifier counts are descriptive only and cannot support a
  causal model comparison.
- Kept Wilson confidence intervals for all synthetic checks in the supplement,
  added a toy structural interpretation, and compressed the historical
  diagnostic so the protocol specification fits in the main paper.
- Prepared release notes and an immutable-release procedure. They explicitly
  state that a public tag and DOI are pending author-controlled remote release,
  not completed artifacts.

## Deliberately not claimed or fabricated

The following requests require external data collection or a newly locked
multi-domain study and cannot be completed truthfully from the present
artifacts:

- blinded ratings from at least three domain experts;
- source closure and comparative ratings for each candidate;
- extraction and entity-resolution precision/recall/F1 with annotator
  agreement;
- candidate-quality, graph-ground, or baseline-superiority results;
- calibration of a replacement structural threshold on development and
  held-out corpora;
- a Zenodo DOI or immutable release tag.

The repository contains the packet, protocol, candidate key separation,
provenance logging, and release procedure needed to perform that study. Human
ratings, a release tag, and a DOI must be supplied by their responsible
authors/reviewers; synthetic, LLM-generated, or post-hoc labels are not valid
substitutes.
