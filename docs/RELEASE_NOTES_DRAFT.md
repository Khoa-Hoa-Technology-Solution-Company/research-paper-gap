# Draft release notes: KG-TABI v0.1.0

## Scope

This release packages KG-TABI as an auditable protocol and diagnostic artifact.
It does not claim candidate quality, novelty, source support, usefulness, or
superiority to any baseline.

## Included artifacts

- main paper and supplementary PDF;
- frozen 185-record lexical input snapshot;
- 185-call forward extraction provenance and raw/resolved triples;
- graph, topology configuration, temporal report, and zero-candidate TABI
  output;
- forward gate/paper-normalized sensitivity and provider-model audit;
- reproducibility manifest, audit packets, blind-review protocol, and source
  trace-chain documentation.

## Known limitations

- The forward rerun overlaps the legacy corpus and is not an independent
  replication.
- Provider-returned model identifiers vary despite one configured model string.
- Historical source provenance is incomplete.
- No human labels or candidate-level closure/blinded comparison are included.

## Publication checklist

1. Review and commit the exact files represented by the final manifest.
2. Create and protect annotated Git tag `v0.1.0`.
3. Create a GitHub Release from that tag and attach the listed artifacts.
4. Archive the release through Zenodo and then add its minted DOI to
   `CITATION.cff`, the paper, and the release notes.
