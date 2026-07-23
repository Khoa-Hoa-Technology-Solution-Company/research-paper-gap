# Immutable release procedure and current status

## Current status

The forward run is complete, and the local release packet is ready for review,
including `paper.pdf`, `supplementary.pdf`, the frozen input snapshot, forward
provenance, `provider_model_audit.json`, and `reproducibility_manifest.json`.
It is **not an immutable public release**: no reviewed commit/tag has been
recorded in the manuscript, and Zenodo has not minted a DOI. Do not cite a DOI
or release identifier until those external steps are complete.

The current repository state is not an immutable public artifact. In
particular, a content-addressed local manifest is not a GitHub release and does
not mint a DOI.

Before submitting or citing a versioned artifact:

1. Inspect the completed forward run and retain all provider-model-variant
   records without rewriting their provenance.
2. Run the full test suite, regenerate `reproducibility_manifest.json` for the
   final run, and compile
   `paper.pdf` and `supplementary.pdf` from the exact sources to be released.
3. Select and add an explicit software license approved by all rights holders.
4. Commit the reviewed files, record the commit SHA, and create an annotated,
   protected Git tag such as `v0.1.0`. Push both commit and tag to GitHub.
5. Create a GitHub Release from that tag. Attach the PDFs, the manifest, the
   frozen input snapshot, and forward-run provenance as release assets or link
   their immutable archival locations.
6. Enable the GitHub--Zenodo integration, archive that release, and add the
   DOI returned by Zenodo to `CITATION.cff`, the paper availability statement,
   and the release notes. Do not insert a DOI before Zenodo mints it.

The manifest must be regenerated after any source, PDF, or E2E-output change;
the current local content-addressed hash is recorded in the manifest itself and
is not a substitute for a protected tag or DOI.
