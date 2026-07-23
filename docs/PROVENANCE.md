# Forward provenance and closure-retrieval manifests

This repository has two distinct reproducibility scopes:

- **Legacy pilot artifacts** are historical diagnostic inputs. Missing model,
  prompt, source-offset, and original-screening metadata are not inferred or
  back-filled.
- **Future reruns** use the forward schema described here. They can record the
  corpus, chunk, LLM, retrieval, and response-hash information needed to audit
  an end-to-end run.

The schema version is `kgtabi-provenance-v1`.

## Forward-run artifacts

For a run isolated with `KG_TABI_RUN_ID=<run-id>`, the pipeline writes these
new artifacts below `data/runs/<run-id>/`:

| Artifact | Purpose |
| --- | --- |
| `raw_papers/stage1_provenance_manifest.json` | Query expansion, Semantic Scholar request parameters and response hashes/IDs, screening method, and hashes of corpus/chunk artifacts. |
| `provenance/llm_calls.jsonl` | One non-secret record per LLM request: provider/model fields, configured revision/API date when supplied, temperature/top-p, prompt hash/version, retry policy, timestamps, and response hash/storage policy. |
| `provenance/llm_responses/*.json` | Optional SDK-normalized provider response payload, only when `LLM_RAW_RESPONSE_POLICY=store`. This directory is not produced by the default policy. |
| `triples/extraction_provenance_manifest.json` | Chunk-to-call index, input/output hashes, checkpoint state, and extraction call/result hashes. |
| `gaps/closure_search_audit.json` | Per-claim candidate-retrieval records, query variants, API requests, retrieved IDs, deduplication, and citation-neighborhood requests. |
| `gaps/closure_search_manifest.json` | Run-level index and configuration for the closure audit. |

Raw prompts are always hash-only. The default response policy is also
hash-only, so the SDK-normalized provider response and assistant message can be
compared across artifacts without automatically storing model output or
source-derived content. Never commit a `store` response directory unless its
content-sharing policy has been reviewed.

## Source and triple fields

Forward chunk records include both legacy-compatible `paperId` and canonical
`paper_id`, along with `chunk_id`, `chunk_index`, section label, source-text
hash, exact chunk text hash, and exclusive source character offsets. A
`chunk_id` includes a content hash, so it changes when a supposedly identical
chunk changes.

Each forward triple retains its legacy `confidence` field for compatibility and
also records `model_reported_extraction_score`. The latter is not a calibrated
probability. It carries the extraction call ID, provider/model fields,
configured revision/date when available, prompt and response hashes,
generation parameters, retry policy, source paper/chunk identifiers, and
evidence offsets.

Evidence offsets are emitted only for a **unique exact** quote match. An
unmatched or repeated quote is explicitly marked and has null offsets. A quote
match establishes text location, not that the quote semantically supports the
triple; that remains a human-audit question.

## Forward temporal accounting

Forward runs retain paper and chunk identifiers, so temporal analysis must
report annual papers, chunks, tokens, raw/accepted triples, unique entities,
and triples per 1,000 tokens. The planned primary activity denominator is
unique supporting papers per entity and year; event- and chunk-normalized
activity are sensitivity variants. This avoids treating extraction density as
though it were independent paper-level evidence.

## Configuration

Set these in `.env` for a future run when the provider exposes the information:

```env
KG_TABI_RUN_ID=rerun-2026-01
LLM_PROVIDER=openai
LLM_MODEL=<locked-model-identifier>
LLM_MODEL_REVISION=<provider-revision-if-known>
LLM_MODEL_RELEASE_DATE=<YYYY-MM-DD-if-known>
LLM_API_VERSION=<endpoint-or-provider-api-version-if-known>
LLM_API_DATE=<provider-api-date-if-known>
LLM_TOP_P=
LLM_MAX_RETRIES=7
LLM_RETRY_BASE_DELAY_SECONDS=5.0
LLM_RETRY_NON_RATE_LIMIT_DELAY_SECONDS=3.0
LLM_RAW_RESPONSE_POLICY=hash-only
SEMANTIC_SCHOLAR_API_BASE_URL=https://api.semanticscholar.org/graph/v1
SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS=30
SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS=3.0
```

Nullable model revision/date fields are intentional. Do not guess them from a
marketing model name. The call record also retains provider-returned response
model and system fingerprint when the compatible API provides them.

## Reproducible closure retrieval

Closure search starts with three deterministic lexical variants in a fixed
order:

1. the normalized verbatim claim;
2. ordered content tokens (up to eight, using a versioned stop-word rule);
3. a first/last content-token anchor-pair heuristic.

The final two are lexical coverage heuristics, not semantic claims about true
concept pairs or method/application structure. Optional LLM-generated variants
are supplemental and are separately marked with prompt version, call
provenance, response hash, and parsed output. Query deduplication uses Unicode
case-folding and normalized whitespace; the audit states exactly which variant
survived.

Each Semantic Scholar request retains endpoint, parameters, timestamps, status,
response hash, and returned paper IDs. Candidate records deduplicate by
Semantic Scholar `paperId`, preserving first-seen query/API-rank order and all
query IDs that retrieved a candidate. Citation expansion is bounded and records
the IDs returned for both reference and citation directions.

To run only deterministic variants (therefore no LLM query-generation call):

```powershell
python -m src.closure_search --deterministic-only
```

Retrieval ranks and API contents can still change over time. A closure manifest
therefore documents an observed candidate set; it does not certify novelty,
source support, or the completeness of a literature search.

## Local validation

The forward-schema tests do not call an LLM or Semantic Scholar:

```powershell
py -3.14 -m unittest tests/test_provenance.py -v
```
