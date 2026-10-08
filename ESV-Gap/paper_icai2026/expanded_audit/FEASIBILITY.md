# Feasibility of a larger ESV-Gap study (2026-09-29)

This is a read-only inventory. Historical run directories and API resources were not modified. Credential checks inspected presence only and did not print key material.

## Immediate finding

A fresh end-to-end online run of 300 or more retained papers cannot be completed in this execution environment as configured. The pipeline needs a Groq-compatible LLM credential for query/screen/extract stages. None is present in process variables, `ESV-Gap/.env`, or `ESV-Gap/config.yaml`. The workspace-root `.env` does contain a Semantic Scholar key under `SEMANTIC_SCHOLAR_API_KEY` and a generic `LLM_API_KEY` for a different OpenAI-compatible provider; neither credential was used or tested. The pipeline's `groq_key_pool.py` does not read `LLM_API_KEY` or its provider base URL/model. Collection obtains its key from `config.yaml`'s `api_keys.semantic_scholar`, whereas the external verification module can read the environment variable. A new provider adapter, quota/cost check, and isolated pilot would be needed before treating that generic credential as an alternative. Key validity and quota were not tested.

Current `config.yaml` targets **Explainable AI (XAI)** and 150 retained papers. Its `collection.max_papers=1000` is a *raw retrieval cap*, not a retained target. `filtering.target_corpus_size` must be raised in a new, isolated config and collection queries/domain reset to the intended IoT scope. All `paths` default to shared `data/` and `outputs/` locations; any new run must point those paths to a new named run directory to preserve historical evidence. Use `python run_pipeline.py --stage <stage> --config <new-isolated-config>` from `ESV-Gap` only after the config and credentials are ready.

`run_pipeline.py --stage all` executes collect, filter, extract, build, detect, validate, verify, author-gaps, extractor-recall, saturation, synthesize, score, visualise. It omits the `fulltext` stage, although `gap_certification` requires source text level `full_text` and at least 1000 characters, and it requires three completed external queries. A proper expanded study therefore needs a staged workflow with candidate generation, `fulltext`, re-detection if enriched, validation, and verification, plus audit of full-text yield. In the frozen 150-paper IoT run, full-text enrichment attempted 14 downloads against a target of 20 and enriched zero, a likely certification blocker that corpus scaling alone will not solve.

## Reusable local data

| Preserved corpus | Rows | Distinct IDs | Distinct DOI | Triple file? |
|---|---:|---:|---:|---|
| IoT IDS, 2026-08-30 | 150 | 150 | 149 | yes |
| IoT IDS, 2026-08-31 | 150 | 150 | 149 | yes |
| Security of microservices | 143 | 143 | 141 | yes |
| Security of MongoDB | 53 | 53 | 53 | yes |

The two IoT runs together contain 300 rows but only **223 distinct paper IDs / 221 distinct DOIs**. Their extracted triples may differ for overlapping papers, so a merged corpus needs a one-paper-one-extraction rule and provenance log. Across all 20 run directories with both `corpus_filtered.jsonl` and `all_triples.json`, there are 749 rows, 582 distinct paper IDs and 577 distinct DOIs. Those runs span IoT IDS, microservice security, MongoDB, monolith/SQL security, and handwritten expression recognition. This union can support an offline *heterogeneity feasibility study* but is not a new 582-paper domain-specific IoT study. No single preserved complete corpus exceeds 150 papers.

An offline cross-run benchmark could reuse cached extractions after title/DOI/ID deduplication, but it would need to harmonize relation schemas, extraction model versions, source dates, and repeated document conflicts. It cannot establish new positive discovery without independent labels, and mixing domains must be reported explicitly. No such benchmark was run during this inventory.

## Runtime and scale

The default Python has `openai`, `dotenv`, `requests`, `networkx`, `yaml`, `pyvis`, and `streamlit`. It lacks `fitz`/PyMuPDF, `spacy`, `pykeen`, `torch`, and `sentence_transformers`. Some pipeline stages have lexical/NumPy fallbacks, but full-text PDF extraction needs PyMuPDF; installing it or finding the bundled runtime must precede a full certificate attempt. The temporal offline replay was already executable with installed NetworkX/NumPy/PyYAML.

For 300 retained papers, screening uses a 1.25 candidate multiplier, so the intended balancing pool is about 375 relevant papers. Each screened paper that passes the lexical guard calls the LLM, and every retained paper receives at least one extraction call (possibly multiple chunks). A planning lower bound is therefore hundreds of calls, roughly **375 screening + 300 extraction calls**, before query generation, retries, external semantic verification, or full text. The 3-second programmed delay after LLM screening alone would be roughly 19 minutes at 375 calls; actual wall time is higher and depends on API response/rate limits. No currency cost estimate is reliable without current provider pricing, token accounting, available quota, and measured chunk counts; none was checked online here.

## Recommended next executable experiment

Prepare a new immutable IoT corpus protocol with retrieval timestamp, year window, query strata, DOI/title deduplication, screening ledger, and reserved cutoff folds. Target at least 300 distinct retained papers *after* deduplication, not a raw API budget. First run a bounded pilot of 30–50 papers in a new path to verify credentials, rate limits, full-text access, and token/cost logs. Then scale to the intended corpus and compare direct evidence, typed graph, and topology components at a fixed candidate/reviewer budget. Human-adjudicated temporal controls remain required before making positive discovery claims. Historical runs remain untouched.
