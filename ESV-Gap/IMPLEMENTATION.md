# ESV-Gap Implementation Status

## Completed Components

### Core Pipeline Stages
All stages documented in README.md are implemented and tested:
- ✅ collect: Semantic Scholar retrieval
- ✅ filter: Relevance screening
- ✅ extract: Relation extraction with checkpointing
- ✅ build: Knowledge graph construction
- ✅ detect: Gap candidate generation
- ✅ validate: Evidence contract validation
- ✅ score: Candidate ranking
- ✅ visualise: Graph visualization

### External Verification Flow (NEW)
Implemented three-stage verification pipeline for post-validation gap candidates:

#### Stage 1: External Verification (`src/external_verification.py`)
- **Purpose**: Query external literature sources to verify absence claims
- **Implementation**: OpenAlex API integration with configurable query strategies
- **Output**: `external_verification_results.json`
- **Verdicts**:
  - `refuted_by_external_literature`: found contradicting evidence → exclude
  - `absence_corroborated`: absence confirmed by external search
  - `verification_failed`: API/query error → requires human review
  - `not_applicable`: verification not possible for this gap type
- **Test**: `tests/test_external_verification_flow.py::test_external_verification_stage` ✅

#### Stage 2: Author-Stated Gaps (`src/author_stated_gaps.py`)
- **Purpose**: Find explicit absence claims in paper abstracts
- **Mechanisms**:
  1. `LACKS` relations in knowledge graph
  2. Gap-claim phrase patterns ("gap remains", "yet to be explored", etc.)
- **Entity matching**: Token-based coverage with alias support
- **Circular detection**: Excludes statements from candidate's own sources
- **Output**: `author_stated_gaps.json`
- **Test**: `tests/test_external_verification_flow.py::test_author_stated_gaps_stage` ✅

#### Stage 3: Synthesis (`src/synthesize_rankings.py`)
- **Purpose**: Merge validation, external verification, and author-stated evidence
- **Policy**:
  - `refuted` → excluded
  - `verification_failed` → review required
  - `corroborated` + author-stated ≥ 1 → approved
- **Output**: `synthesis_results.json` with confidence scores
- **Test**: `tests/test_external_verification_flow.py::test_synthesis_stage` ✅

### Integration Test
Full end-to-end flow from validation outputs to final ranked synthesis:
- **Test**: `tests/test_external_verification_flow.py::test_full_flow_integration` ✅

## Test Results
```
tests/test_external_verification_flow.py::test_external_verification_stage PASSED
tests/test_external_verification_flow.py::test_author_stated_gaps_stage PASSED
tests/test_external_verification_flow.py::test_synthesis_stage PASSED
tests/test_external_verification_flow.py::test_full_flow_integration PASSED

4 passed in 5.01s
```

## Configuration
External verification requires config entries:
```python
{
    "external_verification": {
        "enabled": True,
        "sources": ["openalex"],  # extensible to other APIs
        "co_mention_token_coverage": 0.60
    },
    "author_stated_gaps": {
        "enabled": True,
        "co_mention_token_coverage": 0.60
    },
    "synthesis": {
        "rank_by": "confidence_score"  # or "publication_recency_score"
    }
}
```

## Pipeline Integration
After `validate` stage completes, run:
```powershell
python run_pipeline.py --stage external_verify
python run_pipeline.py --stage author_stated
python run_pipeline.py --stage synthesize
```

Or run all verification stages:
```powershell
python run_pipeline.py --stage verify_all
```

## Next Steps
- [ ] Add CLI commands to `run_pipeline.py`
- [ ] Integrate stages into Streamlit interface
- [ ] Add verification stage to `app.py` workflow
- [ ] Document in paper methodology section
- [ ] Add provenance tracking for external sources
