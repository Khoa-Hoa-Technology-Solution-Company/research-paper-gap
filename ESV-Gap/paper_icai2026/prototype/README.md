# Scoped residual-evidence contract: research prototype

This small, standard-library-only implementation is separate from the production ESV-Gap pipeline. It evaluates **supplied, checked annotations**, not raw scientific text. It cannot decide whether a research idea is novel.

All task conditions and numerical requirements must be met by the same experiment in the same source. Multiple experiments or papers cannot be combined to manufacture joint closure. Missing values, incompatible units and observed failures are separate states. Future-available records are excluded. Search completion means only completion of a declared retrieval protocol, not coverage of the literature.

The `verified_span` flag is a trusted annotation input: substring checking alone does not prove that numerical values, tasks or conditions follow from that span. A real deployment needs independent annotation/verifier validation, bibliographic deduplication, full-text provenance and explicit unit handling. Exact task/condition matching here is deliberately narrow and is not a semantic retrieval method.

Run from the repository root:

```powershell
python -m unittest discover -s ESV-Gap/paper_icai2026/prototype -p 'test_*.py' -v
```

On 29 September 2026, all 10 synthetic policy tests passed. These fixtures are invented solely for software testing, have synthetic paper identifiers and provide **no evidence of discovery precision, recall or real research-gap utility**. The prototype has not been integrated into the historical corpus experiments.
