# Audit packet and source-trace chain

KG-TABI separates protocol compliance from scientific validity. A complete
trace lets an auditor determine how an output was produced; it does not prove
that a triple is true or that a potential-gap hypothesis is novel or useful.

## Forward source-trace chain

```text
retrieved record
  -> inclusion decision
  -> stable abstract chunk
  -> LLM extraction call
  -> typed triple and evidence span
  -> complete-link entity-resolution decision
  -> graph event and topology/temporal signal
  -> TABI candidate (only if the frozen gate passes)
  -> closure queries and retrieved-source bundle
  -> blinded review packet and separate unblinding key
```

Each forward link carries the following minimum fields.

| Link | Keys retained | What an auditor can check | What it cannot prove |
|---|---|---|---|
| Record → decision | source paper ID, title/year, decision, reason, corpus hash | which records entered the run | screening accuracy without human labels |
| Decision → chunk | stable `chunk_id`, section, chunk/source hashes, offsets when supplied | exact input text identity and position | full-text provenance when only an abstract is available |
| Chunk → call | call ID, prompt version/hash, configured/provider model metadata, temperature/top-p, retries | request configuration and provider-exposed identity | provider weights, routing determinism, or hidden system prompt |
| Call → triple | response/content hashes, typed fields, quote, quote-match status and offsets | that a reported quote occurs at the recorded location | semantic support until a reviewer labels it |
| Triple → entity | proposal path, type constraint, component descriptor, complete-link acceptance/rejection | why labels were or were not merged | merge precision/recall without a labelled pair set |
| Entity → signal | event IDs, graph/configuration hash, partition/statistic or temporal report | why a signal or null was emitted under the frozen configuration | a scientifically meaningful gap |
| Candidate → closure/review | deterministic query variants, returned paper IDs/ranks, blind ID, separate key | provenance of closure retrieval and blinding | novelty/importance/feasibility without expert review |

## Packets

`src.audit_artifacts` generates separate packets and keys; reviewers never
receive system identity or the unblinding key.

- **Screening packet**: title, abstract, year, blinded item ID; reviewers enter
  include/exclude/uncertain, exclusion reason, and comments.
- **Triple packet**: typed triple, source metadata, evidence quote, source
  chunk, provenance-recovery status; reviewers label subject/object/relation/
  type correctness, quote support, metadata, and specificity.
- **Entity packet**: sampled merges and high-similarity non-merges; reviewers
  label `same_entity`, `related_but_distinct`, `unrelated`, or `uncertain`.
- **Candidate packet**: Claim, Grounds, Warrant, feasibility label, closure
  bundle, and blinded candidate ID; reviewers rate support, clarity, novelty,
  importance, actionability, feasibility, already-addressed status, and
  unsupported-evidence risk.

Preserve completed reviewer files unchanged. Compute agreement before
adjudication, release the key only after analysis is frozen, and never replace
human labels with NLI, retrieval rank, model scores, or synthetic ratings.

## Reproduction exercise

An independent auditor can execute the following tasks without seeing any
candidate-system label:

1. verify the frozen input and output hashes in `reproducibility_manifest.json`;
2. follow one resolved triple to its chunk, extraction call, quote location, and
   entity-resolution record;
3. reproduce the declared null decision from the locked configuration;
4. compare a forward run with the provider-model audit and identify routing
   variability; and
5. confirm that a blind packet cannot be mapped to a system without its key.

Record completion status, elapsed time, errors, and auditor confidence if this
is used as a formal external audit or usability study. No such external audit
has yet been collected for the current artifact.
