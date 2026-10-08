# Verification status — 30 September 2026

## Supplementary PDF prepared — 1 October 2026

- Created `output/pdf/ESV-Gap_ICAI2026_Supplementary.pdf` for the supplementary-material slot shown for Paper ID 63. Five A4 pages, 126421 bytes; SHA-256 `6584e1f7d145e0e5608fcf0f4639621e22784cefdec7adbfbbcfa56ba3c0d0d3`.
- Contents: frozen replay scope/configuration synopsis; per-fold record/control/candidate accounting; macro metrics; query sensitivity; AI-assisted case-audit and MongoDB provenance limits; full input/config/code/original-report SHA-256 values; recorded runtime; self-contained aggregate-arithmetic listing and conditional offline replay instructions.
- The local aggregate verifier passed before generation. The included arithmetic listing executed and returned ESV R20 0.0139; BM25 R5/R10/R20 0.0556/0.0694/0.0933; Q2/Q3 R20 0.0853/0.0853. No new experiment or annotation claim was introduced.
- Rendered and visually inspected all five final pages. Fixed a narrow BM25 column in Table S1, regenerated, and inspected again. No clipping, numeric line breaks, overlaps or missing accented author glyphs observed. Fonts are embedded; the PDF has no source-data attachments.
- The supplement explicitly omits corpus/triples and does not provide the local code/report bundle or a public URL. Arithmetic verification is not full-pipeline reproduction. The companion manuscript PDF was not edited; its full hash is printed in S4.
- No CMT upload/save or external publication performed. The supplied screenshot's visible PDF format and 20 MB limit are satisfied by the local file.

## Submission polish after final readiness review

- Reordered all 19 bibliography entries by first citation; an automated source check passed. Final reference-column trigger is item 6.
- Disclosed unverified MongoDB source identity/publication year in the abstract, Table IV, case inspection, threats paragraph and bibliography. The retained record is for frozen replay accounting only, not independently established future evidence; corrected metadata could change controls and scores. No source verification or corrected-metadata replay is claimed.
- All empirical counts and proxy values remain unchanged. The aggregate checker passed; no manual human annotation or public-artifact claim was added.
- Native compiler remains unavailable. Existing MiKTeX completed two passes after content edits and two after final reference-column balancing; final log has no undefined citations/references or overfull boxes.
- All five revised pages visually inspected; final rebalanced page 5 inspected again. Five A4 pages, no clipping, overlapping cells or broken glyphs observed. Working and delivered PDFs have matching SHA-256 `0A0609A9A196B26542FA395E4B10543204EA1B7CADD25CCEDA83C73730EA929D`.
- This section supersedes earlier wording that the MongoDB provenance limitation appeared only in internal notes. Original reports, inputs and the local review bundle remain unchanged. No conference submission performed.

## Latest Draft (5) review revision

- Applied explicit RQ1/RQ2/RQ3 labels in the Introduction and replaced `concrete failure taxonomy` with `concrete failure-mode analysis` in the Conclusion. No metric, input, baseline, annotation claim, or experiment was changed.
- Retained AI-assisted abstract-level audit wording: author manual inspection of all five abstracts has not been confirmed. Public artifact availability is not claimed in the manuscript.
- Native compilation retried and failed with the existing standard-directory configuration error. Existing MiKTeX completed two successful passes; the final log has no undefined citations/references or overfull boxes. Underfull spacing messages remain.
- Re-rendered and visually inspected all five A4 pages after these edits; no clipped text, broken glyphs, overlapping cells or equations observed. Kept the existing reference-column trigger and template dimensions. Stable output PDF and working PDF have matching SHA-256 `4F2F123261FDF176B023F6FC396ADD575A1D9E4468930789DC7A37B5D3135004`.
- Prepared unpublished local artifact directory `release_candidate/` and ZIP `output/artifacts/ESV-Gap_ICAI2026_Review_Bundle.zip` (19 payload files). ZIP SHA-256: `29f4b397067eed4996c8ce18acbb721efae70eb33f0b2b8608b8ee8c1082932b`.
- Original runner, detector, temporal code, config and three analysis-script hashes match the saved reports. Config credential fields were checked empty before copying. Dependency lock in the bundle adds `PyYAML==6.0.3`, documented as a change; the original lock is unchanged.
- Aggregate verifier passed both in the prepared directory and in a fresh extraction of the ZIP. Existing four temporal unit tests passed in the prepared bundle using local CPython/NetworkX/NumPy/python-louvain/PyYAML. This verifies arithmetic and implementation tests only, not independent full-pipeline reproduction or scientific closure.
- The bundle excludes corpus abstracts, raw triples, graph files, raw fold replay, candidate/control strings and identifiers, internal inspection notes, and credentials. BM25 and sensitivity matched-ID fields are replaced by distinct-ID counts; projection rules and original/payload hashes are in its manifest.
- README specifies omitted inputs, replay order and limits. No license was selected, no repository or archival URL was created, and nothing was uploaded. Full replay still requires the retained inputs; public distribution requires the author's approval and rights/license review.
- Reference status remains as documented below: four publisher records confirmed, MongoDB DOI/source provenance unresolved. No new certainty claim was added.

## Earlier checks retained for provenance

- PDF mới: 5 A4 pages using IEEEtran conference, unchanged template margins/font sizes; all five pages rendered and inspected after the final review polish. The six-category paragraph in Section VI was shortened from approximately 131 to 108 words (18%).
- Existing MiKTeX compiler: successful build, optional missing balance package removed; no package installation.
- Final MiKTeX compile log (30 September): no undefined citations/references and no overfull boxes. Some underfull spacing messages remain. Native editor compile was retried but still unavailable with `Unable to find standard directories for platform`.
- Visual review: all five pages re-rendered and inspected after the final edits. No clipped text, overlapping table cells or broken equations observed.
- Page 3: Table III and IV occupy the top of the right column as standard IEEE floats. The Results paragraph continues below them; it has no clipped or overlapping text. `brokenpenalty=10000` prevents a hyphenated word from being divided across columns or pages; within-column hyphenation remains enabled.
- Author block: sole author Lê Anh Hòa, FPT University, Vietnam, `leanhhoa3002004@gmail.com`, as supplied by the user. The user should confirm whether a department/campus belongs in the CMT fields.
- Historical artifact recount: run successfully and retained `audit/reproduced_counts.json`.
- Four-domain rerun: current code executed offline on 378 records distinct under the runner's DOI/ID/title key across four preserved corpora. Ten domain-cutoff evaluations completed; nine had silver positives. Matched direct-limitation baselines use the same extracted triples, controls and top-20 budget. Full fingerprints are in `expanded_run/multidomain_summary.json`.
- BM25 post-hoc limitation reranking: completed; macro proxy Recall@5/10/20 = 0.0556/0.0694/0.0933 and 5/64 non-independent control rows matched at top 20. Confidence ranking = 0/0/0; ESV-Gap = 0/0/0.0139. See `expanded_run/bm25_lacks_baseline.json`. This is not an extraction-independent text baseline.
- Five-case AI-assisted abstract inspection: four abstracts do not report a same-scope resolution experiment, and one empirical record pairs with an underspecified candidate; zero independently confirmed same-scope resolutions, **not** five verified false positives. See `expanded_run/BM25_MATCH_INSPECTION.md`.
- Traceability: the five later records are cited in Table IV and the bibliography. Publisher/DOI metadata is confirmed for the Jayawardena, Rahaman, Olaya, and Kampa records. The MongoDB entry has preserved-corpus and secondary-index metadata, but its SSRN/DOI landing page was inaccessible during this check; its abstract has substantial wording overlap with a same-title 2017 course paper credited to different authors. This anomaly remains in the internal inspection notes, not the conference manuscript: it is outside the paper's systematic research question and has not been adjudicated. The frozen replay counts are unchanged, and no MongoDB match is claimed as an independently verified same-scope closure witness.
- Final reference proofreading: title/year/pages/DOI fields for SciFact, SciFact-Open, S2ORC, SPECTER, SciRepEval and ResearchAgent were cross-checked with ACL Anthology; Context24 title, author metadata, year and pages match `https://aclanthology.org/2024.sdp-1.3/` (no DOI listed). No corrections were necessary. This does not resolve the separately documented MongoDB provenance limitation.
- Human inspection role: the user has not confirmed personally reading and comparing all five abstracts. The manuscript therefore retains its AI-assisted description and does not claim author manual inspection.
- Post-hoc query sensitivity: Q1/Q2/Q3 macro Recall@20 = 0.0933/0.0853/0.0853 and matched control rows 5/4/4; Q1 exactly matches the original BM25 report across all ten folds. The check is not preregistered. See `expanded_run/bm25_query_sensitivity.json`.
- Saved-candidate source decomposition: `expanded_run/analyze_candidate_sources.py` reran scoring on the stored lists; all 39 candidate rows came from explicit limitations, zero from typed evidence-map cells. Report and input hashes are in `expanded_run/candidate_source_ablation.json`. This is not evidence of graph-topology benefit.
- Temporal replay: code/config/environment fingerprint in `audit/replay/audit_summary.json`; current and historical control sets differ. This is NOT a measured method improvement.
- Existing temporal unit tests: 4 passed (audit agent).
- New scoped-contract synthetic unit tests: 10 passed (primary agent); not a scientific-discovery evaluation.
- Native editor compile: unavailable, reports `Unable to find standard directories for platform`. Source remains saved/openable; the latest five-page PDF was built by the existing local MiKTeX compiler instead and copied to `output/pdf/`.
- Research-gap typology: Müller-Bloch and Kranz (2015) added to Related Work with the correct category `methodological conflict` and acknowledgement that verification is part of their original framework. Section VI proposes type-dependent evidence, separate type/status labels, concept matrices and logged searches. These six types have NOT been annotated or evaluated on the retained corpora. Operational design is saved in `TAXONOMY_VALIDATION_PLAN_VI.md`.
- Independent expert evaluation, newly collected multi-domain efficacy, full-pipeline certification, and prototype discovery gains: NOT performed. The completed four-domain experiment is a retrospective proxy stress test.
- Author/editorial responsibility and conference submission: NOT completed; no external submission or artifact publication occurred.

The final five-page layout uses an IEEE reference-column trigger at item 8. Temporary render images were removed after visual QA.
