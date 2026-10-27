# AMI 2027 camera-ready: changes and measurements

Paper #162. Submitted version: `main_springer_ami.tex` / `main_springer_ami.pdf` (unchanged, identical to `ESV-gap submit AMI.pdf`).
Camera-ready working copy: `main_springer_ami_camera_ready.tex` (15 pages). Due 2027-01-20.

Authors: Anh Hoa Le, Ly Van Khoa Phan, Dinh Thanh Nguyen, Duc Hoang Nguyen, Dinh Anh Ho, Long Truong (FPT University, Vietnam).

Build:

```
pdflatex main_springer_ami_camera_ready.tex
bibtex main_springer_ami_camera_ready
pdflatex main_springer_ami_camera_ready.tex
pdflatex main_springer_ami_camera_ready.tex
```

## Why the results changed

On 2026-10-27 we checked whether the submitted numbers could be regenerated. Three were never measured:

- Stage A2: no classifier existed in the repository and the frozen run stored no Stage A2 labels.
- Pillar C: `benchmarks/gold_benchmark_extracted.json` was built by `scratch/setup_exact_benchmark.py`, which deletes eight gold triples and adds four hand-written false positives. It is not extractor output.
- K_fetch: the labels in `scratch/eval_k_fetch.py` were keyword rules that mapped every "lightweight" paper to a supporting limitation.

All three were measured on 2026-10-27, and the camera-ready reports the measured values.

| Quantity | Submitted | Measured |
|---|---|---|
| Extractor recall / miss rate (configured model gpt-oss-20b) | 77.8% / 22.2% | 61.1% / 38.9% [24.8, 55.1] |
| Extractor recall / miss rate (gpt-oss-120b) | | 69.4% / 30.6% [18.0, 46.9] |
| C1 zero-day | REFUTED (2) | REFUTED (5/5/5) |
| C2 protocols | REVIEW_REQUIRED | REVIEW_REQUIRED (0/0/0, circular) |
| C3 industrial validation | REFUTED (LiteRT-IDSNet) | REVIEW_REQUIRED (0/1/0, no corroboration) |
| C4 edge computational overhead | SUPPORTED_OPEN | REFUTED (4/4/4) |
| C5 adversarial defenses | REFUTED (1) | REFUTED (2/2/2) |
| Promoted gaps | 1 of 5 | 0 of 5 |
| Stage A2 agreement (3 models, 23 records) | | Fleiss kappa 0.63; 22/23 identical counterevidence decisions |
| C2 counterevidence at K = 5, 10, 20, 50 | not measured | 0 at every K (two models) |
| C4 counterevidence at K = 5, 10, 20, 50 | "none, stays open" | 4, 6, 12-13, 24-25 |
| Fault injection | 6/6 | 5/6 before the code fix below, 6/6 after |

Counts are verified counterevidence records from gpt-oss-120b / gpt-oss-20b / Qwen3.8-27B; dispositions are the majority over models. Llama-3.3-70B, named in the submitted paper, is no longer served by Groq and was not used.

Code fix: `src/autonomous_reasoning.py` let independent corroboration stand in for local provenance, so a candidate with zero supporting papers was retained (fault scenario 6). It now fails closed; regression test `test_2b_zero_local_provenance_fails_closed_despite_corroboration`.

Corpus statistics (150 papers, 1,404 relation events, 1,185 entities, beta = 0.995, decay -10.3%) were reproduced from the frozen run by `experiments/build_canonical_results.py` and are unchanged.

## Reproduction

From `ESV-Gap/` (needs `GROQ_API_KEY` in `../.env`):

```
python experiments/pillar_c_measure.py --model openai/gpt-oss-20b
python experiments/pillar_c_measure.py --model openai/gpt-oss-120b
python experiments/stage_a2_verify.py fetch
python experiments/stage_a2_verify.py classify --model openai/gpt-oss-120b --cache outputs/stage_a2/records_top5.json --tag _top5
python experiments/stage_a2_verify.py decide --labels outputs/stage_a2/labels_openai_gpt-oss-120b_top5.json --cache outputs/stage_a2/records_top5.json --pillar-b outputs/stage_a2/pillar_b.json
```

Per-record labels, spans, decisions and costs are in `paper_v2/measurements/` (record identifiers only; abstracts are not redistributed).

## Reviewer comments and changes

| Reviewer comment | Change |
|---|---|
| R1, R2.1: only n = 5, one domain | Kept as a stated limitation; larger candidate set and second domain named as needed. |
| R1: Stage A2 lacks evaluation | Stage A2 now actually run with three models; agreement reported. Human labels still missing (stated). |
| R2.2: single LLM | Three models for Stage A2, two for extraction. |
| R2.3: small D_gold | Interval width stated; benchmark described as 10 constructed passages outside the IoT domain. |
| R2.4: K_fetch = 5 | Monotonicity argument for REFUTED; C2 and C4 re-run with Stage A2 to K = 50. |
| R2 minor 1: terminology | Sentence on biomedicine and MeSH. |
| R2 minor 2: latency and cost | Measured: 3.7-4.2 s per call, 12.8k-14.3k prompt and 2.4k-4.3k completion tokens per model for 23 records. |
| R1: literature | ResearchAgent, SciMON, Si et al. added. |

## Camera-ready instructions (AMI 2027 checklist)

- Template: `llncs.cls` v2.24, no float-spacing overrides. `lmodern` is kept only so that the T1 Computer Modern design is embedded as Type 1 fonts; the local MiKTeX lacks `cm-super` and would otherwise produce Type 3 fonts. The PDF contains no Type 3 fonts.
- Corresponding author: Anh Hoa Le, marked with a footnote.
- Fig. 2 was redrawn as a vector PDF at text width, with a legend and line styles and grey levels so it does not rely on colour (`experiments/plot_saturation_figure.py`, same data). Alt text for both figures is in `ALT_TEXT.md`.
- `credits` section added with a generative-AI use statement and a disclosure of interests.
- DOIs added for Cortes 2016, Schneider 2022, ResearchAgent and SciMON.
- To stay at 15 pages: the gates table became one paragraph, the fault-injection table became one sentence listing the six scenarios, the Wilson formula is cited inline, and several paragraphs were shortened.
- Upload files: `camera_ready/AMI2027_paper162_final.pdf` and `camera_ready/AMI2027_paper162_source.zip`. The ZIP compiles from a clean folder to the same 15 pages.

## Still to do before submission

- ORCID iDs for the authors (`\orcidID{...}` after each name), if available.
- Funding: confirmed none; the `credits` section states "This study received no external funding." The disclosure of interests says the authors have no competing interests; confirm with all authors.
- Update the abstract in the CMT paper record to match the new abstract.
- Springer licence to publish, signed by Anh Hoa Le with the exact title and author names.

- Decide whether to inform the AMI programme chairs: the reviewed version reported one supported open gap (C4) and refuted C3; the measured results promote no gap.
- Optional, strongest answer to R1: two authors label the 23 Stage A2 records by hand (about one hour) and report agreement with the models.
- Stage A2 lives in `experiments/stage_a2_verify.py`, not in the `src/` pipeline that the paper describes as one architecture.
- Springer copyright form; author registration and payment (due 2027-01-20).
