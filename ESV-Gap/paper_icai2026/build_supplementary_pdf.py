"""Render numeric-only supplementary documentation from retained audit reports."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted,
)
from pypdf import PdfReader


PAPER = Path(__file__).resolve().parent
WORKSPACE = PAPER.parent.parent
BUNDLE = PAPER / "release_candidate"
REPORTS = BUNDLE / "reports"
OUT = WORKSPACE / "output/pdf/ESV-Gap_ICAI2026_Supplementary.pdf"
MAIN = WORKSPACE / "output/pdf/ESV-Gap_ICAI2026_Draft.pdf"
WIDTH = A4[0] - 96
NAMES = {
    "iot_intrusion_detection": "IoT IDS",
    "microservice_security": "Microservices",
    "mongodb_security": "MongoDB",
    "handwritten_math_recognition": "Handwritten math",
}


def load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def main():
    # Fail before authoring if the numeric payload or manifest is inconsistent.
    subprocess.run([sys.executable, "-B", str(BUNDLE / "verify_aggregate_bundle.py")], check=True)
    for name, filename in [("Academic", "times.ttf"), ("Academic-Bold", "timesbd.ttf"),
                           ("Academic-Italic", "timesi.ttf"), ("Code", "consola.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / filename)))
    pdfmetrics.registerFontFamily("Academic", normal="Academic", bold="Academic-Bold",
                                  italic="Academic-Italic", boldItalic="Academic-Bold")
    style = {
        "body": ParagraphStyle("Body", fontName="Academic", fontSize=10, leading=13,
                               spaceAfter=7),
        "small": ParagraphStyle("Small", fontName="Academic", fontSize=8.6, leading=11,
                                spaceAfter=5),
        "title": ParagraphStyle("Title", fontName="Academic-Bold", fontSize=18, leading=21,
                                alignment=TA_CENTER, spaceAfter=9),
        "subtitle": ParagraphStyle("Subtitle", fontName="Academic", fontSize=11, leading=14,
                                   alignment=TA_CENTER, spaceAfter=7),
        "heading": ParagraphStyle("Heading", fontName="Academic-Bold", fontSize=13, leading=16,
                                  spaceAfter=9, spaceBefore=3),
        "cell": ParagraphStyle("Cell", fontName="Academic", fontSize=8.3, leading=10.5),
        "hash": ParagraphStyle("Hash", fontName="Code", fontSize=7.2, leading=10),
        "code": ParagraphStyle("CodeBlock", fontName="Code", fontSize=7.8, leading=10.2,
                               spaceBefore=3, spaceAfter=8),
    }
    story = []

    def p(text, kind="body"):
        story.append(Paragraph(text, style[kind]))

    def h(text):
        p(text, "heading")

    def table(rows, widths, align_numbers=False):
        data = [[Paragraph(escape(str(value)), style["cell"]) for value in row] for row in rows]
        obj = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf0f2")),
            ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        obj.setStyle(TableStyle(commands))
        story.extend([obj, Spacer(1, 9)])

    def hashes(rows):
        data = [[Paragraph(escape(label), style["cell"]), Paragraph(value, style["hash"])]
                for label, value in rows]
        obj = Table(data, colWidths=[140, WIDTH - 140], hAlign="LEFT")
        obj.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
        ]))
        story.extend([obj, Spacer(1, 9)])

    replay = load("multidomain_summary.json")
    bm25 = load("bm25_lacks_baseline.json")
    sensitivity = load("bm25_query_sensitivity.json")
    ablation = load("candidate_source_ablation.json")
    folds = [r | {"domain": d["domain"]} for d in replay["sources"] for r in d["cutoffs"]
             if r["status"] == "completed"]
    bm_by = {(r["domain"], r["cutoff"]): r for r in bm25["rows"]}
    sens_by = {(r["domain"], r["cutoff"]): r for r in sensitivity["rows"]}
    scored = [r for r in folds if r["positive_controls"]]

    p("Supplementary Material", "title")
    p("Auditing AI-Assisted Research-Gap Retrieval:<br/>A Four-Corpus Time-Split Evaluation", "subtitle")
    p("Lê Anh Hòa | FPT University, Vietnam<br/>leanhhoa3002004@gmail.com<br/>ICAI-FAI 2026 | Paper ID 63 | Prepared 1 October 2026", "subtitle")
    h("S1. Scope and evaluation configuration")
    p("This supplement documents the frozen offline replay reported in the manuscript. It adds numeric accounting and implementation details, not new experiments, expert labels, or validated gap-discovery claims. The five retained BM25 matches are not independently confirmed closure witnesses.")
    table([["Preserved corpus", "Keyed records", "Triple events"]] +
          [[NAMES[d["domain"]], d["retained_papers"], d["triples"]] for d in replay["sources"]] +
          [["Total", replay["source_rows"], sum(d["triples"] for d in replay["sources"])]],
          [WIDTH * .55, WIDTH * .22, WIDTH * .23])
    p("Record identity uses lowercased DOI with a DOI-URL prefix stripped, otherwise source ID, otherwise whitespace-normalized lowercase title. All 378 rows have distinct keys under this rule; preprint/publication and identifier variants may remain.", "small")
    table([["Setting", "Recorded implementation"]] + [
        ["Cutoffs / eligibility", "2022, 2023, 2024; at least 10 pre-cutoff and 10 future records."],
        ["Temporal split", "Stored year <= cutoff for candidates; stored year > cutoff for future controls."],
        ["Candidate generators", "Explicit LACKS evidence and typed evidence-map empty cells only."],
        ["Explicit evidence settings", "Confidence >= 0.65; limitation cue required; evidence-cell consolidation enabled (similarity 0.72)."],
        ["Evidence-map settings", "Enabled; minimum marginal papers 2; up to 60 entities/type; top 40 candidates. Allowed pairs: METHOD-DATASET, METHOD-CONCEPT, CONCEPT-DATASET."],
        ["Ranking budget", "k = 5, 10, 20; inspect min(k, number of candidates)."],
        ["BM25 settings", "k1 = 1.2, b = 0.75; subject-capability strings from the direct-LACKS pool, not raw abstracts."],
    ], [130, WIDTH - 130])
    p("This is a configuration synopsis, not the full YAML. The recorded config hash is in S4. TransE, Louvain and temporal-decay detectors from the earlier end-to-end audit were not rerun; the later certification gate was not executed.", "small")

    story.append(PageBreak())
    h("S2. Per-fold accounting and aggregate retrieval")
    p("Table S1. Completed domain/cutoff evaluations. Pre/Future count records; H+ and H- are heuristic positive and negative control rows; ESV and Pool count candidates. R20 denotes proxy recall at 20. N/A means no positive controls.", "small")
    rows = [["Domain / year", "Pre", "Future", "H+", "H-", "ESV", "Pool", "ESV R20", "BM25 R20"]]
    for r in folds:
        b = bm_by[(r["domain"], r["cutoff"])]
        rows.append([NAMES[r["domain"]] + " / " + str(r["cutoff"]), r["pre"], r["future"],
                     r["positive_controls"], r["negative_controls"], r["candidates"],
                     r["direct_lacks_baseline_candidates"], fmt(r["recall_at_20"]),
                     fmt(b["recall_at_k"]["20"])])
    rows.append(["Row totals", "-", "-", sum(r["positive_controls"] for r in folds),
                 sum(r["negative_controls"] for r in folds), sum(r["candidates"] for r in folds),
                 sum(r["direct_lacks_baseline_candidates"] for r in folds), "-", "-"])
    table(rows, [145, 35, 42, 33, 33, 35, 42, 67, WIDTH - 432])
    p("Two folds were excluded by the executability rule: MongoDB/2022 (9 pre, 44 future) and handwritten math/2024 (24 pre, 8 future). Handwritten math/2023 executes but is omitted from macro recall because H+ = 0. The threshold is not a sample-size justification.")
    p("Table S2. Unweighted macro proxy recall over the nine scored folds. Matched rows at 20 have denominator 64; all cutoffs reuse papers.", "small")
    table([["Ranking", "R@5", "R@10", "R@20", "Matched H+ at 20"],
           ["ESV-Gap", *[fmt(bm25["macro_esv_gap_recall_at_k"][k]) for k in ("5", "10", "20")], "1 / 64"],
           ["Confidence-LACKS", *[fmt(bm25["macro_confidence_lacks_recall_at_k"][k]) for k in ("5", "10", "20")], "0 / 64"],
           ["BM25-LACKS (Q1)", *[fmt(bm25["macro_recall_at_k"][k]) for k in ("5", "10", "20")], "5 / 64"]],
          [160, 66, 66, 66, WIDTH - 358])
    p("Control-row micro recall at 20 is 1/64 = 0.0156 for ESV-Gap and 5/64 = 0.0781 for BM25-LACKS. These differ from macro recall because fold denominators differ. Neither measure is recall of true research gaps or an independent population estimate.")
    h("Candidate-source decomposition")
    p(f"All {ablation['candidate_rows_not_independent']} ESV candidate rows are explicit-limitation candidates; the typed evidence-map branch contributes {ablation['typed_evidence_map_rows_not_independent']}. Four completed evaluations emit no ESV candidate. The direct-LACKS pool contains 474 non-independent rows. These observations do not identify a causal graph-topology benefit or isolate why the graph branch is empty.")
    p("H- rows are algorithmically generated controls, not expert-confirmed unresolved gaps. Certificate precision/recall are not measured; serialized zero certificate counts are placeholders. No independence-based confidence interval or significance test is reported.", "small")

    story.append(PageBreak())
    h("S3. Matching, query sensitivity and case validity")
    p("Token matching lowercases text, extracts alphanumeric tokens, excludes one-character tokens and the implementation's stopword set. For token sets A and B, similarity is 0.7 times containment (intersection size / smaller set size) plus 0.3 times Jaccard (intersection / union); an empty set gives zero. A limitation match requires capability similarity >= 0.62 and either subject similarity >= 0.25 or capability similarity >= 0.82. A structural match compares token-normalized endpoint sets.")
    p("Recall@k counts distinct positive control IDs matched by at least one top-k candidate, divided by the fold's H+ count. BM25 tokenization and matching tokenization are separate: BM25 uses lowercase alphanumeric term frequencies and a set of query terms, with IDF = log(1 + (N - df + 0.5)/(df + 0.5)). BM25 ties use confidence and then lexical subject/capability order.", "small")
    p("Queries prepend the domain label to Q1: 'limitation unresolved challenge'; Q2: 'research gap limitation problem'; Q3: 'limitation future work challenge'. All queries are post-hoc, not preregistered or tuned on a held-out set. Future-control text does not enter ranking.")
    p("Table S3. Per-fold proxy R@20 and number of matched control rows (in parentheses). Confidence-LACKS recall is zero in every scored fold.", "small")
    rows = [["Domain / year", "H+", "Q1 R20 (hits)", "Q2 R20 (hits)", "Q3 R20 (hits)"]]
    for r in folds:
        s = sens_by[(r["domain"], r["cutoff"])]["queries"]
        values = [fmt(s[q]["recall_at_k"]["20"]) + " (" + str(s[q]["matched_control_count_at_20"]) + ")"
                  for q in ("Q1", "Q2", "Q3")]
        rows.append([NAMES[r["domain"]] + " / " + str(r["cutoff"]), r["positive_controls"], *values])
    rows.append(["Macro / row hits", "64", *[fmt(sensitivity["summary"][q]["macro_recall_at_k"]["20"]) +
                 " (" + str(sensitivity["summary"][q]["matched_control_rows_at_20_not_independent"]) + ")"
                 for q in ("Q1", "Q2", "Q3")]])
    table(rows, [151, 32, 105, 105, WIDTH - 393])
    p("Q1/Q2/Q3 all have macro R@5 = 0.0556 and R@10 = 0.0694. At 20 they differ by one matched row; this is wording sensitivity, not evidence of certified discovery.", "small")
    h("Interpretation of the five Q1 matches")
    p("Four retained abstracts do not report a same-scope resolution experiment. The empirical microservice/IoT IDS record reports DoS metrics, but its matched 'Retrofitting-Security' candidate leaves task and condition unspecified. The five-case inspection is AI-assisted, abstract-only and non-blinded; it has no independent human adjudication or full-text refutation.")
    p("The MongoDB record's source identity and stored 2024 year are unverified. It remains in the frozen replay solely for accounting, not as independently established future evidence; corrected metadata could alter its temporal assignment and proxy scores. No corrected-metadata replay is claimed. Zero independently confirmed closure witnesses does not mean five proven false positives. The manuscript's Table IV identifies the retained records.", "small")

    story.append(PageBreak())
    h("S4. File identity and recorded runtime")
    p("All digests below are full SHA-256 values. They identify byte sequences, not rights, scientific validity, independent reproduction or historical availability. Input files are omitted from this PDF.", "small")
    h("Preserved inputs")
    hashes([(NAMES[domain] + " / " + kind, value)
            for domain, entries in replay["fingerprints"]["inputs"].items()
            for kind, value in entries.items()])
    h("Replay configuration and code")
    hashes([(label, replay["fingerprints"][field]) for label, field in [
        ("config.yaml", "config"), ("src/temporal_backtest.py", "temporal_code"),
        ("src/detect_gaps.py", "detector_code"), ("Offline replay runner", "runner")]])
    h("Original numeric/analysis reports")
    originals = PAPER / "expanded_run"
    hashes([(name, digest(originals / name)) for name in [
        "multidomain_summary.json", "candidate_source_ablation.json",
        "bm25_lacks_baseline.json", "bm25_query_sensitivity.json"]])
    p("Report hashes identify the original retained reports. The local review bundle projects BM25 and sensitivity reports by removing candidate/control text and matched IDs, retaining distinct-ID counts; these projected files have different hashes, recorded in that bundle's manifest. The other two reports are byte-preserved.", "small")
    p("Recorded replay runtime: CPython 3.14.6, NetworkX 3.6.1, PyYAML 6.0.3. Local implementation-test dependencies also include NumPy 2.5.1 and python-louvain 0.16. These versions describe the retained environment, not a portability guarantee.", "small")
    hashes([("Companion manuscript PDF", digest(MAIN))])

    story.append(PageBreak())
    h("S5. Arithmetic check and offline replay requirements")
    p("The self-contained Python listing below recomputes the printed macro recalls from per-fold counts. Only nine folds with positive controls are included, in the order of Table S1. This checks arithmetic; the counts are retained inputs to the check, not independently regenerated matches.")
    positives = [r["positive_controls"] for r in scored]
    esv = [round(r["recall_at_20"] * r["positive_controls"]) for r in scored]
    bm_hits = {k: [round(bm_by[(r["domain"], r["cutoff"])]["recall_at_k"][k] * r["positive_controls"])
                   for r in scored] for k in ("5", "10", "20")}
    lines = ["# Python 3; standard library only", f"positive = {positives}", f"esv20 = {esv}"]
    for k in ("5", "10", "20"):
        lines.append(f"bm{k} = {bm_hits[k]}")
    for q in ("Q2", "Q3"):
        hits = [sens_by[(r["domain"], r["cutoff"])]["queries"][q]["matched_control_count_at_20"] for r in scored]
        lines.append(f"{q.lower()}20 = {hits}")
    lines += ["def macro(hits):", "    return sum(h/n for h, n in zip(hits, positive)) / len(positive)",
              "assert sum(positive) == 64", "assert sum(esv20) == 1 and sum(bm20) == 5",
              "assert sum(q220) == 4 and sum(q320) == 4",
              "for name, hits in [('ESV@20', esv20), ('BM25@5', bm5),",
              "                   ('BM25@10', bm10), ('BM25@20', bm20),",
              "                   ('Q2@20', q220), ('Q3@20', q320)]:",
              "    print(name, f'{macro(hits):.4f}')"]
    code = "\n".join(lines)
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
    story.append(Preformatted(code, style["code"]))
    h("Full replay is conditional on omitted files")
    p("This PDF is documentation, not a standalone executable artifact. It does not supply the local code/report bundle or a public download URL. Full replay needs the original scripts/config and lawfully held corpus/triple files matching S4. No independent public reproduction is claimed.")
    p("For each run directory below, the runner expects data/processed/corpus_filtered.jsonl and data/triples/all_triples.json under runs/&lt;run&gt;/:", "small")
    for d in replay["sources"]:
        p(escape(NAMES[d["domain"]] + ": " + d["source_run"]), "small")
    p("With matching inputs and scripts restored, run from the ESV-Gap project root, in order:", "small")
    commands = ["python -B paper_icai2026/expanded_run/" + name for name in [
        "run_multidomain_backtest.py", "analyze_candidate_sources.py",
        "run_bm25_lacks_baseline.py", "run_bm25_query_sensitivity.py"]]
    story.append(Preformatted("\n".join(commands), style["code"]))
    p("Analysis scripts require the saved fold outputs generated by the first command. Raw outputs may contain excluded text and identifiers and require rights review before redistribution. Four existing temporal unit tests passed locally; they test implementation behavior, not scientific discovery. The six research-gap types remain proposed future annotation categories and have not been evaluated.", "small")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#8a8a8a"))
        canvas.setLineWidth(.4)
        canvas.line(48, 36, A4[0] - 48, 36)
        canvas.setFont("Academic", 8)
        canvas.drawString(48, 23, "ICAI-FAI 2026 | Paper 63 | Supplementary Material")
        canvas.drawRightString(A4[0] - 48, 23, f"S{doc.page}")
        canvas.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=48, rightMargin=48,
                            topMargin=43, bottomMargin=48,
                            title="Supplementary Material - Auditing AI-Assisted Research-Gap Retrieval",
                            author="Lê Anh Hòa", subject="ICAI-FAI 2026, Paper ID 63: frozen replay documentation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(str(OUT))
    if len(reader.pages) != 5:
        raise ValueError(f"Expected 5 pages; got {len(reader.pages)}")
    for index, expected in enumerate(["S1. Scope", "S2. Per-fold", "S3. Matching", "S4. File", "S5. Arithmetic"]):
        if expected not in reader.pages[index].extract_text():
            raise ValueError("Section flowed to wrong page: " + expected)
    if OUT.stat().st_size >= 20_000_000:
        raise ValueError("CMT attachment exceeds visible 20 MB limit")
    print(f"Created: {OUT}\nPages: {len(reader.pages)}\nBytes: {OUT.stat().st_size}\nSHA-256: {digest(OUT)}")


if __name__ == "__main__":
    main()
