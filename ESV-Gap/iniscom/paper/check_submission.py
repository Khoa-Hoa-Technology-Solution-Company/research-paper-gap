"""Inspect the compiled anonymous submission (requires pypdf and Poppler).

    python check_submission.py

The saved report is a mechanical check. The rendered PDF pages must also be read
visually; this does not certify tagged-PDF accessibility or universal anonymity.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    path = HERE / "main_submission.pdf"
    reader = PdfReader(path)
    pages = [p.extract_text() for p in reader.pages]
    text = "\n".join(pages)
    assert not reader.is_encrypted and all(p.strip() for p in pages)
    assert "Anonymous Authors" in pages[0] and "Anonymous Institution" in pages[0]
    assert not (reader.metadata.author or "").strip()
    refs_start = next(i for i, page in enumerate(pages) if re.search(r"(?m)^References\s*$", page))
    body = "\n".join(pages[:refs_start])
    for name in ("Anh Hoa Le", "Long Truong", "FPT University", "leanhhoa30012004", "longt5@"):
        assert name.lower() not in body.lower()
    assert "??" not in text and "\ufffd" not in text
    assert "We release the benchmark" not in text
    assert "0.797" in text and "0.845" in text and "Adamic" in text
    compact = re.sub(r"\s+", "", text).lower()
    metadata = json.loads((ROOT / "outputs" / "reference_metadata.json").read_text(encoding="utf-8"))
    for ref in metadata["records"]:
        assert ref["metadata"]["DOI"].lower() in compact
        assert ref["openalex_id"].lower() in compact
    log = (HERE / "main_submission.log").read_text(encoding="utf-8", errors="replace")
    for problem in ("Overfull", r"Underfull \hbox", "There were undefined", "Citation `", "Reference `",
                    "Label(s) may have changed", "Rerun to get"):
        assert problem not in log, problem
    assert not re.search(r"(?m)^!|LaTeX Error:", log)
    warnings = re.findall(r"(?m)^.*Warning:.*$", log)
    assert all(w == r"Package amsmath Warning: Unable to redefine math accent \vec." for w in warnings), warnings
    font_output = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True, check=True).stdout
    font_rows = [row.split() for row in font_output.splitlines()[2:] if row.strip()]
    assert font_rows and all(row[-5] == "yes" for row in font_rows), font_output
    bib = (HERE / "main_submission.bbl").read_text(encoding="utf-8")
    cited = re.findall(r"\\bibitem\{([^}]+)\}", bib)
    # Standard third-person references, including the necessary ESV-Gap citation,
    # retain real authors. They are not author-block or metadata identifiers.
    report = {
        "status": "passed", "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pages": len(pages),
        "first_reference_page": refs_start + 1, "pages_before_references": refs_start,
        "author_metadata": reader.metadata.author or "", "anonymous_author_block": True,
        "personal_identifiers_in_body": False, "all_fonts_embedded": True,
        "font_resources": len(font_rows), "bibliography_entries": len(cited),
        "new_dois_and_openalex_ids_present": len(metadata["records"]),
        "undefined_references_or_citations": False, "overfull_boxes": False,
        "underfull_vbox_diagnostics": re.findall(r"(?m)^Underfull.*$", log),
        "remaining_warnings": warnings, "tagged_accessibility_certified": False,
        "self_citation": "Normal third-person bibliographic citation retained; no anonymity guarantee",
        "visual_review": "Separate rendered-page inspection required"
    }
    output = ROOT / "outputs" / "submission_pdf_checks.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
