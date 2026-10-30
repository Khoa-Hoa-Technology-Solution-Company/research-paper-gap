"""Create a TeX-only submission archive with the author branch removed.

    python package_submission.py

Normal bibliographic citations retain their authors. This is a local package, not a
publication or upload. Compile the extracted main_submission.tex to verify the archive.
"""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    source = (HERE / "main.tex").read_text(encoding="utf-8")
    anonymous = "\\author{Anonymous Authors}\n\\authorrunning{Anonymous Authors}\n\\institute{Anonymous Institution}"
    source, replaced = re.subn(r"\\ifdefined\\anonymous\s+\\author\{Anonymous Authors\}.*?\\fi",
                              lambda match: anonymous, source, count=1, flags=re.S)
    assert replaced == 1
    for identifier in ("Anh Hoa Le", "Long Truong", "FPT University", "leanhhoa30012004", "longt5@"):
        assert identifier not in source
    files = [HERE / name for name in ["main_submission.tex", "main_submission.bbl", "references.bib",
                                      "llncs.cls", "splncs04.bst", "ALT_TEXT.md"]]
    files += sorted((HERE / "tables").glob("*.tex"))
    files += sorted((HERE / "tables").glob("*.dat"))
    target = ROOT / "gapcast_iniscom2027_submission_source.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("main.tex", source)
        archive.writestr("README.txt", "Anonymous INISCOM 2027 submission source.\n\n"
                         "Build: pdflatex main_submission; bibtex main_submission; "
                         "pdflatex main_submission; pdflatex main_submission.\n"
                         "Generated tables and plot data are already included.\n"
                         "The named-author branch has been removed. Standard bibliographic "
                         "citations retain their authors.\n"
                         "ALT_TEXT.md supplies descriptions for the accessibility workflow; "
                         "the PDF is not a tagged-accessibility PDF.\n")
        for path in files:
            archive.write(path, path.relative_to(HERE).as_posix())
    pdf = ROOT / "gapcast_iniscom2027_submission.pdf"
    shutil.copyfile(HERE / "main_submission.pdf", pdf)
    assert hashlib.sha256(pdf.read_bytes()).digest() == hashlib.sha256((HERE / "main_submission.pdf").read_bytes()).digest()
    print(json.dumps({"pdf": str(pdf), "source_archive": str(target),
                      "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "source_entries": len(files) + 2, "named_author_branch_removed": True}, indent=2))


if __name__ == "__main__":
    main()
