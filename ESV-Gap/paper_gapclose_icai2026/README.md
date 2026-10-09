# Manuscript: Absence of Evidence Is Not a Research Gap

ICAI-FAI 2026 submission (IEEE conference format, 8 pages including references).

Authors: Anh Hoa Le, Ly Van Khoa Phan, Dinh Thanh Nguyen, Duc Hoang Nguyen, Dinh Anh Ho,
Long Truong (FPT University).

## Build

```
python make_tables.py      # regenerate tables/*.tex and tables/macros.tex from ../gapclose/outputs*/results
pdflatex main.tex
pdflatex main.tex
```

`make_tables.py` writes every table, the figure data and the number macros used in the text
(`\FNfoneOne`, `\RetroBestPow`, ...), so the PDF always matches the latest experiment outputs.
Run the experiments first as described in [`../gapclose/README.md`](../gapclose/README.md).

| File | Content |
|---|---|
| `main.tex` | manuscript |
| `make_tables.py` | generates `tables/` from experiment results |
| `tables/cls.tex` | uncertified accuracy (Table II) |
| `tables/robust.tex`, `fn.tex`, `power.tex`, `fig.tex` | certification under deletion (Tables IV to VI, Fig. 2) |
| `tables/climate.tex` | second domain (Table VII) |
| `tables/retro.tex`, `dossier.tex` | retrospective validation and gap dossiers (Tables VIII, IX) |
| `BAO_CAO_DEM_08-10.md` | internal revision log (Vietnamese) |

Writing conventions: no em or en dashes in the prose, result numbers in the text come from the generated macros wherever possible.
