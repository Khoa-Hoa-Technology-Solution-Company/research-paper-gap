#!/usr/bin/env bash
# Full pipeline after data/corpus_*.jsonl is downloaded (src/fetch_corpus.py).
set -e
cd "$(dirname "$0")/src"
PY=../../gapclose/.venv/Scripts/python
export PYTHONIOENCODING=utf-8
$PY build_index.py
$PY gaps.py --tag main
$PY gaps.py --tag g2 --g 2
$PY gaps.py --tag k1 --k 1
$PY gaps.py --tag k5 --k 5
$PY gaps.py --tag h2 --h 2
$PY gaps.py --tag occ20 --min-occ 20
$PY gaps.py --tag occ100 --min-occ 100
$PY gaps.py --tag sent --label sentence
$PY gaps.py --tag sentns --label sentence_ns
$PY gaps.py --tag vocab2013 --vocab-year 2013
$PY ablation.py
$PY extra.py
$PY subsets.py
$PY audit_examples.py
$PY verify_revision.py
echo ALL_DONE
