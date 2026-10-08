#!/usr/bin/env bash
# Overnight chain: wait for the OpenAlex daily budget to reset, complete the external evidence for
# all 1,109 questions, rerun every certification setting, regenerate tables and rebuild the paper.
# Progress is appended to outputs/overnight_status.md.
set -u
ROOT=/d/Workspace/research-paper-gap/ESV-Gap
cd "$ROOT/gapclose/src"
PY=../.venv/Scripts/python
LOG=../outputs/overnight_status.md
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
export OPENALEX_API_KEY="$(powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENALEX_API_KEY','User')" | tr -d '\r\n')"
say() { echo "- $(date '+%H:%M') $*" | tee -a "$LOG"; }

say "chain started; waiting for OpenAlex budget"
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://api.openalex.org/works?search=aspirin&per-page=1&select=id&api_key=$OPENALEX_API_KEY")
  [ "$code" = "200" ] && break
  sleep 600
done
say "OpenAlex budget available"

$PY -u - >> ../outputs/overnight_fetch.txt 2>&1 <<'EOF'
import json
from common import OUT
from external import fetch_all, score_all, CACHE
from onnx_models import NLI
corpus = json.load(open(OUT / "bench" / "corpus.json", encoding="utf-8"))
nli = NLI("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
for split in ("calib", "test"):
    claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
    for attempt in range(3):
        cache = fetch_all(claims, corpus, split)
        if len(cache) == len(claims):
            break
    have = [c for c in claims if str(c["id"]) in cache]
    print(split, "cached", len(cache), "of", len(claims), flush=True)
    score_all(have, cache, nli, split, "_large")
print("FETCH_DONE", flush=True)
EOF
say "external evidence: $(grep -a 'cached' ../outputs/overnight_fetch.txt | tail -2 | tr '\n' ' ')"

while [ ! -f ../outputs/scores/bge_test.json ]; do sleep 60; done
say "BGE scores present; running 4 certification settings (R=500)"
for prot in pooled holdout; do
  for del in mcar targeted; do
    $PY certify.py --R 500 --protocol $prot --deletion $del > ../outputs/certify_log_${prot}_${del}.txt 2>&1 &
  done
done
wait
say "certification done: $(grep -h 'questions (' ../outputs/certify_log_pooled_mcar.txt)"

$PY make_audit.py >> ../outputs/overnight_fetch.txt 2>&1
cd "$ROOT/paper_gapclose_icai2026"
$PY make_tables.py > ../gapclose/outputs/make_tables_log.txt 2>&1
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1
say "paper rebuilt: $(grep -o 'Output written on main.pdf ([0-9]* pages' main.log) ; errors: $(grep -c '^!' main.log)"
say "CHAIN_DONE"
