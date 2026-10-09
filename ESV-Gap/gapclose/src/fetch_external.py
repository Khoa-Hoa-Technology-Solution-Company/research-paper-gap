"""Fetch OpenAlex evidence for every question and score it with the NLI verifier (cached, resumable).

Requires OPENALEX_API_KEY in the environment (free key: about 1,000 searches per day). Re-run the
script after a budget reset to continue; already cached questions are skipped.
"""
from __future__ import annotations

import json

from common import OUT
from external import BudgetExhausted, fetch_all, score_all
from onnx_models import NLI

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def main():
    corpus = json.load(open(OUT / "bench" / "corpus.json", encoding="utf-8"))
    caches = {}
    for split in ("calib", "test"):
        claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
        try:
            cache = fetch_all(claims, corpus, split)
        except BudgetExhausted:
            print("OpenAlex daily budget exhausted; run again after the reset (midnight UTC).")
            cache = json.load(open(OUT / "external" / f"openalex_{split}.json", encoding="utf-8"))
        caches[split] = ([c for c in claims if str(c["id"]) in cache], cache)
        print(f"{split}: {len(cache)}/{len(claims)} questions cached", flush=True)
    nli = NLI(NLI_MODEL)
    for split, (have, cache) in caches.items():
        score_all(have, cache, nli, split, "_large")
    print("done")


if __name__ == "__main__":
    main()
