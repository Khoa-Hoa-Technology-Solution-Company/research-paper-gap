"""Pillar C: measure the real relation extractor on the gold benchmark.

The paper's Pillar C numbers (TP/FN/FP = 28/8/4) were computed on
benchmarks/gold_benchmark_extracted.json, which scratch/setup_exact_benchmark.py builds by hand
(deleting eight gold triples and adding four false positives). This script instead runs the
pipeline's own extractor (src.extract_triples.extract_triples_from_text, same prompt) on the
ten gold texts and scores the output with src.extractor_recall.evaluate_extracted_triples.

Usage (from ESV-Gap/):
  python experiments/pillar_c_measure.py --model openai/gpt-oss-20b --key-env GROQ_API_KEY
  python experiments/pillar_c_measure.py --model llama-3.3-70b-versatile --key-env GROQ_API_KEY

Writes outputs/pillar_c/extracted_<model>.json and report_<model>.json. Keys are read from the
environment (or ../.env) and never written to disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.extract_triples import extract_triples_from_text, load_extraction_prompt  # noqa: E402
from src.extractor_recall import evaluate_extracted_triples  # noqa: E402

OUT = Path("outputs/pillar_c")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--key-env", default="GROQ_API_KEY")
    ap.add_argument("--benchmark", default="benchmarks/gold_triples_benchmark.json")
    ap.add_argument("--domain", default="deep learning IoT intrusion detection")
    args = ap.parse_args()

    env = ROOT.parent / ".env"
    if env.exists():
        from dotenv import load_dotenv
        load_dotenv(env)
    key = os.environ.get(args.key_env)
    if not key:
        raise SystemExit(f"environment variable {args.key_env} is not set")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url)

    bench = json.load(open(args.benchmark, encoding="utf-8"))
    prompt = load_extraction_prompt("prompts")
    extracted, timing = [], []
    for p in bench["papers"]:
        t0 = time.time()
        triples = extract_triples_from_text(client, args.model, prompt, args.domain, p["title"], p["year"], p["text"],
                                            temperature=0.0)
        timing.append(round(time.time() - t0, 2))
        for t in triples:
            t["source_paper_id"] = p["paper_id"]
        extracted.extend(triples)
        print(p["paper_id"], len(triples), "triples", timing[-1], "s")

    res = evaluate_extracted_triples(bench["papers"], extracted, entity_threshold=0.60, strict_relation=True)
    s = res["summary"]
    tp, fn, fp = s.get("true_positives"), s.get("false_negatives"), s.get("false_positives")
    s["recall_wilson95"] = wilson(tp, tp + fn)
    s["precision_wilson95"] = wilson(tp, tp + fp)
    s["miss_rate_wilson95"] = wilson(fn, tp + fn)
    res["model"], res["latency_s_per_text"] = args.model, timing
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_")
    json.dump(extracted, open(OUT / f"extracted_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(res, open(OUT / f"report_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
