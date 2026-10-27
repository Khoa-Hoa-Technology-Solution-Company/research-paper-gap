"""Stage A2 semantic counterevidence verification, as described in the AMI 2027 paper (Sec. 4.2).

Re-runs Stage A2 on the Stage A1 records stored in a frozen run and derives the candidate
dispositions, so that the numbers in Table 3 can be measured instead of copied.

Steps
  1. fetch   : read the Stage A1 co-mention records of every candidate from
               <run>/outputs/external_verification.json and download their titles and abstracts
               from the Semantic Scholar batch API (cached in --cache).
  2. classify: send each (candidate, record) pair to an OpenAI-compatible chat endpoint with the
               four-label JSON schema of the paper, temperature 0, seed 42, up to 3 retries.
               An evidence span is admitted only if it is a verbatim substring of title or
               abstract. Latency and token usage are logged per call.
  3. decide  : REFUTED if >= 1 VERIFIED_COUNTEREVIDENCE record with an admitted span; otherwise
               REVIEW_REQUIRED if verification failed or no independent corroboration exists
               (Pillar B, from <run>/outputs/author_stated_gaps.json); otherwise SUPPORTED_OPEN
               for an unsaturated corpus.

Usage (from ESV-Gap/):
  python experiments/stage_a2_verify.py fetch
  python experiments/stage_a2_verify.py classify --model llama-3.3-70b-versatile \
      --base-url https://api.groq.com/openai/v1 --key-env GROQ_API_KEY
  python experiments/stage_a2_verify.py decide --labels outputs/stage_a2/labels_llama-3.3-70b-versatile.json

API keys are read from the environment (or a .env file) and never written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

RUN = Path("runs/deep_learning_iot_intrusion_de_20260831_114802")
OUT = Path("outputs/stage_a2")
LABELS = ["VERIFIED_COUNTEREVIDENCE", "SUPPORTING_LIMITATION", "AMBIGUOUS", "IRRELEVANT"]

PROMPT = """You verify whether a scientific paper resolves a hypothesized research gap.

Hypothesized gap: the method "{u}" has not yet achieved or addressed the capability "{v}".

Paper title: {title}
Paper abstract: {abstract}

Classify the relation between the method and the capability in this paper with exactly one label:
- VERIFIED_COUNTEREVIDENCE: the paper demonstrates, implements, or evaluates the capability for the method (it addresses, improves, or evaluates on it), so the gap is not open.
- SUPPORTING_LIMITATION: the paper states that the capability is an unresolved limitation or open challenge for the method.
- AMBIGUOUS: the paper discusses both without a definitive resolution.
- IRRELEVANT: the method and capability only co-occur superficially.

For VERIFIED_COUNTEREVIDENCE and SUPPORTING_LIMITATION, copy the supporting sentence or clause
exactly, character for character, from the title or abstract into evidence_span.

Answer with JSON only:
{{"classification": "<label>", "evidence_span": "<verbatim text or empty>", "confidence": <0..1>, "relation_rationale": "<one sentence>"}}"""


def load_env():
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env)
        except ImportError:
            pass


def candidates():
    ev = json.load(open(RUN / "outputs" / "external_verification.json", encoding="utf-8"))
    out = []
    for i, c in enumerate(ev["candidates"], 1):
        recs = []
        for p in c["external_verification"]["probes"]:
            for h in p["confirmed_hits"]:
                recs.append({"source": p["source"], **h})
        out.append({"id": f"C{i}", "subject": c["subject"], "capability": c["missing_capability"],
                    "stage_a1_verdict": c["external_verification"]["verdict"],
                    "probes": [{k: p[k] for k in ("source", "query", "raw_count", "inspected_records", "confirmed_hit_count", "retrieved_at")}
                               for p in c["external_verification"]["probes"]],
                    "records": recs})
    return out


def cmd_fetch(args):
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    ids = sorted({r["id"] for c in cands for r in c["records"] if r["source"] == "semantic_scholar"})
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,year,externalIds"
    req = urllib.request.Request(url, data=json.dumps({"ids": ids}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        req.add_header("x-api-key", key)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                papers = json.load(r)
            break
        except Exception as e:  # rate limit or network: back off and retry
            print("fetch failed:", type(e).__name__, e)
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit("could not fetch abstracts")
    meta = {i: p for i, p in zip(ids, papers) if p}
    for c in cands:
        for r in c["records"]:
            p = meta.get(r["id"]) or {}
            r["abstract"] = p.get("abstract") or ""
            r["title"] = p.get("title") or r.get("title", "")
    json.dump(cands, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n = sum(len(c["records"]) for c in cands)
    na = sum(1 for c in cands for r in c["records"] if r["abstract"])
    print(f"{n} records, {na} with abstract -> {args.cache}")


def verbatim(span, title, abstract):
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
    s = norm(span).strip(" .\"'")
    return bool(s) and (s in norm(title) or s in norm(abstract))


def cmd_classify(args):
    load_env()
    from openai import OpenAI
    key = os.environ.get(args.key_env)
    if not key:
        raise SystemExit(f"environment variable {args.key_env} is not set")
    client = OpenAI(api_key=key, base_url=args.base_url)
    cands = json.load(open(args.cache, encoding="utf-8"))
    rows = []
    for c in cands:
        for r in c["records"]:
            prompt = PROMPT.format(u=c["subject"], v=c["capability"], title=r["title"],
                                   abstract=r["abstract"] or "(no abstract available)")
            row = {"candidate": c["id"], "record_id": r["id"], "title": r["title"], "has_abstract": bool(r["abstract"])}
            for attempt in range(4):
                t0 = time.time()
                try:
                    extra = {"reasoning_effort": "low"} if "gpt-oss" in args.model else {}
                    resp = client.chat.completions.create(
                        model=args.model, temperature=0.0, seed=42, max_tokens=args.max_tokens,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": prompt}], **extra)
                    dt = time.time() - t0
                    out = json.loads(resp.choices[0].message.content)
                    lab = str(out.get("classification", "")).strip().upper()
                    if lab not in LABELS:
                        raise ValueError(f"bad label {lab!r}")
                    span = out.get("evidence_span") or ""
                    ok_span = verbatim(span, r["title"], r["abstract"])
                    if lab in LABELS[:2] and not ok_span:
                        lab_eff, span_status = "AMBIGUOUS", "span_not_verbatim"
                    else:
                        lab_eff, span_status = lab, ("verbatim" if ok_span else "none")
                    u = resp.usage
                    row.update(label_raw=lab, label=lab_eff, span=span, span_status=span_status,
                               confidence=out.get("confidence"), rationale=out.get("relation_rationale"),
                               latency_s=round(dt, 3), prompt_tokens=getattr(u, "prompt_tokens", None),
                               completion_tokens=getattr(u, "completion_tokens", None), attempts=attempt + 1, ok=True)
                    break
                except Exception as e:
                    row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}", attempts=attempt + 1)
                    time.sleep(2 ** attempt)
            rows.append(row)
            print(c["id"], row.get("label", "FAILED"), row.get("latency_s"), r["title"][:60])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"labels_{args.model.replace('/', '_')}{args.tag}.json"
    json.dump({"model": args.model, "base_url": args.base_url, "seed": 42, "temperature": 0.0, "rows": rows},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", path)


def pillar_b():
    data = json.load(open(RUN / "outputs" / "author_stated_gaps.json", encoding="utf-8"))
    return data


def cmd_decide(args):
    lab = json.load(open(args.labels, encoding="utf-8"))
    cands = json.load(open(args.cache, encoding="utf-8"))
    corrob = {c["id"]: c for c in json.load(open(args.pillar_b, encoding="utf-8"))} if args.pillar_b else {}
    res = []
    for c in cands:
        rows = [r for r in lab["rows"] if r["candidate"] == c["id"]]
        failed = any(not r.get("ok") for r in rows)
        n_ce = sum(r.get("label") == "VERIFIED_COUNTEREVIDENCE" for r in rows)
        n_sl = sum(r.get("label") == "SUPPORTING_LIMITATION" for r in rows)
        indep = corrob.get(c["id"], {}).get("independent", None)
        if n_ce >= 1:
            d = "REFUTED"
        elif failed or indep is None or indep < 1:
            d = "REVIEW_REQUIRED"
        else:
            d = "SUPPORTED_OPEN"
        res.append({"candidate": c["id"], "co_mention_records": len(rows), "verified_counterevidence": n_ce,
                    "supporting_limitation": n_sl, "failed_calls": sum(not r.get("ok") for r in rows),
                    "independent_corroboration": indep, "disposition": d})
    calls = [r for r in lab["rows"] if r.get("ok")]
    cost = {"calls": len(calls),
            "mean_latency_s": round(sum(r["latency_s"] for r in calls) / max(1, len(calls)), 3),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in calls),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in calls)}
    out = {"model": lab["model"], "candidates": res, "cost": cost}
    path = Path(args.labels).with_name(Path(args.labels).stem.replace("labels_", "decisions_") + ".json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "classify", "decide"])
    ap.add_argument("--cache", default=str(OUT / "records.json"))
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--key-env", default="GROQ_API_KEY")
    ap.add_argument("--labels")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _kfetch")
    ap.add_argument("--pillar-b", help="JSON list [{id, independent}] with Pillar B corroboration counts")
    args = ap.parse_args()
    {"fetch": cmd_fetch, "classify": cmd_classify, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    main()
