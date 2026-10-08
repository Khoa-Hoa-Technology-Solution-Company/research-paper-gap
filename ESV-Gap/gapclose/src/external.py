"""External evidence recovery: OpenAlex search + sentence-level NLI on retrieved abstracts.

Results are cached in outputs/external/ so every question is fetched and scored once.
The SciFact gold documents themselves may be returned by OpenAlex; this is intended (the
local corpus is incomplete, the literature is not).  `gold_hit` records whether a returned
title matches a gold witness title, for analysis only - it is never used by any decision.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from common import ENGLISH_STOP_WORDS, OUT, write_json

CACHE = OUT / "external"
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:60]


def _keywords(t):
    return " ".join(w for w in re.findall(r"[A-Za-z0-9\-]+", t) if w.lower() not in ENGLISH_STOP_WORDS)


def _abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


class BudgetExhausted(RuntimeError):
    pass


def _get(url):
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 429 and b"budget" in e.read().lower():
                raise BudgetExhausted(url.split("?")[0]) from e
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def _doc(id_, title, year, abstract):
    return {"id": id_, "title": title or "", "year": year,
            "sentences": [s for s in SENT.split(abstract) if len(s) > 20][:30]}


def fetch(claim: str, source: str = "openalex", n: int = 10) -> list[dict]:
    words = _keywords(claim)
    if source == "openalex":
        key = os.environ.get("OPENALEX_API_KEY")
        url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(words[:400])
               + f"&per-page={n}&select=id,title,publication_year,abstract_inverted_index"
               + (f"&api_key={key}" if key else ""))
        r = _get(url)
        if r is None:
            return None
        return [_doc(w["id"], w.get("title"), w.get("publication_year"), _abstract(w.get("abstract_inverted_index")))
                for w in r.get("results", []) if w.get("abstract_inverted_index")]
    if source == "epmc":
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
               + urllib.parse.quote(words if os.environ.get("EPMC_AND") else " OR ".join(words.split()))
               + f"&resultType=core&pageSize={n}&format=json")
        r = _get(url)
        if r is None:
            return None
        r = r.get("resultList", {}).get("result", [])
        return [_doc(w.get("id"), w.get("title"), w.get("pubYear"), re.sub(r"<[^>]+>", " ", w["abstractText"]))
                for w in r if w.get("abstractText")]
    raise ValueError(source)


def fetch_all(claims: list[dict], corpus: dict, name: str, source: str = "openalex") -> dict:
    path = CACHE / f"{source}_{name}.json"
    cache = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    for i, c in enumerate(claims):
        key = str(c["id"])
        if key in cache:
            continue
        try:
            docs = fetch(c["claim"], source)
        except BudgetExhausted:
            write_json(path, cache)
            raise
        if docs is None:  # request failed: leave uncached so it is retried later
            continue
        gold = {_norm(corpus[d]["title"]) for d in c["witness"]}
        cache[key] = {"docs": docs, "gold_hit": any(_norm(d["title"]) in gold for d in docs)}
        time.sleep(0.15)
        if i % 50 == 0:
            write_json(path, cache)
            print(f"  fetched {i+1}/{len(claims)}", flush=True)
    write_json(path, cache)
    return cache


def score_all(claims: list[dict], cache: dict, nli, name: str, tag: str, batch: int = 32,
              source: str = "openalex") -> dict:
    """Sentence-level NLI [p_ent, p_con, p_neu] for every retrieved sentence (cached)."""
    path = CACHE / f"nli_{name}{tag}.json" if source == "openalex" else CACHE / f"nli_{source}_{name}{tag}.json"
    done = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    lab = {v.lower(): k for k, v in nli.id2label.items()}
    pairs, owners = [], []
    for c in claims:
        key = str(c["id"])
        if key in done:
            continue
        for di, d in enumerate(cache[key]["docs"]):
            for si, s in enumerate(d["sentences"]):
                pairs.append((s, c["claim"]))
                owners.append((key, di, si))
    if pairs:
        t0 = time.time()
        probs = nli.predict(pairs, batch_size=batch)
        print(f"  external NLI: {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)
        for (key, di, si), p in zip(owners, probs):
            done.setdefault(key, {}).setdefault(str(di), []).append(
                [round(float(p[lab["entailment"]]), 5), round(float(p[lab["contradiction"]]), 5),
                 round(float(p[lab["neutral"]]), 5)])
        for c in claims:
            done.setdefault(str(c["id"]), {})
        write_json(path, done)
    return done


def external_scores(split: str, tag: str = "_large", alpha: float = 1.0, source: str = "openalex") -> dict[str, float]:
    """ESV score on external evidence: max over retrieved sentences of non-neutral NLI x scope^alpha."""
    from common import content_terms
    claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
    cache = json.load(open(CACHE / f"{source}_{split}.json", encoding="utf-8"))
    nli = json.load(open(CACHE / (f"nli_{split}{tag}.json" if source == "openalex" else f"nli_{source}_{split}{tag}.json"),
                         encoding="utf-8"))
    out = {}
    for c in claims:
        k = str(c["id"])
        q = content_terms(c["claim"])
        best = 0.0
        for di, d in enumerate(cache.get(k, {}).get("docs", [])):
            tt = content_terms(d["title"])
            for si, (pe, pc, _) in enumerate(nli.get(k, {}).get(str(di), [])):
                cov = len(q & (content_terms(d["sentences"][si]) | tt)) / len(q) if q else 1.0
                best = max(best, max(pe, pc) * cov ** alpha)
        out[k] = best
    return out
