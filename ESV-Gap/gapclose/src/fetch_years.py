"""Publication year, date and DOI for every SciFact corpus document (Semantic Scholar batch API).

SciFact document ids are Semantic Scholar corpus ids. Writes <OUT>/bench/doc_meta.json:
{doc_id: {"year": int|None, "date": str|None, "doi": str|None, "s2_title": str|None}}.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from common import OUT, write_json

URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=year,title,publicationDate,externalIds"


def batch(ids):
    body = json.dumps({"ids": [f"CorpusId:{i}" for i in ids]}).encode()
    for attempt in range(8):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Semantic Scholar batch failed")


def main():
    corpus = json.load(open(OUT / "bench" / "corpus.json", encoding="utf-8"))
    path = OUT / "bench" / "doc_meta.json"
    meta = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    todo = [d for d in corpus if d not in meta]
    for i in range(0, len(todo), 500):
        ids = todo[i:i + 500]
        for d, x in zip(ids, batch(ids)):
            meta[d] = {"year": x.get("year") if x else None, "date": x.get("publicationDate") if x else None,
                       "doi": ((x.get("externalIds") or {}).get("DOI")) if x else None,
                       "s2_title": x.get("title") if x else None}
        write_json(path, meta)
        print(f"{min(i + 500, len(todo))}/{len(todo)}", flush=True)
        time.sleep(3)
    years = [m["year"] for m in meta.values() if m["year"]]
    print(f"docs with year: {len(years)}/{len(meta)}; range {min(years)} to {max(years)}")


if __name__ == "__main__":
    main()
