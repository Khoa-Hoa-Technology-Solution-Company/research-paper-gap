"""Download the 6G/IoT corpus from OpenAlex.

One cursor-paginated filter query per OpenAlex topic (primary topic only, so a work
is never fetched twice). Filter-only list calls cost 0.0001 USD per page of 200.
Output: data/corpus_<topic>.jsonl with id, year, title, abstract.

    OPENALEX_API_KEY=... python src/fetch_corpus.py
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# OpenAlex topics covering the INISCOM 2027 call (wireless PHY, 6G enablers, IoT, sensing).
TOPICS = {
    "T11458": "Advanced Wireless Communication Technologies",
    "T10125": "Advanced Wireless Communication Techniques",
    "T10148": "Advanced MIMO Systems Optimization",
    "T10936": "Millimeter-Wave Propagation and Modeling",
    "T10851": "Optical Wireless Communication Technologies",
    "T12042": "Satellite Communication Systems",
    "T11133": "UAV Applications and Optimization",
    "T10273": "IoT and Edge/Fog Computing",
    "T12079": "IoT Networks and Protocols",
    "T13535": "Wireless Sensor Networks and IoT",
    "T10080": "Energy Efficient Wireless Sensor Networks",
    "T10761": "Vehicular Ad Hoc Networks (VANETs)",
    "T10579": "Cognitive Radio Networks and Spectrum Sensing",
    "T11409": "Advanced Wireless Network Optimization",
    "T13553": "Age of Information Optimization",
}
YEARS = "2010-2025"


def abstract_text(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idx in inv.items() for i in idx]
    return " ".join(w for _, w in sorted(pos))


def get(url, tries=6):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            wait = 5 * (t + 1)
            print(f"  retry {t+1} in {wait}s: {e}", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url[:120]}")


def fetch_topic(tid):
    out = DATA / f"corpus_{tid}.jsonl"
    done = DATA / f"corpus_{tid}.done"
    if done.exists():
        return tid, sum(1 for _ in open(out, encoding="utf-8"))
    key = os.environ.get("OPENALEX_API_KEY", "")
    flt = f"primary_topic.id:{tid},publication_year:{YEARS},type:article,has_abstract:true"
    base = ("https://api.openalex.org/works?filter=" + urllib.parse.quote(flt, safe=":,")
            + "&per_page=200&select=id,title,publication_year,abstract_inverted_index"
            + (f"&api_key={key}" if key else ""))
    cursor, n = "*", 0
    with open(out, "w", encoding="utf-8") as f:
        while cursor:
            d = get(base + "&cursor=" + urllib.parse.quote(cursor))
            for w in d["results"]:
                f.write(json.dumps({"id": w["id"].rsplit("/", 1)[-1], "year": w["publication_year"],
                                    "title": w.get("title") or "",
                                    "abstract": abstract_text(w.get("abstract_inverted_index"))},
                                   ensure_ascii=False) + "\n")
            n += len(d["results"])
            cursor = d["meta"].get("next_cursor")
            if not d["results"]:
                break
            if n % 10000 < 200:
                print(f"{tid}: {n}/{d['meta']['count']}", flush=True)
    done.write_text(str(n))
    return tid, n


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for tid, n in ex.map(fetch_topic, TOPICS):
            print(f"DONE {tid} {TOPICS[tid]}: {n}", flush=True)
