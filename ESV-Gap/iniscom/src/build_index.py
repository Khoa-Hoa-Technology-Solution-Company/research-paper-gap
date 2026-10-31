"""Tag every work with lexicon concepts and build per-year concept and pair counts.

Output: data/index.npz with
  years            (Y,)        publication years covered
  docs             (Y,)        number of works per year
  occ              (Y, C)      works per year mentioning concept c
  co               (Y, C, C)   works per year mentioning both c and c' (title or abstract)
  co_sent          (Y, C, C)   works per year with c and c' in the same sentence (stricter label)
  co_sent_ns       (Y, C, C)   as co_sent, excluding works whose title marks a survey/review/tutorial
  concepts                     concept names (lexicon order)
plus data/work_ids.txt (OpenAlex IDs of the corpus) and outputs/corpus_stats.json.
"""
import glob
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from lexicon import _COMPILED, CONCEPTS, tag

ROOT = Path(__file__).resolve().parents[1]
Y0, Y1 = 2010, 2025
SURVEY = re.compile(r"\b(survey|review|tutorial|overview|roadmap|vision|perspective|trends|challenges and opportunities)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def process(path):
    years = np.arange(Y0, Y1 + 1)
    C = len(CONCEPTS)
    cid = {c: i for i, c in enumerate(CONCEPTS)}
    docs = np.zeros(len(years), int)
    occ = np.zeros((len(years), C), int)
    co = np.zeros((len(years), C, C), int)
    cs = np.zeros((len(years), C, C), int)
    cn = np.zeros((len(years), C, C), int)
    ids, n_tags = [], Counter()
    for line in open(path, encoding="utf-8"):
        w = json.loads(line)
        if not (Y0 <= (w["year"] or 0) <= Y1):
            continue
        ids.append(w["id"])
        y = w["year"] - Y0
        found = tag(f"{w['title']} {w['abstract']}")
        n_tags[len(found)] += 1
        docs[y] += 1
        if not found:
            continue
        ix = np.array(sorted(cid[c] for c in found))
        occ[y, ix] += 1
        co[y][np.ix_(ix, ix)] += 1
        if len(found) < 2:
            continue
        pairs = set()
        for s in [w["title"]] + SENT.split(w["abstract"]):
            here = sorted(cid[c] for c in found if _COMPILED[c].search(s))
            pairs.update((a, b) for i, a in enumerate(here) for b in here[i + 1:])
        survey = bool(SURVEY.search(w["title"]))
        for a, b in pairs:
            cs[y, a, b] += 1
            cs[y, b, a] += 1
            if not survey:
                cn[y, a, b] += 1
                cn[y, b, a] += 1
    return Path(path).stem.split("_")[1], ids, docs, occ, co, cs, n_tags, cn


def main():
    files = sorted(glob.glob(str(ROOT / "data" / "corpus_T*.jsonl")))
    with Pool(min(8, len(files))) as pool:
        parts = pool.map(process, files)
    years = np.arange(Y0, Y1 + 1)
    docs = sum(p[2] for p in parts)
    occ = sum(p[3] for p in parts)
    co = sum(p[4] for p in parts)
    cs = sum(p[5] for p in parts)
    cn = sum(p[7] for p in parts)
    for i in range(len(CONCEPTS)):
        cs[:, i, i] = occ[:, i]
    n_tags = sum((p[6] for p in parts), Counter())
    all_ids = sorted(i for p in parts for i in p[1])
    assert len(all_ids) == len(set(all_ids)), "duplicate works across topics"
    (ROOT / "data" / "work_ids.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    np.savez_compressed(ROOT / "data" / "index.npz", years=years, docs=docs, occ=occ, co=co, co_sent=cs, co_sent_ns=cn,
                        concepts=np.array(CONCEPTS))
    stats = {"works": int(docs.sum()), "per_topic": {p[0]: len(p[1]) for p in parts},
             "per_year": dict(zip(map(int, years), map(int, docs))), "concepts": len(CONCEPTS),
             "tagged_share": 1 - n_tags[0] / max(1, sum(n_tags.values())),
             "mean_concepts_per_work": sum(k * v for k, v in n_tags.items()) / max(1, sum(n_tags.values())),
             "concept_freq": {c: int(occ[:, i].sum()) for i, c in enumerate(CONCEPTS)},
             "pairs_doc": int(np.triu(co.sum(0), 1).sum()), "pairs_sentence": int(np.triu(cs.sum(0), 1).sum()),
             "pairs_sentence_nosurvey": int(np.triu(cn.sum(0), 1).sum())}
    json.dump(stats, open(ROOT / "outputs" / "corpus_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "concept_freq"}, indent=1))


if __name__ == "__main__":
    main()
