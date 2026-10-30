"""Trace the three 2022 case-study pairs back to individual corpus records.

This verifies lexical counts and preserves evidence for reading. It is not a blinded,
independent human annotation study and does not estimate co-study label precision.

    python src/audit_examples.py
"""
import json
from pathlib import Path

import numpy as np

from build_index import SENT
from lexicon import _COMPILED

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("RIS", "semantic communication"),
    ("age of information", "digital twin"),
    ("ISAC", "wireless power transfer"),
]


def main():
    index = np.load(ROOT / "data" / "index.npz")
    concepts = list(index["concepts"])
    years = index["years"]
    groups = [{"pair": list(pair), "prior_records": [], "window_records": []} for pair in PAIRS]
    seen = set()
    for path in sorted((ROOT / "data").glob("corpus_T*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                work = json.loads(line)
                if not 2010 <= (work["year"] or 0) <= 2025:
                    continue
                if work["id"] in seen:
                    raise ValueError(f"Duplicate corpus record: {work['id']}")
                seen.add(work["id"])
                text = f"{work['title']} {work['abstract']}"
                for pair, group in zip(PAIRS, groups):
                    if not all(_COMPILED[c].search(text) for c in pair):
                        continue
                    sentences = [work["title"]] + SENT.split(work["abstract"])
                    matching = [s for s in sentences if all(_COMPILED[c].search(s) for c in pair)]
                    record = {**work, "corpus_topic": path.stem.removeprefix("corpus_"),
                              "url": f"https://openalex.org/{work['id']}",
                              "same_sentence": bool(matching), "matching_sentences": matching}
                    key = "prior_records" if work["year"] <= 2022 else "window_records"
                    group[key].append(record)
    for group in groups:
        a, b = (concepts.index(c) for c in group["pair"])
        group["prior_count"] = len(group["prior_records"])
        group["window_count"] = len(group["window_records"])
        group["same_sentence_count"] = sum(w["same_sentence"] for w in group["window_records"])
        assert group["prior_count"] == int(index["co"][years <= 2022, a, b].sum())
        assert group["window_count"] == int(index["co"][years >= 2023, a, b].sum())
        assert group["same_sentence_count"] == int(index["co_sent"][years >= 2023, a, b].sum())
        group["window_records"].sort(key=lambda w: (w["year"], w["id"]))
    output = {"cutoff": 2022, "horizon": 3, "unique_corpus_works": len(seen),
              "interpretation": "Lexical evidence trace; selected examples are illustrative, not label validation",
              "pairs": groups}
    path = ROOT / "outputs" / "example_evidence.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for group in groups:
        print(json.dumps({k: group[k] for k in ("pair", "prior_count", "window_count", "same_sentence_count")}))
        for work in group["window_records"]:
            print(work["id"], work["year"], "sentence=" + str(work["same_sentence"]), json.dumps(work["title"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
