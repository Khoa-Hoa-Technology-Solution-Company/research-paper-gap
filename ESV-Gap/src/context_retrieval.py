"""Source-linked grounding-context retrieval under a fixed word budget.

The learned prior uses method quotations from other papers. Predictions are
proposals for review, never reviewed annotations or closure certificates.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


def terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold())


def windows(source: str, size: int = 96, stride: int = 80) -> list[dict]:
    if size < 1 or stride < 1 or stride > size:
        raise ValueError("Require 1 <= stride <= size")
    words = list(re.finditer(r"\S+", source))
    result = []
    for start in range(0, len(words), stride):
        end = min(len(words), start + size)
        a, b = words[start].start(), words[end - 1].end()
        result.append({"start": a, "end": b, "text": source[a:b],
                       "word_count": end - start})
        if end == len(words):
            break
    return result


def context_prior(training_claims: list[dict], source_texts: dict[str, str]) -> dict[str, float]:
    positive, background = Counter(), Counter()
    paper_ids = {row["citekey"] for row in training_claims}
    for row in training_claims:
        positive.update(terms(" ".join(row["context"])))
    for pid in sorted(paper_ids):
        background.update(terms(source_texts[pid]))
    vocabulary = set(background) | set(positive)
    if not vocabulary:
        return {}
    vp = len(vocabulary)
    ptotal, btotal = sum(positive.values()) + vp, sum(background.values()) + vp
    return {word: max(0., math.log(((freq + 1) / ptotal) /
                                  ((background[word] + 1) / btotal)))
            for word, freq in positive.items() if freq >= 2}


def lexical_scores(texts: list[str], query: str, *, kind: str = "bm25") -> list[float]:
    docs = [Counter(terms(t)) for t in texts]
    if not docs:
        return []
    df = Counter(word for doc in docs for word in doc)
    n = len(docs)
    query_bag = Counter(terms(query))
    average = sum(sum(d.values()) for d in docs) / n or 1
    if kind == "tfidf":
        idf = {t: math.log((1+n)/(1+freq))+1 for t, freq in df.items()}
        q = {t: (1+math.log(f))*idf.get(t, 0.) for t, f in query_bag.items()}
        qnorm = math.sqrt(sum(v*v for v in q.values())) or 1
        scores = []
        for doc in docs:
            vec = {t: (1+math.log(f))*idf[t] for t, f in doc.items()}
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1
            scores.append(sum(v*q.get(t, 0.) for t, v in vec.items()) / (norm*qnorm))
        return scores
    if kind != "bm25":
        raise ValueError("Unknown lexical score")
    result = []
    for doc in docs:
        score = 0.
        length = sum(doc.values())
        for word in sorted(query_bag):
            freq = doc[word]
            if freq:
                idf = math.log(1+(n-df[word]+.5)/(df[word]+.5))
                score += idf*freq*2.2/(freq+1.2*(.25+.75*length/average))
        result.append(score)
    return result


def _normalize(scores):
    maximum = max(scores, default=0.)
    return [s/maximum if maximum else 0. for s in scores]


def retrieve_context(source: str, query: str, *, prior: dict[str, float] | None = None,
                     method: str = "context_diverse", budget: int = 512) -> list[dict]:
    if budget < 1:
        raise ValueError("Positive word budget required")
    candidates = windows(source)
    texts = [row["text"] for row in candidates]
    bags = [set(terms(t)) for t in texts]
    lexical = _normalize(lexical_scores(texts, query,
                         kind="tfidf" if method == "tfidf" else "bm25"))
    prior = prior or {}
    context = _normalize([sum(prior.get(t, 0.) for t in terms(text)) /
                          max(1, len(terms(text))) for text in texts])
    if method in {"bm25", "tfidf", "bm25_diverse"}:
        scores = lexical
    elif method == "prior_only":
        scores = context
    elif method in {"context", "context_diverse"}:
        scores = [.5*a+.5*b for a, b in zip(lexical, context)]
    else:
        raise ValueError("Unknown retrieval method")
    diversify = method in {"context_diverse", "bm25_diverse"}
    selected, used, available = [], 0, set(range(len(candidates)))
    while available and used < budget:
        scored = []
        for index in sorted(available):
            row = candidates[index]
            if any(row["start"] < old["end"] and old["start"] < row["end"]
                   for old in selected):
                continue
            similarity = max((len(bags[index] & bags[old["window_id"]]) /
                              max(1, len(bags[index] | bags[old["window_id"]]))
                              for old in selected), default=0.)
            value = scores[index] - (.2*similarity if diversify else 0.)
            scored.append((value, -row["start"], index))
        if not scored:
            break
        _, _, index = max(scored)
        available.remove(index)
        row = dict(candidates[index])
        remaining = budget-used
        if row["word_count"] > remaining:
            words = list(re.finditer(r"\S+", row["text"]))
            row["end"] = row["start"] + words[remaining-1].end()
            row["text"] = source[row["start"]:row["end"]]
            row["word_count"] = remaining
        row.update(window_id=index, score=scores[index], reviewed=False)
        selected.append(row)
        used += row["word_count"]
    # Source order improves readability; neither gold text nor label is consulted.
    return sorted(selected, key=lambda row: row["start"])


def word_overlap(prediction: str, references: str) -> dict[str, float]:
    p, g = terms(prediction), terms(references)
    def ngram_score(n):
        pc = Counter(tuple(p[i:i+n]) for i in range(len(p)-n+1))
        gc = Counter(tuple(g[i:i+n]) for i in range(len(g)-n+1))
        overlap = sum((pc & gc).values())
        precision = overlap/max(1, sum(pc.values()))
        recall = overlap/max(1, sum(gc.values()))
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        return precision, recall, f1
    p1, r1, f1 = ngram_score(1)
    _, _, f2 = ngram_score(2)
    masks = {}
    for index, token in enumerate(g):
        masks[token] = masks.get(token, 0) | (1 << index)
    state = 0
    for token in p:
        x = masks.get(token, 0) | state
        state = x & ~(x - ((state << 1) | 1))
    length = state.bit_count()
    precision, recall = length/max(1, len(p)), length/max(1, len(g))
    fl = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return {"rouge1_f1": f1, "rouge2_f1": f2, "rougel_f1": fl,
            "unigram_precision": p1, "unigram_recall": r1}
