"""Shared utilities: tokenisation, BM25, IO, hashing."""
from __future__ import annotations

import hashlib
import os
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# scikit-learn's ENGLISH_STOP_WORDS (BSD-3), inlined to avoid a scipy dependency.
ENGLISH_STOP_WORDS = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all",
    "almost", "alone", "along", "already", "also", "although", "always", "am", "among",
    "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone",
    "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became",
    "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind",
    "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom",
    "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
    "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg",
    "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even",
    "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen",
    "fifty", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has",
    "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein",
    "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred",
    "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its",
    "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many",
    "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most",
    "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never",
    "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not",
    "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
    "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",
    "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem",
    "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something",
    "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than",
    "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter",
    "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus",
    "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un",
    "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were",
    "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas",
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without",
    "would", "yet", "you", "your", "yours", "yourself", "yourselves",
})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(os.environ["GAPCLOSE_OUT"]) if os.environ.get("GAPCLOSE_OUT") else ROOT / "outputs"

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOP = set(ENGLISH_STOP_WORDS) | {
    "study", "studies", "result", "results", "patients", "patient", "associated",
    "increase", "increases", "increased", "decrease", "decreased", "reduces",
    "reduced", "higher", "lower", "effect", "effects", "role", "level", "levels",
    "risk", "use", "used", "using", "cells", "cell",
}


def light_stem(tok: str) -> str:
    """Very light, deterministic suffix stripping (no external deps)."""
    for suf in ("ization", "ations", "ation", "ments", "ment", "ness", "ities", "ity",
                "ings", "ing", "ies", "es", "s", "ed"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            if suf == "ies":
                return tok[: -3] + "y"
            return tok[: -len(suf)]
    return tok


def tokens(text: str, stop: bool = True) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if stop and (t in ENGLISH_STOP_WORDS or len(t) < 2):
            continue
        out.append(light_stem(t))
    return out


def content_terms(text: str) -> set[str]:
    """Content terms used for co-mention / scope coverage (stricter stoplist)."""
    return {light_stem(t) for t in _TOKEN.findall(text.lower())
            if t not in STOP and len(t) > 2 and not t.isdigit()}


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.dl.mean())
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.inv: dict[str, list[int]] = {}
        for i, d in enumerate(self.tf):
            for t in d:
                self.inv.setdefault(t, []).append(i)

    def scores(self, q: list[str], mask: np.ndarray | None = None) -> np.ndarray:
        s = np.zeros(self.N)
        for t in set(q):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in self.inv[t]:
                f = self.tf[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        return s


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LogRegND:
    """L2-regularised logistic regression (C=1, standardised features, unpenalised bias)."""

    def fit(self, X, y, C: float = 1.0, iters: int = 100):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        A = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        w = np.zeros(A.shape[1])
        reg = np.full(A.shape[1], 1 / C)
        reg[0] = 1e-9
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y) + reg * w
            H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-10:
                break
        self.w = w
        return self

    def predict_proba(self, X):
        A = np.hstack([np.ones((len(X), 1)), (np.asarray(X, float) - self.mu) / self.sd])
        return 1 / (1 + np.exp(-A @ self.w))
