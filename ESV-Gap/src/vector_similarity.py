"""Small NumPy-only vector helpers for restricted Windows environments."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "will", "with",
}


def cosine_similarity_matrix(vectors) -> np.ndarray:
    """Pairwise cosine similarity without scikit-learn native extensions."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=norms > 0,
    )
    return normalized @ normalized.T


def tfidf_matrix(texts: Iterable[str], max_features: int = 10000) -> np.ndarray:
    """Build a deterministic dense TF-IDF matrix for small paper corpora."""
    tokenized = []
    document_frequency: Counter[str] = Counter()
    corpus_frequency: Counter[str] = Counter()
    for text in texts:
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", str(text).casefold())
            if len(token) > 1 and token not in _ENGLISH_STOPWORDS
        ]
        counts = Counter(tokens)
        tokenized.append(counts)
        document_frequency.update(counts.keys())
        corpus_frequency.update(counts)

    vocabulary = sorted(
        corpus_frequency,
        key=lambda term: (-corpus_frequency[term], term),
    )[:max(1, int(max_features))]
    term_index = {term: index for index, term in enumerate(vocabulary)}
    matrix = np.zeros((len(tokenized), len(vocabulary)), dtype=np.float64)
    document_count = len(tokenized)
    for row, counts in enumerate(tokenized):
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            column = term_index.get(term)
            if column is None:
                continue
            term_frequency = count / total
            inverse_document_frequency = math.log(
                (1 + document_count) / (1 + document_frequency[term])
            ) + 1.0
            matrix[row, column] = term_frequency * inverse_document_frequency
    return matrix
