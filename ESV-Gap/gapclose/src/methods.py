"""Gap-closure decision methods operating on cached Stage-1 scores.

Each method maps (claim, available corpus) -> (score, witness) where a higher
score means "more evidence that the question is already closed".  `removed`
is the set of documents deleted in the corpus-incompleteness stress test.
"""
from __future__ import annotations

import numpy as np

from common import content_terms, numbers

VERIFY_K = 5  # documents passed to sentence-level verifiers


class Ctx:
    """Per-split context with precomputed lexical sets."""

    def __init__(self, corpus: dict, claims: list[dict], rows: list[dict]):
        self.corpus = corpus
        self.claims = {c["id"]: c for c in claims}
        self.rows = {r["id"]: r for r in rows}
        self.doc_terms = {d: content_terms(v["title"] + " " + " ".join(v["sentences"]))
                          for d, v in corpus.items()}
        self.sent_terms = {d: [content_terms(s) for s in v["sentences"]] for d, v in corpus.items()}
        self.title_terms = {d: content_terms(v["title"]) for d, v in corpus.items()}
        self.doc_nums = {d: numbers(v["title"] + " " + " ".join(v["sentences"])) for d, v in corpus.items()}
        # inverted index for co-mention over the full corpus
        self.inv: dict[str, set[str]] = {}
        for d, ts in self.doc_terms.items():
            for t in ts:
                self.inv.setdefault(t, set()).add(d)


def _avail(lst, removed, n=None):
    out = [x for x in lst if x[0] not in removed]
    return out[:n] if n else out


# ---------------------------------------------------------------- baselines
def comention(ctx: Ctx, cid, removed=frozenset()):
    """KG/co-occurrence style: fraction of claim content terms co-mentioned in one abstract."""
    q = content_terms(ctx.claims[cid]["claim"])
    if not q:
        return 0.0, None
    cnt: dict[str, int] = {}
    for t in q:
        for d in ctx.inv.get(t, ()):
            if d not in removed:
                cnt[d] = cnt.get(d, 0) + 1
    if not cnt:
        return 0.0, None
    d = max(cnt, key=lambda x: (cnt[x], x))
    sents = ctx.sent_terms[d]
    si = int(np.argmax([len(q & s) for s in sents])) if sents else 0
    return cnt[d] / len(q), (d, si)


def bm25_thr(ctx: Ctx, cid, removed=frozenset()):
    top = _avail(ctx.rows[cid]["bm25_top"], removed, 1)
    return (top[0][1], (top[0][0], None)) if top else (0.0, None)


def dense_thr(ctx: Ctx, cid, removed=frozenset()):
    top = _avail(ctx.rows[cid]["dense_top"], removed, 1)
    return (top[0][1], (top[0][0], None)) if top else (0.0, None)


def _sentences(ctx: Ctx, cid, removed, k=VERIFY_K):
    """Yield (doc, sent_idx, p_ent, p_con, p_neu) over the top-k available hybrid docs."""
    row = ctx.rows[cid]
    for d, _ in _avail(row["hybrid_top"], removed, k):
        for si, (pe, pc, pn) in enumerate(row["nli"].get(d, [])):
            yield d, si, pe, pc, pn


def nli_verify(ctx: Ctx, cid, removed=frozenset()):
    """Retrieve-then-verify without scope control: max non-neutral NLI mass."""
    best, wit = 0.0, None
    for d, si, pe, pc, pn in _sentences(ctx, cid, removed):
        s = max(pe, pc)
        if s > best:
            best, wit = s, (d, si)
    return best, wit


# ---------------------------------------------------------------- proposed
def scope_factor(ctx: Ctx, cid, d, si) -> float:
    """Scope agreement between the question and ONE witness sentence (+ its title).

    * facet coverage: share of the claim's content terms present in the witness
      sentence or the title of the same abstract (no pooling across documents);
    * numeric scope: every number stated in the claim must occur in that abstract.
    """
    c = ctx.claims[cid]["claim"]
    q = content_terms(c)
    if not q:
        return 1.0
    local = ctx.sent_terms[d][si] | ctx.title_terms[d]
    cov = len(q & local) / len(q)
    qn = numbers(c)
    num_ok = 1.0 if not qn or qn <= ctx.doc_nums[d] else 0.0
    return cov * num_ok


def esv_scope(ctx: Ctx, cid, removed=frozenset(), alpha: float = 0.5, k: int = VERIFY_K):
    """ESV-Scope: witness score = NLI non-neutral mass x scope^alpha, single sentence."""
    best, wit = 0.0, None
    for d, si, pe, pc, pn in _sentences(ctx, cid, removed, k):
        s = max(pe, pc) * (scope_factor(ctx, cid, d, si) ** alpha)
        if s > best:
            best, wit = s, (d, si)
    return best, wit


def esv_nearmiss(ctx: Ctx, cid, removed=frozenset()):
    """Related-but-not-closing evidence: strongest *topical* support without a witness.

    Used only by the abstention policy: high relevance of non-closing documents
    means the question sits in a studied neighbourhood where a missed witness
    is plausible, so OPEN should be withheld.
    """
    top = _avail(ctx.rows[cid]["dense_top"], removed, 3)
    return float(np.mean([s for _, s in top])) if top else 0.0


def esv_features(ctx: Ctx, cid, removed=frozenset(), k: int = VERIFY_K):
    """Verifier + scope features of the top-k witness sentences (no retrieval scores).

    Returns (features, witness) where the witness is the best scope-weighted sentence.
    """
    per_doc: dict[str, float] = {}
    best, wit, b_pe, b_pc, b_scope = 0.0, None, 0.0, 0.0, 0.0
    max_pe = max_pc = 0.0
    for d, si, pe, pc, pn in _sentences(ctx, cid, removed, k):
        sc = scope_factor(ctx, cid, d, si)
        s = max(pe, pc) * sc ** 0.5
        per_doc[d] = max(per_doc.get(d, 0.0), s)
        max_pe, max_pc = max(max_pe, pe), max(max_pc, pc)
        if s > best:
            best, wit, b_pe, b_pc, b_scope = s, (d, si), pe, pc, sc
    top = sorted(per_doc.values(), reverse=True)
    f = [best, max_pe, max_pc, b_pe, b_pc, b_scope,
         float(np.mean(top[:3])) if top else 0.0,
         sum(v > 0.5 for v in top) / k]
    return np.array(f), wit


class ESVLearned:
    """ESV with a logistic head over esv_features, fitted on the calibration split only."""

    def __init__(self, k: int = VERIFY_K):
        self.k = k

    def fit(self, ctx: Ctx, claims: list[dict], y):
        from common import LogRegND
        X = np.stack([esv_features(ctx, c["id"], k=self.k)[0] for c in claims])
        self.lr = LogRegND().fit(X, y)
        return self

    def __call__(self, ctx: Ctx, cid, removed=frozenset()):
        f, wit = esv_features(ctx, cid, removed, self.k)
        return float(self.lr.predict_proba(f[None])[0]), wit


METHODS = {
    "CoMention": comention,
    "BM25-thr": bm25_thr,
    "Dense-thr": dense_thr,
    "NLI-verify": nli_verify,
    "ESV-Scope": esv_scope,
}
