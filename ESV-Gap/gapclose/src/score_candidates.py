"""Stage 1: retrieval + sentence-level NLI scoring (cached, model-agnostic).

For every claim we keep the top-K documents of a hybrid ranking (reciprocal
rank fusion of BM25 and a dense bi-encoder) over the *full* corpus, plus all
lexical/dense scores needed by retrieval-only baselines.  Sentence-level NLI
probabilities are computed for every sentence of those K documents.  K is set
larger than the verifier depth k so that the corpus-ablation stress test can
re-rank after removing witness documents without new model calls.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import BM25, OUT, tokens, write_json

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def rrf(*rank_lists, k: int = 60) -> dict[int, float]:
    s: dict[int, float] = {}
    for rl in rank_lists:
        for r, i in enumerate(rl):
            s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--nli", default=NLI_MODEL)
    ap.add_argument("--dense", default=DENSE_MODEL)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    ap.add_argument("--tag", default="", help="suffix for outputs/scores/<split><tag>.json")
    args = ap.parse_args()

    corpus = json.load(open(OUT / "bench" / "corpus.json", encoding="utf-8"))
    ids = list(corpus)
    texts = [corpus[d]["title"] + " " + " ".join(corpus[d]["sentences"]) for d in ids]
    bm = BM25([tokens(t) for t in texts])

    if args.backend == "onnx":
        from onnx_models import NLI, Encoder
        enc = Encoder(args.dense)
        nli = NLI(args.nli)
        id2label = nli.id2label
    else:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        enc = SentenceTransformer(args.dense, device=dev)
        nli = CrossEncoder(args.nli, device=dev)
        id2label = nli.model.config.id2label
    print(f"backend={args.backend}", flush=True)
    t0 = time.time()
    demb = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    print(f"encoded corpus in {time.time()-t0:.0f}s", flush=True)
    lab = {v.lower(): int(k) for k, v in id2label.items()}
    i_ent, i_con, i_neu = lab["entailment"], lab["contradiction"], lab["neutral"]

    for split in ("test", "calib"):
        claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
        qemb = enc.encode([c["claim"] for c in claims], batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False)
        rows = []
        pairs, owners = [], []
        for ci, c in enumerate(claims):
            b = bm.scores(tokens(c["claim"]))
            d = demb @ qemb[ci]
            br = np.argsort(-b)[: args.pool]
            dr = np.argsort(-d)[: args.pool]
            fused = rrf(br.tolist(), dr.tolist())
            top = sorted(fused, key=lambda i: -fused[i])[: args.K]
            row = {
                "id": c["id"],
                "bm25_top": [[ids[i], float(b[i])] for i in br[:20]],
                "dense_top": [[ids[i], float(d[i])] for i in dr[:20]],
                "hybrid_top": [[ids[i], float(fused[i])] for i in top],
                "nli": {},
            }
            for i in top:
                for si, s in enumerate(corpus[ids[i]]["sentences"]):
                    pairs.append((c["claim"], s))
                    owners.append((ci, ids[i], si))
            rows.append(row)
        print(f"{split}: {len(pairs)} NLI pairs", flush=True)
        t0 = time.time()
        # premise = abstract sentence, hypothesis = claim
        probs = nli.predict([(s, h) for h, s in pairs], batch_size=args.batch, apply_softmax=True,
                            show_progress_bar=False)
        print(f"{split}: NLI done in {time.time()-t0:.0f}s", flush=True)
        for (ci, did, si), p in zip(owners, probs):
            rows[ci]["nli"].setdefault(did, []).append(
                [round(float(p[i_ent]), 5), round(float(p[i_con]), 5), round(float(p[i_neu]), 5)])
        write_json(OUT / "scores" / f"{split}{args.tag}.json", {"dense_model": args.dense, "nli_model": args.nli,
                                                      "K": args.K, "rows": rows})


if __name__ == "__main__":
    main()
