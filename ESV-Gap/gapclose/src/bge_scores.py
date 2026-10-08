"""Modern retrieval baselines: BGE-large dense retrieval and a BGE cross-encoder reranker.

Writes outputs/scores/bge_<split>.json with, per question,
  bge_dense_top : top-20 corpus documents by BGE-large-en-v1.5 cosine similarity
  bge_rerank    : reranker relevance (sigmoid) for every document of the hybrid pool
Both feed max-over-available-documents scores, so they stay monotone under deletion.
"""
from __future__ import annotations

import json
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from common import OUT, write_json

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def session(repo, max_len):
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tok.enable_truncation(max_len)
    tok.enable_padding(pad_to_multiple_of=64)
    prov = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"), providers=prov)
    return tok, sess, {i.name for i in sess.get_inputs()}


def run(tok, sess, names, items, batch, pair=False):
    tok.no_padding()
    lens = [len(e.ids) for e in tok.encode_batch(items)]
    tok.enable_padding(pad_to_multiple_of=64)
    order = sorted(range(len(items)), key=lambda i: lens[i])
    out = [None] * len(items)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        fill = idx + [idx[-1]] * (batch - len(idx))
        enc = tok.encode_batch([items[j] for j in fill])
        f = {"input_ids": np.array([e.ids for e in enc], np.int64),
             "attention_mask": np.array([e.attention_mask for e in enc], np.int64)}
        if "token_type_ids" in names:
            f["token_type_ids"] = np.array([e.type_ids for e in enc], np.int64)
        y = sess.run(None, f)[0][: len(idx)]
        for j, v in zip(idx, y):
            out[j] = v
    return out


def main():
    corpus = json.load(open(OUT / "bench" / "corpus.json", encoding="utf-8"))
    ids = list(corpus)
    texts = [corpus[d]["title"] + ". " + " ".join(corpus[d]["sentences"]) for d in ids]

    tok, sess, names = session("Xenova/bge-large-en-v1.5", 512)
    t0 = time.time()
    H = run(tok, sess, names, texts, 16)
    D = np.stack([h[0] for h in H])  # CLS pooling
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    print(f"BGE corpus embeddings in {time.time()-t0:.0f}s", flush=True)

    pos = {d: i for i, d in enumerate(ids)}
    rtok, rsess, rnames = session("Xenova/bge-reranker-base", 512)
    for split in ("calib", "test"):
        claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
        rows = {str(r["id"]): r for r in json.load(open(OUT / "scores" / f"{split}_large.json", encoding="utf-8"))["rows"]}
        Q = np.stack([h[0] for h in run(tok, sess, names, [QUERY_PREFIX + c["claim"] for c in claims], 32)])
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        S = Q @ D.T
        pairs, owners = [], []
        for c in claims:
            for d, _ in rows[str(c["id"])]["hybrid_top"]:
                pairs.append((c["claim"], texts[pos[d]]))
                owners.append((str(c["id"]), d))
        t0 = time.time()
        logits = run(rtok, rsess, rnames, pairs, 16, pair=True)
        print(f"{split}: reranked {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)
        out = {}
        for ci, c in enumerate(claims):
            top = np.argsort(-S[ci])[:20]
            out[str(c["id"])] = {"bge_dense_top": [[ids[i], float(S[ci, i])] for i in top], "bge_rerank": []}
        for (k, d), z in zip(owners, logits):
            out[k]["bge_rerank"].append([d, float(1 / (1 + np.exp(-float(np.ravel(z)[0]))))])
        write_json(OUT / "scores" / f"bge_{split}.json", out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
