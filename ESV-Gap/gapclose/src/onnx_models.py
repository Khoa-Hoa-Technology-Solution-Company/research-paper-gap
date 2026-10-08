"""ONNX Runtime backend for the bi-encoder and the NLI cross-encoder.

Used instead of sentence-transformers/torch where torch DLLs cannot be loaded
(e.g. Windows Smart App Control).  Weights are the Xenova ONNX exports of the
same checkpoints (sentence-transformers/all-MiniLM-L6-v2,
cross-encoder/nli-deberta-v3-small).
"""
from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ONNX_REPOS = {
    "sentence-transformers/all-MiniLM-L6-v2": "Xenova/all-MiniLM-L6-v2",
    "cross-encoder/nli-deberta-v3-small": "Xenova/nli-deberta-v3-small",
    # official ONNX export by the model author (MNLI, FEVER-NLI, ANLI, LingNLI, WANLI; no SciFact)
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli":
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}


def _session(repo: str, max_len: int):
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tok.enable_truncation(max_len)
    # fixed length buckets: DirectML re-compiles the graph for every new input shape
    tok.enable_padding(pad_to_multiple_of=64)
    providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"), providers=providers)
    return tok, sess, {i.name for i in sess.get_inputs()}


def _feeds(enc, names):
    f = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
         "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
    if "token_type_ids" in names:
        f["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
    return f


class Encoder:
    def __init__(self, name: str, max_len: int = 256):
        self.tok, self.sess, self.names = _session(ONNX_REPOS[name], max_len)

    def encode(self, texts, batch_size=64, normalize_embeddings=True, **_):
        out = []
        for i in range(0, len(texts), batch_size):
            f = _feeds(self.tok.encode_batch(texts[i:i + batch_size]), self.names)
            h = self.sess.run(None, f)[0]
            m = f["attention_mask"][..., None].astype(np.float32)
            e = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            if normalize_embeddings:
                e /= np.linalg.norm(e, axis=1, keepdims=True)
            out.append(e)
        return np.concatenate(out)


class NLI:
    def __init__(self, name: str, max_len: int = 512):
        repo = ONNX_REPOS[name]
        self.tok, self.sess, self.names = _session(repo, max_len)
        cfg = json.load(open(hf_hub_download(repo, "config.json"), encoding="utf-8"))
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}

    def predict(self, pairs, batch_size=64, **_):
        # tokenise once, sort by token length so padded batches stay small, and keep the
        # batch dimension fixed (last batch is padded with repeats) to limit input shapes
        self.tok.no_padding()
        lens = [len(e.ids) for e in self.tok.encode_batch(list(pairs))]
        self.tok.enable_padding(pad_to_multiple_of=64)
        order = sorted(range(len(pairs)), key=lambda i: lens[i])
        probs = np.zeros((len(pairs), len(self.id2label)), dtype=np.float32)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            fill = idx + [idx[-1]] * (batch_size - len(idx))
            f = _feeds(self.tok.encode_batch([pairs[j] for j in fill]), self.names)
            z = self.sess.run(None, f)[0][: len(idx)]
            z = np.exp(z - z.max(1, keepdims=True))
            probs[idx] = z / z.sum(1, keepdims=True)
        return probs
