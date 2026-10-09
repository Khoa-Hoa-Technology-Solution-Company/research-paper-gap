"""Local LLM-judge baseline (Qwen2.5-1.5B-Instruct, ONNX fp16, no API).

For every (question, pooled document) pair the model is asked whether the abstract contains
evidence that supports or refutes the claim, and the closure score is P(Yes) / (P(Yes) + P(No))
at the first answer token.  A question's score is the maximum over available documents, which
keeps it monotone under deletion.  Writes <OUT>/scores/llm_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from common import OUT, write_json

REPO = "onnx-community/Qwen2.5-1.5B-Instruct"
PROMPT = ("<|im_start|>system\nYou are a careful scientific fact checker.<|im_end|>\n"
          "<|im_start|>user\nText: {doc}\n\nDoes this text contain evidence that supports or refutes the "
          "following claim?\nClaim: {claim}\nAnswer with Yes or No.<|im_end|>\n<|im_start|>assistant\n")


class Judge:
    def __init__(self, max_doc_tokens=320):
        cfg = json.load(open(hf_hub_download(REPO, "config.json"), encoding="utf-8"))
        self.L, self.kv = cfg["num_hidden_layers"], cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        self.tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
        hf_hub_download(REPO, "onnx/model_fp16.onnx_data")
        prov = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(hf_hub_download(REPO, "onnx/model_fp16.onnx"), providers=prov)
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.kv_dtype = np.float16 if "float16" in self.inputs["past_key_values.0.key"].type else np.float32
        self.yes = [self.tok.token_to_id(t) for t in ("Yes", "ĠYes")]
        self.no = [self.tok.token_to_id(t) for t in ("No", "ĠNo")]
        self.yes, self.no = [t for t in self.yes if t is not None], [t for t in self.no if t is not None]
        self.max_doc_tokens = max_doc_tokens
        self.pad = self.tok.token_to_id("<|endoftext|>")

    def _ids(self, claim, doc):
        d = self.tok.encode(doc).ids[: self.max_doc_tokens]
        doc = self.tok.decode(d)
        return self.tok.encode(PROMPT.format(doc=doc, claim=claim)).ids

    def score(self, pairs, batch=2, bucket=64):
        enc = [self._ids(c, d) for c, d in pairs]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        out = np.zeros(len(enc), dtype=np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            fill = idx + [idx[-1]] * (batch - len(idx))
            T = max(len(enc[i]) for i in fill)
            T = ((T + bucket - 1) // bucket) * bucket
            ids = np.full((batch, T), self.pad, np.int64)
            mask = np.zeros((batch, T), np.int64)
            for r, i in enumerate(fill):  # left padding: the last position is the answer slot
                ids[r, T - len(enc[i]):] = enc[i]
                mask[r, T - len(enc[i]):] = 1
            pos = np.clip(np.cumsum(mask, 1) - 1, 0, None)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "position_ids" in self.inputs:
                feed["position_ids"] = pos
            for l in range(self.L):
                for kv in ("key", "value"):
                    feed[f"past_key_values.{l}.{kv}"] = np.zeros((batch, self.kv, 0, self.hd), self.kv_dtype)
            logits = self.sess.run(["logits"], feed)[0][:, -1, :].astype(np.float32)
            ly = np.logaddexp.reduce(logits[:, self.yes], axis=1)
            ln = np.logaddexp.reduce(logits[:, self.no], axis=1)
            out[idx] = (1 / (1 + np.exp(ln - ly)))[: len(idx)]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_large", help="score file whose hybrid pool is judged")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    args = ap.parse_args()
    corpus = json.load(open(OUT / "bench" / "corpus.json", encoding="utf-8"))
    judge = Judge()
    for split in ("calib", "test"):
        claims = json.load(open(OUT / "bench" / f"{split}.json", encoding="utf-8"))
        if args.limit:
            claims = claims[: args.limit]
        rows = {str(r["id"]): r for r in json.load(open(OUT / "scores" / f"{split}{args.tag}.json", encoding="utf-8"))["rows"]}
        pairs, owners = [], []
        for c in claims:
            for d, _ in rows[str(c["id"])]["hybrid_top"]:
                pairs.append((c["claim"], corpus[d]["title"] + ". " + " ".join(corpus[d]["sentences"])))
                owners.append((str(c["id"]), d))
        t0 = time.time()
        p = judge.score(pairs)
        print(f"{split}: judged {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)
        res = {}
        for (k, d), v in zip(owners, p):
            res.setdefault(k, []).append([d, float(v)])
        if not args.limit:
            write_json(OUT / "scores" / f"llm_{split}.json", res)
        else:
            y = [c["status"] == "CLOSED" for c in claims]
            s = [max(v for _, v in res[str(c["id"])]) for c in claims]
            print("smoke test (question, gold, max P(Yes)):", list(zip([c["claim"][:40] for c in claims], y, np.round(s, 3)))[:6])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
