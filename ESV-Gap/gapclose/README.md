# GapClose-SciFact (ICAI-FAI 2026 experiment)

Scope-aware gap-closure verification with abstention, evaluated on SciFact expert labels.

1. Download SciFact: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
   and extract to `data/scifact/` (so that `data/scifact/data/corpus.jsonl` exists).
2. `pip install -r requirements.txt` (ONNX Runtime backend; torch is optional, `--backend torch`).
3. From `src/`:

```
python build_benchmark.py
python score_candidates.py --nli MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --batch 32 --tag _large
python cv_dev.py --tag _large      # 5-fold CV on the calibration split only; selects the ESV variant
python evaluate.py --tag _large    # frozen config, single pass over the test split + stress test
```

Outputs go to `outputs/` (bench/, scores/, results/). `results/main_run1_small_untuned.json` is the
first test run (DeBERTa-v3-small, untuned ESV defaults) and is kept for disclosure.

Protocol: every design choice (ESV alpha, verifier depth k, scope-only vs learned head) is made from
`cv_dev.py` on SciFact-train; SciFact-dev (test) is evaluated once per verifier model.
