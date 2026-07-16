# Experimental Evaluation Results

This report summarizes the comparison between our proposed **KG-TABI** framework and the baselines.

| Method | Total Gaps | Unique Gaps | Avg Words/Claim | NLI Entailment Rate | Jaccard Overlap vs KG-TABI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KG-TABI (Ours)** | 46 | 45 | 24.4 | 76.1% | N/A |
| **B1 (Mulla RAG)** | 100 | 100 | 61.3 | 50.0% | 0.166 |
| **B2 (Simple LLM)** | 300 | 296 | 22.9 | 65.7% | 0.147 |
| **B3 (GAPMAP Text)** | 101 | 100 | 26.1 | 63.4% | 0.212 |
| **B4 (GraphRAG)** | 97 | 97 | 28.0 | 72.2% | 0.213 |
| **B5 (LightRAG)** | 100 | 98 | 30.6 | 83.0% | 0.186 |
| **B6 (HippoRAG)** | 92 | 88 | 31.6 | 79.3% | 0.204 |
