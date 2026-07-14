# Experimental Evaluation Results

This report summarizes the comparison between our proposed **KG-TABI** framework and the baselines.

| Method | Total Gaps | Unique Gaps | Avg Words/Claim | NLI Entailment Rate | Jaccard Overlap vs KG-TABI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KG-TABI (Ours)** | 9 | 9 | 27.7 | 100.0% | N/A |
| **B1 (Mulla RAG)** | 21 | 21 | 61.2 | 71.4% | 0.106 |
| **B2 (Simple LLM)** | 63 | 62 | 23.2 | 63.5% | 0.081 |
| **B3 (GAPMAP Text)** | 21 | 21 | 25.2 | 57.1% | 0.122 |
| **B4 (GraphRAG)** | 21 | 21 | 29.5 | 76.2% | 0.122 |
| **B5 (LightRAG)** | 21 | 21 | 32.0 | 61.9% | 0.126 |
| **B6 (HippoRAG)** | 21 | 21 | 30.2 | 66.7% | 0.103 |
