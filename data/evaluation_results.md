# Experimental Evaluation Results

This report summarizes the comparison between our proposed **KG-TABI** framework and the baselines.

| Method | Total Gaps | Unique Gaps | Avg Words/Claim | NLI Entailment Rate | Jaccard Overlap vs KG-TABI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KG-TABI (Ours)** | 31 | 29 | 20.3 | 100.0% | N/A |
| **B1 (Mulla RAG)** | 5 | 5 | 35.4 | 0.0% | 0.185 |
| **B2 (Simple LLM)** | 15 | 15 | 17.7 | 66.7% | 0.162 |
| **B3 (GAPMAP Text)** | 5 | 5 | 26.8 | 100.0% | 0.158 |
