# v2 vs no-instruction control: inter-response divergence

Same cell definition and metric pipeline as evaluate_hard.py
(per-question mean pairwise cosine distance over all responses in the
(question, condition) cell; metric type matches results_ablation.json).

| Model | Metric | mean v2 | mean control | diff | q's v2>control | sign p |
|---|---|---|---|---|---|---|
| deepseek-r1-32b | tfidf_cosine | 0.7325 | 0.6294 | 0.1031 | 5/5 | 0.0625 |
| llama3.1-8b | ollama_cosine | 0.1537 | 0.0976 | 0.056 | 5/5 | 0.0625 |
| qwen3-32b | ollama_cosine | 0.1613 | 0.0522 | 0.1091 | 5/5 | 0.0625 |
| qwen3.5-122b | ollama_cosine | 0.9721 | 1.0 | -0.0279 | 0/5 | 0.0625 |

Reading: supports 'injection changes outputs vs no instruction' only on the
models whose metric discriminates; on Qwen3.5-122B the embedding metric
saturates (control near 1.0) and the comparison is uninformative.
