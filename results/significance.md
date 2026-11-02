Each system vs. the TF-IDF baseline on nDCG@10, on the same test queries: two-sided Wilcoxon signed-rank test (zero differences dropped), Holm correction over the 2 comparisons. W/T/L = queries where the system scores higher / equal / lower.

| system                                      |   TF-IDF mean |   system mean |   delta | W/T/L     |   wilcoxon_p |    holm_p | significant (α=0.05)   |
|:--------------------------------------------|--------------:|--------------:|--------:|:----------|-------------:|----------:|:-----------------------|
| Semantic search: bge-small-en-v1.5          |        0.6589 |        0.7127 | 0.05384 | 85/164/51 |     0.002744 | 0.002744  | yes                    |
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |        0.6589 |        0.7721 | 0.1132  | 96/171/33 |     1.02e-11 | 2.041e-11 | yes                    |
