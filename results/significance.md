Each system vs. the TF-IDF baseline on nDCG@10, on the same test queries: two-sided Wilcoxon signed-rank test (zero differences dropped), Holm correction over the 6 comparisons. W/T/L = queries where the system scores higher / equal / lower.

| system                                      |   TF-IDF mean |   system mean |    delta | W/T/L     |   wilcoxon_p |    holm_p | significant (α=0.05)   |
|:--------------------------------------------|--------------:|--------------:|---------:|:----------|-------------:|----------:|:-----------------------|
| Dense: all-MiniLM-L6-v2                     |        0.6589 |        0.6451 | -0.01378 | 72/161/67 |    0.3619    | 0.3619    | no                     |
| Dense: bge-small-en-v1.5                    |        0.6589 |        0.7127 |  0.05384 | 85/164/51 |    0.002744  | 0.008231  | yes                    |
| Hybrid: TF-IDF + bge (RRF)                  |        0.6589 |        0.7215 |  0.06264 | 83/184/33 |    3.719e-08 | 1.488e-07 | yes                    |
| TF-IDF → cross-encoder                      |        0.6589 |        0.6896 |  0.03078 | 74/171/55 |    0.04278   | 0.08555   | no                     |
| TF-IDF → LLM zero-shot (Qwen3-0.6B)         |        0.6589 |        0.2757 | -0.3832  | 9/86/205  |    5.152e-33 | 3.091e-32 | yes                    |
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |        0.6589 |        0.7721 |  0.1132  | 96/171/33 |    1.02e-11  | 5.102e-11 | yes                    |
