Each system vs. the TF-IDF baseline on the same test queries. Two-sided Wilcoxon signed-rank (zero differences dropped) and paired t-test; Holm correction over the 6 comparisons per metric applied to the Wilcoxon p-values. W/T/L = queries where the system scores higher / equal / lower.

## nDCG@10

| system                                      |   TF-IDF mean |   system mean |    delta | W/T/L     |   wilcoxon_p |   ttest_p |    holm_p | significant (α=0.05)   |
|:--------------------------------------------|--------------:|--------------:|---------:|:----------|-------------:|----------:|----------:|:-----------------------|
| Dense: all-MiniLM-L6-v2                     |        0.6589 |        0.6451 | -0.01378 | 72/161/67 |    0.3619    | 0.4426    | 0.3619    | no                     |
| Dense: bge-small-en-v1.5                    |        0.6589 |        0.7127 |  0.05384 | 85/164/51 |    0.002744  | 0.001937  | 0.008231  | yes                    |
| Hybrid: TF-IDF + bge (RRF)                  |        0.6589 |        0.7215 |  0.06264 | 83/184/33 |    3.719e-08 | 9.983e-08 | 1.488e-07 | yes                    |
| TF-IDF → cross-encoder                      |        0.6589 |        0.6896 |  0.03078 | 74/171/55 |    0.04278   | 0.05995   | 0.08555   | no                     |
| TF-IDF → LLM zero-shot (Qwen3-0.6B)         |        0.6589 |        0.2757 | -0.3832  | 9/86/205  |    5.152e-33 | 2.026e-47 | 3.091e-32 | yes                    |
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |        0.6589 |        0.7721 |  0.1132  | 96/171/33 |    1.02e-11  | 1.188e-11 | 5.102e-11 | yes                    |

## R@100

| system                                      |   TF-IDF mean |   system mean |     delta | W/T/L     |   wilcoxon_p |   ttest_p |   holm_p | significant (α=0.05)   |
|:--------------------------------------------|--------------:|--------------:|----------:|:----------|-------------:|----------:|---------:|:-----------------------|
| Dense: all-MiniLM-L6-v2                     |        0.9299 |        0.925  | -0.004889 | 13/274/13 |    0.7145    | 0.7659    | 1        | no                     |
| Dense: bge-small-en-v1.5                    |        0.9299 |        0.9417 |  0.01178  | 15/275/10 |    0.503     | 0.4637    | 1        | no                     |
| Hybrid: TF-IDF + bge (RRF)                  |        0.9299 |        0.9683 |  0.03844  | 13/287/0  |    0.0006318 | 0.0005011 | 0.003791 | yes                    |
| TF-IDF → cross-encoder                      |        0.9299 |        0.9299 |  0        | 0/300/0   |    1         | 1         | 1        | no                     |
| TF-IDF → LLM zero-shot (Qwen3-0.6B)         |        0.9299 |        0.9299 |  0        | 0/300/0   |    1         | 1         | 1        | no                     |
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |        0.9299 |        0.9299 |  0        | 0/300/0   |    1         | 1         | 1        | no                     |
