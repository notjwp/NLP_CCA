Index build = TF-IDF indexing or corpus embedding time. Latency = one query at a time, top-100, after a warm-up query. Hybrid/rerank include their first-stage cost.

| System                                      |   Index build (s) |   Index size (MB) |   Model params (M) |   Mean latency (ms) |   p95 latency (ms) | Device   |
|:--------------------------------------------|------------------:|------------------:|-------------------:|--------------------:|-------------------:|:---------|
| TF-IDF cosine (lsc.lsc)                     |              1.00 |              5.10 |               0.00 |                0.17 |               0.25 | cpu      |
| Dense: all-MiniLM-L6-v2                     |             13.03 |              7.59 |              22.71 |                7.89 |               8.96 | cuda:0   |
| Dense: bge-small-en-v1.5                    |             47.30 |              7.59 |              33.36 |               12.46 |              13.56 | cuda:0   |
| Hybrid: TF-IDF + bge (RRF)                  |             48.30 |             12.69 |              33.36 |               14.48 |              15.38 | cuda:0   |
| TF-IDF → cross-encoder                      |              1.00 |              5.10 |              22.71 |              484.23 |             559.22 | cuda:0   |
| TF-IDF → LLM zero-shot (Qwen3-0.6B)         |              1.00 |              5.10 |             596.05 |             5728.74 |            6420.42 | cuda:0   |
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |              1.00 |              5.10 |             595.78 |             5789.13 |            6452.95 | cuda:0   |
