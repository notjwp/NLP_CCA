Index build = TF-IDF indexing or corpus embedding time. Time per claim = one query at a time, top-100, after a warm-up query; hybrid and rerankers include their first stage.

| System                                      |   Index build (s) |   Mean time per claim (ms) | Device   |
|:--------------------------------------------|------------------:|---------------------------:|:---------|
| TF-IDF cosine (lsc.lsc)                     |              1.00 |                       0.17 | cpu      |
| Dense: all-MiniLM-L6-v2                     |             13.03 |                       7.89 | cuda:0   |
| Dense: bge-small-en-v1.5                    |             47.30 |                      12.46 | cuda:0   |
| Hybrid: TF-IDF + bge (RRF)                  |             48.30 |                      14.48 | cuda:0   |
| TF-IDF → cross-encoder                      |              1.00 |                     484.23 | cuda:0   |
| TF-IDF → LLM zero-shot (Qwen3-0.6B)         |              1.00 |                    5728.74 | cuda:0   |
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |              1.00 |                    5789.13 | cuda:0   |
