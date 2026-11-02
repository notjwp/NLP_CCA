Index build = TF-IDF indexing or corpus embedding time. Time per claim = one query at a time, top-100, after a warm-up query; the LLM reranker includes its TF-IDF first stage.

| System                                      |   Index build (s) |   Mean time per claim (ms) | Device   |
|:--------------------------------------------|------------------:|---------------------------:|:---------|
| TF-IDF cosine (lsc.lsc)                     |              1.00 |                       0.17 | cpu      |
| Semantic search: bge-small-en-v1.5          |             47.30 |                      12.46 | cuda:0   |
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |              1.00 |                    5789.13 | cuda:0   |
