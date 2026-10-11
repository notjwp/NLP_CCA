Index build = TF-IDF indexing or corpus embedding time. Latency = one query at a time, top-100, after a warm-up query. Hybrid/rerank include their first-stage cost.

| System                     |   Index build (s) |   Index size (MB) |   Model params (M) |   Mean latency (ms) |   p95 latency (ms) | Device   |
|:---------------------------|------------------:|------------------:|-------------------:|--------------------:|-------------------:|:---------|
| TF-IDF cosine (lsc.lsc)    |              1.69 |              5.10 |               0.00 |                0.26 |               0.40 | cpu      |
| Dense: all-MiniLM-L6-v2    |             13.03 |              7.59 |              22.71 |               11.69 |              14.25 | cuda:0   |
| Dense: bge-small-en-v1.5   |             47.30 |              7.59 |              33.36 |               18.91 |              25.27 | cuda:0   |
| Hybrid: TF-IDF + bge (RRF) |             48.99 |             12.69 |              33.36 |               21.03 |              26.42 | cuda:0   |
| TF-IDF → cross-encoder     |              1.69 |              5.10 |              22.71 |              497.47 |             581.72 | cuda:0   |
