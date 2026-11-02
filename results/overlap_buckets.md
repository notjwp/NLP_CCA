Mean nDCG@10 per lexical-overlap tercile. Overlap = share of a query's index terms (analyzer of the selected TF-IDF config: lsc.lsc + stopwords + stemming) that occur in its relevant abstract(s).

| bucket   | overlap range   |   n |   TF-IDF cosine (lsc.lsc) |   Semantic search: bge-small-en-v1.5 |   TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |
|:---------|:----------------|----:|--------------------------:|-------------------------------------:|----------------------------------------------:|
| low      | 0.00–0.45       | 100 |                     0.369 |                                0.438 |                                         0.530 |
| medium   | 0.47–0.75       | 100 |                     0.719 |                                0.785 |                                         0.835 |
| high     | 0.75–1.00       | 100 |                     0.888 |                                0.914 |                                         0.951 |
