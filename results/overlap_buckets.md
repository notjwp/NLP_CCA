Mean nDCG@10 per lexical-overlap tercile. Overlap = share of a query's index terms (analyzer: stemming=True, stopwords=True) that occur in its relevant abstract(s). Spearman ρ between overlap and per-query nDCG@10: TF-IDF cosine (lsc.lsc) 0.60, Dense: all-MiniLM-L6-v2 0.48, Dense: bge-small-en-v1.5 0.54, Hybrid: TF-IDF + bge (RRF) 0.57, TF-IDF → cross-encoder 0.61.

| bucket   | overlap range   |   n |   TF-IDF cosine (lsc.lsc) |   Dense: all-MiniLM-L6-v2 |   Dense: bge-small-en-v1.5 |   Hybrid: TF-IDF + bge (RRF) |   TF-IDF → cross-encoder |
|:---------|:----------------|----:|--------------------------:|--------------------------:|---------------------------:|-----------------------------:|-------------------------:|
| low      | 0.00–0.45       | 100 |                     0.369 |                     0.371 |                      0.438 |                        0.448 |                    0.360 |
| medium   | 0.47–0.75       | 100 |                     0.719 |                     0.733 |                      0.785 |                        0.796 |                    0.792 |
| high     | 0.75–1.00       | 100 |                     0.888 |                     0.831 |                      0.914 |                        0.921 |                    0.917 |
