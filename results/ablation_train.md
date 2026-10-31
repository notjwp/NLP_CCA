TF-IDF variants on the SciFact **train** split (809 queries). Selected by nDCG@10: **lsc.lsc + stopwords + stemming**.

| config                                                 | scheme   | stemming   | stopwords   | lemmatization   | noun phrases   |   nDCG@10 |   MAP@100 |   MRR@10 |   P@10 |   R@10 |   R@100 | selected   |
|:-------------------------------------------------------|:---------|:-----------|:------------|:----------------|:---------------|----------:|----------:|---------:|-------:|-------:|--------:|:-----------|
| ntc.ntc, raw tokens                                    | ntc.ntc  | False      | False       | False           | False          |    0.5957 |    0.5468 |   0.5526 | 0.0842 | 0.7559 |  0.9077 | False      |
| ltc.ltc + stopwords                                    | ltc.ltc  | False      | True        | False           | False          |    0.6456 |    0.6029 |   0.6067 | 0.0875 | 0.7865 |  0.9149 | False      |
| ltc.ltc + stopwords + stemming                         | ltc.ltc  | True       | True        | False           | False          |    0.6688 |    0.6250 |   0.6300 | 0.0897 | 0.8097 |  0.9279 | False      |
| lnc.ltc + stopwords + stemming                         | lnc.ltc  | True       | True        | False           | False          |    0.6758 |    0.6343 |   0.6413 | 0.0896 | 0.8073 |  0.9221 | False      |
| lsc.lsc + stopwords + stemming                         | lsc.lsc  | True       | True        | False           | False          |    0.6789 |    0.6364 |   0.6423 | 0.0904 | 0.8141 |  0.9309 | True       |
| lsc.lsc + stopwords + POS lemmatization                | lsc.lsc  | False      | True        | True            | False          |    0.6644 |    0.6206 |   0.6259 | 0.0896 | 0.8043 |  0.9198 | False      |
| lsc.lsc + stopwords + stemming + noun phrases          | lsc.lsc  | True       | True        | False           | True           |    0.6595 |    0.6147 |   0.6181 | 0.0897 | 0.8063 |  0.9371 | False      |
| lsc.lsc + stopwords + POS lemmatization + noun phrases | lsc.lsc  | False      | True        | True            | True           |    0.6411 |    0.5978 |   0.6012 | 0.0875 | 0.7853 |  0.9285 | False      |
| scikit-learn TfidfVectorizer (reference = lsc.lsc)     | lsc.lsc  | True       | True        | False           | False          |    0.6789 |    0.6364 |   0.6423 | 0.0904 | 0.8141 |  0.9309 | False      |
