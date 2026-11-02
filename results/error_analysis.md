# Error analysis (SciFact test, 300 claims)

A claim **fails** for a system when none of its relevant abstracts is in that system's top 10 (nDCG@10 = 0).

## 1. Failures by cause

Each claim is put in one category using only the claim and its relevant abstract(s), so the categories are the same for every system. Rules, applied in order: **spelling variant** = a claim word is written differently in the abstract (e.g. *Bcl2* / *BCL-2*); **vocabulary mismatch** = fewer than half of the claim's index terms occur in the abstract; **rare shared term** = a shared term occurs in at most 10 of the 5,183 abstracts; **common words only** = everything else. Cells: failures (share of the category's claims).

| category            |   claims | TF-IDF cosine (lsc.lsc)   | Semantic search: bge-small-en-v1.5   | TF-IDF → LLM reranker (Qwen3-Reranker-0.6B)   |
|:--------------------|---------:|:--------------------------|:-------------------------------------|:----------------------------------------------|
| spelling variant    |       15 | 2 (13%)                   | 0 (0%)                               | 0 (0%)                                        |
| vocabulary mismatch |       99 | 34 (34%)                  | 38 (38%)                             | 27 (27%)                                      |
| rare shared term    |      106 | 0 (0%)                    | 1 (1%)                               | 1 (1%)                                        |
| common words only   |       80 | 12 (15%)                  | 7 (9%)                               | 3 (4%)                                        |
| all claims          |      300 | 48 (16%)                  | 46 (15%)                             | 31 (10%)                                      |

**spelling variant**, e.g.
- [1088] “Silencing of Bcl2 is important for the maintenance and progression of tumors.” → *Antiapoptotic BCL-2 is required for maintenance of a model leukemia.*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) 32; Semantic search: bge-small-en-v1.5 1; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) 1.
- [1370] “Vitamin D deficiency is unrelated to birth weight.” → *Association between maternal serum 25-hydroxyvitamin D level and pregnancy and neonatal outcomes: systematic review and meta-analysis of observational studies.*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) 15; Semantic search: bge-small-en-v1.5 1; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) 1.

**vocabulary mismatch**, e.g.
- [535] “Hypertension is frequently observed in type 1 diabetes patients.” → *Glucose tolerance and blood pressure: long term follow up in middle aged men.*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) >100; Semantic search: bge-small-en-v1.5 2; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) >100.
- [1363] “Venules have a thinner or absent smooth layer compared to arterioles.” → *Scaffold-based three-dimensional human fibroblast culture provides a structural matrix that supports angiogenesis in infarcted heart tissue.*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) 2; Semantic search: bge-small-en-v1.5 >100; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) 6.

**rare shared term**, e.g.
- [914] “PPAR-RXRs can be activated by PPAR ligands.” → *Unique response pathways are established by allosteric interactions among nuclear hormone receptors*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) 7; Semantic search: bge-small-en-v1.5 18; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) 17.
- [294] “Crossover hot spots are not found within gene promoters in Saccharomyces cerevisiae.” → *Mapping Meiotic Single-Strand DNA Reveals a New Landscape of DNA Double-Strand Breaks in Saccharomyces cerevisiae*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) 2; Semantic search: bge-small-en-v1.5 9; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) 7.

**common words only**, e.g.
- [132] “Aspirin inhibits the production of PGE2.” → *Cyclooxygenase-Dependent Tumor Growth through Evasion of Immunity*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) >100; Semantic search: bge-small-en-v1.5 12; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) >100.
- [452] “Gene expression does not vary appreciably across genetically identical cells.” → *Nature, Nurture, or Chance: Stochastic Gene Expression and Its Consequences*. Rank of the relevant abstract: TF-IDF cosine (lsc.lsc) 43; Semantic search: bge-small-en-v1.5 1; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) 2.

## 2. LLM reranker: first-stage misses vs. reranking mistakes

The LLM reranker only reorders TF-IDF's top 100. A failure is either a relevant abstract TF-IDF never retrieved (no reranker can fix it) or one the reranker saw but placed below rank 10.

| system                                      |   failures |   first stage missed it |   reranker ranked it low |
|:--------------------------------------------|-----------:|------------------------:|-------------------------:|
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |         31 |                      20 |                       11 |

## 3. How the LLM answers

The LLM is asked whether an abstract meets the claim's information need and may only answer yes or no; its score is log P(yes)/P(no). Share of the TF-IDF candidates it answered *yes* (most candidates are not relevant: about 1 in 100 is):

| system                                      |   candidates judged | answered yes (P(yes) > 0.5)   | relevant candidates answered yes   |
|:--------------------------------------------|--------------------:|:------------------------------|:-----------------------------------|
| TF-IDF → LLM reranker (Qwen3-Reranker-0.6B) |               29889 | 2.3%                          | 75.0%                              |
