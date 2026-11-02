# Classical vs. Semantic Information Retrieval: TF-IDF with Cosine Similarity vs. Semantic Search and an LLM on BEIR SciFact

## 1. Aim

Build a classical Information Retrieval (IR) system using **TF-IDF term weighting and cosine
similarity**, then measure its retrieval effectiveness against an **LLM / semantic-search approach**
built on pretrained transformer language models. The questions are:

1. How good is a well-configured TF-IDF system on a realistic scientific retrieval task?
2. Do semantic search (embeddings) and an LLM reranker beat it, by how much, and is the gap
   statistically reliable?
3. *Where* does each approach win, and why?
4. What does the extra effectiveness cost in time and hardware?

Every number in this report comes from the files the pipeline writes to [`results/`](results/)
(`uv run irsys evaluate && uv run irsys analyze`).

## 2. Dataset: BEIR SciFact

SciFact (Wadden et al., 2020), in its BEIR packaging (Thakur et al., 2021), pairs short scientific
**claims** (the queries) with the **research abstracts** that support or refute them.

| | |
|---|---|
| Corpus | 5,183 abstracts (title + abstract; on average 146 index terms after preprocessing) |
| Test queries | 300 claims (on average 9.2 index terms) |
| Train queries | 809 claims, disjoint from test, used **only** to choose the TF-IDF configuration |
| Relevance judgments | binary; 339 judgments over the 300 test queries (277 queries have exactly one relevant abstract) |

It is a good test bed for this comparison. Claims are written in different words from the abstracts
that answer them (for example, a claim about "silencing of Bcl2" is answered by an abstract on
"antiapoptotic BCL-2"), yet both use dense technical vocabulary that rewards exact term matching.

## 3. Methods

Three systems answer every claim:

| System | Approach |
|---|---|
| **TF-IDF + cosine similarity** | Classical: matches words, rare words count more (built from scratch) |
| **Semantic search (bge-small)** | Matches meaning: compares embedding vectors |
| **LLM reranker (Qwen3-Reranker)** | An LLM reads the claim with each of TF-IDF's top 100 abstracts and judges relevance |

### 3.1 The classical system: TF-IDF + cosine similarity (built from scratch)

**Text analysis** (`src/irsys/preprocess.py`). Lowercase → split into `[a-z0-9]+` tokens (numbers are
kept, since claims often quote figures) → remove English stopwords → Porter stemming.

Two classical NLP options build on **part-of-speech (POS) tagging**. NLTK's averaged-perceptron tagger
(Bird et al., 2009) labels every token with a Penn Treebank tag (Marcus et al., 1993), with stopwords
still in place so the tagger sees the full context:

- **POS-based lemmatization** replaces stemming. Each word becomes its WordNet dictionary form for its
  part of speech (Miller, 1995): *studies* (NNS) → *study*, *increased* (VBD) → *increase*, and *meeting*
  stays *meeting* as a noun but becomes *meet* as a verb.
- **Noun-phrase chunking** (shallow parsing) runs NLTK's regular-expression chunk parser with the grammar
  `NP: {<JJ.*|NN.*>*<NN.*>}`. That is the adjective/noun-ending-in-a-noun pattern of technical terms
  (Justeson & Katz, 1995). For example, *"LDL cholesterol has no involvement in the development of
  cardiovascular disease"* is parsed as [LDL cholesterol] … [involvement] … [development] … [cardiovascular
  disease]. Each pair of adjacent words inside a phrase becomes an extra index term (`ldl_cholesterol`,
  `cardiovascular_diseas`), a POS-filtered form of biword indexing (Manning et al., 2008, §2.4.1).

Both options are evaluated in the ablation (§4.1). The demo shows every step for any claim.

**Inverted index** (`src/irsys/tfidf.py`). One postings list per term holds the documents it occurs in,
with weights pre-computed and length-normalized at index time. The SciFact index has 25,963 terms
and takes 5.1 MB.

**Weighting** uses SMART notation `ddd.qqq` (Manning et al., 2008, ch. 6): document weighting, then
query weighting, each written as *tf / idf / normalization*:

| letter | term frequency | document frequency | normalization |
|---|---|---|---|
| `n` | raw $tf$ | none (1) | none |
| `l` | $1 + \ln tf$ | — | — |
| `t` | — | $\ln \frac{N}{df}$ | — |
| `s` | — | $\ln \frac{N}{df} + 1$ (scikit-learn's variant; not an original SMART letter) | — |
| `c` | — | — | cosine (L2 length) |

For the selected scheme `lsc.lsc`, a term $t$ in a document (or the query) gets the weight

$$w_{t,d} = (1 + \ln tf_{t,d}) \cdot \left(\ln\frac{N}{df_t} + 1\right),$$

and the vectors are L2-normalized. The relevance score is then the **cosine similarity**

$$\text{score}(q,d) = \cos(\vec q, \vec d) = \sum_{t \in q \cap d} \hat w_{t,q}\,\hat w_{t,d},$$

computed term-at-a-time. Only the postings lists of the query's terms are visited, so documents that
share no term with the query are never touched and never returned.

**Correctness checks.** Unit tests compare the scores against values worked out by hand from the
formulas (`tests/test_tfidf.py`). A parity test also requires that our `lsc.lsc` scores equal
scikit-learn's `TfidfVectorizer(sublinear_tf=True, smooth_idf=False, norm="l2")` + `cosine_similarity`
to within $10^{-9}$ on every document. A mutation check confirmed that the parity test fails when the
scheme is changed.

### 3.2 Semantic search and the LLM reranker

| System | How it works | Model |
|---|---|---|
| **Semantic search: bge-small-en-v1.5** | Bi-encoder. Claim and abstracts are embedded *independently* into a 384-dimensional space; relevance = cosine of the embeddings (exact search over all 5,183 vectors). Claims get the instruction prefix *"Represent this sentence for searching relevant passages: "* | BAAI BGE small, 33.4M params, trained for retrieval (Xiao et al., 2023) |
| **TF-IDF → LLM reranker** | Two-stage. TF-IDF retrieves the top 100; an LLM reads the claim together with each abstract, may only answer "yes" or "no", and the abstracts are re-sorted by its confidence in "yes". | Qwen3-Reranker-0.6B, 596M params, a decoder-only Qwen3 LLM (Qwen Team, 2025) fine-tuned for relevance judgement (Zhang et al., 2025) |

Both models are used off the shelf; we fine-tune nothing. bge-small is a compact *encoder* language model
(BERT family); Qwen3-Reranker is a *decoder-only*, GPT-style large language model, small enough (0.6B
parameters, 1.2 GB in bfloat16) to run on a 4 GB laptop GPU.

**LLM scoring** (`src/irsys/llm.py`). Each (claim, abstract) pair becomes one chat prompt, in the format
Qwen3-Reranker was trained on, with the task instruction its authors use for SciFact:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided.
Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Given a scientific claim, retrieve documents that support or refute the claim
<Query>: {claim}
<Document>: {title + abstract}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The LLM does not generate any text. One forward pass gives its next-token logits, and the relevance
score is the log-odds $\log P(\text{yes}) / P(\text{no}) = z_\text{yes} - z_\text{no}$: pointwise relevance
judgement, as in monoT5 (Nogueira et al., 2020). Prompts are capped at 512 tokens by cutting the end of the
abstract.

### 3.3 Experimental protocol

- **No tuning on test data.** The TF-IDF configuration is selected by nDCG@10 on the **train** queries
  (§4.1). The models, the reranking depth (100) and the LLM prompt (the model authors' SciFact
  instruction) are fixed in advance. The 300 test queries are used once, for the final comparison.
- **Metrics.** Six ranking metrics, implemented from scratch with trec_eval/BEIR definitions and
  unit-tested on hand-computed cases: **nDCG@k**, **MAP**, **MRR**, **Precision@k**, **Recall@k** and
  **F1@k** (the harmonic mean of P@k and R@k), at k = 5 and 10. **nDCG@10** is the primary metric, as in
  BEIR. Every system returns the top 100 documents per query; MAP and MRR use that whole ranking.
- **Significance.** Semantic search and the LLM reranker are each compared with TF-IDF on nDCG@10 on the
  *same* queries using the two-sided Wilcoxon signed-rank test, with a Holm correction over the 2
  comparisons.
- **Speed.** Index build time and the mean time to answer one query.
- **Hardware:** NVIDIA RTX 3050 Laptop GPU (4 GB), PyTorch 2.14.1 (CUDA 13.0), Python 3.13.5.
  TF-IDF runs on the CPU.

## 4. Results

### 4.1 Choosing the TF-IDF configuration (train split, 809 queries)

From [`results/ablation_train.md`](results/ablation_train.md), which also has the k = 5 columns:

| Configuration | nDCG@10 | MAP | MRR | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|---:|
| `ntc.ntc`, raw tokens (no stopword removal, no stemming) | 0.5957 | 0.5468 | 0.5584 | 0.0842 | 0.7559 | 0.1497 |
| `ltc.ltc` + stopwords | 0.6456 | 0.6029 | 0.6119 | 0.0875 | 0.7865 | 0.1557 |
| `ltc.ltc` + stopwords + stemming | 0.6688 | 0.6250 | 0.6347 | 0.0897 | 0.8097 | 0.1598 |
| `lnc.ltc` + stopwords + stemming | 0.6758 | 0.6343 | 0.6460 | 0.0896 | 0.8073 | 0.1595 |
| **`lsc.lsc` + stopwords + stemming (selected)** | **0.6789** | **0.6364** | **0.6468** | **0.0904** | **0.8141** | **0.1608** |
| `lsc.lsc` + stopwords + POS lemmatization | 0.6644 | 0.6206 | 0.6304 | 0.0896 | 0.8043 | 0.1594 |
| `lsc.lsc` + stopwords + stemming + noun phrases | 0.6595 | 0.6147 | 0.6229 | 0.0897 | 0.8063 | 0.1596 |
| `lsc.lsc` + stopwords + POS lemmatization + noun phrases | 0.6411 | 0.5978 | 0.6067 | 0.0875 | 0.7853 | 0.1556 |
| scikit-learn `TfidfVectorizer` reference (= `lsc.lsc`) | 0.6789 | 0.6364 | 0.6468 | 0.0904 | 0.8141 | 0.1608 |

The selected configuration is the best on all six metrics, at k = 10 and at k = 5.

- **Sublinear tf + stopword removal** gives the largest gain (+0.050 nDCG@10 over raw counts). One
  repeated term can no longer dominate a document, and function words stop adding noise.
- **Porter stemming** adds +0.023, by matching *mutation/mutations/mutated*.
- The `lnc`/`ltc`/`lsc` variants with stemming are within about 0.01 of each other. idf is the only thing
  they change: none on documents (`lnc`), or the "+1" variant that keeps common terms from dropping to
  zero weight (`lsc`).
- **POS-based lemmatization is worse than stemming** (−0.015). Lemmatization is the more conservative
  normalizer: the corpus has 35,450 distinct surface words, 32,017 lemmas but only 25,963 stems. Stemming
  maps *activate, activation, active* all to `activ`, and *indicate, indication, indicator* to `indic`,
  while lemmatization keeps the verb and the noun apart. Claims and abstracts often state the same fact
  with a different form of the word (claim *discontinue* / abstract *discontinuation*; *positivity* /
  *positive*). On the training claims, stems match more of a claim's terms in its relevant abstract than
  lemmas do for **228** of the 809 claims, and fewer for only 27. That extra matching outweighs stemming's
  over-merging (*organ/organism/organization* → `organ`).
- **Noun-phrase terms lower nDCG@10 (−0.019)**, and every other metric with it. Phrases are very
  specific: 36% of the claims' phrase terms occur in no abstract, and the rest are rare (median document
  frequency 6). Their large idf lets one exact shared phrase dominate the score, which often promotes the
  wrong abstract to the top. Combining both options is the worst of the POS variants (0.641).
- The selection rule (best nDCG@10 on train) therefore keeps plain stemming. POS tagging and chunking stay
  in the system as analysis tools: the demo shows the tags and phrases for any claim.
- The scikit-learn reference reproduces our `lsc.lsc` system exactly, which cross-checks the whole
  pipeline end to end.

### 4.2 Main comparison (test split, 300 queries)

From [`results/metrics.md`](results/metrics.md). At k = 10, with MAP and MRR over each system's top 100:

| System | nDCG@10 | MAP | MRR | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF cosine** (`lsc.lsc`) | 0.659 | 0.609 | 0.623 | 0.090 | 0.819 | 0.160 |
| Semantic search: bge-small-en-v1.5 | 0.713 | 0.674 | 0.687 | 0.094 | 0.836 | 0.167 |
| **TF-IDF → LLM reranker (Qwen3-Reranker-0.6B)** | **0.772** | **0.734** | **0.743** | **0.100** | **0.889** | **0.177** |

At k = 5:

| System | nDCG@5 | P@5 | R@5 | F1@5 |
|---|---:|---:|---:|---:|
| **TF-IDF cosine** (`lsc.lsc`) | 0.633 | 0.161 | 0.745 | 0.260 |
| Semantic search: bge-small-en-v1.5 | 0.685 | 0.167 | 0.759 | 0.269 |
| **TF-IDF → LLM reranker** | **0.744** | **0.180** | **0.808** | **0.288** |

![Effectiveness on SciFact test](results/figures/main_metrics.png)

- **The same order on every metric:** the LLM reranker is the best, semantic search second and TF-IDF
  third, on all six metrics at both k = 5 and k = 10.
- **Precision@k and F1@k are small for every system** because most queries have only one relevant
  abstract, which caps P@5 at 0.2 and P@10 at 0.1. F1@k sits close to the smaller of the two, precision.
  Going from k = 5 to k = 10, recall rises (TF-IDF 0.745 → 0.819, LLM reranker 0.808 → 0.889) while
  precision halves.
- **Beyond the top 10:** TF-IDF's top 100 contains none of the relevant abstracts for 20 of the 300
  queries, semantic search's for 17.

These results agree with published numbers: bge-small-en-v1.5 reproduces the SciFact nDCG@10 that an
independent BEIR evaluation reports for the same checkpoint (0.713; mxbai-edge-colbert-v0 tech report,
Table 16), and our TF-IDF (0.659) is within 0.01 of BEIR's official BM25 baseline (0.665).

### 4.3 Statistical significance (vs. TF-IDF)

From [`results/significance.md`](results/significance.md). W/T/L counts the queries where the system
does better / the same / worse than TF-IDF.

| System | Δ nDCG@10 | W/T/L | Wilcoxon p | Holm-adjusted p | Significant? |
|---|---:|:---:|---:|---:|:---:|
| Semantic search: bge-small-en-v1.5 | +0.054 | 85/164/51 | 0.0027 | 0.0027 | **yes** |
| TF-IDF → LLM reranker | +0.113 | 96/171/33 | 1.0 × 10⁻¹¹ | 2.0 × 10⁻¹¹ | **yes** |

A further paired Wilcoxon test shows that the LLM reranker is also significantly better than semantic
search: +0.059 nDCG@10, 69/197/34, p = 2.5 × 10⁻⁵.

### 4.4 Speed

From [`results/efficiency.md`](results/efficiency.md) (time per query = one query at a time, top 100):

| System | Index build | Time per query | Hardware |
|---|---:|---:|:---:|
| TF-IDF cosine | 1.0 s | **0.17 ms** | CPU |
| Semantic search: bge-small-en-v1.5 | 47.3 s | 12.5 ms | GPU |
| TF-IDF → LLM reranker | 1.0 s | 5,789 ms | GPU |

TF-IDF answers a query about **70× faster than semantic search and ~34,000× faster than LLM
reranking**. It needs no GPU, no model download and about a second to index. The LLM is expensive
because it runs a full forward pass for each of the 100 candidates per query; the run over the 300 test
queries took about 29 minutes on the laptop GPU.

## 5. Analysis: where does each approach win?

### 5.1 Per-query differences

![Per-query difference](results/figures/per_query_delta.png)

The average hides a lot of disagreement between the systems. Semantic search beats TF-IDF on 85
queries, **TF-IDF beats semantic search on 51**, and 164 are tied. 154 of those ties are queries where
both systems are perfect (129) or both score zero (25). The two methods fail on largely *different*
queries: TF-IDF scores zero on 48 queries and semantic search on 46, but only 25 of these are shared.
The LLM reranker scores zero on 31: it rescues 24 of TF-IDF's 48 failures and loses only 7 queries that
TF-IDF had found.

### 5.2 Vocabulary mismatch

The usual argument for semantic search is the **vocabulary-mismatch problem**: a relevant document may
use different words from the query, and term matching cannot bridge that gap. To test this directly,
each test query gets a **lexical overlap** score: the share of its index terms (after stopword removal
and stemming) that appear in its relevant abstract(s). The queries are then split into three equal
groups (from [`results/overlap_buckets.md`](results/overlap_buckets.md)):

| Overlap tercile | Overlap range | TF-IDF | Semantic search | LLM reranker |
|---|:---:|---:|---:|---:|
| low | 0.00–0.45 | 0.369 | 0.438 | **0.530** |
| medium | 0.47–0.75 | 0.719 | 0.785 | **0.835** |
| high | 0.75–1.00 | 0.888 | 0.914 | **0.951** |

![nDCG@10 by vocabulary overlap](results/figures/overlap_buckets.png)

- **Every** system degrades as overlap falls. Even the neural models do much better when the
  vocabularies agree, and all of them find the low-overlap third hard.
- TF-IDF depends on overlap the most: its nDCG@10 climbs by 0.52 from the low to the high tercile, against
  0.48 for semantic search and 0.42 for the LLM reranker.
- Semantic search's advantage over TF-IDF is largest when overlap is low or medium (+0.069, +0.066) and
  shrinks to +0.026 when overlap is high.
- The LLM reranker is the best system in every tercile, and its gain over TF-IDF is largest exactly where
  TF-IDF is weakest: +0.161 at low overlap, +0.116 at medium, +0.063 at high. It recognises a relevant
  abstract that shares few words with the claim, as long as TF-IDF put it somewhere in its top 100.

### 5.3 Illustrative queries

Selected from [`results/examples.md`](results/examples.md). Rank of the relevant abstract:

| Claim | TF-IDF | Semantic search | LLM reranker |
|---|---:|---:|---:|
| "Silencing of Bcl2 is important for the maintenance and progression of tumors." | 32 | 1 | 1 |
| "The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events." | 11 | 1 | 1 |
| "Ivermectin is used to treat onchocerciasis." | 1 | 59 | 1 |
| "LDL cholesterol has no involvement in the development of cardiovascular disease." | 1 | 10 | 1 |

**Semantic search wins — morphology and tokenization.** The Bcl2 claim is answered by *"Antiapoptotic
**BCL-2** is required for maintenance of a model leukemia"*. The tokenizer turns "BCL-2" into `bcl` + `2`,
while the claim's "Bcl2" becomes `bcl2`, so the gene name, the most informative word in the claim, never
matches, and TF-IDF ranks the abstract **32nd**. Semantic search and the LLM rank it **1st**, because their
subword tokenizers and learned representations treat *Bcl2* and *BCL-2* as the same gene.

**Semantic search wins — paraphrase.** The co-IR claim is answered by *"Is autoimmunity the Achilles'
heel of cancer immunotherapy?"*: the same idea in different words. TF-IDF ranks it 11th; semantic search
and the LLM rank it 1st.

**TF-IDF wins — a rare, exact technical term.** In the ivermectin claim, *ivermectin* is not in the corpus
at all, but *onchocerciasis* occurs in exactly **one** of the 5,183 abstracts, so it gets the maximum
possible idf, and TF-IDF ranks that abstract **1st**. Semantic search ranks it 59th, drifting to general
drug-therapy papers: it has only a vague representation of such a rare disease name. The LLM reranker,
which reads the actual abstract, keeps it 1st.

**TF-IDF wins — high overlap, subtle topic.** Five of six query terms of the LDL claim occur in the
relevant abstract (a genome-wide association study of LDL cholesterol), and TF-IDF ranks it 1st. Semantic
search ranks it 10th, preferring a review of *HDL* cholesterol: the embedding blurs the LDL/HDL
distinction that exact term matching keeps. The LLM reranker keeps it 1st.

### 5.4 Error analysis

From [`results/error_analysis.md`](results/error_analysis.md). A claim **fails** for a system when none of
its relevant abstracts is in the top 10 (nDCG@10 = 0). To ask *why* systems fail, each claim is put in
one category using only the claim and its relevant abstract(s), so the categories are the same for
every system. The rules apply in order:

1. **spelling variant**: a claim word is written differently in the abstract (*Bcl2* / *BCL-2*,
   *birth weight* / *birthweight*);
2. **vocabulary mismatch**: fewer than half of the claim's index terms occur in the abstract;
3. **rare shared term**: a shared term occurs in at most 10 of the 5,183 abstracts;
4. **common words only**: everything else.

| Category | Claims | TF-IDF | Semantic search | LLM reranker |
|---|---:|---:|---:|---:|
| spelling variant | 15 | 2 (13%) | 0 | 0 |
| vocabulary mismatch | 99 | 34 (34%) | 38 (38%) | 27 (27%) |
| rare shared term | 106 | **0 (0%)** | 1 (1%) | 1 (1%) |
| common words only | 80 | 12 (15%) | 7 (9%) | 3 (4%) |
| **all claims** | 300 | 48 (16%) | 46 (15%) | 31 (10%) |

- **Vocabulary mismatch is the hard core of the task.** A third of the claims fall in it, and it holds
  most failures of *every* system, the LLM reranker included (27 of its 31). Neural models reduce the
  problem; they do not remove it.
- **Rare shared terms are TF-IDF's home ground.** In the 106 claims that share a rare term with their
  relevant abstract, TF-IDF never fails.
- **Spelling variants are the one category where TF-IDF fails and both neural systems succeed** (2 of 15
  claims, e.g. *Bcl2*). Subword tokenizers make the neural models robust to hyphenation.

**LLM reranker failures: first stage or reranker?** Of the LLM reranker's 31 failures, **20** are relevant
abstracts that TF-IDF never placed in its top 100, so the reranker never saw them, and only **11** are
abstracts it saw but ranked below 10. It almost never misjudges an abstract it is shown; its ceiling is
TF-IDF's candidate list.

**How the LLM answers.** Out of 29,889 judged (claim, abstract) pairs, where roughly 1 candidate in 100
is relevant, the LLM answered "yes" (P(yes) > 0.5) to **2.3%** of all candidates and to **75%** of the
relevant ones: it is well calibrated, saying yes rarely and mostly to the right abstracts.

## 6. Discussion

**How good is TF-IDF?** Very competitive. With sensible preprocessing it reaches nDCG@10 = 0.659, within
0.01 of BEIR's published BM25 figure for SciFact (0.665). Scientific claims and abstracts share a lot of
rare, precise vocabulary, which is exactly what idf weighting rewards; TF-IDF never fails when a claim
shares a rare term with its answer.

**Does semantic search beat it?** Yes, by a clear and statistically significant margin: +0.054 nDCG@10
(+8%), with gains concentrated on queries phrased differently from their answers. The two approaches fail
on largely different queries.

**Does the LLM beat it?** Yes, by the widest margin: the LLM reranker reaches nDCG@10 0.772, +0.113
(+17%) over TF-IDF, and is significantly better than semantic search too. It is the best system on all
six metrics. Because it reranks TF-IDF's candidates, it combines both worlds: classical retrieval finds
100 candidates fast, and the LLM judges which of them really answers the claim. Its remaining failures
come mostly from TF-IDF's first stage.

**At what cost?** TF-IDF is about 70× faster than semantic search and ~34,000× faster than LLM
reranking (0.17 ms vs. 5.8 s per query). It runs on any CPU, indexes in about a second, and its scores
can be explained term by term. The neural systems need a GPU (or much more time) and model downloads,
and semantic search must re-encode the corpus when it changes. Their scores are opaque.

| Prefer TF-IDF when | Prefer semantic search / an LLM when |
|---|---|
| queries use the corpus's own technical terms (names, codes, rare entities) | queries paraphrase or use different vocabulary from the documents |
| low latency / CPU-only / large corpora matter | effectiveness matters more than compute cost |
| results must be explainable (which terms matched) | users write natural-language questions |
| no suitable pretrained model exists for the domain or language | a model trained for retrieval exists for the domain |

## 7. Limitations

- **One dataset.** SciFact is scientific and small (5k documents). The balance could shift on other
  domains, for example conversational queries or long documents.
- **Truncation.** bge reads at most 512 tokens per document and the LLM 512 tokens per prompt, so the
  ends of long abstracts are ignored. TF-IDF sees the whole text.
- **Sparse judgments.** Most queries have one relevant abstract, so a single rank change moves a query's
  score a lot, and unjudged-but-relevant documents count as misses. It is also why Precision@k and F1@k are
  low for every system.
- **Tie-splitting.** The overlap terciles break ties in query order to keep the groups equal in size, so
  0.75 appears in two buckets.
- **Model scale.** Only models up to 0.6B parameters fit the 4 GB laptop GPU. Larger LLMs, or API models,
  would likely score higher, at a much greater computational cost.
- **One prompt.** The LLM uses the instruction its authors use for SciFact. It was fixed in advance and
  not tuned (tuning it on the test claims would not be fair).
- **Reranking depth.** The LLM sees only TF-IDF's top 100, so it can never find a relevant abstract
  TF-IDF missed (20 of the 300 queries).
- **Efficiency numbers** come from one laptop and include Python overhead; they are only meant as
  relative comparisons.

## 8. Conclusion

A from-scratch TF-IDF/cosine system with sublinear tf, idf weighting, stopword removal and stemming is
a strong, fast and transparent baseline on scientific claim retrieval (nDCG@10 = 0.659). POS tagging and
noun-phrase chunking, tested as lemmatization and phrase indexing, did not improve on plain stemming.
Semantic search with a retrieval-trained embedding model is significantly more effective (0.713), mainly
because it copes with vocabulary mismatch, and an LLM reranking TF-IDF's candidates is the most effective
system (0.772): the best on all six metrics (nDCG@k, MAP, MRR, Precision@k, Recall@k, F1@k) at both
k = 5 and 10. TF-IDF still wins on queries that hinge on rare, exact terms, and it is thousands of times
cheaper. Classical and LLM-based retrieval are complementary, and the best design combines them: fast
TF-IDF retrieval of candidates, then an LLM to judge them.

## 9. Reproducing the results

```sh
uv sync
uv run pytest                 # 127 tests (11 marked slow: data download + real models)
uv run irsys evaluate         # → results/*.md, *.csv, runs/*.trec, config.json  (~35 min on an RTX 3050)
uv run irsys rescore          # recompute the metric files from the saved runs (no models)
uv run irsys analyze          # → results/figures/*.png, overlap_buckets.md, error_analysis.md, examples.md
uv run irsys search "your claim here" -k 5
uv run irsys demo             # interactive demo (DEMO.md)
```

The selected TF-IDF configuration, model names, device and library versions are recorded in
[`results/config.json`](results/config.json). Ranked runs for every system are in TREC format in
[`results/runs/`](results/runs/), ready for `trec_eval`.

## References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python* (NLTK). O'Reilly.
- *Fantastic (small) Retrievers and How to Train Them: mxbai-edge-colbert-v0 Tech Report* (2025). *arXiv:2510.14880* (BEIR appendix, Table 16: reference SciFact nDCG@10 for bge-small-en-v1.5).
- Justeson, J. S., & Katz, S. M. (1995). Technical terminology: some linguistic properties and an algorithm for identification in text. *Natural Language Engineering*, 1(1), 9–27.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*, §2.4.1 (biword indexes) and ch. 6 (Scoring, term weighting and the vector space model). Cambridge University Press.
- Marcus, M. P., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English: The Penn Treebank. *Computational Linguistics*, 19(2), 313–330.
- Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39–41.
- Nogueira, R., Jiang, Z., Pradeep, R., & Lin, J. (2020). Document ranking with a pretrained sequence-to-sequence model. *Findings of EMNLP*.
- Qwen Team (2025). Qwen3 technical report. *arXiv:2505.09388*.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5).
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS Datasets and Benchmarks*.
- Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or fiction: Verifying scientific claims. *EMNLP*.
- Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese embedding (BGE). *arXiv:2309.07597*.
- Zhang, Y., Li, M., Long, D., et al. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. *arXiv:2506.05176*.
