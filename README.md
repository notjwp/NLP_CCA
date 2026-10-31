# TF-IDF vs. Semantic Search on BEIR SciFact

A classical Information Retrieval system (**TF-IDF weighting + cosine similarity**, built from
scratch on an inverted index, with tokenization, stopword removal, stemming, POS tagging and
noun-phrase chunking) compared against **LLM/semantic search** (transformer bi-encoders, a
cross-encoder, two LLM rerankers and a hybrid), evaluated on the BEIR SciFact benchmark.

The write-up with results and analysis is in **[REPORT.md](REPORT.md)**.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 and all dependencies are installed by:

```sh
uv sync
```

`pyproject.toml` pulls PyTorch from the CUDA 13.0 wheel index on Windows/Linux. Without an NVIDIA
GPU everything still runs on CPU, only slower. Check with:

```sh
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

```sh
uv run irsys evaluate      # download data, TF-IDF ablation (train), all systems (test) -> results/
uv run irsys analyze       # figures, vocabulary-overlap and error analysis, examples -> results/
uv run irsys search "vitamin D deficiency increases risk of fractures" -k 5   # side-by-side, in the terminal
uv run irsys demo          # interactive presentation demo in the browser (see DEMO.md)
uv run irsys demo --no-llm # the same without the LLM column (no GPU needed)
```

`irsys demo` opens a local web app with four tabs:
- **Tab 1:** live search with word matching (TF-IDF), meaning matching (bge-small embeddings), the two
  combined, and the LLM reranker, with matching words highlighted and a ✓ on the correct paper for
  benchmark claims.
- **Tab 2:** inside TF-IDF: tokens, part-of-speech tags, noun phrases, stopwords, stems, then how each
  word's weight adds up to the score.
- **Tab 3:** inside the LLM: the exact yes/no prompt and the probability of "yes" for the papers that matter.
- **Tab 4:** the evaluation results, with a plain-language guide to the grades.

[DEMO.md](DEMO.md) has a presenter script, a pre-flight checklist and backup screenshots.

On the first run the dataset (~5 MB), three small encoder models (~90–130 MB each), two 0.6B-parameter
Qwen3 LLMs (~1.2–1.5 GB each) and NLTK's POS tagger and WordNet (into `data/nltk_data/`) download
automatically. On an RTX 3050 Laptop GPU (4 GB), `evaluate` takes about an hour: the two LLMs each read
100 candidates for every one of the 300 test claims (~6 s per claim); everything else takes ~5 minutes.
Corpus embeddings are cached in `cache/`.

## Tests

```sh
uv run pytest -m "not slow"   # fast unit tests (metrics, TF-IDF maths, sklearn parity, fusion, stats, ...)
uv run pytest                 # + smoke tests that download data and run the real models
```

## Layout

| Path | Contents |
|---|---|
| `src/irsys/preprocess.py` | Tokenizer, stopword removal, Porter stemmer, POS tagging, POS-based lemmatization, noun-phrase chunking |
| `src/irsys/tfidf.py` | From-scratch inverted index, SMART TF-IDF weighting, cosine scoring; scikit-learn reference |
| `src/irsys/dense.py` | Bi-encoder semantic retriever (sentence-transformers) |
| `src/irsys/rerank.py` | Cross-encoder reranking of first-stage candidates |
| `src/irsys/llm.py` | LLM reranking: a yes/no relevance prompt, scored from the next-token probabilities |
| `src/irsys/fusion.py` | Reciprocal Rank Fusion hybrid |
| `src/irsys/metrics.py` | nDCG, MAP, MRR, P@k, Recall@k (from scratch) |
| `src/irsys/stats.py` | Wilcoxon / paired t-test, Holm correction, bootstrap CIs |
| `src/irsys/evaluate.py` | The experiment: ablation on train, all systems on test, result files |
| `src/irsys/analysis.py` | Figures, vocabulary-overlap analysis, error analysis, qualitative examples |
| `src/irsys/demo.py` | Interactive presentation demo (Gradio) |
| `src/irsys/cli.py` | `irsys` command line |
| `results/` | Generated tables (`*.md`, `*.csv`), TREC run files (`runs/`), figures (`figures/`) |
| `tests/` | pytest suite |
