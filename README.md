# TF-IDF vs. Semantic Search on BEIR SciFact

A classical Information Retrieval system (**TF-IDF weighting + cosine similarity**, built from
scratch on an inverted index) compared against **LLM/semantic search** (transformer bi-encoders,
a cross-encoder reranker and a hybrid), evaluated on the BEIR SciFact benchmark.

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
uv run irsys analyze       # figures + vocabulary-overlap analysis + examples -> results/
uv run irsys search "vitamin D deficiency increases risk of fractures" -k 5   # side-by-side, in the terminal
uv run irsys demo          # interactive presentation demo in the browser (see DEMO.md)
```

`irsys demo` opens a local web app with three tabs:
- **Tab 1:** live TF-IDF vs. semantic vs. hybrid search, with matching words highlighted and a ✓ on the
  correct abstract for benchmark claims.
- **Tab 2:** a step-by-step breakdown of how TF-IDF computed the score.
- **Tab 3:** the evaluation results.

[DEMO.md](DEMO.md) has a 7-minute presenter script, a pre-flight checklist and backup screenshots.

On the first run the dataset (~5 MB) and three small models (~90–130 MB each) download from the
Hugging Face Hub. On an RTX 3050 Laptop GPU, `evaluate` takes about 4 minutes; most of that is the
cross-encoder. On CPU, expect roughly 15–25 minutes. Corpus embeddings are cached in `cache/`.

## Tests

```sh
uv run pytest -m "not slow"   # fast unit tests (metrics, TF-IDF maths, sklearn parity, fusion, stats, ...)
uv run pytest                 # + smoke tests that download data and run the real models
```

## Layout

| Path | Contents |
|---|---|
| `src/irsys/preprocess.py` | Tokenizer, stopword removal, Porter stemmer |
| `src/irsys/tfidf.py` | From-scratch inverted index, SMART TF-IDF weighting, cosine scoring; scikit-learn reference |
| `src/irsys/dense.py` | Bi-encoder semantic retriever (sentence-transformers) |
| `src/irsys/rerank.py` | Cross-encoder reranking of first-stage candidates |
| `src/irsys/fusion.py` | Reciprocal Rank Fusion hybrid |
| `src/irsys/metrics.py` | nDCG, MAP, MRR, P@k, Recall@k (from scratch) |
| `src/irsys/stats.py` | Wilcoxon / paired t-test, Holm correction, bootstrap CIs |
| `src/irsys/evaluate.py` | The experiment: ablation on train, all systems on test, result files |
| `src/irsys/analysis.py` | Figures, vocabulary-overlap analysis, qualitative examples |
| `src/irsys/demo.py` | Interactive presentation demo (Gradio) |
| `src/irsys/cli.py` | `irsys` command line |
| `results/` | Generated tables (`*.md`, `*.csv`), TREC run files (`runs/`), figures (`figures/`) |
| `tests/` | pytest suite |
