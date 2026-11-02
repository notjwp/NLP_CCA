"""End-to-end experiment: TF-IDF ablation on train, then TF-IDF vs. semantic search vs. an LLM reranker on the
SciFact test split."""

import gc
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from irsys.base import Retriever, Run
from irsys.data import RESULTS_DIR, doc_text, load_scifact
from irsys.dense import DenseRetriever
from irsys.llm import INSTRUCTION, LLMReranker
from irsys.metrics import METRICS, evaluate_run
from irsys.preprocess import Analyzer
from irsys.rerank import RerankRetriever
from irsys.stats import paired_tests
from irsys.tfidf import SklearnTfidfRetriever, TfidfRetriever

BGE = "BAAI/bge-small-en-v1.5"  # semantic search: embeddings trained for retrieval
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
LLM_RERANKER = "Qwen/Qwen3-Reranker-0.6B"  # LLM fine-tuned for relevance judgements
RUN_DEPTH = 100  # docs kept per query; also the reranking depth
SIGNIFICANCE_METRIC = "nDCG@10"  # the main metric; the only one tested for significance


@dataclass(frozen=True)
class TfidfConfig:
    label: str
    scheme: str
    stem: bool
    stopwords: bool
    lemmatize: bool = False
    phrases: bool = False

    def analyzer(self) -> Analyzer:
        return Analyzer(self.stem, self.stopwords, self.lemmatize, self.phrases)

    def build(self) -> TfidfRetriever:
        return TfidfRetriever(self.analyzer(), self.scheme)


ABLATION_CONFIGS = [
    TfidfConfig("ntc.ntc, raw tokens", "ntc.ntc", stem=False, stopwords=False),
    TfidfConfig("ltc.ltc + stopwords", "ltc.ltc", stem=False, stopwords=True),
    TfidfConfig("ltc.ltc + stopwords + stemming", "ltc.ltc", stem=True, stopwords=True),
    TfidfConfig("lnc.ltc + stopwords + stemming", "lnc.ltc", stem=True, stopwords=True),
    TfidfConfig("lsc.lsc + stopwords + stemming", "lsc.lsc", stem=True, stopwords=True),
    # Classical NLP on top: POS tagging drives lemmatization and noun-phrase chunking.
    TfidfConfig("lsc.lsc + stopwords + POS lemmatization", "lsc.lsc", stem=False, stopwords=True, lemmatize=True),
    TfidfConfig("lsc.lsc + stopwords + stemming + noun phrases", "lsc.lsc", stem=True, stopwords=True, phrases=True),
    TfidfConfig("lsc.lsc + stopwords + POS lemmatization + noun phrases", "lsc.lsc", stem=False, stopwords=True,
                lemmatize=True, phrases=True),
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_trec_run(run: Run, path: Path, tag: str) -> None:
    """TREC run format: ``qid Q0 docid rank score tag``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for qid, scored in run.items():
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        lines += [f"{qid} Q0 {doc} {rank} {score:.6f} {tag}" for rank, (doc, score) in enumerate(ranked, start=1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize(system: Retriever | Callable[[], Retriever]) -> Retriever:
    """Systems are given as retrievers or as factories, so large models are loaded only when their turn comes."""
    return system if isinstance(system, Retriever) else system()


def run_with_latency(retriever: Retriever, queries: dict[str, str], k: int) -> tuple[Run, list[float]]:
    """Answer each query on its own (as an interactive user would), timing each in ms."""
    run, latencies = {}, []
    for qid, text in tqdm(queries.items(), desc=getattr(retriever, "name", None), leave=False, mininterval=5):
        start = time.perf_counter()
        run[qid] = dict(retriever.search(text, k))
        latencies.append((time.perf_counter() - start) * 1000)
    return run, latencies


def tfidf_ablation(doc_ids, texts, queries, qrels) -> tuple[pd.DataFrame, TfidfConfig]:
    """Evaluate each TF-IDF configuration; select by nDCG@10 (call this on the *train* split)."""
    rows = []
    for config in ABLATION_CONFIGS:
        r = config.build().index(doc_ids, texts)
        means = evaluate_run(r.search_batch(queries, RUN_DEPTH), qrels).mean()
        rows.append({"config": config.label, "scheme": config.scheme, "stemming": config.stem,
                     "stopwords": config.stopwords, "lemmatization": config.lemmatize,
                     "noun phrases": config.phrases, **means[METRICS].to_dict()})
    sk = SklearnTfidfRetriever(Analyzer(stem=True, stopwords=True)).index(doc_ids, texts)
    means = evaluate_run(sk.search_batch(queries, RUN_DEPTH), qrels).mean()
    rows.append({"config": "scikit-learn TfidfVectorizer (reference = lsc.lsc)", "scheme": "lsc.lsc",
                 "stemming": True, "stopwords": True, "lemmatization": False, "noun phrases": False,
                 **means[METRICS].to_dict()})

    table = pd.DataFrame(rows)
    best_idx = int(table.iloc[: len(ABLATION_CONFIGS)]["nDCG@10"].idxmax())
    table["selected"] = table.index == best_idx
    return table, ABLATION_CONFIGS[best_idx]


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"


def evaluate(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _log("Loading BEIR SciFact")
    corpus, train_queries, train_qrels = load_scifact("train")
    _, test_queries, test_qrels = load_scifact("test")
    doc_ids = list(corpus)
    texts = [doc_text(corpus[d]) for d in doc_ids]
    _log(f"{len(doc_ids)} docs | {len(train_queries)} train / {len(test_queries)} test queries")

    # 1. Choose the TF-IDF configuration on the train split only.
    _log("TF-IDF ablation on the train split")
    ablation, best = tfidf_ablation(doc_ids, texts, train_queries, train_qrels)
    _write_ablation(results_dir, ablation, best, len(train_queries))
    _log(f"Selected TF-IDF config: {best.label}")

    # 2. Build every system.
    start = time.perf_counter()
    tfidf = best.build().index(doc_ids, texts)
    tfidf_seconds = time.perf_counter() - start
    tfidf.name = f"TF-IDF cosine ({best.scheme})"
    _log(f"Encoding corpus with {BGE}")
    bge = DenseRetriever(BGE, query_prompt=BGE_QUERY_PROMPT, name="Semantic search: bge-small-en-v1.5").index(
        doc_ids, texts)
    doc_texts = dict(zip(doc_ids, texts))

    # Classical retrieval vs. semantic search vs. an LLM reranking TF-IDF's top 100.
    systems: dict[str, Retriever | Callable[[], Retriever]] = {
        "tfidf": tfidf,
        "dense-bge": bge,
        "tfidf-llm": lambda: RerankRetriever(tfidf, LLMReranker(LLM_RERANKER), doc_texts, depth=RUN_DEPTH,
                                             name="TF-IDF → LLM reranker (Qwen3-Reranker-0.6B)"),
    }
    device = str(bge.model.device)

    # 3. Retrieve + evaluate on test. The LLM is loaded only when its turn comes and freed after its run.
    per_query: dict[str, pd.DataFrame] = {}
    latencies: dict[str, list[float]] = {}
    names: dict[str, str] = {}
    for key, entry in systems.items():
        system = materialize(entry)
        names[key] = system.name
        _log(f"Retrieving: {system.name}")
        system.search(next(iter(test_queries.values())), RUN_DEPTH)  # warm-up (CUDA init, caches)
        run, latencies[key] = run_with_latency(system, test_queries, RUN_DEPTH)
        write_trec_run(run, results_dir / "runs" / f"{key}.trec", key)
        per_query[key] = evaluate_run(run, test_qrels)
        if isinstance(system, RerankRetriever):
            del system
            gc.collect()
            torch.cuda.empty_cache()

    _write_metrics(per_query, names, results_dir)
    _write_significance(per_query, names, results_dir)
    build_seconds = {"tfidf": tfidf_seconds, "dense-bge": bge.index_seconds}  # the reranker needs only TF-IDF's index
    _write_efficiency(results_dir, [{
        "System": names[key],
        "Index build (s)": build_seconds.get(key, tfidf_seconds),
        "Mean time per claim (ms)": float(np.mean(latencies[key])),
        "Device": "cpu" if key == "tfidf" else device,
    } for key in systems])
    (results_dir / "config.json").write_text(json.dumps({
        "dataset": "BEIR SciFact",
        "n_docs": len(doc_ids),
        "n_test_queries": len(test_queries),
        "n_train_queries": len(train_queries),
        "tfidf": {"label": best.label, "scheme": best.scheme, **best.analyzer().config()},
        "models": {"bge": BGE, "bge_query_prompt": BGE_QUERY_PROMPT, "llm_reranker": LLM_RERANKER,
                   "llm_instruction": INSTRUCTION},
        "run_depth": RUN_DEPTH,
        "systems": names,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }, indent=2), encoding="utf-8")
    _log(f"Done. Results in {results_dir}")


def _write_ablation(results_dir: Path, table: pd.DataFrame, best: "TfidfConfig", n_queries: int) -> None:
    (results_dir / "ablation_train.md").write_text(
        f"TF-IDF variants on the SciFact **train** split ({n_queries} queries). "
        f"Selected by nDCG@10: **{best.label}**.\n\n" + _md(table),
        encoding="utf-8",
    )


def _write_metrics(per_query: dict[str, pd.DataFrame], names: dict[str, str], results_dir: Path) -> None:
    long = pd.concat({k: df for k, df in per_query.items()}, names=["system", "qid"]).reset_index()
    long.to_csv(results_dir / "per_query.csv", index=False)

    metrics = pd.DataFrame([{"system": key, "name": names[key], **df[METRICS].mean().to_dict()}
                            for key, df in per_query.items()])
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    table = metrics[["name", *METRICS]].rename(columns={"name": "System"})
    n = len(next(iter(per_query.values())))
    (results_dir / "metrics.md").write_text(
        f"Mean over the {n} SciFact test queries. k = 5 and 10; MAP and MRR use each system's whole top-{RUN_DEPTH} "
        "ranking.\n\n" + _md(table),
        encoding="utf-8",
    )


def _write_significance(per_query: dict[str, pd.DataFrame], names: dict[str, str], results_dir: Path) -> None:
    base = per_query["tfidf"][SIGNIFICANCE_METRIC]
    table = paired_tests(base, {k: df[SIGNIFICANCE_METRIC] for k, df in per_query.items() if k != "tfidf"})
    table["system"] = table["system"].map(names)
    table["W/T/L"] = [f"{w}/{t}/{l}" for w, t, l in zip(table["wins"], table["ties"], table["losses"])]
    table["significant (α=0.05)"] = np.where(table["holm_p"] < 0.05, "yes", "no")
    out = table[["system", "baseline_mean", "system_mean", "delta", "W/T/L", "wilcoxon_p", "holm_p",
                 "significant (α=0.05)"]].rename(columns={"baseline_mean": "TF-IDF mean", "system_mean": "system mean"})
    (results_dir / "significance.md").write_text(
        f"Each system vs. the TF-IDF baseline on {SIGNIFICANCE_METRIC}, on the same test queries: two-sided Wilcoxon "
        f"signed-rank test (zero differences dropped), Holm correction over the {len(per_query) - 1} comparisons. "
        "W/T/L = queries where the system scores higher / equal / lower.\n\n"
        + out.to_markdown(index=False, floatfmt=".4g") + "\n",
        encoding="utf-8",
    )


def _write_efficiency(results_dir: Path, rows: list[dict]) -> None:
    (results_dir / "efficiency.md").write_text(
        "Index build = TF-IDF indexing or corpus embedding time. Time per claim = one query at a time, "
        f"top-{RUN_DEPTH}, after a warm-up query; the LLM reranker includes its TF-IDF first stage.\n\n"
        + _md(pd.DataFrame(rows), floatfmt=".2f"),
        encoding="utf-8",
    )


def read_markdown_table(text: str) -> list[dict[str, str]]:
    """Rows of the first pipe table in ``text`` (header row, alignment row, then data) as strings."""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    return [dict(zip(cells(lines[0]), cells(line))) for line in lines[2:]] if len(lines) > 2 else []


def rescore(results_dir: Path = RESULTS_DIR, qrels: dict[str, dict[str, int]] | None = None,
            ablation: bool = True) -> None:
    """Recompute every metric file from the saved runs, without loading any model.

    Rankings come from ``runs/*.trec``; timings are carried over from ``efficiency.md`` (they were
    measured when the runs were made). With ``ablation``, the TF-IDF train-split table is rebuilt too.
    """
    from irsys.analysis import read_trec_run

    names = json.loads((results_dir / "config.json").read_text(encoding="utf-8"))["systems"]
    if qrels is None:
        _, _, qrels = load_scifact("test")
    per_query = {}
    for key in names:
        ranked = read_trec_run(results_dir / "runs" / f"{key}.trec")
        run = {qid: {d: float(len(docs) - i) for i, d in enumerate(docs)} for qid, docs in ranked.items()}
        per_query[key] = evaluate_run(run, qrels)
    _write_metrics(per_query, names, results_dir)
    _write_significance(per_query, names, results_dir)

    timings = [row for row in read_markdown_table((results_dir / "efficiency.md").read_text(encoding="utf-8"))
               if row["System"] in names.values()]
    _write_efficiency(results_dir, [{
        "System": row["System"],
        "Index build (s)": float(row["Index build (s)"]),
        "Mean time per claim (ms)": float(row.get("Mean time per claim (ms)", row.get("Mean latency (ms)"))),
        "Device": row["Device"],
    } for row in timings])

    if ablation:
        corpus, train_queries, train_qrels = load_scifact("train")
        doc_ids = list(corpus)
        table, best = tfidf_ablation(doc_ids, [doc_text(corpus[d]) for d in doc_ids], train_queries, train_qrels)
        _write_ablation(results_dir, table, best, len(train_queries))
    _log(f"Rescored from saved runs. Results in {results_dir}")
