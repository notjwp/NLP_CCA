"""End-to-end experiment: TF-IDF ablation on train, then all systems on the SciFact test split."""

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irsys.base import Retriever, Run
from irsys.data import RESULTS_DIR, doc_text, load_scifact
from irsys.dense import DenseRetriever
from irsys.fusion import RRFHybrid
from irsys.metrics import evaluate_run
from irsys.preprocess import Analyzer
from irsys.rerank import CrossEncoderReranker, RerankRetriever
from irsys.stats import bootstrap_ci, paired_tests
from irsys.tfidf import SklearnTfidfRetriever, TfidfRetriever

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RUN_DEPTH = 100  # docs kept per query; also the reranking depth
SIGNIFICANCE_METRICS = ("nDCG@10", "R@100")
MAIN_METRICS = ["nDCG@10", "MAP@100", "MRR@10", "P@10", "R@10", "R@100"]


@dataclass(frozen=True)
class TfidfConfig:
    label: str
    scheme: str
    stem: bool
    stopwords: bool

    def build(self) -> TfidfRetriever:
        return TfidfRetriever(Analyzer(stem=self.stem, stopwords=self.stopwords), self.scheme)


ABLATION_CONFIGS = [
    TfidfConfig("ntc.ntc, raw tokens", "ntc.ntc", stem=False, stopwords=False),
    TfidfConfig("ltc.ltc + stopwords", "ltc.ltc", stem=False, stopwords=True),
    TfidfConfig("ltc.ltc + stopwords + stemming", "ltc.ltc", stem=True, stopwords=True),
    TfidfConfig("lnc.ltc + stopwords + stemming", "lnc.ltc", stem=True, stopwords=True),
    TfidfConfig("lsc.lsc + stopwords + stemming", "lsc.lsc", stem=True, stopwords=True),
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


def run_with_latency(retriever: Retriever, queries: dict[str, str], k: int) -> tuple[Run, list[float]]:
    """Answer each query on its own (as an interactive user would), timing each in ms."""
    run, latencies = {}, []
    for qid, text in queries.items():
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
                     "stopwords": config.stopwords, **means[MAIN_METRICS].to_dict()})
    sk = SklearnTfidfRetriever(Analyzer(stem=True, stopwords=True)).index(doc_ids, texts)
    means = evaluate_run(sk.search_batch(queries, RUN_DEPTH), qrels).mean()
    rows.append({"config": "scikit-learn TfidfVectorizer (reference = lsc.lsc)", "scheme": "lsc.lsc",
                 "stemming": True, "stopwords": True, **means[MAIN_METRICS].to_dict()})

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
    (results_dir / "ablation_train.md").write_text(
        f"TF-IDF variants on the SciFact **train** split ({len(train_queries)} queries). "
        f"Selected by nDCG@10: **{best.label}**.\n\n" + _md(ablation),
        encoding="utf-8",
    )
    _log(f"Selected TF-IDF config: {best.label}")

    # 2. Build every system.
    start = time.perf_counter()
    tfidf = best.build().index(doc_ids, texts)
    tfidf_seconds = time.perf_counter() - start
    tfidf.name = f"TF-IDF cosine ({best.scheme})"
    _log(f"Encoding corpus with {MINILM}")
    minilm = DenseRetriever(MINILM, name="Dense: all-MiniLM-L6-v2").index(doc_ids, texts)
    _log(f"Encoding corpus with {BGE}")
    bge = DenseRetriever(BGE, query_prompt=BGE_QUERY_PROMPT, name="Dense: bge-small-en-v1.5").index(doc_ids, texts)
    cross_encoder = CrossEncoderReranker(CROSS_ENCODER)
    systems: dict[str, Retriever] = {
        "tfidf": tfidf,
        "dense-minilm": minilm,
        "dense-bge": bge,
        "hybrid-rrf": RRFHybrid([tfidf, bge], name="Hybrid: TF-IDF + bge (RRF)"),
        "tfidf-ce": RerankRetriever(tfidf, cross_encoder, dict(zip(doc_ids, texts)), depth=RUN_DEPTH,
                                    name="TF-IDF → cross-encoder"),
    }
    device = str(minilm.model.device)

    # 3. Retrieve + evaluate on test.
    per_query: dict[str, pd.DataFrame] = {}
    latencies: dict[str, list[float]] = {}
    for key, system in systems.items():
        _log(f"Retrieving: {system.name}")
        system.search(next(iter(test_queries.values())), RUN_DEPTH)  # warm-up (CUDA init, caches)
        run, latencies[key] = run_with_latency(system, test_queries, RUN_DEPTH)
        write_trec_run(run, results_dir / "runs" / f"{key}.trec", key)
        per_query[key] = evaluate_run(run, test_qrels)

    names = {key: s.name for key, s in systems.items()}
    _write_metrics(per_query, names, results_dir)
    _write_significance(per_query, names, results_dir)
    _write_efficiency(results_dir, names, latencies, device, {
        "tfidf": (tfidf_seconds, tfidf.nbytes, 0),
        "dense-minilm": (minilm.index_seconds, minilm.nbytes, minilm.n_params),
        "dense-bge": (bge.index_seconds, bge.nbytes, bge.n_params),
        "hybrid-rrf": (tfidf_seconds + bge.index_seconds, tfidf.nbytes + bge.nbytes, bge.n_params),
        "tfidf-ce": (tfidf_seconds, tfidf.nbytes, cross_encoder.n_params),
    })
    (results_dir / "config.json").write_text(json.dumps({
        "dataset": "BEIR SciFact",
        "n_docs": len(doc_ids),
        "n_test_queries": len(test_queries),
        "n_train_queries": len(train_queries),
        "tfidf": {"label": best.label, "scheme": best.scheme, "stem": best.stem, "stopwords": best.stopwords},
        "models": {"minilm": MINILM, "bge": BGE, "bge_query_prompt": BGE_QUERY_PROMPT, "cross_encoder": CROSS_ENCODER},
        "run_depth": RUN_DEPTH,
        "rrf_k": 60,
        "systems": names,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }, indent=2), encoding="utf-8")
    _log(f"Done. Results in {results_dir}")


def _write_metrics(per_query: dict[str, pd.DataFrame], names: dict[str, str], results_dir: Path) -> None:
    long = pd.concat({k: df for k, df in per_query.items()}, names=["system", "qid"]).reset_index()
    long.to_csv(results_dir / "per_query.csv", index=False)

    rows = []
    for key, df in per_query.items():
        lo, hi = bootstrap_ci(df["nDCG@10"])
        rows.append({"system": key, "name": names[key], **df.mean().to_dict(), "nDCG@10_ci_lo": lo, "nDCG@10_ci_hi": hi})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    table = metrics[["name", *MAIN_METRICS]].rename(columns={"name": "System"})
    table.insert(2, "nDCG@10 95% CI", [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in
                                       zip(metrics["nDCG@10_ci_lo"], metrics["nDCG@10_ci_hi"])])
    n = len(next(iter(per_query.values())))
    (results_dir / "metrics.md").write_text(
        f"Mean over the {n} SciFact test queries. CI: 10,000-sample percentile bootstrap over queries.\n\n" + _md(table),
        encoding="utf-8",
    )


def _write_significance(per_query: dict[str, pd.DataFrame], names: dict[str, str], results_dir: Path) -> None:
    parts = [(
        "Each system vs. the TF-IDF baseline on the same test queries. Two-sided Wilcoxon signed-rank "
        "(zero differences dropped) and paired t-test; Holm correction over the 4 comparisons per metric "
        "applied to the Wilcoxon p-values. W/T/L = queries where the system scores higher / equal / lower.\n"
    )]
    for metric in SIGNIFICANCE_METRICS:
        base = per_query["tfidf"][metric]
        table = paired_tests(base, {k: df[metric] for k, df in per_query.items() if k != "tfidf"})
        table["system"] = table["system"].map(names)
        table["W/T/L"] = [f"{w}/{t}/{l}" for w, t, l in zip(table["wins"], table["ties"], table["losses"])]
        table["significant (α=0.05)"] = np.where(table["holm_p"] < 0.05, "yes", "no")
        out = table[["system", "baseline_mean", "system_mean", "delta", "W/T/L", "wilcoxon_p", "ttest_p",
                     "holm_p", "significant (α=0.05)"]].rename(columns={"baseline_mean": "TF-IDF mean",
                                                                        "system_mean": "system mean"})
        parts.append(f"\n## {metric}\n\n" + out.to_markdown(index=False, floatfmt=".4g") + "\n")
    (results_dir / "significance.md").write_text("".join(parts), encoding="utf-8")


def _write_efficiency(results_dir, names, latencies, device, build) -> None:
    rows = []
    for key, (index_s, nbytes, params) in build.items():
        lat = np.asarray(latencies[key])
        rows.append({
            "System": names[key],
            "Index build (s)": index_s,
            "Index size (MB)": nbytes / 2**20,
            "Model params (M)": params / 1e6,
            "Mean latency (ms)": lat.mean(),
            "p95 latency (ms)": np.percentile(lat, 95),
            "Device": "cpu" if key == "tfidf" else device,
        })
    (results_dir / "efficiency.md").write_text(
        "Index build = TF-IDF indexing or corpus embedding time. Latency = one query at a time, "
        f"top-{RUN_DEPTH}, after a warm-up query. Hybrid/rerank include their first-stage cost.\n\n"
        + _md(pd.DataFrame(rows), floatfmt=".2f"),
        encoding="utf-8",
    )
