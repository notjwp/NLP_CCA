"""Command line: ``irsys evaluate | analyze | search "query"``."""

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irsys", description="TF-IDF/cosine vs. semantic search on BEIR SciFact")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluate", help="run the TF-IDF ablation (train) and all systems (test); write results/")
    sub.add_parser("analyze", help="figures and example queries from results/")
    search = sub.add_parser("search", help="show TF-IDF and dense results side by side for one query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5, help="results per system (default 5)")
    sub.add_parser("demo", help="open the interactive presentation demo in the browser")
    return parser


def _demo() -> None:
    import os
    from pathlib import Path

    # Once the models are downloaded, run fully offline: no Hub round-trips, so no stalls on bad Wi-Fi.
    # (Must be set before huggingface_hub is imported.)
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    if (hub / "models--BAAI--bge-small-en-v1.5").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    from irsys.demo import launch_demo

    launch_demo()


def _search(query: str, k: int) -> None:
    from irsys.data import RESULTS_DIR, doc_text, load_scifact
    from irsys.dense import DenseRetriever
    from irsys.evaluate import BGE, BGE_QUERY_PROMPT
    from irsys.preprocess import Analyzer
    from irsys.tfidf import TfidfRetriever

    corpus, _, _ = load_scifact("test")
    doc_ids = list(corpus)
    texts = [doc_text(corpus[d]) for d in doc_ids]
    config_path = RESULTS_DIR / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))["tfidf"] if config_path.exists() else {
        "scheme": "ltc.ltc", "stem": True, "stopwords": True}
    tfidf = TfidfRetriever(Analyzer(cfg["stem"], cfg["stopwords"]), cfg["scheme"]).index(doc_ids, texts)
    dense = DenseRetriever(BGE, query_prompt=BGE_QUERY_PROMPT, name="Dense (bge-small-en-v1.5)").index(doc_ids, texts)

    print(f"\nQuery: {query}")
    for retriever in (tfidf, dense):
        print(f"\n{retriever.name}")
        results = retriever.search(query, k)
        if not results:
            print("   (no document shares a term with the query)")
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"  {rank:>2}. {score:6.3f}  [{doc}] {corpus[doc]['title'][:100]}")


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        from irsys.evaluate import evaluate

        evaluate()
    elif args.command == "analyze":
        from irsys.analysis import analyze

        analyze()
    elif args.command == "demo":
        _demo()
    else:
        _search(args.query, args.k)
