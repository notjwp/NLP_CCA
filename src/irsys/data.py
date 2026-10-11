"""BEIR SciFact (Wadden et al., 2020; Thakur et al., 2021), fetched from the Hugging Face Hub."""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]  # project root (src/irsys/data.py -> ../..)
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

Corpus = dict[str, dict[str, str]]  # doc id -> {"title", "text"}
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]  # query id -> {doc id: grade}


def _fetch(repo_id: str, filename: str, data_dir: Path) -> Path:
    local = data_dir / repo_id / filename
    if local.exists():  # downloaded on an earlier run; skip the Hub round-trip
        return local
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=data_dir / repo_id))


def load_scifact(split: str = "test", data_dir: Path = DATA_DIR) -> tuple[Corpus, Queries, Qrels]:
    """Corpus plus the queries and relevance judgments of one split ("train" or "test")."""
    corpus_df = pd.read_parquet(_fetch("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", data_dir))
    queries_df = pd.read_parquet(_fetch("BeIR/scifact", "queries/queries-00000-of-00001.parquet", data_dir))
    qrels_df = pd.read_csv(_fetch("BeIR/scifact-qrels", f"{split}.tsv", data_dir), sep="\t", dtype=str)

    corpus = {
        str(did): {"title": title or "", "text": text or ""}
        for did, title, text in zip(corpus_df["_id"], corpus_df["title"], corpus_df["text"])
    }
    qrels: Qrels = {}
    for qid, did, grade in qrels_df[["query-id", "corpus-id", "score"]].itertuples(index=False):
        qrels.setdefault(qid, {})[did] = int(grade)
    all_queries = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
    queries = {qid: all_queries[qid] for qid in qrels}
    return corpus, queries, qrels


def doc_text(doc: dict[str, str]) -> str:
    """Text every system indexes: title followed by abstract."""
    return f"{doc['title']} {doc['text']}".strip()
