"""Ranked-retrieval metrics, computed per query (definitions follow trec_eval / BEIR).

``ranking`` is a list of doc ids, best first. ``qrels`` maps doc id -> relevance grade;
grade > 0 means relevant. nDCG uses the grade as a linear gain with a log2 discount.
"""

from math import log2

import pandas as pd

from irsys.base import Run

RECALL_KS = (1, 2, 3, 5, 10, 20, 50, 100)


def _relevant(qrels: dict[str, int]) -> set[str]:
    return {d for d, g in qrels.items() if g > 0}


def precision_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    rel = _relevant(qrels)
    return sum(d in rel for d in ranking[:k]) / k


def recall_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    rel = _relevant(qrels)
    return sum(d in rel for d in ranking[:k]) / len(rel) if rel else 0.0


def reciprocal_rank(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    rel = _relevant(qrels)
    for rank, d in enumerate(ranking[:k], start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def average_precision(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    rel = _relevant(qrels)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for rank, d in enumerate(ranking[:k], start=1):
        if d in rel:
            hits += 1
            total += hits / rank
    return total / len(rel)


def ndcg_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    dcg = sum(qrels.get(d, 0) / log2(rank + 1) for rank, d in enumerate(ranking[:k], start=1) if qrels.get(d, 0) > 0)
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def query_metrics(ranking: list[str], qrels: dict[str, int]) -> dict[str, float]:
    row = {
        "nDCG@10": ndcg_at_k(ranking, qrels, 10),
        "MAP@100": average_precision(ranking, qrels, 100),
        "MRR@10": reciprocal_rank(ranking, qrels, 10),
        "P@10": precision_at_k(ranking, qrels, 10),
    }
    row.update({f"R@{k}": recall_at_k(ranking, qrels, k) for k in RECALL_KS})
    return row


def evaluate_run(run: Run, qrels: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Per-query metrics for every judged query with at least one relevant doc.

    Judged queries missing from the run score zero rather than being dropped.
    """
    rows = {}
    for qid, judged in qrels.items():
        if not _relevant(judged):
            continue
        scored = run.get(qid, {})
        ranking = sorted(scored, key=scored.get, reverse=True)  # stable: ties keep run order
        rows[qid] = query_metrics(ranking, judged)
    return pd.DataFrame.from_dict(rows, orient="index")
