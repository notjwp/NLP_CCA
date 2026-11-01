"""Ranked-retrieval metrics, computed per query (definitions follow trec_eval / BEIR).

``ranking`` is a list of doc ids, best first. ``qrels`` maps doc id -> relevance grade;
grade > 0 means relevant. nDCG uses the grade as a linear gain with a log2 discount.

The evaluation reports exactly six metrics: nDCG@k, MAP, MRR, Precision@k, Recall@k and
F1@k, at k = 5 and 10. MAP and MRR use the whole ranking (every system returns its top 100).
"""

from math import log2

import pandas as pd

from irsys.base import Run

CUTOFFS = (5, 10)  # k for nDCG@k, P@k, R@k and F1@k
METRICS = [*(f"nDCG@{k}" for k in CUTOFFS), "MAP", "MRR",
           *(f"{name}@{k}" for name in ("P", "R", "F1") for k in CUTOFFS)]


def _relevant(qrels: dict[str, int]) -> set[str]:
    return {d for d, g in qrels.items() if g > 0}


def precision_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    rel = _relevant(qrels)
    return sum(d in rel for d in ranking[:k]) / k


def recall_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    rel = _relevant(qrels)
    return sum(d in rel for d in ranking[:k]) / len(rel) if rel else 0.0


def f1_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    """Harmonic mean of P@k and R@k (0 when both are 0)."""
    p, r = precision_at_k(ranking, qrels, k), recall_at_k(ranking, qrels, k)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def reciprocal_rank(ranking: list[str], qrels: dict[str, int], k: int | None = None) -> float:
    rel = _relevant(qrels)
    for rank, d in enumerate(ranking[:k], start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def average_precision(ranking: list[str], qrels: dict[str, int], k: int | None = None) -> float:
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
    row = {f"nDCG@{k}": ndcg_at_k(ranking, qrels, k) for k in CUTOFFS}
    row["MAP"] = average_precision(ranking, qrels)
    row["MRR"] = reciprocal_rank(ranking, qrels)
    for name, fn in (("P", precision_at_k), ("R", recall_at_k), ("F1", f1_at_k)):
        row.update({f"{name}@{k}": fn(ranking, qrels, k) for k in CUTOFFS})
    return {m: row[m] for m in METRICS}


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
    return pd.DataFrame.from_dict(rows, orient="index", columns=METRICS)
