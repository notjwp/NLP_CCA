"""Common retriever interface so evaluation, fusion and the CLI treat every system alike."""

import numpy as np

Ranking = list[tuple[str, float]]
Run = dict[str, dict[str, float]]  # query id -> {doc id: score}


class Retriever:
    name: str = "retriever"

    def search(self, query: str, k: int = 10) -> Ranking:
        raise NotImplementedError

    def search_batch(self, queries: dict[str, str], k: int = 100) -> Run:
        return {qid: dict(self.search(text, k)) for qid, text in queries.items()}


def top_k(scores: np.ndarray, doc_ids: list[str], k: int, positive_only: bool = False) -> Ranking:
    """Best-k (doc id, score) pairs by descending score; ties broken by corpus order."""
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    idx = idx[np.lexsort((idx, -scores[idx]))]
    if positive_only:
        idx = idx[scores[idx] > 0]
    return [(doc_ids[i], float(scores[i])) for i in idx]
