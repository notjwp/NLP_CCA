"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009): score(d) = sum_runs 1 / (k + rank)."""

from collections import defaultdict

from irsys.base import Ranking, Retriever, Run


def _fuse(rankings: list[list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def rrf(runs: list[Run], k: int = 60, depth: int = 1000) -> Run:
    """Fuse runs by rank alone, so differently-scaled scores (cosine, logits) combine safely."""
    qids = dict.fromkeys(q for run in runs for q in run)
    return {
        qid: _fuse([sorted(s, key=s.get, reverse=True)[:depth] for s in (run.get(qid, {}) for run in runs)], k)
        for qid in qids
    }


class RRFHybrid(Retriever):
    def __init__(self, retrievers: list[Retriever], k: int = 60, depth: int = 1000, name: str | None = None):
        self.retrievers = retrievers
        self.k = k
        self.depth = depth
        self.name = name or "Hybrid RRF (" + " + ".join(r.name for r in retrievers) + ")"

    def search(self, query: str, k: int = 10) -> Ranking:
        fused = _fuse([[d for d, _ in r.search(query, self.depth)] for r in self.retrievers], self.k)
        return list(fused.items())[:k]

    def search_batch(self, queries: dict[str, str], k: int = 100) -> Run:
        fused = rrf([r.search_batch(queries, self.depth) for r in self.retrievers], self.k, self.depth)
        return {qid: dict(list(fused.get(qid, {}).items())[:k]) for qid in queries}
