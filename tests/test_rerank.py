import numpy as np

from irsys.base import Retriever
from irsys.rerank import RerankRetriever

TEXTS = {"short": "a b", "longest": "a b c d e f", "mid": "a b c d"}


class _FirstStage(Retriever):
    def search(self, query, k=10):
        return [("short", 0.9), ("mid", 0.8), ("longest", 0.7)][:k]


class _LengthScorer:
    """Stand-in for a cross-encoder: longer text = more relevant."""

    def __init__(self):
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return np.array([len(t) for t in texts], dtype=float)


def test_rerank_reorders_candidates_by_reranker_score():
    results = RerankRetriever(_FirstStage(), _LengthScorer(), TEXTS, depth=3).search("q", k=3)
    assert [d for d, _ in results] == ["longest", "mid", "short"]
    assert results[0][1] == len(TEXTS["longest"])


def test_rerank_only_scores_first_stage_depth():
    scorer = _LengthScorer()
    results = RerankRetriever(_FirstStage(), scorer, TEXTS, depth=2).search("q", k=10)
    assert scorer.calls == [("q", [TEXTS["short"], TEXTS["mid"]])]
    assert [d for d, _ in results] == ["mid", "short"]


def test_rerank_returns_top_k():
    assert len(RerankRetriever(_FirstStage(), _LengthScorer(), TEXTS, depth=3).search("q", k=1)) == 1


def test_rerank_with_no_candidates_skips_reranker():
    class _Empty(Retriever):
        def search(self, query, k=10):
            return []

    scorer = _LengthScorer()
    assert RerankRetriever(_Empty(), scorer, TEXTS).search("q") == []
    assert scorer.calls == []
