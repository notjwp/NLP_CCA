import pytest

from irsys.base import Retriever
from irsys.fusion import RRFHybrid, rrf

RUN_A = {"q": {"x": 3.0, "y": 2.0, "z": 1.0}}  # ranks x, y, z
RUN_B = {"q": {"y": 0.9, "w": 0.5}, "q2": {"w": 1.0}}  # ranks y, w


def test_rrf_sums_reciprocal_ranks():
    fused = rrf([RUN_A, RUN_B], k=60)["q"]
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["w"] == pytest.approx(1 / 62)
    assert fused["z"] == pytest.approx(1 / 63)
    assert list(fused) == ["y", "x", "w", "z"]  # stored best-first


def test_rrf_ignores_scores_beyond_rank_depth():
    fused = rrf([RUN_A, RUN_B], k=60, depth=1)["q"]
    assert set(fused) == {"x", "y"}


def test_rrf_keeps_queries_found_in_only_one_run():
    assert rrf([RUN_A, RUN_B])["q2"] == {"w": pytest.approx(1 / 61)}


class _Fixed(Retriever):
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, k=10):
        return self.ranking[:k]


def test_hybrid_retriever_fuses_component_searches():
    hybrid = RRFHybrid([_Fixed([("x", 3.0), ("y", 2.0)]), _Fixed([("y", 0.9), ("w", 0.5)])])
    results = hybrid.search("anything", k=2)
    assert [d for d, _ in results] == ["y", "x"]
    assert results[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_search_batch_matches_single_search():
    hybrid = RRFHybrid([_Fixed([("x", 3.0), ("y", 2.0)]), _Fixed([("y", 0.9), ("w", 0.5)])])
    run = hybrid.search_batch({"q1": "a", "q2": "b"}, k=2)
    assert run["q1"] == pytest.approx(dict(hybrid.search("a", k=2)))
    assert list(run["q1"]) == ["y", "x"]
