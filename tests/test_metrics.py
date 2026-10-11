from math import log2

import pytest

from irsys.metrics import (
    average_precision,
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKING = ["a", "b", "c", "d"]
QRELS = {"b": 1, "d": 1, "e": 1}  # 3 relevant, "e" never retrieved


def test_precision_at_k():
    assert precision_at_k(RANKING, QRELS, 4) == pytest.approx(2 / 4)
    assert precision_at_k(RANKING, QRELS, 2) == pytest.approx(1 / 2)


def test_precision_divides_by_k_even_if_fewer_results():
    assert precision_at_k(["b"], QRELS, 10) == pytest.approx(1 / 10)


def test_recall_at_k():
    assert recall_at_k(RANKING, QRELS, 4) == pytest.approx(2 / 3)
    assert recall_at_k(RANKING, QRELS, 1) == 0.0


def test_reciprocal_rank_of_first_relevant():
    assert reciprocal_rank(RANKING, QRELS, 10) == pytest.approx(1 / 2)
    assert reciprocal_rank(RANKING, QRELS, 1) == 0.0  # cut off before "b"


def test_average_precision_divides_by_all_relevant():
    # precision at the relevant ranks 2 and 4 is 1/2 and 2/4; R = 3
    assert average_precision(RANKING, QRELS, 100) == pytest.approx((1 / 2 + 2 / 4) / 3)


def test_ndcg_binary():
    dcg = 1 / log2(3) + 1 / log2(5)
    idcg = 1 / log2(2) + 1 / log2(3) + 1 / log2(4)
    assert ndcg_at_k(RANKING, QRELS, 4) == pytest.approx(dcg / idcg)


def test_ndcg_uses_graded_gains():
    qrels = {"a": 2, "b": 1}
    dcg = 1 / log2(2) + 2 / log2(3)
    idcg = 2 / log2(2) + 1 / log2(3)
    assert ndcg_at_k(["b", "a"], qrels, 10) == pytest.approx(dcg / idcg)


def test_perfect_ranking_has_ndcg_one():
    assert ndcg_at_k(["b", "d", "e"], QRELS, 10) == pytest.approx(1.0)


def test_relevant_below_cutoff_does_not_count():
    ranking = [f"x{i}" for i in range(10)] + ["b"]
    assert ndcg_at_k(ranking, QRELS, 10) == 0.0


def test_zero_grade_judgments_are_not_relevant():
    qrels = {"a": 0, "b": 1}
    assert precision_at_k(["a"], qrels, 1) == 0.0
    assert recall_at_k(["b"], qrels, 1) == 1.0


def test_no_relevant_retrieved_scores_zero():
    for fn in (ndcg_at_k, average_precision, reciprocal_rank, precision_at_k, recall_at_k):
        assert fn(["x", "y"], QRELS, 10) == 0.0


def test_evaluate_run_ranks_by_score_and_covers_judged_queries():
    run = {
        "q1": {"a": 0.1, "b": 0.9},  # "b" ranked first despite dict order
        # "q2" missing from the run: must count as all-zero, not be dropped
    }
    qrels = {"q1": {"b": 1}, "q2": {"z": 1}, "q3": {"z": 0}}  # q3 has nothing relevant: skipped
    df = evaluate_run(run, qrels)
    assert sorted(df.index) == ["q1", "q2"]
    assert df.loc["q1", "nDCG@10"] == pytest.approx(1.0)
    assert df.loc["q1", "MRR@10"] == pytest.approx(1.0)
    assert df.loc["q2", "nDCG@10"] == 0.0
    assert {"nDCG@10", "MAP@100", "MRR@10", "P@10", "R@10", "R@100", "R@1", "R@50"} <= set(df.columns)
