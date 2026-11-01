from math import log2

import pytest

from irsys.metrics import (
    METRICS,
    average_precision,
    evaluate_run,
    f1_at_k,
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


def test_f1_is_harmonic_mean_of_precision_and_recall():
    # P@4 = 2/4, R@4 = 2/3 -> F1 = 2PR / (P + R) = 4/7
    assert f1_at_k(RANKING, QRELS, 4) == pytest.approx(4 / 7)
    assert f1_at_k(RANKING, QRELS, 1) == 0.0  # P = R = 0


def test_map_and_mrr_use_the_whole_ranking_by_default():
    ranking = [f"x{i}" for i in range(49)] + ["b"]  # first relevant at rank 50
    assert reciprocal_rank(ranking, QRELS) == pytest.approx(1 / 50)
    assert average_precision(ranking, QRELS) == pytest.approx((1 / 50) / 3)


def test_no_relevant_retrieved_scores_zero():
    for fn in (ndcg_at_k, average_precision, reciprocal_rank, precision_at_k, recall_at_k, f1_at_k):
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
    assert df.loc["q1", "MRR"] == pytest.approx(1.0)
    assert df.loc["q1", "F1@5"] == pytest.approx(2 * (1 / 5) * 1 / (1 / 5 + 1))
    assert df.loc["q2", "nDCG@10"] == 0.0


def test_exactly_the_six_metrics_at_k_5_and_10():
    assert METRICS == ["nDCG@5", "nDCG@10", "MAP", "MRR", "P@5", "P@10", "R@5", "R@10", "F1@5", "F1@10"]
    df = evaluate_run({"q1": {"b": 1.0}}, {"q1": {"b": 1}})
    assert list(df.columns) == METRICS
