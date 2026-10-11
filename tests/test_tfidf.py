from math import log, sqrt

import numpy as np
import pytest

from irsys.preprocess import Analyzer
from irsys.tfidf import SklearnTfidfRetriever, TfidfRetriever

DOC_IDS = ["d1", "d2", "d3"]
TEXTS = ["apple banana apple", "banana cherry", "cherry cherry date"]
RAW = Analyzer(stem=False, stopwords=False)


def build(scheme="ltc.ltc", doc_ids=DOC_IDS, texts=TEXTS):
    return TfidfRetriever(analyzer=RAW, scheme=scheme).index(doc_ids, texts)


def test_ltc_cosine_matches_hand_computation():
    # df: apple=1, banana=2, cherry=2, date=1; N=3
    w_apple = (1 + log(2)) * log(3)  # tf=2 in d1
    w_banana = 1 * log(3 / 2)
    results = build().search("banana", k=10)
    assert [d for d, _ in results] == ["d2", "d1"]  # d3 has no query term
    assert results[0][1] == pytest.approx(1 / sqrt(2))  # banana, cherry equal weight
    assert results[1][1] == pytest.approx(w_banana / sqrt(w_apple**2 + w_banana**2))


def test_lnc_doc_side_has_no_idf():
    results = dict(build("lnc.ltc").search("banana", k=10))
    assert results["d1"] == pytest.approx(1 / sqrt((1 + log(2)) ** 2 + 1))
    assert results["d2"] == pytest.approx(1 / sqrt(2))


def test_ntc_uses_raw_term_frequency():
    results = dict(build("ntc.ntc").search("banana", k=10))
    w_apple, w_banana = 2 * log(3), log(3 / 2)
    assert results["d1"] == pytest.approx(w_banana / sqrt(w_apple**2 + w_banana**2))


def test_repeated_query_terms_raise_their_weight():
    # ltc query: "cherry cherry date" weights cherry by 1+ln2, date by 1 (times idf)
    q_cherry, q_date = (1 + log(2)) * log(1.5), log(3)
    q_norm = sqrt(q_cherry**2 + q_date**2)
    d3_cherry, d3_date = (1 + log(2)) * log(1.5), log(3)
    d3_norm = sqrt(d3_cherry**2 + d3_date**2)
    expected = (q_cherry * d3_cherry + q_date * d3_date) / (q_norm * d3_norm)
    results = dict(build().search("cherry cherry date", k=10))
    assert results["d3"] == pytest.approx(expected)
    assert results["d3"] == pytest.approx(1.0)  # identical vectors


def test_k_limits_number_of_results():
    assert len(build().search("banana cherry", k=1)) == 1


def test_out_of_vocabulary_query_returns_nothing():
    assert build().search("zebra", k=10) == []


def test_empty_query_returns_nothing():
    assert build().search("", k=10) == []


def test_empty_document_scores_zero_not_nan():
    r = build(doc_ids=["d1", "empty"], texts=["apple banana", ""])
    scores = r.scores("apple")
    assert not np.isnan(scores).any()
    assert scores[1] == 0.0


def test_invalid_scheme_is_rejected():
    with pytest.raises(ValueError):
        TfidfRetriever(scheme="xyz.ltc")


def test_search_batch_returns_run_dict():
    run = build().search_batch({"q1": "banana", "q2": "zebra"}, k=10)
    assert set(run) == {"q1", "q2"}
    assert run["q1"] == dict(build().search("banana", k=10))
    assert run["q2"] == {}


def test_index_size_is_reported():
    assert build().nbytes > 0


def _synthetic_corpus(n_docs=200, vocab=300, seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(vocab)]
    p = 1 / np.arange(1, vocab + 1)  # Zipf-like term distribution
    p /= p.sum()
    texts = [" ".join(rng.choice(words, size=rng.integers(5, 60), p=p)) for _ in range(n_docs)]
    queries = [" ".join(rng.choice(words, size=rng.integers(1, 6), p=p)) for _ in range(25)]
    return [f"d{i}" for i in range(n_docs)], texts, queries


@pytest.mark.parametrize("corpus", ["toy", "synthetic"])
def test_matches_sklearn_tfidf_cosine(corpus):
    # sklearn: sublinear tf (1+ln tf), idf = ln(N/df)+1 (smooth_idf=False), L2 norm == SMART lsc.lsc
    if corpus == "toy":
        doc_ids, texts, queries = DOC_IDS, TEXTS, ["banana", "cherry date", "apple apple cherry"]
    else:
        doc_ids, texts, queries = _synthetic_corpus()
    ours = TfidfRetriever(analyzer=RAW, scheme="lsc.lsc").index(doc_ids, texts)
    ref = SklearnTfidfRetriever(analyzer=RAW).index(doc_ids, texts)
    for q in queries:
        np.testing.assert_allclose(ours.scores(q), ref.scores(q), atol=1e-9)


@pytest.mark.parametrize("scheme", ["ltc.ltc", "lnc.ltc", "lsc.lsc"])
def test_explain_contributions_sum_to_cosine_score(scheme):
    r = build(scheme)
    for doc_index, doc_id in enumerate(DOC_IDS):
        parts = r.explain("banana cherry zebra", doc_id)
        assert sum(w_q * w_d for _, w_q, w_d in parts) == pytest.approx(r.scores("banana cherry zebra")[doc_index])


def test_explain_lists_indexed_query_terms_with_zero_for_absent():
    parts = {term: (w_q, w_d) for term, w_q, w_d in build().explain("banana cherry zebra", "d1")}
    assert set(parts) == {"banana", "cherry"}  # "zebra" is not in the index
    assert parts["cherry"][1] == 0.0  # d1 has no "cherry"
    assert parts["banana"][1] > 0


def test_idf_uses_the_query_side_scheme():
    assert build("ltc.ltc").idf("banana") == pytest.approx(log(3 / 2))
    assert build("lsc.lsc").idf("banana") == pytest.approx(log(3 / 2) + 1)
    assert build("lnc.ltc").idf("apple") == pytest.approx(log(3))
