import pandas as pd
import pytest

from irsys.analysis import (
    error_table,
    failure_category,
    first_relevant_rank,
    lexical_overlap,
    overlap_terciles,
    read_trec_run,
    read_trec_scores,
    rerank_failures,
    spelling_variant,
)
from irsys.evaluate import write_trec_run
from irsys.preprocess import Analyzer

RAW = Analyzer(stem=False, stopwords=False)


def test_lexical_overlap_is_share_of_query_terms_in_relevant_docs():
    corpus = {"d1": "apple banana", "d2": "cherry", "d3": "apple cherry date"}
    queries = {"q1": "apple date zebra", "q2": "zebra"}
    qrels = {"q1": {"d1": 1, "d3": 1, "d2": 0}, "q2": {"d2": 1}}
    overlap = lexical_overlap(queries, qrels, corpus, RAW)
    assert overlap["q1"] == pytest.approx(2 / 3)  # apple, date found; zebra not; d2 is grade 0
    assert overlap["q2"] == 0.0


def test_lexical_overlap_counts_unique_query_terms():
    overlap = lexical_overlap({"q": "apple apple pear"}, {"q": {"d": 1}}, {"d": "apple"}, RAW)
    assert overlap["q"] == pytest.approx(1 / 2)


def test_overlap_terciles_split_evenly_even_with_ties():
    overlap = pd.Series([0.0, 0.0, 0.0, 0.5, 0.5, 1.0], index=list("abcdef"))
    buckets = overlap_terciles(overlap)
    assert buckets.value_counts().to_dict() == {"low": 2, "medium": 2, "high": 2}
    assert buckets["f"] == "high" and buckets["a"] == "low"


def test_trec_run_round_trip(tmp_path):
    run = {"q1": {"d1": 0.9, "d2": 0.5}, "q2": {"d3": 1.0}}
    write_trec_run(run, tmp_path / "r.trec", "tag")
    assert read_trec_run(tmp_path / "r.trec") == {"q1": ["d1", "d2"], "q2": ["d3"]}


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], {"c": 1, "b": 0}) == 3
    assert first_relevant_rank(["a"], {"z": 1}) is None


# ---------------------------------------------------------------- error analysis

def test_spelling_variant_detects_hyphenated_names_both_ways():
    assert spelling_variant("Silencing of Bcl2 is important", "Antiapoptotic BCL-2 is required")
    assert spelling_variant("IL-6 levels rise", "IL6 was elevated")
    assert not spelling_variant("vitamin D prevents fractures", "a vitamin D trial on fractures")
    # Joining a stopword to its neighbour is not a spelling variant ("is" + "land").
    assert not spelling_variant("island", "it is land")


STEM = Analyzer(stem=True, stopwords=True)
DF = {"vitamin": 50, "fractur": 40, "elder": 200, "onchocerciasi": 1, "treat": 900, "drug": 700}


def test_failure_category_rules_in_order():
    assert failure_category("Bcl2 silencing", ["BCL-2 silencing"], STEM, DF) == "spelling variant"
    # 1 of 3 claim terms in the abstract.
    assert failure_category("vitamin fractures elderly", ["elderly people"], STEM, DF) == "vocabulary mismatch"
    # Most terms shared, one of them in only 1 abstract.
    assert failure_category("drugs treat onchocerciasis", ["drug to treat onchocerciasis"], STEM,
                            DF) == "rare shared term"
    assert failure_category("vitamin fractures", ["vitamin and fractures"], STEM, DF) == "common words only"


def test_failure_category_uses_any_relevant_abstract():
    assert failure_category("vitamin fractures", ["unrelated", "vitamin fractures"], STEM, DF) == "common words only"


def test_error_table_counts_failures_per_category_and_system():
    category = pd.Series({"q1": "vocabulary mismatch", "q2": "vocabulary mismatch", "q3": "rare shared term"})
    ranks = pd.DataFrame({"tfidf": [None, 3, 1], "dense": [2, 11, 40]}, index=["q1", "q2", "q3"])
    table = error_table(category, ranks, order=["vocabulary mismatch", "rare shared term", "spelling variant"])
    assert list(table["category"]) == ["vocabulary mismatch", "rare shared term", "all claims"]  # empty ones dropped
    assert list(table["claims"]) == [2, 1, 3]
    assert list(table["tfidf"]) == [1, 0, 1]  # missing from the top 100 counts as a failure
    assert list(table["dense"]) == [1, 1, 2]  # rank 11 is outside the top 10


def test_rerank_failures_split_first_stage_misses_from_reranker_mistakes():
    first_stage = pd.Series({"q1": None, "q2": 5, "q3": 50, "q4": 2})  # None: not in the first-stage top 100
    reranked = pd.Series({"q1": None, "q2": 30, "q3": 1, "q4": 1})
    assert rerank_failures(first_stage, reranked) == {"failures": 2, "first stage missed it": 1,
                                                     "reranker ranked it low": 1}


def test_read_trec_scores(tmp_path):
    write_trec_run({"q1": {"d1": 0.9, "d2": -1.5}}, tmp_path / "r.trec", "tag")
    assert read_trec_scores(tmp_path / "r.trec") == {"q1": {"d1": 0.9, "d2": -1.5}}
