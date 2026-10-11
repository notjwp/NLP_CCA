import pandas as pd
import pytest

from irsys.analysis import first_relevant_rank, lexical_overlap, overlap_terciles, read_trec_run
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
