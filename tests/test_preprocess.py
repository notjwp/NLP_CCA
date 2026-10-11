from irsys.preprocess import Analyzer


def test_lowercases_and_splits_on_non_alphanumerics():
    analyze = Analyzer(stem=False, stopwords=False)
    assert analyze("Hello, World! T-cells") == ["hello", "world", "t", "cells"]


def test_keeps_numbers():
    analyze = Analyzer(stem=False, stopwords=False)
    assert analyze("IL-6 rose 0.02% in 1990") == ["il", "6", "rose", "0", "02", "in", "1990"]


def test_removes_english_stopwords():
    analyze = Analyzer(stem=False, stopwords=True)
    assert analyze("The cat is on the mat") == ["cat", "mat"]


def test_applies_porter_stemming():
    analyze = Analyzer(stem=True, stopwords=False)
    assert analyze("running cells") == ["run", "cell"]


def test_stopwords_removed_before_stemming():
    # "this" is a stopword; Porter would turn it into "thi", which is not.
    analyze = Analyzer(stem=True, stopwords=True)
    assert analyze("this mutation") == ["mutat"]


def test_empty_text_gives_no_tokens():
    assert Analyzer()("") == []
    assert Analyzer()("!!! ...") == []


def test_stages_expose_each_pipeline_step():
    stages = Analyzer(stem=True, stopwords=True).stages("The mutations of IL-6")
    assert stages == {
        "tokens": ["the", "mutations", "of", "il", "6"],
        "without_stopwords": ["mutations", "il", "6"],
        "terms": ["mutat", "il", "6"],
    }


def test_stages_agree_with_call():
    analyze = Analyzer(stem=True, stopwords=False)
    assert analyze.stages("Cells were running")["terms"] == analyze("Cells were running")
