import pytest

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


# ---------------------------------------------------------------- POS tagging and parsing

def test_pos_tags_align_with_tokens():
    tagged = Analyzer().tag("Silencing of Bcl2 is important")
    assert [token for token, _ in tagged] == ["silencing", "of", "bcl2", "is", "important"]
    assert [tag for _, tag in tagged][1:] == ["IN", "NN", "VBZ", "JJ"]  # Penn Treebank tags


def test_lemmatization_uses_the_pos_tag():
    analyze = Analyzer(stem=False, stopwords=True, lemmatize=True)
    assert analyze("the studies increased") == ["study", "increase"]
    # Same word, different part of speech: verb -> base form, noun -> unchanged.
    assert analyze("they were meeting today") == ["meet", "today"]
    assert analyze("the meeting ended early") == ["meeting", "end", "early"]


def test_stemming_and_lemmatization_are_exclusive():
    with pytest.raises(ValueError):
        Analyzer(stem=True, lemmatize=True)


def test_noun_phrase_chunks():
    chunks = Analyzer().noun_phrases("LDL cholesterol has no involvement in the development of cardiovascular disease")
    assert chunks == [["ldl", "cholesterol"], ["involvement"], ["development"], ["cardiovascular", "disease"]]


def test_phrase_terms_join_adjacent_words_of_each_noun_phrase():
    analyze = Analyzer(stem=True, stopwords=True, phrases=True)
    assert analyze("LDL cholesterol and cardiovascular disease") == [
        "ldl", "cholesterol", "cardiovascular", "diseas", "ldl_cholesterol", "cardiovascular_diseas"]
    # A three-word phrase gives two adjacent pairs.
    assert analyze("low density lipoprotein")[-2:] == ["low_densiti", "densiti_lipoprotein"]


def test_phrase_terms_skip_stopwords_inside_the_phrase():
    # "several" is tagged JJ, so it is inside the chunk, but it is a stopword.
    assert Analyzer(stem=False, phrases=True)("several cell lines") == ["cell", "lines", "cell_lines"]


def test_phrase_terms_use_lemmas_when_lemmatizing():
    assert Analyzer(stem=False, lemmatize=True, phrases=True)("cell lines")[-1] == "cell_line"


def test_single_word_noun_phrases_add_no_phrase_terms():
    claim = "Ivermectin is used to treat onchocerciasis."  # NN VBZ VBN TO VB NN: two one-word chunks
    assert Analyzer(stem=True, phrases=True)(claim) == Analyzer(stem=True)(claim)


def test_config_round_trip():
    analyzer = Analyzer(stem=False, stopwords=True, lemmatize=True, phrases=True)
    assert Analyzer.from_config(analyzer.config()).config() == analyzer.config()
    # Configs written before POS tagging existed have only stem/stopwords.
    old = Analyzer.from_config({"scheme": "lsc.lsc", "stem": True, "stopwords": True})
    assert old.config() == {"stem": True, "stopwords": True, "lemmatize": False, "phrases": False}


def test_pos_options_handle_empty_text():
    assert Analyzer(stem=False, lemmatize=True, phrases=True)("!!!") == []
    assert Analyzer().tag("") == []
    assert Analyzer().noun_phrases("") == []
