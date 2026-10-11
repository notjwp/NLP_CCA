import pytest

from irsys.base import Retriever
from irsys.demo import DemoEngine, highlight, render_explanation, render_results, scheme_formula
from irsys.preprocess import Analyzer
from irsys.tfidf import TfidfRetriever

ANALYZER = Analyzer(stem=True, stopwords=True)
CORPUS = {
    "d1": {"title": "Gene mutations", "text": "Two mutations in the BRCA gene raise cancer risk."},
    "d2": {"title": "Vitamin D", "text": "Vitamin D deficiency is linked to bone fractures."},
    "d3": {"title": "Sleep", "text": "Sleep loss impairs memory consolidation."},
    "d4": {"title": "Exercise", "text": "Running improves cardiovascular health."},
}
QUERIES = {"q1": "Low vitamin D causes fractures", "q2": "BRCA gene mutations and vitamin intake"}
QRELS = {"q1": {"d2": 1}, "q2": {"d2": 1}}  # q2: TF-IDF ranks d1 first, the correct one is d2


class _Fixed(Retriever):
    name = "Fake semantic"

    def search(self, query, k=10):
        return [("d4", 0.9), ("d3", 0.8), ("d2", 0.7), ("d1", 0.1)][:k]


@pytest.fixture
def engine():
    ids = list(CORPUS)
    tfidf = TfidfRetriever(ANALYZER, "lsc.lsc").index(ids, [f"{CORPUS[d]['title']} {CORPUS[d]['text']}" for d in ids])
    return DemoEngine(CORPUS, QUERIES, QRELS, tfidf, {"TF-IDF": tfidf, "Semantic": _Fixed()})


def test_highlight_marks_stemmed_variants_but_not_stopwords():
    terms = set(ANALYZER("the mutated genes"))
    assert highlight("Two mutations in the gene", ANALYZER, terms) == (
        "Two <mark>mutations</mark> in the <mark>gene</mark>"
    )


def test_highlight_escapes_html():
    assert highlight("<b>gene</b> & co", ANALYZER, {"gene"}) == "&lt;b&gt;<mark>gene</mark>&lt;/b&gt; &amp; co"


def test_lookup_recognises_test_claims_ignoring_outer_whitespace(engine):
    assert engine.lookup("  Low vitamin D causes fractures ") == "q1"
    assert engine.lookup("something else") is None


def test_compare_flags_relevant_hits_and_finds_rank_beyond_top_k(engine):
    tfidf, semantic = engine.compare(QUERIES["q1"], k=2)
    assert tfidf.hits[0].doc_id == "d2" and tfidf.hits[0].relevant
    assert tfidf.correct_rank == 1
    assert [h.doc_id for h in semantic.hits] == ["d4", "d3"]  # relevant d2 is not shown...
    assert semantic.correct_rank == 3  # ...but its rank is still reported
    assert tfidf.latency_ms >= 0


def test_compare_free_text_has_no_relevance_information(engine):
    for result in engine.compare("sleep and memory", k=3):
        assert result.correct_rank is None
        assert not any(h.relevant for h in result.hits)


def test_term_table_lists_indexed_terms_with_idf_and_unit_query_weights(engine):
    table = engine.term_table("vitamin fractures zebra")
    assert list(table["term"]) == ["vitamin", "fractur"]  # zebra is not in the corpus
    assert list(table["idf"]) == pytest.approx([engine.tfidf.idf("vitamin"), engine.tfidf.idf("fractur")])
    assert (table["query weight"] ** 2).sum() == pytest.approx(1.0)


def test_contributions_sum_to_score_and_tfidf_rank(engine):
    table, score, rank = engine.contributions("vitamin D fractures", "d2")
    assert table["contribution"].sum() == pytest.approx(score)
    assert score == pytest.approx(engine.tfidf.scores("vitamin D fractures")[1])
    assert rank == 1


def test_oov_terms_are_reported(engine):
    assert engine.oov_terms("ivermectin and vitamin") == ["ivermectin"]


def test_rendered_results_show_check_mark_and_rank(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"])
    assert "✓" in html
    assert "rank 3" in html  # semantic system's correct abstract
    assert "<mark>" in html


def test_scheme_formula_describes_weighting():
    text = scheme_formula("lsc.lsc")
    assert "1 + ln tf" in text and "ln(N/df) + 1" in text and "length 1" in text


def test_explanation_walks_through_every_step(engine):
    html = render_explanation(engine, QUERIES["q1"])
    for step in ("Step 1", "Step 2", "Step 3"):
        assert step in html
    assert "the correct abstract ✓" in html  # TF-IDF's top hit is the correct one...
    assert "<b>Correct abstract</b>" not in html  # ...so it is not explained twice


def test_explanation_contrasts_top_hit_with_correct_abstract(engine):
    html = render_explanation(engine, QUERIES["q2"])
    assert html.count('class="doc"') == 2
    assert "#1 result" in html and "<b>Correct abstract</b>" in html
    assert "Gene mutations" in html and "Vitamin D" in html


def test_explanation_flags_terms_missing_from_corpus(engine):
    html = render_explanation(engine, "ivermectin vitamin")
    assert "occur in <b>no</b> abstract" in html and "ivermectin" in html


def test_explanation_when_no_query_term_is_indexed(engine):
    html = render_explanation(engine, "zebra quokka")
    assert "None of the query terms occur in the corpus" in html
    assert "Step 3" not in html


def test_correct_abstract_below_the_visible_list_is_flagged_not_ticked(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"])
    assert "✓ correct abstract at rank 1" in html  # TF-IDF: shown in the list
    assert "correct abstract only at rank 3" in html  # semantic: below the 2 shown results
    assert "✓ correct abstract at rank 3" not in html
