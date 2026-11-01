import numpy as np
import pytest

from irsys.base import Retriever
from irsys.demo import (
    DemoEngine,
    highlight,
    render_explanation,
    render_llm_explanation,
    render_results,
    scheme_formula,
)
from irsys.preprocess import Analyzer
from irsys.rerank import RerankRetriever
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


class _FakeLLM:
    """Stand-in for LLMReranker: says yes (log-odds +2) to papers mentioning vitamins, no (-3) otherwise."""

    instruction = "Given a scientific claim, retrieve documents that support or refute the claim"

    def score(self, query, texts):
        return np.array([2.0 if "vitamin" in t.lower() else -3.0 for t in texts])


@pytest.fixture
def engine():
    ids = list(CORPUS)
    texts = [f"{CORPUS[d]['title']} {CORPUS[d]['text']}" for d in ids]
    tfidf = TfidfRetriever(ANALYZER, "lsc.lsc").index(ids, texts)
    return DemoEngine(CORPUS, QUERIES, QRELS, tfidf, {"TF-IDF": tfidf, "Semantic": _Fixed()})


@pytest.fixture
def llm_engine(engine):
    texts = {d: f"{CORPUS[d]['title']} {CORPUS[d]['text']}" for d in CORPUS}
    engine.systems["LLM"] = RerankRetriever(engine.tfidf, _FakeLLM(), texts, depth=100)
    engine.llm_name = "LLM"
    return engine


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
    assert "position 3" in html  # semantic system's correct abstract
    assert "<mark>" in html


def test_rendered_results_explain_each_system_in_plain_words(engine):
    engine.descriptions = {"TF-IDF": "Matches the words.", "Semantic": "Matches the meaning."}
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"])
    assert "Matches the words." in html and "Matches the meaning." in html


def test_result_cards_show_positions_not_raw_scores(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"])
    assert "#1" in html and "score" not in html.lower()


def test_simple_results_table_has_the_four_metrics_that_matter(tmp_path):
    import pandas as pd

    from irsys.demo import simple_results_table

    names = {"tfidf": "TF-IDF cosine", "dense-bge": "Dense: bge", "tfidf-llm-zs": "LLM zero-shot"}
    pd.DataFrame({"system": list(names), "name": list(names.values()), "nDCG@10": [0.5, 0.7, 0.1],
                  "R@100": [0.93, 0.95, 0.93]}).to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "efficiency.md").write_text(
        "Latency note.\n\n" + pd.DataFrame({"System": list(names.values()), "Mean latency (ms)": [0.17, 12.46, 5728.7]})
        .to_markdown(index=False), encoding="utf-8")
    scores = {"tfidf": [0.5] * 30, "dense-bge": [1.0] * 25 + [0.5] * 5, "tfidf-llm-zs": [0.0] * 25 + [0.5] * 5}
    pd.DataFrame([{"system": s, "qid": str(i), "nDCG@10": v} for s, vs in scores.items() for i, v in enumerate(vs)]
                 ).to_csv(tmp_path / "per_query.csv", index=False)

    table = simple_results_table(tmp_path)
    assert "| System | Grade (nDCG@10) | Correct paper in top 100 | Time per claim | Better than TF-IDF? |" in table
    assert "| Word matching · TF-IDF (ours) | 0.500 | 93% | 0.17 ms | — (baseline) |" in table
    assert "| Meaning matching · bge-small | 0.700 | 95% | 12.5 ms | yes |" in table
    assert "| LLM, prompted only | 0.100 | 93% | 5.7 s | no, worse |" in table


def test_slow_answers_are_shown_in_seconds():
    from irsys.demo import SystemResult

    fast = SystemResult("Fast", [], False, None, 27.43)
    slow = SystemResult("Slow", [], False, None, 6898.6)
    html = render_results([fast, slow], ANALYZER, "q")
    assert "answered in 27.4 ms" in html and "answered in 6.9 s" in html


def test_rendered_results_say_when_the_correct_paper_is_missing(engine):
    class _Nothing(Retriever):
        def search(self, query, k=10):
            return [("d4", 0.9)]

    engine.systems = {"Nothing": _Nothing()}
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"])
    assert "✗ correct paper not in its top 100" in html


def test_scheme_formula_describes_weighting():
    text = scheme_formula("lsc.lsc")
    assert "1 + ln tf" in text and "ln(N/df) + 1" in text and "length 1" in text


def test_explanation_walks_through_every_step(engine):
    html = render_explanation(engine, QUERIES["q1"])
    for step in ("Step 1", "Step 2", "Step 3"):
        assert step in html
    assert "the correct paper ✓" in html  # TF-IDF's top hit is the correct one...
    assert "<b>Correct paper</b>" not in html  # ...so it is not explained twice


def test_explanation_contrasts_top_hit_with_correct_abstract(engine):
    html = render_explanation(engine, QUERIES["q2"])
    assert html.count('class="doc"') == 2
    assert "#1 result" in html and "<b>Correct paper</b>" in html
    assert "Gene mutations" in html and "Vitamin D" in html


def test_explanation_flags_terms_missing_from_corpus(engine):
    html = render_explanation(engine, "ivermectin vitamin")
    assert "appear in <b>no</b> paper" in html and "ivermectin" in html


def test_explanation_when_no_query_term_is_indexed(engine):
    html = render_explanation(engine, "zebra quokka")
    assert "None of the claim's words appear in any paper" in html
    assert "Step 3" not in html


def test_correct_abstract_below_the_visible_list_is_flagged_not_ticked(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"])
    assert "✓ found the correct paper at position 1" in html  # TF-IDF: shown in the list
    assert "▼ correct paper only at position 3" in html  # semantic: below the 2 shown results
    assert "✓ found the correct paper at position 3" not in html


def test_fast_and_slow_systems_can_be_compared_separately(llm_engine):
    fast = llm_engine.compare(QUERIES["q2"], k=2, names=llm_engine.fast_names())
    assert [r.name for r in fast] == ["TF-IDF", "Semantic"]
    (llm,) = llm_engine.compare(QUERIES["q2"], k=2, names=["LLM"])
    assert llm.hits[0].doc_id == "d2" and llm.correct_rank == 1  # TF-IDF had it 2nd; the LLM says yes to it


def test_engine_without_llm_has_only_fast_systems(engine):
    assert engine.llm_name is None and engine.fast_names() == ["TF-IDF", "Semantic"]


def test_pending_column_is_shown_while_the_llm_works(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"], pending=["LLM reranker"])
    assert html.count('class="col') == 3 and "LLM reranker" in html and "reading" in html


def test_llm_explanation_shows_the_prompt_and_yes_probabilities(llm_engine):
    html = render_llm_explanation(llm_engine, QUERIES["q2"])
    assert "&lt;Query&gt;: BRCA gene mutations and vitamin intake" in html  # the exact prompt, escaped
    assert "88.1%" in html  # sigmoid(+2): the vitamin paper
    assert "4.7%" in html  # sigmoid(-3): TF-IDF's #1, the BRCA paper
    assert "Correct paper" in html and "TF-IDF's #1" in html


def test_llm_explanation_without_an_llm(engine):
    assert "not loaded" in render_llm_explanation(engine, QUERIES["q1"])


def test_explanation_shows_parts_of_speech_and_noun_phrases(engine):
    html = render_explanation(engine, QUERIES["q1"])  # "Low vitamin D causes fractures"
    assert "adjective" in html and "noun" in html and "verb" in html
    assert "low vitamin d" in html  # noun-phrase chunk from the parser
