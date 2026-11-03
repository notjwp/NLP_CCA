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


def test_results_tables_show_the_six_metrics_then_speed_and_significance(tmp_path):
    import pandas as pd

    from irsys.demo import results_tables
    from irsys.metrics import METRICS

    names = {"tfidf": "TF-IDF cosine", "dense-bge": "Semantic: bge", "tfidf-llm": "LLM reranker"}
    grades = {"tfidf": 0.5, "dense-bge": 0.7, "tfidf-llm": 0.1}
    pd.DataFrame([{"system": s, "name": n, **{m: grades[s] for m in METRICS}} for s, n in names.items()]
                 ).to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "efficiency.md").write_text(
        "Note.\n\n" + pd.DataFrame({"System": list(names.values()), "Mean time per claim (ms)": [0.17, 12.46, 5728.7]})
        .to_markdown(index=False), encoding="utf-8")
    scores = {"tfidf": [0.5] * 30, "dense-bge": [1.0] * 25 + [0.5] * 5, "tfidf-llm": [0.0] * 25 + [0.5] * 5}
    pd.DataFrame([{"system": s, "qid": str(i), "nDCG@10": v} for s, vs in scores.items() for i, v in enumerate(vs)]
                 ).to_csv(tmp_path / "per_query.csv", index=False)

    tables = results_tables(tmp_path)
    assert "| System | " + " | ".join(METRICS) + " |" in tables
    assert "| Word matching · TF-IDF + cosine (ours) | " + " | ".join(["0.500"] * len(METRICS)) + " |" in tables
    assert "| System | Time per claim | Really better than TF-IDF? |" in tables
    assert "| Word matching · TF-IDF + cosine (ours) | 0.17 ms | — (baseline) |" in tables
    assert "| Meaning matching · semantic search (bge-small) | 12.5 ms | yes |" in tables
    assert "| LLM reranker (Qwen3) | 5.7 s | no, worse |" in tables


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


# ---------------------------------------------------------------- presenter-friendly layout


def test_one_big_tile_per_engine_says_where_the_correct_paper_landed(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"])
    assert html.count('class="tile ') == 2
    assert ">#1<" in html and ">#3<" in html  # TF-IDF first, the fake semantic system third


def test_tiles_and_lists_render_separately_so_the_why_buttons_fit_between(engine):
    from irsys.demo import render_lists, render_tiles

    results = engine.compare(QUERIES["q1"], k=2)
    tiles, lists = render_tiles(results), render_lists(results, ANALYZER, QUERIES["q1"])
    assert 'class="tile ' in tiles and 'class="col' not in tiles
    assert 'class="col' in lists and 'class="tile ' not in lists


def test_pending_engine_gets_a_waiting_tile(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=2), ANALYZER, QUERIES["q1"], pending=["LLM reranker"])
    assert html.count('class="tile ') == 3 and 'class="tile pending"' in html


def test_own_claims_have_no_answer_key_tile(engine):
    html = render_results(engine.compare("sleep and memory", k=2), ANALYZER, "sleep and memory")
    assert "no answer key" in html and "✓" not in html


def test_result_lists_keep_titles_and_put_the_abstract_in_a_tooltip(engine):
    html = render_results(engine.compare(QUERIES["q1"], k=1), ANALYZER, QUERIES["q1"])
    assert 'title="Vitamin D deficiency is linked to bone fractures."' in html


def test_story_card_explains_the_example_in_one_sentence(engine):
    from irsys.demo import render_story

    stories = {"q1": ("A made-up story", "TF-IDF wins because of <b>vitamin</b>.")}
    preset = render_story(engine, QUERIES["q1"], stories)
    assert "A made-up story" in preset and "<b>vitamin</b>" in preset
    assert "known answer" in render_story(engine, QUERIES["q2"], stories)  # a test claim without a story
    assert "Your own claim" in render_story(engine, "sleep and memory", stories)
    assert render_story(engine, "   ", stories) == ""


def test_every_preset_has_a_story():
    from irsys.demo import PRESETS, STORIES

    assert {qid for _, qid in PRESETS} == set(STORIES)


def test_tfidf_takeaway_when_the_top_hit_is_correct(engine):
    from irsys.demo import tfidf_takeaway

    text = tfidf_takeaway(engine, QUERIES["q1"])
    assert "is the correct paper" in text and "most important word" in text


def test_tfidf_takeaway_names_the_word_that_decided_it(engine):
    from irsys.demo import tfidf_takeaway

    text = tfidf_takeaway(engine, QUERIES["q2"])  # TF-IDF's #1 is the BRCA paper; the correct one is 2nd
    assert "TF-IDF's #1" in text and "position 2" in text
    assert "“brca”" in text or "“gene”" in text or "“mutat”" in text  # a word only the #1 paper shares


def test_tfidf_explanation_starts_step_3_with_the_takeaway(engine):
    html = render_explanation(engine, QUERIES["q2"])
    assert 'class="takeaway"' in html and html.index('class="takeaway"') < html.index('class="doc"')


def test_llm_explanation_says_how_far_the_correct_paper_moved(llm_engine):
    html = render_llm_explanation(llm_engine, QUERIES["q2"])
    assert "moves from position 2 to 1" in html
    assert "<details" in html  # the long prompt is folded away until the presenter opens it


def test_results_headline_shows_one_tile_per_system(tmp_path):
    import pandas as pd

    from irsys.demo import results_headline
    from irsys.metrics import METRICS

    names = {"tfidf": "TF-IDF cosine", "dense-bge": "Semantic: bge"}
    grades = {"tfidf": 0.5, "dense-bge": 0.7}
    pd.DataFrame([{"system": s, "name": n, **{m: grades[s] for m in METRICS}} for s, n in names.items()]
                 ).to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "efficiency.md").write_text(
        "Note.\n\n" + pd.DataFrame({"System": list(names.values()), "Mean time per claim (ms)": [0.17, 12.46]})
        .to_markdown(index=False), encoding="utf-8")
    pd.DataFrame([{"system": s, "qid": str(i), "nDCG@10": v} for s, vs in
                  {"tfidf": [0.5] * 30, "dense-bge": [1.0] * 25 + [0.5] * 5}.items() for i, v in enumerate(vs)]
                 ).to_csv(tmp_path / "per_query.csv", index=False)

    html = results_headline(tmp_path)
    assert html.count('class="tile ') == 2
    assert "0.500" in html and "0.700" in html and "0.17 ms" in html and "12.5 ms" in html
    assert "Semantic search" in html
