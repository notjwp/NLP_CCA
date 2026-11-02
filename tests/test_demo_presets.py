"""The preset buttons must tell the story DEMO.md promises, on the real data and models."""

import pytest

from irsys.demo import PRESETS, DemoEngine

pytestmark = pytest.mark.slow

# Which of the two first-stage systems finds the correct paper at position 1 (the other one ranks it below 5).
WINNER = {"1088": "semantic", "1278": "semantic", "660": "tfidf", "674": "tfidf"}


@pytest.fixture(scope="module")
def engine():
    return DemoEngine.load(llm=True)


def test_every_preset_has_a_story():
    assert {qid for _, qid in PRESETS} == set(WINNER)


@pytest.mark.parametrize("label, qid", PRESETS)
def test_preset_story_holds(engine, label, qid):
    tfidf, semantic, llm = engine.compare(engine.queries[qid], k=5)
    tfidf_rank, semantic_rank = tfidf.correct_rank or 10**6, semantic.correct_rank or 10**6
    if WINNER[qid] == "semantic":
        assert semantic_rank == 1 and tfidf_rank > 5
    else:
        assert tfidf_rank == 1 and semantic_rank > 5
    assert llm.correct_rank == 1  # the LLM reranker gets all four right
