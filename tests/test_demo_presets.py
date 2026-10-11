"""The preset buttons must tell the story their labels promise, on the real data and models."""

import pytest

from irsys.demo import PRESETS, DemoEngine

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine():
    return DemoEngine.load()


@pytest.mark.parametrize("label, qid", PRESETS)
def test_preset_story_holds(engine, label, qid):
    tfidf, semantic, _hybrid = engine.compare(engine.queries[qid], k=5)
    tfidf_rank, semantic_rank = tfidf.correct_rank or 10**6, semantic.correct_rank or 10**6
    if "semantic wins" in label:
        assert semantic_rank == 1 and tfidf_rank > 5
    else:
        assert tfidf_rank == 1 and semantic_rank > 5
