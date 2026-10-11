"""Smoke tests with the real (small) transformer models. Slow: downloads weights on first run."""

import numpy as np
import pytest

from irsys.dense import DenseRetriever
from irsys.rerank import CrossEncoderReranker

pytestmark = pytest.mark.slow

DOC_IDS = ["photo", "stock", "volcano", "football", "piano"]
TEXTS = [
    "Photosynthesis lets plants convert sunlight, water and carbon dioxide into glucose.",
    "The stock market fell sharply after the central bank raised interest rates.",
    "Lava from the volcano destroyed several villages on the island.",
    "The football team won the championship after a penalty shoot-out.",
    "She practised the piano sonata every morning before school.",
]
QUERY = "how do green leaves turn light into chemical energy"  # no word overlap with "photo"


def test_dense_retriever_finds_paraphrase(tmp_path):
    r = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2", cache_dir=tmp_path).index(DOC_IDS, TEXTS)
    results = r.search(QUERY, k=5)
    assert results[0][0] == "photo"
    assert len(results) == 5 and results[0][1] > results[1][1]


def test_dense_embeddings_are_cached_and_reused(tmp_path):
    r1 = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2", cache_dir=tmp_path).index(DOC_IDS, TEXTS)
    assert len(list(tmp_path.glob("*.npy"))) == 1
    r2 = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2", cache_dir=tmp_path).index(DOC_IDS, TEXTS)
    np.testing.assert_array_equal(r1.embeddings, r2.embeddings)
    np.testing.assert_allclose(np.linalg.norm(r1.embeddings, axis=1), 1.0, rtol=1e-5)


def test_dense_search_batch_matches_search(tmp_path):
    r = DenseRetriever("BAAI/bge-small-en-v1.5", query_prompt="Represent this sentence for searching relevant passages: ",
                       cache_dir=tmp_path).index(DOC_IDS, TEXTS)
    run = r.search_batch({"q": QUERY}, k=3)
    single = dict(r.search(QUERY, k=3))
    assert list(run["q"]) == list(single)
    assert list(run["q"].values()) == pytest.approx(list(single.values()), abs=1e-5)


def test_cross_encoder_prefers_relevant_passage():
    scores = CrossEncoderReranker().score(QUERY, TEXTS)
    assert scores.shape == (5,)
    assert int(np.argmax(scores)) == 0
