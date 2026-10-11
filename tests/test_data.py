import pytest

from irsys.data import doc_text, load_scifact


def test_doc_text_joins_title_and_body():
    assert doc_text({"title": "Title", "text": "Body text."}) == "Title Body text."
    assert doc_text({"title": "", "text": "Body"}) == "Body"


@pytest.mark.slow
@pytest.mark.parametrize("split, n_queries", [("test", 300), ("train", 809)])
def test_load_scifact_split(split, n_queries):
    corpus, queries, qrels = load_scifact(split)
    assert len(corpus) == 5183
    assert len(queries) == len(qrels) == n_queries
    assert set(queries) == set(qrels)
    assert all(doc in corpus for judged in qrels.values() for doc in judged)
    assert all(isinstance(q, str) and q for q in queries.values())
