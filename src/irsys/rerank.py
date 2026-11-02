"""Two-stage retrieval: a first-stage retriever proposes candidates, a slower model reorders them.

The reranker (here the LLM in ``llm.py``) reads the query and each candidate *together*, which is
far more accurate than matching words or comparing two independent embeddings, but too slow to
run over the whole corpus, hence reranking only the first stage's top ``depth``.
"""

import numpy as np

from irsys.base import Ranking, Retriever


class RerankRetriever(Retriever):
    def __init__(
        self,
        first_stage: Retriever,
        reranker,  # anything with .score(query, texts) -> np.ndarray
        doc_texts: dict[str, str],
        depth: int = 100,
        name: str | None = None,
    ):
        self.first_stage = first_stage
        self.reranker = reranker
        self.doc_texts = doc_texts
        self.depth = depth
        self.name = name or f"{getattr(first_stage, 'name', 'first stage')} -> reranker"

    def search(self, query: str, k: int = 10) -> Ranking:
        candidates = [d for d, _ in self.first_stage.search(query, self.depth)]
        if not candidates:
            return []
        scores = self.reranker.score(query, [self.doc_texts[d] for d in candidates])
        order = np.argsort(-scores, kind="stable")[:k]
        return [(candidates[i], float(scores[i])) for i in order]
