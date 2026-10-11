"""Two-stage retrieval: a first-stage retriever proposes candidates, a cross-encoder reorders them.

A cross-encoder reads the query and the document *together* through one transformer
(Nogueira & Cho, 2019), which is far more accurate than comparing two independent
embeddings but too slow to run over the whole corpus, hence reranking only the top-``depth``.
"""

import numpy as np
from sentence_transformers import CrossEncoder

from irsys.base import Ranking, Retriever


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        pairs = [(query, t) for t in texts]
        return np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False), dtype=float
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())


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
        self.name = name or f"{getattr(first_stage, 'name', 'first stage')} -> cross-encoder"

    def search(self, query: str, k: int = 10) -> Ranking:
        candidates = [d for d, _ in self.first_stage.search(query, self.depth)]
        if not candidates:
            return []
        scores = self.reranker.score(query, [self.doc_texts[d] for d in candidates])
        order = np.argsort(-scores, kind="stable")[:k]
        return [(candidates[i], float(scores[i])) for i in order]
