"""Semantic search with a sentence-transformers bi-encoder (Reimers & Gurevych, 2019).

Documents and queries are embedded independently into the same vector space; with
L2-normalized embeddings the dot product is the cosine similarity. 5k documents fit in
memory, so search is exact (one matrix-vector product), no ANN index needed.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from irsys.base import Ranking, Retriever, Run, top_k
from irsys.data import CACHE_DIR


class DenseRetriever(Retriever):
    def __init__(
        self,
        model_name: str,
        query_prompt: str | None = None,
        name: str | None = None,
        cache_dir: Path = CACHE_DIR,
        batch_size: int = 64,
        device: str | None = None,  # None: CUDA when available, else CPU
    ):
        self.model_name = model_name
        self.query_prompt = query_prompt
        self.name = name or f"Dense ({model_name.split('/')[-1]})"
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        return self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)

    def index(self, doc_ids: list[str], texts: list[str]) -> "DenseRetriever":
        """Embed the corpus, reusing a cached matrix for the same model + texts.

        The original encoding time is stored beside the cache so efficiency numbers
        stay meaningful on cached runs.
        """
        self.doc_ids = list(doc_ids)
        digest = hashlib.sha1("\x00".join([self.model_name, *texts]).encode("utf-8")).hexdigest()[:12]
        path = self.cache_dir / f"emb_{self.model_name.replace('/', '__')}_{digest}.npy"
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists():
            self.embeddings = np.load(path)
            self.index_seconds = json.loads(meta_path.read_text(encoding="utf-8"))["seconds"]
        else:
            start = time.perf_counter()
            self.embeddings = self._encode(list(texts))
            self.index_seconds = time.perf_counter() - start
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, self.embeddings)
            meta_path.write_text(json.dumps({"seconds": self.index_seconds, "device": str(self.model.device)}), encoding="utf-8")
        return self

    def search(self, query: str, k: int = 10) -> Ranking:
        q = self._encode([query], self.query_prompt)[0]
        return top_k(self.embeddings @ q, self.doc_ids, k)

    def search_batch(self, queries: dict[str, str], k: int = 100) -> Run:
        qids = list(queries)
        scores = self._encode([queries[q] for q in qids], self.query_prompt) @ self.embeddings.T
        return {qid: dict(top_k(row, self.doc_ids, k)) for qid, row in zip(qids, scores)}

    @property
    def nbytes(self) -> int:
        return self.embeddings.nbytes

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
