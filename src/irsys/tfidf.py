"""Classical vector-space retrieval: TF-IDF weighting + cosine similarity over an inverted index.

Weighting schemes use SMART notation ``ddd.qqq`` (Manning, Raghavan & Schuetze, 2008, ch. 6):
the first triple weights documents, the second weights queries, and each triple is
term frequency / document frequency / normalization.

    tf:    n = raw tf            l = 1 + ln(tf)
    idf:   n = none              t = ln(N/df)          s = ln(N/df) + 1  (scikit-learn's variant)
    norm:  n = none              c = cosine (L2 length)

With ``c`` on both sides the dot product of the two weighted vectors *is* their cosine
similarity, so scoring is a sum over the query's postings lists.
"""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from irsys.base import Ranking, Retriever, top_k
from irsys.preprocess import Analyzer

_TF = {
    "n": lambda tf: tf,
    "l": lambda tf: 1.0 + np.log(tf),
}
_IDF = {
    "n": lambda df, n: np.ones_like(df, dtype=float),
    "t": lambda df, n: np.log(n / df),
    "s": lambda df, n: np.log(n / df) + 1.0,
}
_NORM = "nc"


def _parse_scheme(scheme: str) -> tuple[str, str]:
    parts = scheme.split(".")
    if len(parts) != 2 or not all(
        len(p) == 3 and p[0] in _TF and p[1] in _IDF and p[2] in _NORM for p in parts
    ):
        raise ValueError(f"invalid SMART scheme {scheme!r}; expected e.g. 'ltc.ltc'")
    return parts[0], parts[1]


class TfidfRetriever(Retriever):
    """TF-IDF / cosine retriever built from scratch on an inverted index."""

    def __init__(self, analyzer: Analyzer | None = None, scheme: str = "ltc.ltc", name: str | None = None):
        self.analyzer = analyzer or Analyzer()
        self.scheme = scheme
        self.doc_scheme, self.query_scheme = _parse_scheme(scheme)
        self.name = name or f"TF-IDF ({scheme})"

    def index(self, doc_ids: list[str], texts: list[str]) -> "TfidfRetriever":
        self.doc_ids = list(doc_ids)
        self.n_docs = n = len(self.doc_ids)

        # Pass 1: term -> postings of (doc index, raw tf).
        raw: dict[str, tuple[list[int], list[int]]] = {}
        for i, text in enumerate(texts):
            for term, tf in Counter(self.analyzer(text)).items():
                docs, tfs = raw.setdefault(term, ([], []))
                docs.append(i)
                tfs.append(tf)
        self.df = {term: len(docs) for term, (docs, _) in raw.items()}

        # Pass 2: weight every posting; accumulate squared document lengths.
        tf_fn, idf_fn, norm = _TF[self.doc_scheme[0]], _IDF[self.doc_scheme[1]], self.doc_scheme[2]
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        length_sq = np.zeros(n)
        for term, (docs, tfs) in raw.items():
            docs = np.asarray(docs, dtype=np.int32)
            weights = tf_fn(np.asarray(tfs, dtype=np.float64)) * idf_fn(len(docs), n)
            self.postings[term] = (docs, weights)
            length_sq[docs] += weights**2  # a doc appears at most once per postings list

        # Pass 3: fold the cosine length normalization into the stored weights.
        if norm == "c":
            lengths = np.sqrt(length_sq)
            lengths[lengths == 0] = 1.0  # empty documents keep all-zero weights
            for term, (docs, weights) in self.postings.items():
                self.postings[term] = (docs, weights / lengths[docs])
        return self

    def query_weights(self, query: str) -> dict[str, float]:
        counts = Counter(t for t in self.analyzer(query) if t in self.postings)
        if not counts:
            return {}
        terms = list(counts)
        tf_fn, idf_fn, norm = _TF[self.query_scheme[0]], _IDF[self.query_scheme[1]], self.query_scheme[2]
        tfs = np.array([counts[t] for t in terms], dtype=np.float64)
        dfs = np.array([self.df[t] for t in terms], dtype=np.float64)
        weights = tf_fn(tfs) * idf_fn(dfs, self.n_docs)
        if norm == "c":
            length = np.sqrt((weights**2).sum())
            if length > 0:
                weights = weights / length
        return dict(zip(terms, weights))

    def scores(self, query: str) -> np.ndarray:
        """Score of every document (aligned with ``doc_ids``), term-at-a-time."""
        acc = np.zeros(self.n_docs)
        for term, w_q in self.query_weights(query).items():
            docs, w_d = self.postings[term]
            acc[docs] += w_q * w_d
        return acc

    def search(self, query: str, k: int = 10) -> Ranking:
        return top_k(self.scores(query), self.doc_ids, k, positive_only=True)

    def idf(self, term: str) -> float:
        """Query-side idf of an indexed term under this scheme."""
        return float(_IDF[self.query_scheme[1]](self.df[term], self.n_docs))

    def explain(self, query: str, doc_id: str) -> list[tuple[str, float, float]]:
        """(term, query weight, document weight) for every indexed query term.

        The products w_q * w_d sum to the document's cosine score; a document weight of 0
        means the term does not occur in that document.
        """
        i = self.doc_ids.index(doc_id)
        parts = []
        for term, w_q in self.query_weights(query).items():
            docs, w_d = self.postings[term]
            pos = int(np.searchsorted(docs, i))  # postings are in ascending doc order
            parts.append((term, float(w_q), float(w_d[pos]) if pos < len(docs) and docs[pos] == i else 0.0))
        return parts

    @property
    def nbytes(self) -> int:
        return sum(docs.nbytes + weights.nbytes for docs, weights in self.postings.values())


class SklearnTfidfRetriever(Retriever):
    """Reference implementation with scikit-learn, used to cross-check ``TfidfRetriever``.

    sublinear_tf + smooth_idf=False + L2 norm is SMART ``lsc.lsc``.
    """

    def __init__(self, analyzer: Analyzer | None = None, name: str = "TF-IDF (scikit-learn)"):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer or Analyzer(), sublinear_tf=True, smooth_idf=False, norm="l2"
        )
        self.name = name

    def index(self, doc_ids: list[str], texts: list[str]) -> "SklearnTfidfRetriever":
        self.doc_ids = list(doc_ids)
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.matrix, self.vectorizer.transform([query])).ravel()

    def search(self, query: str, k: int = 10) -> Ranking:
        return top_k(self.scores(query), self.doc_ids, k, positive_only=True)
