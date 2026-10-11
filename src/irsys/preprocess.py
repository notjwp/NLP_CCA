"""Text analysis shared by the classical retriever: tokenize, drop stopwords, stem."""

import re
from functools import cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()


@cache
def _stem(token: str) -> str:
    return _stemmer.stem(token)


class Analyzer:
    """Callable text -> list of index terms.

    Lowercases, splits on anything that is not [a-z0-9] (numbers are kept, SciFact
    claims are full of them), then optionally removes English stopwords and applies
    the Porter stemmer. Stopwords are removed before stemming so the stopword list
    matches surface forms.
    """

    def __init__(self, stem: bool = True, stopwords: bool = True):
        self.stem = stem
        self.stopwords = stopwords

    def __call__(self, text: str) -> list[str]:
        return self.stages(text)["terms"]

    def stages(self, text: str) -> dict[str, list[str]]:
        """Output of each pipeline step: tokens -> without_stopwords -> terms (stemmed)."""
        tokens = _TOKEN_RE.findall(text.lower())
        kept = [t for t in tokens if t not in ENGLISH_STOP_WORDS] if self.stopwords else tokens
        terms = [_stem(t) for t in kept] if self.stem else kept
        return {"tokens": tokens, "without_stopwords": kept, "terms": terms}

    def __repr__(self) -> str:
        return f"Analyzer(stem={self.stem}, stopwords={self.stopwords})"
