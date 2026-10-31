"""Text analysis shared by the classical retriever: tokenize, POS-tag, chunk, drop stopwords, stem or lemmatize."""

import re
from functools import cache
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()

NLTK_DIR = Path(__file__).resolve().parents[2] / "data" / "nltk_data"
# NLTK resource -> package; WordNet ships as a zip that NLTK reads in place.
_NLTK_RESOURCES = {"taggers/averaged_perceptron_tagger_eng/": "averaged_perceptron_tagger_eng",
                   "corpora/wordnet.zip": "wordnet"}

# Noun phrase = adjectives/nouns ending in a noun, the (A|N)*N technical-term pattern
# (Justeson & Katz, 1995). Penn Treebank tags: JJ* adjectives, NN* nouns.
NP_GRAMMAR = "NP: {<JJ.*|NN.*>*<NN.*>}"
_WORDNET_POS = {"J": "a", "V": "v", "R": "r"}  # first letter of the Penn tag; everything else is a noun


@cache
def _nltk_ready() -> None:
    """Make the POS tagger model and WordNet available, downloading them into data/ on first use."""
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    for resource, package in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=NLTK_DIR, quiet=True)


@cache
def _tagger():
    _nltk_ready()
    return nltk.tag.PerceptronTagger()


@cache
def _lemmatizer():
    _nltk_ready()
    return nltk.stem.WordNetLemmatizer()


@cache
def _chunker():
    return nltk.RegexpParser(NP_GRAMMAR)


@cache
def _stem(token: str) -> str:
    return _stemmer.stem(token)


@cache
def _lemma(token: str, wordnet_pos: str) -> str:
    return _lemmatizer().lemmatize(token, wordnet_pos)


@cache
def _pos_tags(lowered: str) -> tuple[str, ...]:
    """Penn Treebank tag of every token. Cached: each ablation config re-analyzes the same corpus."""
    tokens = _TOKEN_RE.findall(lowered)
    return tuple(tag for _, tag in _tagger().tag(tokens)) if tokens else ()


def _np_spans(tokens: list[str], tags: tuple[str, ...]) -> list[range]:
    """Token positions of each noun-phrase chunk, from NLTK's regular-expression chunk parser."""
    if not tokens:
        return []
    spans, i = [], 0
    for node in _chunker().parse(list(zip(tokens, tags))):
        width = len(node) if isinstance(node, nltk.Tree) else 1
        if isinstance(node, nltk.Tree):
            spans.append(range(i, i + width))
        i += width
    return spans


class Analyzer:
    """Callable text -> list of index terms.

    Lowercases, splits on anything that is not [a-z0-9] (numbers are kept, SciFact
    claims are full of them), then optionally removes English stopwords and reduces
    each word to a root: the Porter stemmer (``stem``) or a WordNet lemma chosen by
    the word's POS tag (``lemmatize``). Stopwords are removed before stemming so the
    stopword list matches surface forms; POS tags are computed on the full token
    sequence, so the tagger still sees the stopwords as context.

    ``phrases`` adds one extra term per pair of adjacent words inside a noun phrase
    (e.g. ``ldl_cholesterol``), found by POS tagging + chunk parsing.
    """

    def __init__(self, stem: bool = True, stopwords: bool = True, lemmatize: bool = False, phrases: bool = False):
        if stem and lemmatize:
            raise ValueError("choose stemming or lemmatization, not both")
        self.stem = stem
        self.stopwords = stopwords
        self.lemmatize = lemmatize
        self.phrases = phrases

    @classmethod
    def from_config(cls, config: dict) -> "Analyzer":
        """Rebuild from ``config()`` output (or from a results/config.json "tfidf" entry)."""
        return cls(config["stem"], config["stopwords"], config.get("lemmatize", False), config.get("phrases", False))

    def config(self) -> dict[str, bool]:
        return {"stem": self.stem, "stopwords": self.stopwords, "lemmatize": self.lemmatize, "phrases": self.phrases}

    def __call__(self, text: str) -> list[str]:
        return self.stages(text)["terms"]

    def tag(self, text: str) -> list[tuple[str, str]]:
        """(token, Penn Treebank POS tag) for every token."""
        lowered = text.lower()
        return list(zip(_TOKEN_RE.findall(lowered), _pos_tags(lowered)))

    def noun_phrases(self, text: str) -> list[list[str]]:
        """Tokens of each noun-phrase chunk, in order."""
        lowered = text.lower()
        tokens = _TOKEN_RE.findall(lowered)
        return [[tokens[i] for i in span] for span in _np_spans(tokens, _pos_tags(lowered))]

    def stages(self, text: str) -> dict[str, list[str]]:
        """Output of each pipeline step: tokens -> without_stopwords -> terms (normalized, plus phrases)."""
        lowered = text.lower()
        tokens = _TOKEN_RE.findall(lowered)
        keep = [not (self.stopwords and t in ENGLISH_STOP_WORDS) for t in tokens]
        tags = _pos_tags(lowered) if self.lemmatize or self.phrases else ()
        normal = [self._normalize(t, tags[i] if tags else "") if keep[i] else None for i, t in enumerate(tokens)]
        terms = [n for n in normal if n is not None]
        if self.phrases:
            for span in _np_spans(tokens, tags):
                words = [normal[i] for i in span if normal[i] is not None]
                terms += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return {"tokens": tokens, "without_stopwords": [t for t, k in zip(tokens, keep) if k], "terms": terms}

    def _normalize(self, token: str, tag: str) -> str:
        if self.lemmatize:
            return _lemma(token, _WORDNET_POS.get(tag[:1], "n"))
        return _stem(token) if self.stem else token

    def __repr__(self) -> str:
        return (f"Analyzer(stem={self.stem}, stopwords={self.stopwords}, "
                f"lemmatize={self.lemmatize}, phrases={self.phrases})")
