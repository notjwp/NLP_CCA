"""Presentation demo: live TF-IDF vs. semantic search vs. LLM comparison, and step-by-step explanations.

``uv run irsys demo`` opens it in the browser. The engine and HTML renderers are plain
functions (tested in tests/test_demo.py); ``build_app`` only wires them into Gradio.

The layout follows the order of the talk: a one-sentence story for the example, one big tile per engine
saying where it put the correct paper, the three result lists, then "why?" buttons that jump to the
explanation tabs.
"""

import html
import json
import math
import random
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from irsys.base import Retriever
from irsys.data import RESULTS_DIR, doc_text, load_scifact
from irsys.preprocess import Analyzer
from irsys.tfidf import TfidfRetriever

# The four stories from REPORT.md §5.3 (SciFact test query ids).
PRESETS = [
    ("① Bcl2 vs “BCL-2” (spelling)", "1088"),
    ("② Same idea, other words", "1278"),
    ("③ Rare disease name", "660"),
    ("④ LDL vs HDL", "674"),
]
# One headline and one or two sentences per preset: what the example shows, in plain words.
STORIES = {
    "1088": ("Same gene, different spelling",
             "The claim writes <b>Bcl2</b>; the correct paper writes <b>BCL-2</b>. Word matching splits “BCL-2” "
             "into “bcl” + “2”, so the claim's most important word never matches. The AI models know both "
             "spellings mean the same gene."),
    "1278": ("Same idea, different words",
             "The correct paper (“Is autoimmunity the Achilles' heel of cancer immunotherapy?”) says what the claim "
             "says, in other words. Word matching has little to hold on to; meaning matching and the LLM recognise "
             "the idea."),
    "660": ("A rare word is a perfect fingerprint",
            "<b>Onchocerciasis</b> appears in just one of the 5,183 papers, so TF-IDF gives it the highest possible "
            "weight and ranks that paper first. The embedding model has only a vague idea of this disease; the LLM "
            "reads the paper and agrees with TF-IDF."),
    "674": ("One letter matters: LDL vs. HDL",
            "The claim is about <b>LDL</b> cholesterol. Word matching keeps the exact term; the embedding model "
            "blurs LDL and HDL and puts a paper about <b>HDL</b> cholesterol first. The LLM reads the paper and keeps "
            "the right one first."),
}
TFIDF_NAME = "Word matching · TF-IDF + cosine"
SEMANTIC_NAME = "Meaning matching · semantic search"
LLM_NAME = "LLM reranker · Qwen3"
DESCRIPTIONS = {
    TFIDF_NAME: "Our classical system: finds papers that use the claim's words. Rare words count more.",
    SEMANTIC_NAME: "A neural language model (bge-small) turns each text into a “meaning vector” and compares those.",
    LLM_NAME: "A large language model reads the claim with each of TF-IDF's top 100 papers and answers yes or no.",
}
SYSTEM_COLORS = ["#2a78d6", "#1baf7a", "#6250d6"]  # same colors as the report figures and the slides
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    score: float
    relevant: bool


@dataclass
class SystemResult:
    name: str
    hits: list[Hit]
    judged: bool  # the query is a SciFact test claim, so the correct abstract is known
    correct_rank: int | None  # rank of the first correct abstract within the search depth
    latency_ms: float
    description: str = ""  # one plain-language sentence about the system
    ranking: list[tuple[str, float]] = field(default_factory=list)  # the full ranking, up to the search depth


class DemoEngine:
    def __init__(self, corpus, queries, qrels, tfidf: TfidfRetriever, systems: dict[str, Retriever],
                 descriptions: dict[str, str] | None = None, llm_name: str | None = None):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.tfidf = tfidf
        self.systems = systems
        self.descriptions = descriptions or {}  # system name -> one plain-language sentence
        self.llm_name = llm_name  # the slow LLM reranker, shown after the fast systems
        self._qid_by_text = {text.strip(): qid for qid, text in queries.items()}

    def fast_names(self) -> list[str]:
        return [name for name in self.systems if name != self.llm_name]

    @classmethod
    def load(cls, llm: bool = True) -> "DemoEngine":
        from irsys.dense import DenseRetriever
        from irsys.evaluate import BGE, BGE_QUERY_PROMPT, LLM_RERANKER, RUN_DEPTH

        corpus, queries, qrels = load_scifact("test")
        doc_ids = list(corpus)
        texts = [doc_text(corpus[d]) for d in doc_ids]
        config_path = RESULTS_DIR / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))["tfidf"] if config_path.exists() else {
            "scheme": "lsc.lsc", "stem": True, "stopwords": True}
        tfidf = TfidfRetriever(Analyzer.from_config(cfg), cfg["scheme"]).index(doc_ids, texts)
        bge = DenseRetriever(BGE, query_prompt=BGE_QUERY_PROMPT).index(doc_ids, texts)
        systems: dict[str, Retriever] = {TFIDF_NAME: tfidf, SEMANTIC_NAME: bge}
        if llm:
            from irsys.llm import LLMReranker
            from irsys.rerank import RerankRetriever

            systems[LLM_NAME] = RerankRetriever(tfidf, LLMReranker(LLM_RERANKER), dict(zip(doc_ids, texts)),
                                                depth=RUN_DEPTH)
        engine = cls(corpus, queries, qrels, tfidf, systems, DESCRIPTIONS, LLM_NAME if llm else None)
        engine.compare("warm-up query", k=1)  # CUDA initialisation happens here, not on stage
        return engine

    def lookup(self, text: str) -> str | None:
        return self._qid_by_text.get(text.strip())

    def random_claim(self) -> str:
        return self.queries[random.choice(list(self.queries))]

    def relevant(self, text: str) -> list[str]:
        qid = self.lookup(text)
        return [d for d, g in self.qrels.get(qid, {}).items() if g > 0] if qid else []

    def compare(self, text: str, k: int = 5, depth: int = 100, names: list[str] | None = None) -> list[SystemResult]:
        relevant = set(self.relevant(text))
        results = []
        for name in names or list(self.systems):
            start = time.perf_counter()
            ranking = self.systems[name].search(text, depth)
            latency = (time.perf_counter() - start) * 1000
            hits = [Hit(d, self.corpus[d]["title"], self.corpus[d]["text"], s, d in relevant) for d, s in ranking[:k]]
            rank = next((r for r, (d, _) in enumerate(ranking, start=1) if d in relevant), None)
            results.append(SystemResult(name, hits, bool(relevant), rank, latency, self.descriptions.get(name, ""),
                                        ranking))
        return results

    def oov_terms(self, text: str) -> list[str]:
        """Index terms of the query that occur in no abstract (TF-IDF cannot use them)."""
        return list(dict.fromkeys(t for t in self.tfidf.analyzer(text) if t not in self.tfidf.postings))

    def term_table(self, text: str) -> pd.DataFrame:
        counts = pd.Series(self.tfidf.analyzer(text)).value_counts()
        rows = [
            {"term": term, "query tf": int(counts[term]), "df": self.tfidf.df[term],
             "idf": self.tfidf.idf(term), "query weight": w_q}
            for term, w_q in self.tfidf.query_weights(text).items()
        ]
        return pd.DataFrame(rows, columns=["term", "query tf", "df", "idf", "query weight"])

    def contributions(self, text: str, doc_id: str) -> tuple[pd.DataFrame, float, int | None]:
        """Per-term w_q * w_d for one abstract, its cosine score, and its TF-IDF rank (None if unscored)."""
        table = pd.DataFrame(self.tfidf.explain(text, doc_id), columns=["term", "query weight", "doc weight"])
        table["contribution"] = table["query weight"] * table["doc weight"]
        table = table.sort_values("contribution", ascending=False, kind="stable").reset_index(drop=True)
        scores = self.tfidf.scores(text)
        score = float(scores[self.tfidf.doc_ids.index(doc_id)])
        rank = 1 + int((scores > score).sum()) if score > 0 else None
        return table, score, rank


# ---------------------------------------------------------------- HTML rendering

def highlight(text: str, analyzer: Analyzer, terms: set[str]) -> str:
    """HTML-escape ``text`` and wrap words whose index term is in ``terms`` in <mark>."""
    out, last = [], 0
    for m in _WORD_RE.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        word = html.escape(m.group())
        analyzed = analyzer(m.group())
        out.append(f"<mark>{word}</mark>" if analyzed and analyzed[0] in terms else word)
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def _duration(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.2f} ms" if ms < 1 else f"{ms:.1f} ms"


def _snippet(text: str, limit: int = 230) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _dot(color: str) -> str:
    return f'<i style="--c:{color}"></i>'


def render_story(engine: DemoEngine, text: str, stories: dict[str, tuple[str, str]] | None = None) -> str:
    """One card above the tabs: what this example shows, in one or two sentences."""
    text = (text or "").strip()
    if not text:
        return ""
    stories = STORIES if stories is None else stories
    qid = engine.lookup(text)
    if qid in stories:
        kicker, (head, body) = "What this example shows", stories[qid]
    elif qid:
        kicker, head = "A test claim", "A test claim with a known answer"
        body = ("The benchmark says which paper supports or refutes this claim. It is marked ✓ below: watch where "
                "each engine puts it.")
    else:
        kicker, head = "Typed claim", "Your own claim"
        body = ("There is no answer key for a typed claim, so compare the lists by eye. Words in yellow also appear "
                "in the claim.")
    return (f'<div class="irsys"><div class="story"><div class="kicker">💡 {kicker}</div>'
            f'<div class="story-head">{head}</div><p>{body}</p></div></div>')


def _tile(color: str, res: SystemResult) -> str:
    """One big tile per engine: where it put the correct paper, and how long it took."""
    r = res.correct_rank
    if not res.judged:
        cls, big, status = "muted", "—", "no answer key for your own claim"
    elif r and r <= len(res.hits):
        cls, big, status = "good", f"#{r}", f"✓ found the correct paper at position {r}"
    elif r:
        cls, big, status = "warn", f"#{r}", f"▼ correct paper only at position {r}"
    else:
        cls, big, status = "bad", "✗", "✗ correct paper not in its top 100"
    caption = '<div class="cap">position of the correct paper</div>' if res.judged else ""
    return (f'<div class="tile {cls}" style="--c:{color}"><div class="name">{_dot(color)}{html.escape(res.name)}</div>'
            f'<div class="big">{big}</div>{caption}<div class="status">{status}</div>'
            f'<div class="time">answered in {_duration(res.latency_ms)}</div></div>')


def _pending_tile(color: str, name: str) -> str:
    return (f'<div class="tile pending" style="--c:{color}"><div class="name">{_dot(color)}{html.escape(name)}</div>'
            '<div class="spin"></div><div class="status">reading the 100 candidate papers …</div>'
            '<div class="time">about 5 seconds</div></div>')


def _tiles(results: list[SystemResult], pending: list[str] | None) -> str:
    colors = SYSTEM_COLORS * 2
    tiles = [_tile(color, res) for color, res in zip(colors, results)]
    tiles += [_pending_tile(color, name) for name, color in zip(pending or [], colors[len(results):])]
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _lists(results: list[SystemResult], analyzer: Analyzer, query: str, pending: list[str] | None) -> str:
    terms = set(analyzer(query))
    colors = SYSTEM_COLORS * 2
    columns = []
    for color, res in zip(colors, results):
        hits = []
        for i, h in enumerate(res.hits, start=1):
            tag = '<span class="tag">✓ correct paper</span>' if h.relevant else ""
            tooltip = html.escape(_snippet(h.text, 300), quote=True)
            hits.append(
                f'<li class="hit{" relevant" if h.relevant else ""}" title="{tooltip}"><span class="rank">#{i}</span>'
                f'<div>{tag}<div class="title">{highlight(h.title, analyzer, terms)}</div></div></li>'
            )
        body = "".join(hits) or '<li class="empty">No paper shares a word with the claim.</li>'
        columns.append(
            f'<section class="col" style="--c:{color}"><header><h4>{_dot(color)}{html.escape(res.name)}</h4>'
            + (f'<div class="blurb">{html.escape(res.description)}</div>' if res.description else "")
            + f"</header><ol>{body}</ol></section>"
        )
    for name, color in zip(pending or [], colors[len(results):]):
        columns.append(
            f'<section class="col pending" style="--c:{color}"><header><h4>{_dot(color)}{html.escape(name)}</h4>'
            '<div class="blurb">The LLM is reading the 100 candidate papers one by one …</div></header></section>'
        )
    shown = max((len(r.hits) for r in results), default=5) or 5
    legend = (f'<p class="note">Each list shows the engine\'s top {shown}; hover over a title to read the start of '
              "the paper. <mark>Yellow</mark> words also appear in the claim (after cutting words to their root, "
              "so <i>tumors</i> matches <i>tumor</i>).</p>")
    return f'{legend}<div class="grid">{"".join(columns)}</div>'


def render_tiles(results: list[SystemResult], pending: list[str] | None = None) -> str:
    """One big tile per engine: where it put the correct paper (``pending`` engines get a waiting tile)."""
    return f'<div class="irsys">{_tiles(results, pending)}</div>'


def render_lists(results: list[SystemResult], analyzer: Analyzer, query: str,
                 pending: list[str] | None = None) -> str:
    """One result list per engine (``pending`` engines get an empty column while they are still running)."""
    return f'<div class="irsys">{_lists(results, analyzer, query, pending)}</div>'


def render_results(results: list[SystemResult], analyzer: Analyzer, query: str,
                   pending: list[str] | None = None) -> str:
    """The tiles, then the lists: the whole of tab 1 in one piece of HTML."""
    return f'<div class="irsys">{_tiles(results, pending)}{_lists(results, analyzer, query, pending)}</div>'


def scheme_formula(scheme: str) -> str:
    tf = {"n": "tf", "l": "(1 + ln tf)"}
    idf = {"n": "1", "t": "ln(N/df)", "s": "(ln(N/df) + 1)"}
    norm = {"c": ", then each vector is scaled to length 1 (cosine normalisation)", "n": ""}
    parts = []
    for side, code in zip(("document", "query"), scheme.split(".")):
        parts.append(f"{side} term weight = {tf[code[0]]} × {idf[code[1]]}{norm[code[2]]}")
    return "; ".join(parts) + ". Score = Σ w<sub>q</sub> × w<sub>d</sub> over shared terms = cosine similarity."


# Penn Treebank tag (first two letters) -> everyday name, for the audience.
POS_NAMES = {"NN": "noun", "JJ": "adjective", "VB": "verb", "RB": "adverb", "IN": "preposition", "DT": "determiner",
             "CD": "number", "CC": "conjunction", "PR": "pronoun", "MD": "modal verb", "TO": "to", "WD": "wh-word",
             "WP": "wh-word", "WR": "wh-word", "EX": "there", "RP": "particle", "PD": "determiner"}


def pos_name(tag: str) -> str:
    return POS_NAMES.get(tag[:2], tag)


def _pos_chips(tagged: list[tuple[str, str]]) -> str:
    return "".join(
        f'<span class="chip pos" title="Penn Treebank tag {html.escape(tag)}">{html.escape(word)}'
        f"<small>{html.escape(pos_name(tag))}</small></span>"
        for word, tag in tagged
    ) or '<span class="muted">(no words)</span>'


def _phrase_chips(phrases: list[list[str]]) -> str:
    return "".join(
        f'<span class="chip{" phrase" if len(p) > 1 else ""}">{html.escape(" ".join(p))}</span>' for p in phrases
    ) or '<span class="muted">(no noun phrases)</span>'


def _chips(words: list[str], dim: set[str] = frozenset(), dim_title: str = "") -> str:
    if not words:
        return '<span class="muted">(nothing left)</span>'
    return "".join(
        f'<span class="chip{" dim" if w in dim else ""}"{f" title={html.escape(dim_title)!r}" if w in dim else ""}>'
        f"{html.escape(w)}</span>"
        for w in words
    )


def _contribution_table(table: pd.DataFrame, scale: float) -> str:
    rows = []
    for term, w_q, w_d, contribution in table.itertuples(index=False, name=None):
        absent = w_d == 0
        width = 0 if absent or scale <= 0 else 100 * contribution / scale
        rows.append(
            f'<tr class="{"absent" if absent else ""}"><td class="term">{html.escape(term)}</td>'
            f"<td>{w_q:.3f}</td><td>{'✗ missing' if absent else f'{w_d:.3f}'}</td>"
            f'<td>{contribution:.4f}</td><td class="barcell"><div class="bar" style="width:{width:.1f}%"></div></td></tr>'
        )
    return ('<table class="tbl"><tr><th>word root</th><th>in claim (w<sub>q</sub>)</th>'
            '<th>in paper (w<sub>d</sub>)</th><th>w<sub>q</sub> × w<sub>d</sub></th><th></th></tr>'
            + "".join(rows) + "</table>")


def _present(engine: DemoEngine, query: str, doc_id: str) -> set[str]:
    table, _, _ = engine.contributions(query, doc_id)
    return set(table.loc[table["doc weight"] > 0, "term"])


def tfidf_takeaway(engine: DemoEngine, query: str) -> str:
    """One sentence on why TF-IDF ranked the papers as it did, naming the word that decided it."""
    terms = engine.term_table(query)
    top = engine.tfidf.search(query, 1)
    if terms.empty or not top:
        return ""
    ordered = terms.sort_values("query weight", ascending=False, kind="stable")
    key = ordered.iloc[0]
    tied = (ordered["query weight"] >= key["query weight"] - 1e-12).sum() > 1
    importance = "one of the claim's most important words" if tied else "the claim's most important word"
    key_text = f"“{key['term']}” (in {int(key['df']):,} of {engine.tfidf.n_docs:,} papers)"
    top_id = top[0][0]
    in_top = _present(engine, query, top_id)
    relevant = engine.relevant(query)
    if not relevant:
        found = f", including {importance}, {key_text}" if key["term"] in in_top else ""
        return f"TF-IDF's #1 paper contains {len(in_top)} of the claim's {len(terms)} indexed word roots{found}."
    if top_id in relevant:
        if key["term"] in in_top:
            return f"TF-IDF's #1 result is the correct paper ✓. It contains {key_text}, {importance}."
        return "TF-IDF's #1 result is the correct paper ✓: it shares the most important vocabulary with the claim."
    correct = relevant[0]
    _, _, rank = engine.contributions(query, correct)
    where = (f"so the correct paper ends up at position {rank}" if rank
             else "so TF-IDF does not find the correct paper at all")
    in_correct = _present(engine, query, correct)
    missing = [t for t in ordered["term"] if t in in_top and t not in in_correct]
    if key["term"] in missing:
        return f"{importance.capitalize()}, {key_text}, is in TF-IDF's #1 paper but not in the correct paper, {where}."
    if missing:
        return f"TF-IDF's #1 paper shares “{missing[0]}” with the claim and the correct paper does not, {where}."
    return f"The correct paper shares less of the claim's important vocabulary than TF-IDF's #1, {where}."


def render_explanation(engine: DemoEngine, query: str) -> str:
    tfidf = engine.tfidf
    analyzer = tfidf.analyzer
    stages = analyzer.stages(query)
    removed = [t for t in stages["tokens"] if t not in stages["without_stopwords"]]
    oov = set(engine.oov_terms(query))
    roots = "lemmatization" if analyzer.lemmatize else "stemming" if analyzer.stem else "none"
    parts = [
        "<h3>Step 1 · Clean up the claim's text</h3>"
        '<p class="note">The same steps run on all 5,183 papers when the index is built, so claim and papers '
        "are compared word root to word root.</p>",
        '<table class="steps">',
        f"<tr><th>1a · Split into words</th><td>{_chips(stages['tokens'], set(removed), 'filler word')}</td></tr>",
        f"<tr><th>1b · Part of speech<br><small>(POS tagging)</small></th><td>{_pos_chips(analyzer.tag(query))}</td></tr>",
        f"<tr><th>1c · Noun phrases<br><small>(parsing)</small></th>"
        f"<td>{_phrase_chips(analyzer.noun_phrases(query))}</td></tr>",
        f"<tr><th>1d · Drop filler words<br><small>(stopwords)</small></th><td>{_chips(stages['without_stopwords'])}</td></tr>",
        f"<tr><th>1e · Cut to word roots<br><small>({roots})</small></th>"
        f"<td>{_chips(stages['terms'], oov, 'in no paper')}</td></tr></table>",
        '<details class="aside"><summary>How do 1b and 1c work?</summary><p>NLTK labels every word with its part of '
        "speech, then a chunk parser with the grammar <code>NP: {&lt;JJ.*|NN.*&gt;*&lt;NN.*&gt;}</code> (adjectives "
        "and nouns ending in a noun) finds noun phrases. We tested indexing dictionary forms (lemmas) and noun "
        "phrases; on the 809 training claims plain stemming worked best, so the final system indexes word roots "
        "only (report §4.1).</p></details>",
    ]
    if oov:
        parts.append(f'<p class="note">Faded word roots appear in <b>no</b> paper, so TF-IDF cannot use them: '
                     f'{html.escape(", ".join(sorted(oov)))}.</p>')

    terms = engine.term_table(query)
    if terms.empty:
        parts.append("<p class=\"note\"><b>None of the claim's words appear in any paper</b>, "
                     "so TF-IDF cannot score anything.</p>")
        return '<div class="irsys">' + "".join(parts) + "</div>"
    terms = terms.sort_values("query weight", ascending=False, kind="stable")
    top_weight = terms["query weight"].max()
    rows = "".join(
        f'<tr><td class="term">{html.escape(term)}</td><td>{qtf}</td><td>{df:,}</td>'
        f"<td>{idf:.2f}</td><td>{w_q:.3f}</td>"
        f'<td class="barcell"><div class="bar" style="width:{100 * w_q / top_weight:.1f}%"></div></td></tr>'
        for term, qtf, df, idf, w_q in terms.itertuples(index=False, name=None)
    )
    parts.append(
        "<h3>Step 2 · How important is each word?</h3>"
        f'<p class="note">The fewer of the {tfidf.n_docs:,} papers contain a word (<b>df</b>), the higher its rarity '
        "score (<b>idf</b>), so the more it counts. The longest bar is the claim's most important word.</p>"
        '<table class="tbl"><tr><th>word root</th><th>times in claim</th><th>papers containing it (df)</th>'
        f"<th>rarity (idf)</th><th>weight in claim (w<sub>q</sub>)</th><th></th></tr>{rows}</table>"
    )

    # Step 3: why TF-IDF's top hit won, and (for test claims) how the correct abstract scored.
    relevant = engine.relevant(query)
    top = tfidf.search(query, 1)
    top_id = top[0][0] if top else None
    docs = []
    if top_id:
        label = "TF-IDF's #1 result = the correct paper ✓" if top_id in relevant else "TF-IDF's #1 result"
        docs.append((label, top_id))
    docs += [("Correct paper", d) for d in relevant if d != top_id]
    explained = [(label, d, *engine.contributions(query, d)) for label, d in docs]
    scale = max((t["contribution"].max() for _, _, t, _, _ in explained if not t.empty), default=0)
    takeaway = tfidf_takeaway(engine, query)
    parts.append(
        "<h3>Step 3 · Which paper wins, and why?</h3>"
        + (f'<p class="takeaway">{html.escape(takeaway, quote=False)}</p>' if takeaway else "")
        + '<p class="note">For each word root: weight in the claim × weight in the paper. The sum is the '
        "<b>cosine similarity</b> that TF-IDF sorts the papers by. A word missing from the paper adds 0, however "
        "important it is.</p><div class=\"duel\">"
    )
    for label, d, table, score, rank in explained:
        rank_text = f"TF-IDF position {rank}" if rank else "not found by TF-IDF"
        parts.append(
            f'<div class="doc"><div class="doc-head"><b>{html.escape(label)}</b></div>'
            f'<div class="title">{html.escape(engine.corpus[d]["title"])}</div>'
            f'<div class="doc-meta">{rank_text} · cosine similarity <b>{score:.3f}</b></div>'
            f"{_contribution_table(table, scale)}</div>"
        )
    parts.append(
        "</div>"
        f'<details class="aside"><summary>The exact formula (SMART scheme <code>{html.escape(tfidf.scheme)}</code>)'
        f"</summary><p>{scheme_formula(tfidf.scheme)} N = {tfidf.n_docs:,} papers.</p></details>"
    )
    return '<div class="irsys">' + "".join(parts) + "</div>"


def _p_yes(log_odds: float) -> float:
    return 100 / (1 + math.exp(-log_odds))


def render_llm_explanation(engine: DemoEngine, query: str, llm_result: SystemResult | None = None) -> str:
    """How the LLM decides, its P(yes) for the papers that matter for this claim, and the exact prompt."""
    from irsys.llm import format_pair

    if engine.llm_name is None:
        return '<div class="irsys"><p class="note">The LLM reranker is not loaded in this demo.</p></div>'
    llm = engine.systems[engine.llm_name]
    if llm_result is None:
        (llm_result,) = engine.compare(query, names=[engine.llm_name])
    if not llm_result.ranking:
        return ('<div class="irsys"><p class="note">TF-IDF found no candidate papers, so there was nothing for the '
                "LLM to judge.</p></div>")
    log_odds = dict(llm_result.ranking)
    llm_rank = {d: r for r, (d, _) in enumerate(llm_result.ranking, start=1)}
    tfidf_rank = {d: r for r, (d, _) in enumerate(engine.tfidf.search(query, 100), start=1)}
    relevant = engine.relevant(query)
    top_tfidf = llm.first_stage.search(query, 1)[0][0]

    labels: dict[str, list[str]] = {}
    for label, docs in (("TF-IDF's #1", [top_tfidf]), ("Correct paper", relevant),
                        ("LLM's #1", [llm_result.ranking[0][0]])):
        for d in docs:
            labels.setdefault(d, []).append(label)
    color = SYSTEM_COLORS[2]
    rows = []
    for d, names in labels.items():
        if d in log_odds:
            p = _p_yes(log_odds[d])
            bar = (f'<div class="pwrap"><div class="pbar"><div style="width:{p:.1f}%"></div></div>'
                   f"<b>{p:.1f}%</b></div>")
        else:
            bar = '<span class="muted">not among the 100 candidates</span>'
        moved = f"#{tfidf_rank.get(d, '>100')} → #{llm_rank.get(d, '—')}"
        rows.append(
            f'<tr><td class="left role">{html.escape(" · ".join(names), quote=False)}</td><td class="left">'
            f'{html.escape(engine.corpus[d]["title"])}</td><td class="left">{bar}</td><td>{moved}</td></tr>'
        )

    correct = next((d for d in relevant if d in log_odds), None)
    if correct:
        tr, lr = tfidf_rank.get(correct), llm_rank[correct]
        move = f"stays at position {lr}." if tr == lr else f"moves from position {tr} to {lr}."
        p_correct = f"{_p_yes(log_odds[correct]):.1f}%"
        if correct == top_tfidf:
            takeaway = (f"The LLM says {p_correct} “yes” to the correct paper, which was already TF-IDF's #1, "
                        f"so the correct paper {move}")
        else:
            takeaway = (f"The LLM says {p_correct} “yes” to the correct paper but only "
                        f"{_p_yes(log_odds[top_tfidf]):.1f}% to TF-IDF's #1, so the correct paper {move}")
    elif relevant:
        takeaway = ("The correct paper is not among TF-IDF's 100 candidates, so the LLM never sees it: a reranker "
                    "cannot fix a first-stage miss.")
    else:
        takeaway = (f"The LLM's #1 paper got {_p_yes(llm_result.ranking[0][1]):.1f}% “yes”. There is no answer key "
                    "for a typed claim.")

    shown = correct or llm_result.ranking[0][0]
    prefix, body, suffix = format_pair(query, _snippet(engine.corpus[shown]["text"], 300),
                                       getattr(llm.reranker, "instruction", ""))
    flow = [("TF-IDF picks 100 candidate papers", "in a fraction of a millisecond"),
            ("The LLM reads the claim with one paper", "and may only answer “yes” or “no”"),
            ("Papers are sorted by P(“yes”)", "the model's probability that the next word is “yes”")]
    steps = '<span class="arrow">→</span>'.join(
        f'<div class="stepbox"><b class="n">{i}</b><b>{html.escape(h, quote=False)}</b><br>'
        f"<span class=\"muted\">{html.escape(s, quote=False)}</span></div>"
        for i, (h, s) in enumerate(flow, start=1)
    )
    return (
        f'<div class="irsys" style="--c:{color}">'
        "<h3>How the LLM reranker decides</h3>"
        f'<div class="flow">{steps}</div>'
        f'<p class="takeaway" style="--c:{color}">{html.escape(takeaway, quote=False)}</p>'
        "<h3>What it said about the papers that matter here</h3>"
        '<table class="tbl pyes"><tr><th class="left">role</th><th class="left">paper</th><th class="left">P(“yes”)</th>'
        f'<th>TF-IDF → LLM</th></tr>{"".join(rows)}</table>'
        f'<details class="aside"><summary>Show the exact prompt the LLM reads for “'
        f'{html.escape(engine.corpus[shown]["title"])}”</summary>'
        f'<pre class="prompt">{html.escape(prefix + body + suffix)}<b>yes / no ?</b></pre>'
        '<p class="note">The paper is shortened here for display; the model reads up to 512 tokens. The same '
        "question is asked of all 100 candidates.</p></details></div>"
    )


def header_html(n_papers: int, llm: bool) -> str:
    systems = [(SYSTEM_COLORS[0], "Word matching · TF-IDF + cosine (ours)"),
               (SYSTEM_COLORS[1], "Meaning matching · semantic search")]
    if llm:
        systems.append((SYSTEM_COLORS[2], "Reading · LLM reranker"))
    legend = "".join(f'<span class="sys">{_dot(c)}{html.escape(n)}</span>' for c, n in systems)
    return (f'<div class="irsys hero"><h1>Which search engine finds the paper behind a claim?</h1>'
            f'<p class="lead">{n_papers:,} scientific papers. Pick an example or type a claim: '
            f"{len(systems)} engines search at once.</p><div class=\"legend\">{legend}</div></div>")


# ---------------------------------------------------------------- Gradio app

CSS = """
.irsys { font-size: 16px; line-height: 1.45; }
.irsys mark { background: #fde68a; color: #111; padding: 0 2px; border-radius: 3px; }
.irsys .note { color: var(--body-text-color-subdued); margin: 6px 0 12px; }
.irsys .muted { color: var(--body-text-color-subdued); }
.irsys h3 { font-size: 20px; margin: 22px 0 8px; }
.irsys i[style] { display: inline-block; width: 14px; height: 14px; border-radius: 50%; background: var(--c); flex: none; vertical-align: -1px; margin-right: 8px; }
.hero h1 { font-size: 30px; margin: 2px 0 4px; }
.hero .lead { font-size: 17px; margin: 0 0 8px; color: var(--body-text-color-subdued); }
.hero .legend { display: flex; flex-wrap: wrap; gap: 8px 22px; font-weight: 600; }
.irsys .story { border-radius: 14px; padding: 14px 18px; background: color-mix(in srgb, #f2b544 16%, var(--block-background-fill)); border: 1px solid color-mix(in srgb, #f2b544 55%, transparent); }
.irsys .story .kicker { font-size: 12.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued); }
.irsys .story .story-head { font-size: 23px; font-weight: 700; margin: 2px 0 4px; }
.irsys .story p { margin: 0; font-size: 17px; }
.irsys .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 6px 0 10px; }
.irsys .tile { border-radius: 14px; padding: 14px 16px; background: color-mix(in srgb, var(--c) 9%, var(--block-background-fill)); border: 2px solid color-mix(in srgb, var(--c) 55%, transparent); }
.irsys .tile .name { display: flex; align-items: center; font-weight: 700; font-size: 16px; }
.irsys .tile .big { font-size: 48px; font-weight: 800; line-height: 1.05; margin-top: 8px; font-variant-numeric: tabular-nums; }
.irsys .tile .cap { font-size: 13px; color: var(--body-text-color-subdued); }
.irsys .tile .status { font-weight: 700; margin-top: 6px; }
.irsys .tile.good .big, .irsys .tile.good .status { color: #0b8a3e; }
.irsys .tile.warn .big, .irsys .tile.warn .status { color: #b45309; }
.irsys .tile.bad .big, .irsys .tile.bad .status { color: #c62828; }
.irsys .tile .time { font-size: 13.5px; color: var(--body-text-color-subdued); margin-top: 4px; }
.irsys .tile.stat .big { color: var(--body-text-color); }
.irsys .tile.stat .verdict { font-weight: 700; margin-top: 6px; }
.irsys .spin { width: 42px; height: 42px; margin: 10px 0 6px; border: 4px solid color-mix(in srgb, var(--c) 25%, transparent); border-top-color: var(--c); border-radius: 50%; animation: irsys-spin .9s linear infinite; }
@keyframes irsys-spin { to { transform: rotate(360deg); } }
.irsys .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.irsys .col { border: 1px solid var(--border-color-primary); border-radius: 12px; background: var(--block-background-fill); overflow: hidden; }
.irsys .col header { border-top: 6px solid var(--c); padding: 10px 14px; border-bottom: 1px solid var(--border-color-primary); }
.irsys .col h4 { margin: 0; font-size: 16px; display: flex; align-items: center; }
.irsys .blurb { font-size: 13.5px; color: var(--body-text-color-subdued); margin-top: 3px; }
.irsys .col.pending .blurb { font-style: italic; padding: 6px 0; }
.irsys ol { list-style: none; margin: 0; padding: 0; }
.irsys .hit { display: flex; gap: 10px; padding: 10px 14px; border-bottom: 1px solid var(--border-color-primary); cursor: help; }
.irsys .hit.relevant { background: rgba(11, 138, 62, 0.14); }
.irsys .rank { font-weight: 800; min-width: 30px; color: var(--body-text-color-subdued); }
.irsys .hit.relevant .rank, .irsys .tag { color: #0b8a3e; }
.irsys .tag { display: block; font-weight: 800; font-size: 13px; }
.irsys .title { font-weight: 600; }
.irsys .empty { padding: 12px 14px; color: var(--body-text-color-subdued); }
.irsys .takeaway { font-size: 17.5px; font-weight: 600; padding: 12px 16px; margin: 14px 0 10px; border-radius: 12px; background: color-mix(in srgb, var(--c, #2a78d6) 11%, var(--block-background-fill)); border: 1px solid color-mix(in srgb, var(--c, #2a78d6) 45%, transparent); }
.irsys .aside { margin: 8px 0 12px; }
.irsys .aside summary { cursor: pointer; font-weight: 600; color: var(--body-text-color-subdued); }
.irsys .aside p { margin: 6px 0 0 18px; }
.irsys .steps th { text-align: left; padding: 6px 12px 6px 0; white-space: nowrap; font-weight: 600; vertical-align: top; }
.irsys .steps td { padding: 6px 8px; }
.irsys .steps small { font-weight: 400; color: var(--body-text-color-subdued); }
.irsys .chip { display: inline-block; padding: 2px 9px; margin: 2px; border-radius: 12px; background: rgba(42, 120, 214, 0.15); font-family: ui-monospace, Consolas, monospace; font-size: 14px; }
.irsys .chip.dim { opacity: 0.45; text-decoration: line-through; }
.irsys .chip.pos small { display: block; font-family: system-ui, sans-serif; font-size: 11px; color: var(--body-text-color-subdued); text-align: center; }
.irsys .chip.phrase { background: rgba(237, 161, 0, 0.25); font-weight: 600; }
.irsys .tbl { border-collapse: collapse; margin: 4px 0 8px; font-size: 14.5px; width: 100%; }
.irsys .tbl th, .irsys .tbl td { border-bottom: 1px solid var(--border-color-primary); padding: 5px 10px; text-align: right; }
.irsys .tbl th:first-child, .irsys .tbl td.term { text-align: left; font-family: ui-monospace, Consolas, monospace; }
.irsys .tbl td.left, .irsys .tbl th.left { text-align: left; font-family: inherit; }
.irsys .tbl td.role { font-weight: 700; white-space: nowrap; }
.irsys .tbl.pyes td:last-child { white-space: nowrap; font-weight: 700; }
.irsys .tbl.pyes th { white-space: nowrap; }
.irsys .tbl tr.absent td { color: var(--body-text-color-subdued); }
.irsys .barcell { width: 26%; text-align: left !important; }
.irsys .doc .tbl td, .irsys .doc .tbl th { padding: 4px 7px; }
.irsys .bar { height: 12px; background: #2a78d6; border-radius: 0 4px 4px 0; }
.irsys .duel { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
.irsys .doc { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px; background: var(--block-background-fill); }
.irsys .doc-head { font-size: 15px; }
.irsys .doc-meta { font-size: 14px; color: var(--body-text-color-subdued); margin-bottom: 4px; }
.irsys .flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; }
.irsys .flow .stepbox { flex: 1 1 220px; padding: 10px 14px; border-radius: 12px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.irsys .flow .arrow { align-self: center; font-size: 24px; color: var(--body-text-color-subdued); }
.irsys .stepbox .n { display: inline-flex; width: 24px; height: 24px; border-radius: 50%; background: var(--c); color: #fff; align-items: center; justify-content: center; margin-right: 8px; font-size: 13px; }
.irsys .pwrap { display: flex; align-items: center; gap: 10px; }
.irsys .pbar { flex: 1; min-width: 160px; height: 18px; border-radius: 6px; background: color-mix(in srgb, var(--c) 14%, transparent); overflow: hidden; }
.irsys .pbar > div { height: 100%; background: var(--c); }
.irsys .prompt { white-space: pre-wrap; font-size: 13px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px; }
.nav-row button { font-size: 16px !important; font-weight: 600 !important; }
.preset-row button { font-size: 15px !important; }
"""


METRICS_GUIDE = """\
**How to read this.** Every system searched for each of the 300 test claims, and we checked where it put the
correct paper. Each metric is between 0 and 1 (higher is better), averaged over the 300 claims; @5 and @10 mean
"in the first 5 / 10 results".
- **nDCG@k**: the main grade. How high the correct paper appears in the top k: 1 = always first, 0.5 ≈ usually
  third, 0 = not in the top k.
- **MAP** (mean average precision): how early *all* the correct papers appear in the ranking.
- **MRR** (mean reciprocal rank): 1 ÷ the position of the first correct paper (position 2 → 0.5).
- **Precision@k**: the share of the top k that is correct. Small for everyone (at most 0.2 at k = 5, 0.1 at
  k = 10) because most claims have only one correct paper.
- **Recall@k**: the share of the correct papers found in the top k.
- **F1@k**: precision and recall combined into one number (their harmonic mean).

Below the metrics: how long one search takes, and whether a statistical test on the same 300 claims (Wilcoxon,
p < 0.05) says the difference to TF-IDF is real or just luck.
"""
FIGURE_CAPTIONS = {
    "main_metrics": "The six metrics at k = 10, one panel each, on the same 0–1 scale.",
    "overlap_buckets": ("Claims split into three equal groups by how many of their words appear in the correct "
                        "paper. Every system does worse when fewer words are shared; word matching suffers most."),
    "per_query_delta": ("One thin bar per claim. Green: semantic search did better; blue: TF-IDF did better; "
                        "flat: a tie. They win on different claims."),
}
PLAIN_NAMES = {
    "tfidf": "Word matching · TF-IDF + cosine (ours)",
    "dense-bge": "Meaning matching · semantic search (bge-small)",
    "tfidf-llm": "LLM reranker (Qwen3)",
}
SHORT_NAMES = {"tfidf": "TF-IDF + cosine (ours)", "dense-bge": "Semantic search (bge-small)",
               "tfidf-llm": "LLM reranker (Qwen3)"}
KEY_COLORS = dict(zip(("tfidf", "dense-bge", "tfidf-llm"), SYSTEM_COLORS))


def _result_rows(results_dir) -> list[dict]:
    """Per system: its metrics, time per claim and the significance verdict vs. TF-IDF."""
    from irsys.evaluate import read_markdown_table
    from irsys.stats import paired_tests

    metrics = pd.read_csv(results_dir / "metrics.csv")
    latency = {row["System"]: float(row["Mean time per claim (ms)"])
               for row in read_markdown_table((results_dir / "efficiency.md").read_text(encoding="utf-8"))}
    ndcg = pd.read_csv(results_dir / "per_query.csv", dtype={"qid": str}).pivot(
        index="qid", columns="system", values="nDCG@10")
    tests = paired_tests(ndcg["tfidf"], {s: ndcg[s] for s in ndcg if s != "tfidf"}).set_index("system")
    rows = []
    for row in metrics.to_dict("records"):
        key, name = row["system"], row["name"]
        if key == "tfidf":
            verdict = "— (baseline)"
        elif tests.loc[key, "holm_p"] < 0.05:
            verdict = "yes" if tests.loc[key, "delta"] > 0 else "no, worse"
        else:
            verdict = "no real difference"
        rows.append({"key": key, "metrics": row, "verdict": verdict,
                     "time": _duration(latency[name]) if name in latency else "—"})
    return rows


def results_tables(results_dir=RESULTS_DIR) -> str:
    """The six metrics per system, then time per claim and the significance verdict vs. TF-IDF."""
    from irsys.metrics import METRICS

    scores = ["| System | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    extras = ["| System | Time per claim | Really better than TF-IDF? |", "|---|---|---|"]
    for r in _result_rows(results_dir):
        label = PLAIN_NAMES.get(r["key"], r["metrics"]["name"])
        scores.append(f"| {label} | " + " | ".join(f"{r['metrics'][m]:.3f}" for m in METRICS) + " |")
        extras.append(f"| {label} | {r['time']} | {r['verdict']} |")
    return ("**The six metrics** (k = 5 and 10)\n\n" + "\n".join(scores)
            + "\n\n**Speed and significance** (significance tested on nDCG@10)\n\n" + "\n".join(extras) + "\n")


def results_headline(results_dir=RESULTS_DIR) -> str:
    """One big tile per system: nDCG@10, time per claim, and whether it really beats TF-IDF."""
    tiles = []
    for r in _result_rows(results_dir):
        color = KEY_COLORS.get(r["key"], SYSTEM_COLORS[0])
        verdict = "the baseline" if r["key"] == "tfidf" else f"really better than TF-IDF? {r['verdict']}"
        tiles.append(
            f'<div class="tile stat" style="--c:{color}"><div class="name">{_dot(color)}'
            f'{html.escape(SHORT_NAMES.get(r["key"], r["metrics"]["name"]))}</div>'
            f'<div class="big">{r["metrics"]["nDCG@10"]:.3f}</div>'
            '<div class="cap">nDCG@10, the main grade (1 = correct paper always first)</div>'
            f'<div class="verdict">{verdict}</div><div class="time">{r["time"]} per claim</div></div>'
        )
    return f'<div class="irsys"><div class="tiles">{"".join(tiles)}</div></div>'


def build_app(engine: DemoEngine):
    import gradio as gr

    analyzer = engine.tfidf.analyzer
    latest = {"token": 0}  # the LLM column of an older search must not overwrite a newer one
    llm_waiting = ('<div class="irsys"><p class="note">The LLM is still reading the candidate papers … this tab '
                   "fills in about 5 seconds after a search.</p></div>")

    def run_fast(text: str):
        text = (text or "").strip()
        if not text:
            return ("", '<div class="irsys"><p class="note">Type a claim or pick an example.</p></div>', "", "", "",
                    None)
        latest["token"] += 1
        fast = engine.compare(text, k=5, names=engine.fast_names())
        pending = [engine.llm_name] if engine.llm_name else []
        return (render_story(engine, text), render_tiles(fast, pending), render_lists(fast, analyzer, text, pending),
                render_explanation(engine, text),
                llm_waiting if pending else render_llm_explanation(engine, text), (latest["token"], fast))

    def run_llm(text: str, state):
        if not engine.llm_name or state is None:
            return gr.skip(), gr.skip(), gr.skip()
        token, fast = state
        text = text.strip()
        (llm,) = engine.compare(text, k=5, names=[engine.llm_name])
        if token != latest["token"]:  # a newer search started meanwhile
            return gr.skip(), gr.skip(), gr.skip()
        both = fast + [llm]
        return render_tiles(both), render_lists(both, analyzer, text), render_llm_explanation(engine, text, llm)

    def setter(value: str):
        return lambda: value

    def go(tab: str):
        return lambda: gr.Tabs(selected=tab)

    first = engine.queries[PRESETS[0][1]]
    with gr.Blocks(title="Classical vs AI Search", analytics_enabled=False) as app:
        gr.HTML(header_html(len(engine.corpus), engine.llm_name is not None))
        with gr.Row(equal_height=True):
            query = gr.Textbox(value=first, label="Scientific claim (click an example, or type your own and press Enter)",
                               scale=6, lines=1)
            search = gr.Button("Search", variant="primary", scale=1)
        with gr.Row(elem_classes="preset-row"):
            presets = [(gr.Button(label), engine.queries[qid]) for label, qid in PRESETS]
            shuffle = gr.Button("🎲 Random test claim")
        story = gr.HTML()
        with gr.Tabs(selected="compare") as tabs:
            with gr.Tab("1 · Compare the search engines", id="compare"):
                tiles = gr.HTML()
                with gr.Row(elem_classes="nav-row"):
                    to_tfidf = gr.Button("Why did word matching rank them like this?  →  2 · Inside TF-IDF")
                    to_llm = gr.Button("How did the LLM decide?  →  3 · Inside the LLM",
                                       visible=engine.llm_name is not None)
                lists = gr.HTML()
            with gr.Tab("2 · Inside TF-IDF", id="tfidf"):
                explanation = gr.HTML()
                back_tfidf = gr.Button("←  Back to 1 · Compare", size="sm")
            with gr.Tab("3 · Inside the LLM", id="llm"):
                llm_explanation = gr.HTML()
                back_llm = gr.Button("←  Back to 1 · Compare", size="sm")
            with gr.Tab("4 · Results on all 300 test claims", id="results"):
                if (RESULTS_DIR / "metrics.csv").exists():
                    gr.HTML(results_headline())
                    with gr.Accordion("What do these numbers mean?", open=False):
                        gr.Markdown(METRICS_GUIDE)
                    gr.Markdown(results_tables())
                else:
                    gr.Markdown("Run `uv run irsys evaluate` to produce the results.")
                figures = RESULTS_DIR / "figures"
                for name in FIGURE_CAPTIONS:  # one per row: big enough to read on a projector
                    if (figures / f"{name}.png").exists():
                        gr.Image(str(figures / f"{name}.png"), show_label=False, interactive=False)
                        gr.Markdown(f"*{FIGURE_CAPTIONS[name]}*")

        to_tfidf.click(go("tfidf"), None, tabs)
        to_llm.click(go("llm"), None, tabs)
        back_tfidf.click(go("compare"), None, tabs)
        back_llm.click(go("compare"), None, tabs)

        state = gr.State()
        fast_outputs = [story, tiles, lists, explanation, llm_explanation, state]
        llm_outputs = [tiles, lists, llm_explanation]
        search.click(run_fast, query, fast_outputs).then(run_llm, [query, state], llm_outputs)
        query.submit(run_fast, query, fast_outputs).then(run_llm, [query, state], llm_outputs)
        for button, text in presets:
            (button.click(setter(text), None, query)
             .then(run_fast, query, fast_outputs).then(run_llm, [query, state], llm_outputs))
        (shuffle.click(engine.random_claim, None, query)
         .then(run_fast, query, fast_outputs).then(run_llm, [query, state], llm_outputs))
        app.load(run_fast, query, fast_outputs).then(run_llm, [query, state], llm_outputs)
    return app


def launch_demo(inbrowser: bool = True, server_port: int | None = None, llm: bool = True) -> None:
    import gradio as gr

    print("Loading the papers, the TF-IDF index, bge-small" + (" and the Qwen3 LLM" if llm else "") + " ...",
          flush=True)
    engine = DemoEngine.load(llm=llm)
    theme = gr.themes.Default(font=["Segoe UI", "system-ui", "sans-serif"], font_mono=["Consolas", "monospace"])
    build_app(engine).launch(inbrowser=inbrowser, server_port=server_port, css=CSS, theme=theme)
