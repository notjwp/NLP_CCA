"""Presentation demo: live TF-IDF vs. semantic comparison and a step-by-step TF-IDF explanation.

``uv run irsys demo`` opens it in the browser. The engine and HTML renderers are plain
functions (tested in tests/test_demo.py); ``build_app`` only wires them into Gradio.
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
TFIDF_NAME = "Word matching · TF-IDF"
SEMANTIC_NAME = "Meaning matching · AI embeddings"
HYBRID_NAME = "Combined · words + meaning"
LLM_NAME = "LLM reranker · Qwen3"
DESCRIPTIONS = {
    TFIDF_NAME: "Your classical system: finds papers that use the claim's words. Rare words count more.",
    SEMANTIC_NAME: "A neural language model (bge-small) turns each text into a “meaning vector” and compares those.",
    HYBRID_NAME: "Merges the two lists on the left: papers ranked high by either one move up.",
    LLM_NAME: "A large language model reads the claim with each of TF-IDF's top 100 papers and answers yes or no.",
}
SYSTEM_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#6250d6"]  # same colors as the report figures
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
        from irsys.fusion import RRFHybrid

        corpus, queries, qrels = load_scifact("test")
        doc_ids = list(corpus)
        texts = [doc_text(corpus[d]) for d in doc_ids]
        config_path = RESULTS_DIR / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))["tfidf"] if config_path.exists() else {
            "scheme": "lsc.lsc", "stem": True, "stopwords": True}
        tfidf = TfidfRetriever(Analyzer.from_config(cfg), cfg["scheme"]).index(doc_ids, texts)
        bge = DenseRetriever(BGE, query_prompt=BGE_QUERY_PROMPT).index(doc_ids, texts)
        systems: dict[str, Retriever] = {
            TFIDF_NAME: tfidf,
            SEMANTIC_NAME: bge,
            HYBRID_NAME: RRFHybrid([tfidf, bge]),
        }
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


def render_results(results: list[SystemResult], analyzer: Analyzer, query: str,
                   pending: list[str] | None = None) -> str:
    """One column per system; ``pending`` systems get a placeholder column while they are still running."""
    terms = set(analyzer(query))
    columns = []
    for color, res in zip(SYSTEM_COLORS * 2, results):
        if not res.judged:
            badge = ""
        elif res.correct_rank and res.correct_rank <= len(res.hits):
            badge = f'<div class="badge good">✓ found the correct paper at position {res.correct_rank}</div>'
        elif res.correct_rank:
            badge = (f'<div class="badge warn">▼ correct paper only at position {res.correct_rank} '
                     f'(too low to show)</div>')
        else:
            badge = '<div class="badge bad">✗ correct paper not in its top 100</div>'
        hits = []
        for i, h in enumerate(res.hits, start=1):
            tag = '<span class="tag">✓ correct paper</span>' if h.relevant else ""
            hits.append(
                f'<li class="hit{" relevant" if h.relevant else ""}"><div class="hit-head"><span class="rank">#{i}</span>'
                f'{tag}</div>'
                f'<div class="title">{highlight(h.title, analyzer, terms)}</div>'
                f'<div class="snippet">{highlight(_snippet(h.text), analyzer, terms)}</div></li>'
            )
        body = "".join(hits) or '<li class="empty">No paper shares a word with the claim.</li>'
        columns.append(
            f'<section class="col"><header style="border-top-color:{color}"><h3>{html.escape(res.name)}</h3>'
            + (f'<div class="blurb">{html.escape(res.description)}</div>' if res.description else "")
            + f'<div class="meta">answered in {_duration(res.latency_ms)}</div>{badge}</header><ol>{body}</ol></section>'
        )
    for name, color in zip(pending or [], (SYSTEM_COLORS * 2)[len(results):]):
        columns.append(
            f'<section class="col pending"><header style="border-top-color:{color}"><h3>{html.escape(name)}</h3>'
            '<div class="meta">The LLM is reading the 100 candidate papers one by one … (about 5 seconds)</div>'
            "</header></section>"
        )
    note = ("This claim is from the test set, so we know which paper is correct: it is marked ✓ and shaded green."
            if results and results[0].judged
            else "Your own claim: nobody has marked a correct paper, so compare the lists by eye.")
    legend = (f'<p class="note">{note} <mark>Yellow</mark> words also appear in the claim (after cutting words to '
              "their root, so <i>tumors</i> matches <i>tumor</i>). Each column shows its top 5.</p>")
    return f'<div class="irsys">{legend}<div class="grid">{"".join(columns)}</div></div>'


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
            f"<td>{w_q:.3f}</td><td>{'not in this paper' if absent else f'{w_d:.3f}'}</td>"
            f'<td>{contribution:.4f}</td><td class="barcell"><div class="bar" style="width:{width:.1f}%"></div></td></tr>'
        )
    return ('<table class="tbl"><tr><th>word root</th><th>weight in claim (w<sub>q</sub>)</th>'
            '<th>weight in paper (w<sub>d</sub>)</th><th>w<sub>q</sub> × w<sub>d</sub></th><th></th></tr>'
            + "".join(rows) + "</table>")


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
        '<p class="note">1b and 1c use NLTK: a trained POS tagger, then a chunk parser with the grammar '
        "<code>NP: {&lt;JJ.*|NN.*&gt;*&lt;NN.*&gt;}</code> (adjectives and nouns ending in a noun). We tested "
        "using them to index dictionary forms (lemmas) and noun phrases; on the 809 training claims plain stemming "
        "worked best, so the final system indexes word roots only (report §4.1).</p>",
    ]
    if oov:
        parts.append(f'<p class="note">Faded word roots appear in <b>no</b> paper, so TF-IDF cannot use them: '
                     f'{html.escape(", ".join(sorted(oov)))}.</p>')

    terms = engine.term_table(query)
    if terms.empty:
        parts.append("<p class=\"note\"><b>None of the claim's words appear in any paper</b>, "
                     "so TF-IDF cannot score anything.</p>")
        return '<div class="irsys">' + "".join(parts) + "</div>"
    rows = "".join(
        f'<tr><td class="term">{html.escape(term)}</td><td>{qtf}</td><td>{df:,}</td>'
        f"<td>{idf:.3f}</td><td>{w_q:.3f}</td></tr>"
        for term, qtf, df, idf, w_q in terms.sort_values("query weight", ascending=False).itertuples(
            index=False, name=None)
    )
    parts.append(
        "<h3>Step 2 · How important is each word?</h3>"
        f'<p class="note"><b>df</b> = in how many of the {tfidf.n_docs:,} papers the word appears. The rarer the '
        "word, the higher its <b>idf</b> (rarity score), so the more it counts. The weights are then scaled so the "
        "claim's weights form a vector of length 1.</p>"
        '<table class="tbl"><tr><th>word root</th><th>times in claim (tf)</th><th>papers containing it (df)</th>'
        f"<th>rarity (idf)</th><th>weight in claim (w<sub>q</sub>)</th></tr>{rows}</table>"
    )

    # Step 3: why TF-IDF's top hit won, and (for test claims) how the correct abstract scored.
    qid = engine.lookup(query)
    relevant = [d for d, g in engine.qrels.get(qid, {}).items() if g > 0] if qid else []
    top = tfidf.search(query, 1)
    top_id = top[0][0] if top else None
    docs = []
    if top_id:
        label = "TF-IDF's #1 result = the correct paper ✓" if top_id in relevant else "TF-IDF's #1 result"
        docs.append((label, top_id))
    docs += [("Correct paper", d) for d in relevant if d != top_id]
    explained = [(label, d, *engine.contributions(query, d)) for label, d in docs]
    scale = max((t["contribution"].max() for _, _, t, _, _ in explained if not t.empty), default=0)
    parts.append(
        "<h3>Step 3 · Score = how much important vocabulary the claim and the paper share</h3>"
        '<p class="note">For each word root: weight in the claim × weight in the paper. Adding these up gives the '
        "<b>cosine similarity</b>, the number TF-IDF sorts the papers by. A word root missing from the paper adds 0, however "
        "important it is.</p>"
    )
    for label, d, table, score, rank in explained:
        rank_text = f"TF-IDF position {rank}" if rank else "not found by TF-IDF (score 0)"
        parts.append(
            f'<div class="doc"><div class="doc-head"><b>{html.escape(label)}</b> · {rank_text} · '
            f'score = <b>{score:.3f}</b></div><div class="title">{html.escape(engine.corpus[d]["title"])}</div>'
            f"{_contribution_table(table, scale)}</div>"
        )
    parts.append(
        f'<p class="formula"><b>The exact formula</b> (SMART scheme <code>{html.escape(tfidf.scheme)}</code>): '
        f"{scheme_formula(tfidf.scheme)} N = {tfidf.n_docs:,} papers.</p>"
    )
    return '<div class="irsys">' + "".join(parts) + "</div>"


def render_llm_explanation(engine: DemoEngine, query: str, llm_result: SystemResult | None = None) -> str:
    """The exact prompt the LLM sees and its P(yes) for the papers that matter for this claim."""
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

    labels: dict[str, list[str]] = {}
    for label, docs in (("TF-IDF's #1", [llm.first_stage.search(query, 1)[0][0]]),
                        ("Correct paper", engine.relevant(query)), ("LLM's #1", [llm_result.ranking[0][0]])):
        for d in docs:
            labels.setdefault(d, []).append(label)
    rows = []
    for d, names in labels.items():
        p_yes = f"{100 / (1 + math.exp(-log_odds[d])):.1f}%" if d in log_odds else "not a candidate"
        rows.append(
            f'<tr><td class="term">{html.escape(" · ".join(names), quote=False)}</td><td class="left">'
            f'{html.escape(engine.corpus[d]["title"])}</td><td>{tfidf_rank.get(d, ">100")}</td>'
            f"<td><b>{p_yes}</b></td><td>{llm_rank.get(d, '—')}</td></tr>"
        )

    shown = next((d for d in engine.relevant(query) if d in log_odds), llm_result.ranking[0][0])
    prefix, body, suffix = format_pair(query, _snippet(engine.corpus[shown]["text"], 300),
                                       getattr(llm.reranker, "instruction", ""))
    return (
        '<div class="irsys">'
        "<h3>How the LLM reranker works</h3>"
        '<p class="note">TF-IDF first picks 100 candidate papers. The LLM (Qwen3-Reranker, 0.6 billion parameters) '
        "then reads the claim together with <b>one</b> paper at a time, 100 times, and is told it may only answer "
        "“yes” or “no”. We don't let it write anything: we read the probability it gives to “yes” as its next word, "
        "and sort the papers by that probability.</p>"
        f"<h3>What it said about the papers that matter here</h3>"
        '<table class="tbl"><tr><th>role</th><th class="left">paper</th><th>TF-IDF position</th>'
        f'<th>LLM: P(“yes”)</th><th>LLM position</th></tr>{"".join(rows)}</table>'
        f"<h3>The exact prompt, for “{html.escape(engine.corpus[shown]['title'])}”</h3>"
        f'<pre class="prompt">{html.escape(prefix + body + suffix)}<b>yes / no ?</b></pre>'
        '<p class="note">The paper is shortened here for display; the model reads up to 512 tokens. The same '
        "question is asked of all 100 candidates.</p></div>"
    )


# ---------------------------------------------------------------- Gradio app

CSS = """
.irsys { font-size: 14px; }
.irsys mark { background: #fde68a; color: #111; padding: 0 2px; border-radius: 3px; }
.irsys .note { color: var(--body-text-color); opacity: 0.8; margin: 4px 0 12px; }
.irsys .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(235px, 1fr)); gap: 12px; }
.irsys .col { border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--block-background-fill); overflow: hidden; }
.irsys .col header { border-top: 5px solid; padding: 10px 12px 8px; border-bottom: 1px solid var(--border-color-primary); }
.irsys .col h3 { margin: 0; font-size: 15px; }
.irsys .meta { color: var(--body-text-color-subdued); font-size: 12px; }
.irsys .badge { margin-top: 6px; font-weight: 600; font-size: 13px; }
.irsys .badge.good { color: #0ca30c; } .irsys .badge.warn { color: #c98500; } .irsys .badge.bad { color: #d03b3b; }
.irsys ol { list-style: none; margin: 0; padding: 0; }
.irsys .hit { padding: 9px 12px; border-bottom: 1px solid var(--border-color-primary); }
.irsys .hit.relevant { background: rgba(12, 163, 12, 0.10); }
.irsys .hit-head { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--body-text-color-subdued); }
.irsys .rank { font-weight: 700; color: var(--body-text-color); }
.irsys .tag { color: #0ca30c; font-weight: 700; }
.irsys .title { font-weight: 600; margin: 2px 0; }
.irsys .snippet { font-size: 12.5px; color: var(--body-text-color-subdued); }
.irsys .empty { padding: 12px; color: var(--body-text-color-subdued); }
.irsys .formula { background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 8px 12px; }
.irsys h3 { margin: 18px 0 6px; }
.irsys .steps th { text-align: left; padding-right: 12px; white-space: nowrap; font-weight: 600; vertical-align: top; }
.irsys .chip { display: inline-block; padding: 1px 8px; margin: 2px; border-radius: 12px; background: rgba(42, 120, 214, 0.15); font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
.irsys .chip.dim { opacity: 0.45; text-decoration: line-through; }
.irsys .chip.pos small { display: block; font-family: system-ui, sans-serif; font-size: 10.5px; color: var(--body-text-color-subdued); text-align: center; }
.irsys .chip.phrase { background: rgba(237, 161, 0, 0.25); font-weight: 600; }
.irsys .steps small { font-weight: 400; color: var(--body-text-color-subdued); }
.irsys .steps th, .irsys .steps td { padding: 4px 8px; }
.irsys .blurb { font-size: 12.5px; margin: 2px 0 2px; }
.irsys .score { font-variant-numeric: tabular-nums; }
.irsys .col.pending .meta { padding: 8px 0; font-style: italic; }
.irsys .tbl td.left, .irsys .tbl th.left { text-align: left; font-family: inherit; }
.irsys .prompt { white-space: pre-wrap; font-size: 12.5px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px; }
.irsys .tbl { border-collapse: collapse; margin: 4px 0 8px; font-size: 13px; min-width: 60%; }
.irsys .tbl th, .irsys .tbl td { border-bottom: 1px solid var(--border-color-primary); padding: 4px 10px; text-align: right; }
.irsys .tbl th:first-child, .irsys .tbl td.term { text-align: left; font-family: ui-monospace, Consolas, monospace; }
.irsys .tbl tr.absent td { color: var(--body-text-color-subdued); }
.irsys .barcell { width: 220px; text-align: left !important; }
.irsys .bar { height: 12px; background: #2a78d6; border-radius: 0 4px 4px 0; }
.irsys .doc { margin: 10px 0 14px; }
.irsys .doc-head { font-size: 14px; }
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
    "per_query_delta": ("One thin bar per claim. Green: the AI embeddings did better; blue: TF-IDF did better; "
                        "flat: a tie. They win on different claims, which is why combining them works."),
}
PLAIN_NAMES = {
    "tfidf": "Word matching · TF-IDF (ours)",
    "dense-minilm": "Meaning matching · MiniLM",
    "dense-bge": "Meaning matching · bge-small",
    "hybrid-rrf": "Combined · words + meaning",
    "tfidf-ce": "Cross-encoder reranker",
    "tfidf-llm-zs": "LLM, prompted only",
    "tfidf-llm": "LLM reranker, fine-tuned",
}


def results_tables(results_dir=RESULTS_DIR) -> str:
    """The six metrics per system, then time per claim and the significance verdict vs. TF-IDF."""
    from irsys.evaluate import read_markdown_table
    from irsys.metrics import METRICS
    from irsys.stats import paired_tests

    metrics = pd.read_csv(results_dir / "metrics.csv")
    latency = {row["System"]: float(row["Mean time per claim (ms)"])
               for row in read_markdown_table((results_dir / "efficiency.md").read_text(encoding="utf-8"))}
    ndcg = pd.read_csv(results_dir / "per_query.csv", dtype={"qid": str}).pivot(
        index="qid", columns="system", values="nDCG@10")
    tests = paired_tests(ndcg["tfidf"], {s: ndcg[s] for s in ndcg if s != "tfidf"}).set_index("system")

    scores = ["| System | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    extras = ["| System | Time per claim | Really better than TF-IDF? |", "|---|---|---|"]
    for row in metrics.to_dict("records"):
        key, name = row["system"], row["name"]
        label = PLAIN_NAMES.get(key, name)
        scores.append(f"| {label} | " + " | ".join(f"{row[m]:.3f}" for m in METRICS) + " |")
        if key == "tfidf":
            verdict = "— (baseline)"
        elif tests.loc[key, "holm_p"] < 0.05:
            verdict = "yes" if tests.loc[key, "delta"] > 0 else "no, worse"
        else:
            verdict = "no real difference"
        extras.append(f"| {label} | {_duration(latency[name]) if name in latency else '—'} | {verdict} |")
    return ("**The six metrics** (k = 5 and 10)\n\n" + "\n".join(scores)
            + "\n\n**Speed and significance** (significance tested on nDCG@10)\n\n" + "\n".join(extras) + "\n")


def _metrics_markdown() -> str:
    if not (RESULTS_DIR / "metrics.csv").exists():
        return "Run `uv run irsys evaluate` to produce the results table."
    return METRICS_GUIDE + "\n" + results_tables()


def build_app(engine: DemoEngine):
    import gradio as gr

    analyzer = engine.tfidf.analyzer
    latest = {"token": 0}  # the LLM column of an older search must not overwrite a newer one
    llm_waiting = '<div class="irsys"><p class="note">The LLM is still reading the candidate papers …</p></div>'

    def run_fast(text: str):
        text = (text or "").strip()
        if not text:
            return "<p>Type a claim or pick an example.</p>", "", "", None
        latest["token"] += 1
        fast = engine.compare(text, k=5, names=engine.fast_names())
        pending = [engine.llm_name] if engine.llm_name else []
        return (render_results(fast, analyzer, text, pending), render_explanation(engine, text),
                llm_waiting if pending else render_llm_explanation(engine, text), (latest["token"], fast))

    def run_llm(text: str, state):
        if not engine.llm_name or state is None:
            return gr.skip(), gr.skip()
        token, fast = state
        text = text.strip()
        (llm,) = engine.compare(text, k=5, names=[engine.llm_name])
        if token != latest["token"]:  # a newer search started meanwhile
            return gr.skip(), gr.skip()
        return render_results(fast + [llm], analyzer, text), render_llm_explanation(engine, text, llm)

    def setter(value: str):
        return lambda: value

    first = engine.queries[PRESETS[0][1]]
    with gr.Blocks(title="Classical vs AI Search", analytics_enabled=False) as app:
        gr.Markdown(
            f"## Classical vs. AI search on {len(engine.corpus):,} scientific papers\n"
            "Type a scientific claim, or click an example. Each search engine lists the papers it thinks best "
            "support or refute the claim. For the examples we know which paper is correct, so you can see which "
            "engine finds it."
        )
        with gr.Row():
            query = gr.Textbox(value=first, label="Scientific claim", scale=6, lines=1)
            search = gr.Button("Search", variant="primary", scale=1)
        with gr.Row():
            presets = [(gr.Button(label, size="sm"), engine.queries[qid]) for label, qid in PRESETS]
            shuffle = gr.Button("🎲 Random test claim", size="sm")
        with gr.Tabs():
            with gr.Tab("1 · Compare the search engines"):
                results = gr.HTML()
            with gr.Tab("2 · Inside TF-IDF"):
                explanation = gr.HTML()
            with gr.Tab("3 · Inside the LLM"):
                llm_explanation = gr.HTML()
            with gr.Tab("4 · Results on all 300 test claims"):
                gr.Markdown(_metrics_markdown())
                figures = RESULTS_DIR / "figures"
                for name in FIGURE_CAPTIONS:  # one per row: big enough to read on a projector
                    if (figures / f"{name}.png").exists():
                        gr.Image(str(figures / f"{name}.png"), show_label=False, interactive=False)
                        gr.Markdown(f"*{FIGURE_CAPTIONS[name]}*")

        state = gr.State()
        fast_outputs = [results, explanation, llm_explanation, state]
        llm_outputs = [results, llm_explanation]
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
