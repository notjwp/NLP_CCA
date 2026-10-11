"""Presentation demo: live TF-IDF vs. semantic comparison and a step-by-step TF-IDF explanation.

``uv run irsys demo`` opens it in the browser. The engine and HTML renderers are plain
functions (tested in tests/test_demo.py); ``build_app`` only wires them into Gradio.
"""

import html
import json
import random
import re
import time
from dataclasses import dataclass

import pandas as pd

from irsys.base import Retriever
from irsys.data import RESULTS_DIR, doc_text, load_scifact
from irsys.preprocess import Analyzer
from irsys.tfidf import TfidfRetriever

# The four stories from REPORT.md §5.3 (SciFact test query ids).
PRESETS = [
    ("Bcl2 vs “BCL-2” · semantic wins", "1088"),
    ("Paraphrase · semantic wins", "1278"),
    ("Rare term · TF-IDF wins", "660"),
    ("Exact terms · TF-IDF wins", "674"),
]
SYSTEM_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]  # same colors as the report figures
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


class DemoEngine:
    def __init__(self, corpus, queries, qrels, tfidf: TfidfRetriever, systems: dict[str, Retriever]):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.tfidf = tfidf
        self.systems = systems
        self._qid_by_text = {text.strip(): qid for qid, text in queries.items()}

    @classmethod
    def load(cls) -> "DemoEngine":
        from irsys.dense import DenseRetriever
        from irsys.evaluate import BGE, BGE_QUERY_PROMPT
        from irsys.fusion import RRFHybrid

        corpus, queries, qrels = load_scifact("test")
        doc_ids = list(corpus)
        texts = [doc_text(corpus[d]) for d in doc_ids]
        config_path = RESULTS_DIR / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))["tfidf"] if config_path.exists() else {
            "scheme": "ltc.ltc", "stem": True, "stopwords": True}
        tfidf = TfidfRetriever(Analyzer(cfg["stem"], cfg["stopwords"]), cfg["scheme"]).index(doc_ids, texts)
        bge = DenseRetriever(BGE, query_prompt=BGE_QUERY_PROMPT).index(doc_ids, texts)
        systems = {
            f"TF-IDF cosine ({cfg['scheme']})": tfidf,
            "Semantic: bge-small-en-v1.5": bge,
            "Hybrid: TF-IDF + bge (RRF)": RRFHybrid([tfidf, bge]),
        }
        engine = cls(corpus, queries, qrels, tfidf, systems)
        engine.compare("warm-up query", k=1)  # CUDA initialisation happens here, not on stage
        return engine

    def lookup(self, text: str) -> str | None:
        return self._qid_by_text.get(text.strip())

    def random_claim(self) -> str:
        return self.queries[random.choice(list(self.queries))]

    def compare(self, text: str, k: int = 5, depth: int = 100) -> list[SystemResult]:
        qid = self.lookup(text)
        relevant = {d for d, g in self.qrels.get(qid, {}).items() if g > 0} if qid else set()
        results = []
        for name, retriever in self.systems.items():
            start = time.perf_counter()
            ranking = retriever.search(text, depth)
            latency = (time.perf_counter() - start) * 1000
            hits = [Hit(d, self.corpus[d]["title"], self.corpus[d]["text"], s, d in relevant) for d, s in ranking[:k]]
            rank = next((r for r, (d, _) in enumerate(ranking, start=1) if d in relevant), None)
            results.append(SystemResult(name, hits, bool(relevant), rank, latency))
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


def _snippet(text: str, limit: int = 230) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def render_results(results: list[SystemResult], analyzer: Analyzer, query: str) -> str:
    terms = set(analyzer(query))
    columns = []
    for color, res in zip(SYSTEM_COLORS * 2, results):
        if not res.judged:
            badge = ""
        elif res.correct_rank and res.correct_rank <= len(res.hits):
            badge = f'<div class="badge good">✓ correct abstract at rank {res.correct_rank}</div>'
        elif res.correct_rank:
            badge = f'<div class="badge warn">▼ correct abstract only at rank {res.correct_rank}</div>'
        else:
            badge = '<div class="badge bad">✗ correct abstract not in top 100</div>'
        hits = []
        for i, h in enumerate(res.hits, start=1):
            tag = '<span class="tag">✓ correct</span>' if h.relevant else ""
            hits.append(
                f'<li class="hit{" relevant" if h.relevant else ""}"><div class="hit-head"><span class="rank">{i}</span>'
                f'<span class="score">{h.score:.3f}</span>{tag}</div>'
                f'<div class="title">{highlight(h.title, analyzer, terms)}</div>'
                f'<div class="snippet">{highlight(_snippet(h.text), analyzer, terms)}</div></li>'
            )
        body = "".join(hits) or '<li class="empty">No abstract shares a word with the query.</li>'
        columns.append(
            f'<section class="col"><header style="border-top-color:{color}"><h3>{html.escape(res.name)}</h3>'
            f'<div class="meta">{res.latency_ms:.1f} ms</div>{badge}</header><ol>{body}</ol></section>'
        )
    note = ("This is a SciFact test claim, so the correct abstract is known (✓)." if results and results[0].judged
            else "Free-text query: no relevance judgments, compare the results by eye.")
    legend = '<p class="note"><mark>Highlighted</mark> words share an index term (stem) with the query. ' + note + "</p>"
    return f'<div class="irsys">{legend}<div class="grid">{"".join(columns)}</div></div>'


def scheme_formula(scheme: str) -> str:
    tf = {"n": "tf", "l": "(1 + ln tf)"}
    idf = {"n": "1", "t": "ln(N/df)", "s": "(ln(N/df) + 1)"}
    norm = {"c": ", then each vector is scaled to length 1 (cosine normalisation)", "n": ""}
    parts = []
    for side, code in zip(("document", "query"), scheme.split(".")):
        parts.append(f"{side} term weight = {tf[code[0]]} × {idf[code[1]]}{norm[code[2]]}")
    return "; ".join(parts) + ". Score = Σ w<sub>q</sub> × w<sub>d</sub> over shared terms = cosine similarity."


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
            f"<td>{w_q:.3f}</td><td>{'not in abstract' if absent else f'{w_d:.3f}'}</td>"
            f'<td>{contribution:.4f}</td><td class="barcell"><div class="bar" style="width:{width:.1f}%"></div></td></tr>'
        )
    return ('<table class="tbl"><tr><th>term</th><th>w<sub>q</sub></th><th>w<sub>d</sub></th>'
            '<th>w<sub>q</sub> × w<sub>d</sub></th><th></th></tr>' + "".join(rows) + "</table>")


def render_explanation(engine: DemoEngine, query: str) -> str:
    tfidf = engine.tfidf
    stages = tfidf.analyzer.stages(query)
    removed = [t for t in stages["tokens"] if t not in stages["without_stopwords"]]
    oov = set(engine.oov_terms(query))
    parts = [
        (f'<p class="formula"><b>Scheme {html.escape(tfidf.scheme)}:</b> {scheme_formula(tfidf.scheme)} '
         f"N = {tfidf.n_docs:,} abstracts.</p>"),
        '<h3>Step 1 · Text → index terms</h3><table class="steps">',
        f"<tr><th>Tokens</th><td>{_chips(stages['tokens'], set(removed), 'stopword')}</td></tr>",
        f"<tr><th>Stopwords removed</th><td>{_chips(stages['without_stopwords'])}</td></tr>",
        f"<tr><th>Stemmed index terms</th><td>{_chips(stages['terms'], oov, 'in no abstract')}</td></tr></table>",
    ]
    if oov:
        parts.append(f'<p class="note">Faded terms occur in <b>no</b> abstract, so TF-IDF ignores them: '
                     f'{html.escape(", ".join(sorted(oov)))}.</p>')

    terms = engine.term_table(query)
    if terms.empty:
        parts.append('<p class="note"><b>None of the query terms occur in the corpus</b>, '
                     "so TF-IDF cannot score any abstract.</p>")
        return '<div class="irsys">' + "".join(parts) + "</div>"
    rows = "".join(
        f'<tr><td class="term">{html.escape(term)}</td><td>{qtf}</td><td>{df:,}</td>'
        f"<td>{idf:.3f}</td><td>{w_q:.3f}</td></tr>"
        for term, qtf, df, idf, w_q in terms.sort_values("query weight", ascending=False).itertuples(
            index=False, name=None)
    )
    parts.append(
        "<h3>Step 2 · Weight each query term</h3>"
        '<p class="note">Rare terms (small df) get a large idf, so they dominate the query vector.</p>'
        '<table class="tbl"><tr><th>term</th><th>tf in query</th><th>df (abstracts containing it)</th>'
        f"<th>idf</th><th>query weight w<sub>q</sub></th></tr>{rows}</table>"
    )

    # Step 3: why TF-IDF's top hit won, and (for test claims) how the correct abstract scored.
    qid = engine.lookup(query)
    relevant = [d for d, g in engine.qrels.get(qid, {}).items() if g > 0] if qid else []
    top = tfidf.search(query, 1)
    top_id = top[0][0] if top else None
    docs = []
    if top_id:
        label = "TF-IDF's #1 result = the correct abstract ✓" if top_id in relevant else "TF-IDF's #1 result"
        docs.append((label, top_id))
    docs += [("Correct abstract", d) for d in relevant if d != top_id]
    explained = [(label, d, *engine.contributions(query, d)) for label, d in docs]
    scale = max((t["contribution"].max() for _, _, t, _, _ in explained if not t.empty), default=0)
    parts.append("<h3>Step 3 · Cosine score = sum of term contributions</h3>")
    for label, d, table, score, rank in explained:
        rank_text = f"TF-IDF rank {rank}" if rank else "not retrieved by TF-IDF (score 0)"
        parts.append(
            f'<div class="doc"><div class="doc-head"><b>{html.escape(label)}</b> · {rank_text} · '
            f'cosine = <b>{score:.3f}</b></div><div class="title">{html.escape(engine.corpus[d]["title"])}</div>'
            f"{_contribution_table(table, scale)}</div>"
        )
    return '<div class="irsys">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------- Gradio app

CSS = """
.irsys { font-size: 14px; }
.irsys mark { background: #fde68a; color: #111; padding: 0 2px; border-radius: 3px; }
.irsys .note { color: var(--body-text-color-subdued); margin: 4px 0 12px; }
.irsys .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
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
.irsys .tbl { border-collapse: collapse; margin: 4px 0 8px; font-size: 13px; min-width: 60%; }
.irsys .tbl th, .irsys .tbl td { border-bottom: 1px solid var(--border-color-primary); padding: 4px 10px; text-align: right; }
.irsys .tbl th:first-child, .irsys .tbl td.term { text-align: left; font-family: ui-monospace, Consolas, monospace; }
.irsys .tbl tr.absent td { color: var(--body-text-color-subdued); }
.irsys .barcell { width: 220px; text-align: left !important; }
.irsys .bar { height: 12px; background: #2a78d6; border-radius: 0 4px 4px 0; }
.irsys .doc { margin: 10px 0 14px; }
.irsys .doc-head { font-size: 14px; }
"""


def _metrics_markdown() -> str:
    path = RESULTS_DIR / "metrics.md"
    if not path.exists():
        return "Run `uv run irsys evaluate` to produce the results table."
    return path.read_text(encoding="utf-8")


def build_app(engine: DemoEngine):
    import gradio as gr

    def run(text: str):
        text = (text or "").strip()
        if not text:
            return "<p>Type a claim or pick an example.</p>", ""
        return (render_results(engine.compare(text, k=5), engine.tfidf.analyzer, text),
                render_explanation(engine, text))

    def setter(value: str):
        return lambda: value

    first = engine.queries[PRESETS[0][1]]
    with gr.Blocks(title="TF-IDF vs Semantic Search", analytics_enabled=False) as app:
        gr.Markdown(
            "## TF-IDF vs. semantic search · live demo on BEIR SciFact\n"
            f"{len(engine.corpus):,} scientific abstracts. Type a claim, or click an example. "
            "The same query goes to all three systems."
        )
        with gr.Row():
            query = gr.Textbox(value=first, label="Scientific claim (query)", scale=6, lines=1)
            search = gr.Button("Search", variant="primary", scale=1)
        with gr.Row():
            presets = [(gr.Button(label, size="sm"), engine.queries[qid]) for label, qid in PRESETS]
            shuffle = gr.Button("🎲 Random test claim", size="sm")
        with gr.Tabs():
            with gr.Tab("1 · Search & compare"):
                results = gr.HTML()
            with gr.Tab("2 · How TF-IDF scored it"):
                explanation = gr.HTML()
            with gr.Tab("3 · Evaluation results (300 test claims)"):
                gr.Markdown(_metrics_markdown())
                figures = RESULTS_DIR / "figures"
                for row in (["main_metrics"], ["overlap_buckets", "recall_at_k"], ["per_query_delta"]):
                    with gr.Row():
                        for name in row:
                            if (figures / f"{name}.png").exists():
                                gr.Image(str(figures / f"{name}.png"), show_label=False, interactive=False)

        outputs = [results, explanation]
        search.click(run, query, outputs)
        query.submit(run, query, outputs)
        for button, text in presets:
            button.click(setter(text), None, query).then(run, query, outputs)
        shuffle.click(engine.random_claim, None, query).then(run, query, outputs)
        app.load(run, query, outputs)
    return app


def launch_demo(inbrowser: bool = True, server_port: int | None = None) -> None:
    import gradio as gr

    print("Loading corpus, TF-IDF index and the bge-small model ...", flush=True)
    engine = DemoEngine.load()
    theme = gr.themes.Default(font=["Segoe UI", "system-ui", "sans-serif"], font_mono=["Consolas", "monospace"])
    build_app(engine).launch(inbrowser=inbrowser, server_port=server_port, css=CSS, theme=theme)
