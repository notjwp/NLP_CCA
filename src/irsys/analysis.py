"""Figures and error analysis from the files ``evaluate`` wrote (no retrieval is re-run)."""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from irsys.data import RESULTS_DIR, doc_text, load_scifact
from irsys.metrics import RECALL_KS
from irsys.preprocess import Analyzer

# Validated categorical palette (dataviz reference, slots 1-7, light surface); color follows the system.
COLORS = {
    "tfidf": "#2a78d6",
    "dense-minilm": "#eb6834",
    "dense-bge": "#1baf7a",
    "hybrid-rrf": "#eda100",
    "tfidf-ce": "#e87ba4",
    "tfidf-llm-zs": "#008300",
    "tfidf-llm": "#6250d6",
}
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BUCKETS = ["low", "medium", "high"]


# ---------------------------------------------------------------- analysis helpers

def lexical_overlap(queries: dict[str, str], qrels: dict[str, dict[str, int]], doc_texts: dict[str, str],
                    analyzer: Analyzer) -> pd.Series:
    """Share of a query's unique index terms that occur in at least one of its relevant docs."""
    out = {}
    for qid, text in queries.items():
        terms = set(analyzer(text))
        relevant_terms = set().union(*(analyzer(doc_texts[d]) for d, g in qrels.get(qid, {}).items() if g > 0))
        out[qid] = len(terms & relevant_terms) / len(terms) if terms else 0.0
    return pd.Series(out, name="overlap")


def overlap_terciles(overlap: pd.Series) -> pd.Series:
    """Equal-sized low/medium/high groups (ties broken by query order so the split stays even)."""
    return pd.qcut(overlap.rank(method="first"), 3, labels=BUCKETS).astype(str)


def read_trec_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        qid, _, doc, _rank, _score, _tag = line.split()
        run.setdefault(qid, []).append(doc)
    return run


def read_trec_scores(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        qid, _, doc, _rank, score, _tag = line.split()
        run.setdefault(qid, {})[doc] = float(score)
    return run


def first_relevant_rank(ranking: list[str], qrels: dict[str, int]) -> int | None:
    for rank, doc in enumerate(ranking, start=1):
        if qrels.get(doc, 0) > 0:
            return rank
    return None


# ---------------------------------------------------------------- error analysis

CATEGORIES = ["spelling variant", "vocabulary mismatch", "rare shared term", "common words only"]
RARE_DF = 10  # "rare" = occurs in at most 10 of the 5,183 abstracts
_RAW_TOKEN_RE = re.compile(r"[a-z0-9]+")


def spelling_variant(query: str, doc: str) -> bool:
    """A claim word the abstract only contains when split differently (claim "Bcl2", abstract "BCL-2"), or the reverse."""
    def tokens_and_joins(text: str) -> tuple[set[str], set[str]]:
        tokens = _RAW_TOKEN_RE.findall(text.lower())
        joins = {a + b for a, b in zip(tokens, tokens[1:]) if a not in ENGLISH_STOP_WORDS and b not in ENGLISH_STOP_WORDS}
        return set(tokens), joins

    q, q_joins = tokens_and_joins(query)
    d, d_joins = tokens_and_joins(doc)
    return bool((q - d) & d_joins or (d - q) & q_joins)


def failure_category(query: str, relevant_texts: list[str], analyzer: Analyzer, df: dict[str, int]) -> str:
    """What makes a claim hard, judged from the claim and its relevant abstract(s) only, so it is the same for
    every system. Rules apply in order:

    1. spelling variant:    a claim word is written differently in the abstract (Bcl2 / BCL-2);
    2. vocabulary mismatch: fewer than half of the claim's index terms occur in the abstract;
    3. rare shared term:    otherwise, a shared term occurs in at most ``RARE_DF`` abstracts;
    4. common words only:   the shared terms are all common, so many abstracts match them.
    """
    if any(spelling_variant(query, t) for t in relevant_texts):
        return "spelling variant"
    terms = set(analyzer(query))
    shared = terms & set().union(*(analyzer(t) for t in relevant_texts))
    if not terms or len(shared) / len(terms) < 0.5:
        return "vocabulary mismatch"
    if any(df.get(t, 0) <= RARE_DF for t in shared):
        return "rare shared term"
    return "common words only"


def error_table(category: pd.Series, ranks: pd.DataFrame, order: list[str] = CATEGORIES,
                cutoff: int = 10) -> pd.DataFrame:
    """Claims per category and, per system, how many have no relevant abstract in the top ``cutoff``.

    ``ranks``: rank of the first relevant abstract per claim (rows) and system (columns), NaN if not in the run.
    """
    failed = ranks.isna() | (ranks > cutoff)
    rows = []
    for cat in order:
        qids = category.index[category == cat]
        if len(qids):
            rows.append({"category": cat, "claims": len(qids), **failed.loc[qids].sum().astype(int).to_dict()})
    rows.append({"category": "all claims", "claims": len(category),
                 **failed.loc[category.index].sum().astype(int).to_dict()})
    return pd.DataFrame(rows)


def rerank_failures(first_stage: pd.Series, reranked: pd.Series, cutoff: int = 10) -> dict[str, int]:
    """Split a reranker's failures: the relevant abstract was never a candidate, or it was but got ranked low."""
    failed = reranked.isna() | (reranked > cutoff)
    missed = first_stage.isna()
    return {"failures": int(failed.sum()), "first stage missed it": int((failed & missed).sum()),
            "reranker ranked it low": int((failed & ~missed).sum())}


# ---------------------------------------------------------------- figures

def _style() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "Segoe UI" if "Segoe UI" in installed else "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    })


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_main_metrics(metrics: pd.DataFrame, path: Path) -> None:
    """Small multiples: one panel per metric, one bar per system, value at the bar tip."""
    panels = [("nDCG@10", "nDCG@10 (95% CI)"), ("MAP@100", "MAP@100"), ("R@100", "Recall@100")]
    m = metrics.iloc[::-1]  # first system on top
    y = np.arange(len(m))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        ax.barh(y, m[col], height=0.55, color=[COLORS[s] for s in m["system"]])
        if col == "nDCG@10":
            ax.errorbar(m[col], y, xerr=[m[col] - m["nDCG@10_ci_lo"], m["nDCG@10_ci_hi"] - m[col]],
                        fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        label_x = m["nDCG@10_ci_hi"] if col == "nDCG@10" else m[col]  # clear of the error bar
        for yi, v, lx in zip(y, m[col], label_x):
            ax.text(lx + 0.015, yi, f"{v:.3f}", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, m["name"])
    fig.suptitle("Retrieval effectiveness on SciFact test (300 queries)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def plot_recall_at_k(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for _, row in metrics.iterrows():
        values = [row[f"R@{k}"] for k in RECALL_KS]
        ax.plot(RECALL_KS, values, color=COLORS[row["system"]], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round", label=row["name"])
    ax.set_xscale("log")
    ax.set_xticks(RECALL_KS, [str(k) for k in RECALL_KS])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cut-off k (log scale)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k: share of relevant abstracts found in the top k")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_per_query_delta(per_query: pd.DataFrame, names: dict[str, str], path: Path) -> None:
    pivot = per_query.pivot(index="qid", columns="system", values="nDCG@10")
    delta = (pivot["dense-bge"] - pivot["tfidf"]).sort_values(ascending=False).to_numpy()
    wins, ties, losses = (delta > 1e-12).sum(), (np.abs(delta) <= 1e-12).sum(), (delta < -1e-12).sum()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(delta))
    colors = np.where(delta > 0, COLORS["dense-bge"], COLORS["tfidf"])
    ax.bar(x, delta, width=1.0, color=colors, edgecolor=SURFACE, linewidth=0.3)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlim(-1, len(delta))
    ax.set_xticks([])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Test queries, sorted by difference")
    ax.set_ylabel("ΔnDCG@10 (bge − TF-IDF)")
    ax.set_title("Per-query difference: dense bge-small vs. TF-IDF")
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS["dense-bge"]), plt.Rectangle((0, 0), 1, 1, color=COLORS["tfidf"])]
    ax.legend(handles, [f"{names['dense-bge']} better ({wins} queries)", f"{names['tfidf']} better ({losses} queries)"],
              loc="upper right")
    ax.text(wins + ties / 2, -0.08, f"Tied: {ties} queries (mostly both perfect or both zero)", ha="center", va="top",
            color=INK_2, fontsize=9)
    _save(fig, path)


def plot_overlap_buckets(table: pd.DataFrame, systems: list[str], names: dict[str, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    n = len(systems)
    width = 0.8 / n
    x = np.arange(len(table))
    for i, s in enumerate(systems):
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(xs, table[s], width=width, color=COLORS[s], edgecolor=SURFACE, linewidth=2, label=names[s])
    for xi, v in zip(x - 0.4 + width * 0.5, table["tfidf"]):  # label the baseline only
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, [f"{b} overlap\n{r}\n(n={c})" for b, r, c in zip(table["bucket"], table["range"], table["n"])])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean nDCG@10")
    ax.set_title("Effectiveness by query–document vocabulary overlap")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    _save(fig, path)


# ---------------------------------------------------------------- entry point

def analyze(results_dir: Path = RESULTS_DIR) -> None:
    _style()
    metrics = pd.read_csv(results_dir / "metrics.csv")
    per_query = pd.read_csv(results_dir / "per_query.csv", dtype={"qid": str})
    config = json.loads((results_dir / "config.json").read_text(encoding="utf-8"))
    names = dict(zip(metrics["system"], metrics["name"]))
    systems = list(metrics["system"])
    corpus, queries, qrels = load_scifact("test")
    texts = {d: doc_text(doc) for d, doc in corpus.items()}
    fig_dir = results_dir / "figures"

    plot_main_metrics(metrics, fig_dir / "main_metrics.png")
    plot_recall_at_k(metrics, fig_dir / "recall_at_k.png")
    plot_per_query_delta(per_query, names, fig_dir / "per_query_delta.png")

    # Vocabulary-mismatch analysis, using the same analyzer the selected TF-IDF system indexes with.
    cfg = config["tfidf"]
    overlap = lexical_overlap(queries, qrels, texts, Analyzer.from_config(cfg))
    bucket = overlap_terciles(overlap)
    ndcg = per_query.pivot(index="qid", columns="system", values="nDCG@10")
    rows = []
    for b in BUCKETS:
        qids = bucket.index[bucket == b]
        lo, hi = overlap[qids].min(), overlap[qids].max()
        rows.append({"bucket": b, "range": f"{lo:.2f}–{hi:.2f}", "n": len(qids), **ndcg.loc[qids, systems].mean().to_dict()})
    table = pd.DataFrame(rows)
    plot_overlap_buckets(table, systems, names, fig_dir / "overlap_buckets.png")
    (results_dir / "overlap_buckets.md").write_text(
        "Mean nDCG@10 per lexical-overlap tercile. Overlap = share of a query's index terms "
        f"(analyzer of the selected TF-IDF config: {cfg['label']}) that occur in its relevant abstract(s). "
        f"Spearman ρ between overlap and per-query nDCG@10: "
        + ", ".join(f"{names[s]} {overlap.corr(ndcg[s].reindex(overlap.index), method='spearman'):.2f}"
                    for s in systems)
        + ".\n\n" + table.rename(columns={**names, "range": "overlap range"}).to_markdown(index=False, floatfmt=".3f")
        + "\n",
        encoding="utf-8",
    )
    _write_examples(results_dir, queries, qrels, corpus, ndcg, overlap, names, systems)
    _write_error_analysis(results_dir, queries, qrels, corpus, texts, Analyzer.from_config(cfg), names, systems)
    print(f"Figures and analysis written to {results_dir}")


def _write_error_analysis(results_dir, queries, qrels, corpus, texts, analyzer, names, systems) -> None:
    df = Counter(t for text in texts.values() for t in set(analyzer(text)))
    relevant = {qid: [d for d, g in qrels[qid].items() if g > 0] for qid in queries}
    category = pd.Series({qid: failure_category(queries[qid], [texts[d] for d in relevant[qid]], analyzer, df)
                          for qid in queries})
    runs = {s: read_trec_run(results_dir / "runs" / f"{s}.trec") for s in systems}
    ranks = pd.DataFrame({s: {qid: first_relevant_rank(runs[s].get(qid, []), qrels[qid]) for qid in queries}
                          for s in systems}, dtype=float)

    table = error_table(category, ranks)
    for s in systems:
        table[s] = [f"{n} ({100 * n / c:.0f}%)" for n, c in zip(table[s], table["claims"])]
    parts = [
        f"# Error analysis (SciFact test, {len(queries)} claims)\n\n"
        "A claim **fails** for a system when none of its relevant abstracts is in that system's top 10 "
        "(nDCG@10 = 0).\n\n"
        "## 1. Failures by cause\n\n"
        "Each claim is put in one category using only the claim and its relevant abstract(s), so the "
        "categories are the same for every system. Rules, applied in order: **spelling variant** = a claim word "
        "is written differently in the abstract (e.g. *Bcl2* / *BCL-2*); **vocabulary mismatch** = fewer than half "
        "of the claim's index terms occur in the abstract; **rare shared term** = a shared term occurs in at most "
        f"{RARE_DF} of the {len(texts):,} abstracts; **common words only** = everything else. "
        "Cells: failures (share of the category's claims).\n\n",
        table.rename(columns=names).to_markdown(index=False) + "\n",
    ]
    for cat in CATEGORIES:
        qids = sorted(category.index[category == cat], key=int)[:2]
        if qids:
            parts.append(f"\n**{cat}**, e.g.\n")
            for qid in qids:
                rank_text = "; ".join(f"{names[s]} {int(r) if r == r else '>100'}" for s, r in ranks.loc[qid].items())
                parts.append(f"- [{qid}] “{queries[qid]}” → *{corpus[relevant[qid][0]]['title']}*. "
                             f"Rank of the relevant abstract: {rank_text}.\n")

    rerankers = [s for s in systems if s.startswith("tfidf-")]
    if rerankers:
        rows = [{"system": names[s], **rerank_failures(ranks["tfidf"], ranks[s])} for s in rerankers]
        parts += [
            "\n## 2. Rerankers: first-stage misses vs. reranking mistakes\n\n"
            "The rerankers only reorder TF-IDF's top 100. A failure is either a relevant abstract TF-IDF never "
            "retrieved (no reranker can fix it) or one the reranker saw but placed below rank 10.\n\n",
            pd.DataFrame(rows).to_markdown(index=False) + "\n",
        ]

    llms = [s for s in systems if s.startswith("tfidf-llm")]
    if llms:
        rows = []
        for s in llms:
            scores = read_trec_scores(results_dir / "runs" / f"{s}.trec")
            every = np.array([v for run in scores.values() for v in run.values()])
            rel = np.array([v for qid, run in scores.items() for d, v in run.items() if d in relevant.get(qid, [])])
            rows.append({"system": names[s], "candidates judged": len(every),
                         "answered yes (P(yes) > 0.5)": f"{100 * (every > 0).mean():.1f}%",
                         "relevant candidates answered yes": f"{100 * (rel > 0).mean():.1f}%"})
        parts += [
            "\n## 3. How the LLMs answer\n\n"
            "Each LLM is asked whether an abstract meets the claim's information need and may only answer "
            "yes or no; its score is log P(yes)/P(no). Share of the TF-IDF candidates it answered *yes* "
            "(most candidates are not relevant: about 1 in 100 is):\n\n",
            pd.DataFrame(rows).to_markdown(index=False) + "\n",
        ]
    (results_dir / "error_analysis.md").write_text("".join(parts), encoding="utf-8")


def _write_examples(results_dir, queries, qrels, corpus, ndcg, overlap, names, systems, n=3) -> None:
    runs = {s: read_trec_run(results_dir / "runs" / f"{s}.trec") for s in systems}
    delta = (ndcg["dense-bge"] - ndcg["tfidf"]).rename("delta")
    frame = pd.concat([delta, overlap], axis=1)
    # Biggest gaps first; among equal gaps prefer the clearest cases (lowest overlap for dense wins, highest for TF-IDF wins).
    dense_wins = frame.sort_values(["delta", "overlap"], ascending=[False, True]).head(n)
    tfidf_wins = frame.sort_values(["delta", "overlap"], ascending=[True, False]).head(n)

    def block(title: str, chosen: pd.DataFrame) -> str:
        out = [f"## {title}\n"]
        for qid, row in chosen.iterrows():
            relevant = [d for d, g in qrels[qid].items() if g > 0]
            out.append(f"### Query {qid}: “{queries[qid]}”\n")
            out.append(f"- Lexical overlap with relevant abstract(s): {row['overlap']:.2f}; "
                       f"ΔnDCG@10 (bge − TF-IDF) = {row['delta']:+.3f}")
            out += [f"- Relevant: [{d}] {corpus[d]['title']}" for d in relevant]
            ranks = []
            for s in systems:
                r = first_relevant_rank(runs[s].get(qid, []), qrels[qid])
                ranks.append(f"{names[s]}: {r if r else '>100'}")
            out.append("- Rank of first relevant abstract — " + "; ".join(ranks))
            top = runs["tfidf"].get(qid, [])[:1]
            if top and top[0] not in relevant:
                out.append(f"- TF-IDF's top hit instead: [{top[0]}] {corpus[top[0]]['title']}")
            top = runs["dense-bge"].get(qid, [])[:1]
            if top and top[0] not in relevant:
                out.append(f"- bge-small's top hit instead: [{top[0]}] {corpus[top[0]]['title']}")
            out.append("")
        return "\n".join(out)

    (results_dir / "examples.md").write_text(
        "# Qualitative examples (SciFact test)\n\n"
        + block("Where dense retrieval (bge-small) beats TF-IDF most", dense_wins) + "\n"
        + block("Where TF-IDF beats dense retrieval (bge-small) most", tfidf_wins),
        encoding="utf-8",
    )
