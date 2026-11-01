import pytest

from irsys.base import Retriever
from irsys.evaluate import ABLATION_CONFIGS, run_with_latency, tfidf_ablation, write_trec_run
from irsys.metrics import METRICS

DOC_IDS = ["d1", "d2", "d3", "d4"]
TEXTS = [
    "Vitamin D deficiency increases fracture risk in elderly patients",
    "Running improves cardiovascular health",
    "Bone fractures heal slowly when vitamin levels are low",
    "Cardiac arrest outcomes after cooling therapy",
]
QUERIES = {"q1": "vitamin D and fractures", "q2": "heart health from running"}
QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}}


def test_write_trec_run_format(tmp_path):
    path = tmp_path / "runs" / "sys.trec"
    write_trec_run({"q1": {"d2": 0.5, "d1": 0.9}}, path, "sys")
    assert path.read_text().splitlines() == [
        "q1 Q0 d1 1 0.900000 sys",
        "q1 Q0 d2 2 0.500000 sys",
    ]


def test_run_with_latency_times_every_query():
    class _Fixed(Retriever):
        def search(self, query, k=10):
            return [("d1", 1.0), ("d2", 0.5)][:k]

    run, latencies = run_with_latency(_Fixed(), QUERIES, k=1)
    assert run == {"q1": {"d1": 1.0}, "q2": {"d1": 1.0}}
    assert len(latencies) == 2 and all(t >= 0 for t in latencies)


def test_ablation_selects_best_own_config_and_matches_sklearn():
    table, best = tfidf_ablation(DOC_IDS, TEXTS, QUERIES, QRELS)
    assert len(table) == len(ABLATION_CONFIGS) + 1  # + scikit-learn reference row
    own = table[~table["config"].str.contains("scikit-learn")]
    assert best.label == own.loc[own["nDCG@10"].idxmax(), "config"]
    assert table["selected"].sum() == 1
    sk = table[table["config"].str.contains("scikit-learn")].iloc[0]
    lsc = table[table["config"].str.startswith("lsc.lsc")].iloc[0]
    assert sk["nDCG@10"] == pytest.approx(lsc["nDCG@10"])


def test_ablation_covers_pos_lemmatization_and_noun_phrases():
    assert any(c.lemmatize and not c.stem for c in ABLATION_CONFIGS)
    assert any(c.phrases for c in ABLATION_CONFIGS)
    config = next(c for c in ABLATION_CONFIGS if c.phrases)
    retriever = config.build()
    assert retriever.analyzer.config() == {"stem": config.stem, "stopwords": config.stopwords,
                                           "lemmatize": config.lemmatize, "phrases": True}
    assert retriever.scheme == config.scheme


def test_result_files_are_utf8_for_non_ascii_system_names(tmp_path):
    import pandas as pd

    from irsys.evaluate import _write_metrics, _write_significance

    per_query = {
        "tfidf": pd.DataFrame({m: [0.5, 1.0] for m in METRICS}, index=["q1", "q2"]),
        "tfidf-ce": pd.DataFrame({m: [1.0, 1.0] for m in METRICS}, index=["q1", "q2"]),
    }
    names = {"tfidf": "TF-IDF", "tfidf-ce": "TF-IDF → cross-encoder"}
    _write_metrics(per_query, names, tmp_path)
    _write_significance(per_query, names, tmp_path)
    assert "TF-IDF → cross-encoder" in (tmp_path / "metrics.md").read_text(encoding="utf-8")
    assert "TF-IDF → cross-encoder" in (tmp_path / "significance.md").read_text(encoding="utf-8")


def test_metric_files_hold_exactly_the_six_metrics(tmp_path):
    import pandas as pd

    from irsys.evaluate import _write_metrics

    per_query = {"tfidf": pd.DataFrame({m: [0.5, 1.0] for m in METRICS}, index=["q1", "q2"])}
    _write_metrics(per_query, {"tfidf": "TF-IDF"}, tmp_path)
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert list(metrics.columns) == ["system", "name", *METRICS]
    header = (tmp_path / "metrics.md").read_text(encoding="utf-8").splitlines()[2]
    assert [c.strip() for c in header.strip("|").split("|")] == ["System", *METRICS]


def test_significance_note_counts_the_comparisons(tmp_path):
    import pandas as pd

    from irsys.evaluate import _write_significance

    per_query = {k: pd.DataFrame({m: [0.5, v, 1.0] for m in METRICS}, index=["q1", "q2", "q3"])
                 for k, v in [("tfidf", 0.0), ("a", 1.0), ("b", 0.5)]}
    _write_significance(per_query, {"tfidf": "TF-IDF", "a": "A", "b": "B"}, tmp_path)
    text = (tmp_path / "significance.md").read_text(encoding="utf-8")
    assert "Holm correction over the 2 comparisons" in text
    assert "t-test" not in text and "ttest" not in text


def test_efficiency_table_keeps_only_timings(tmp_path):
    from irsys.evaluate import _write_efficiency, read_markdown_table

    _write_efficiency(tmp_path, [{"System": "TF-IDF", "Index build (s)": 1.0, "Mean time per claim (ms)": 0.17,
                                  "Device": "cpu"}])
    rows = read_markdown_table((tmp_path / "efficiency.md").read_text(encoding="utf-8"))
    assert rows == [{"System": "TF-IDF", "Index build (s)": "1.00", "Mean time per claim (ms)": "0.17",
                     "Device": "cpu"}]


def test_rescore_recomputes_every_metric_file_from_saved_runs(tmp_path):
    import json

    import pandas as pd

    from irsys.evaluate import read_markdown_table, rescore

    names = {"tfidf": "TF-IDF", "other": "Other"}
    (tmp_path / "config.json").write_text(json.dumps({"systems": names}), encoding="utf-8")
    # "other" puts the relevant doc first; TF-IDF puts it third.
    write_trec_run({"q1": {"x": 0.9, "y": 0.8, "d1": 0.7}}, tmp_path / "runs" / "tfidf.trec", "tfidf")
    write_trec_run({"q1": {"d1": 0.9, "x": 0.8}}, tmp_path / "runs" / "other.trec", "other")
    old_efficiency = pd.DataFrame({"System": ["TF-IDF", "Other"], "Index build (s)": [1.0, 2.0],
                                   "Index size (MB)": [5.1, 7.6], "Mean latency (ms)": [0.17, 12.5],
                                   "Device": ["cpu", "cuda:0"]})
    (tmp_path / "efficiency.md").write_text("Note.\n\n" + old_efficiency.to_markdown(index=False), encoding="utf-8")

    rescore(tmp_path, qrels={"q1": {"d1": 1}}, ablation=False)

    metrics = pd.read_csv(tmp_path / "metrics.csv").set_index("system")
    assert list(metrics.columns) == ["name", *METRICS]
    assert metrics.loc["tfidf", "MRR"] == pytest.approx(1 / 3)
    assert metrics.loc["other", "nDCG@10"] == pytest.approx(1.0)
    assert (tmp_path / "significance.md").exists() and (tmp_path / "per_query.csv").exists()
    efficiency = read_markdown_table((tmp_path / "efficiency.md").read_text(encoding="utf-8"))
    assert [r["Mean time per claim (ms)"] for r in efficiency] == ["0.17", "12.50"]
    assert "Index size (MB)" not in efficiency[0]


def test_systems_can_be_built_lazily_one_at_a_time():
    from irsys.evaluate import materialize

    built = []

    class _Fixed(Retriever):
        def search(self, query, k=10):
            return []

    def factory():
        built.append(1)
        return _Fixed()

    eager = _Fixed()
    assert materialize(eager) is eager
    assert isinstance(materialize(factory), _Fixed) and built == [1]
