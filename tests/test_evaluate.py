import pytest

from irsys.base import Retriever
from irsys.evaluate import ABLATION_CONFIGS, run_with_latency, tfidf_ablation, write_trec_run

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


def test_result_files_are_utf8_for_non_ascii_system_names(tmp_path):
    import pandas as pd

    from irsys.evaluate import _write_metrics, _write_significance

    per_query = {
        "tfidf": pd.DataFrame({m: [0.5, 1.0] for m in ["nDCG@10", "MAP@100", "MRR@10", "P@10", "R@10", "R@100"]},
                              index=["q1", "q2"]),
        "tfidf-ce": pd.DataFrame({m: [1.0, 1.0] for m in ["nDCG@10", "MAP@100", "MRR@10", "P@10", "R@10", "R@100"]},
                                 index=["q1", "q2"]),
    }
    names = {"tfidf": "TF-IDF", "tfidf-ce": "TF-IDF → cross-encoder"}
    _write_metrics(per_query, names, tmp_path)
    _write_significance(per_query, names, tmp_path)
    assert "TF-IDF → cross-encoder" in (tmp_path / "metrics.md").read_text(encoding="utf-8")
    assert "TF-IDF → cross-encoder" in (tmp_path / "significance.md").read_text(encoding="utf-8")
