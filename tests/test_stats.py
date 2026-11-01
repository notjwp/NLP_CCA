import numpy as np
import pandas as pd
import pytest

from irsys.stats import holm, paired_tests


def test_holm_step_down_adjustment():
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone: 0.03, 0.06, 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def _scores(seed, shift):
    rng = np.random.default_rng(seed)
    base = rng.random(200)
    return pd.Series(base, index=[f"q{i}" for i in range(200)]), pd.Series(
        np.clip(base + shift + rng.normal(0, 0.05, 200), 0, 1), index=[f"q{i}" for i in range(200)]
    )


def test_paired_tests_detect_a_real_improvement_and_count_wins():
    base, better = _scores(0, 0.1)
    same = base.copy()
    table = paired_tests(base, {"better": better, "same": same})
    row = table.set_index("system").loc["better"]
    assert row["delta"] == pytest.approx(better.mean() - base.mean())
    assert row["wilcoxon_p"] < 0.001 and row["holm_p"] < 0.05
    assert "ttest_p" not in table.columns  # the Wilcoxon test is the only significance test
    assert row["wins"] + row["ties"] + row["losses"] == 200
    same_row = table.set_index("system").loc["same"]
    assert same_row["wilcoxon_p"] == 1.0  # all-zero differences
    assert same_row["ties"] == 200


def test_paired_tests_align_queries_by_id():
    base = pd.Series([0.0, 1.0], index=["a", "b"])
    other = pd.Series([1.0, 0.0], index=["b", "a"])  # identical after alignment
    row = paired_tests(base, {"other": other}).iloc[0]
    assert row["ties"] == 2
