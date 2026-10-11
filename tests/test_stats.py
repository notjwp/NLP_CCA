import numpy as np
import pandas as pd
import pytest

from irsys.stats import bootstrap_ci, holm, paired_tests


def test_holm_step_down_adjustment():
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone: 0.03, 0.06, 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(1).random(300)
    lo, hi = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert lo < values.mean() < hi
    assert hi - lo < 0.15


def test_bootstrap_ci_of_constant_is_a_point():
    assert bootstrap_ci(np.full(50, 0.5)) == pytest.approx((0.5, 0.5))


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
    assert row["wilcoxon_p"] < 0.001 and row["ttest_p"] < 0.001 and row["holm_p"] < 0.05
    assert row["wins"] + row["ties"] + row["losses"] == 200
    same_row = table.set_index("system").loc["same"]
    assert same_row["wilcoxon_p"] == 1.0 and same_row["ttest_p"] == 1.0  # all-zero differences
    assert same_row["ties"] == 200


def test_paired_tests_align_queries_by_id():
    base = pd.Series([0.0, 1.0], index=["a", "b"])
    other = pd.Series([1.0, 0.0], index=["b", "a"])  # identical after alignment
    row = paired_tests(base, {"other": other}).iloc[0]
    assert row["ties"] == 2
