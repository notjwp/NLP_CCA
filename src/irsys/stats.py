"""Significance testing over per-query scores (paired by query id)."""

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment for a family of comparisons."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_ci(values, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling queries with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_tests(baseline: pd.Series, systems: dict[str, pd.Series]) -> pd.DataFrame:
    """Each system vs. the baseline on the same queries: two-sided Wilcoxon signed-rank and
    paired t-test, with Holm correction (on the Wilcoxon p-values) across the systems."""
    rows = []
    for name, scores in systems.items():
        other = scores.reindex(baseline.index)
        diff = (other - baseline).to_numpy()
        if np.allclose(diff, 0):
            p_w = p_t = 1.0  # both tests are undefined for all-zero differences
        else:
            p_w = float(stats.wilcoxon(other, baseline).pvalue)
            p_t = float(stats.ttest_rel(other, baseline).pvalue)
        rows.append({
            "system": name,
            "baseline_mean": baseline.mean(),
            "system_mean": other.mean(),
            "delta": other.mean() - baseline.mean(),
            "wins": int((diff > 1e-12).sum()),
            "ties": int((np.abs(diff) <= 1e-12).sum()),
            "losses": int((diff < -1e-12).sum()),
            "wilcoxon_p": p_w,
            "ttest_p": p_t,
        })
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["wilcoxon_p"].tolist())
    return table
