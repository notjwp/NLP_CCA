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


def paired_tests(baseline: pd.Series, systems: dict[str, pd.Series]) -> pd.DataFrame:
    """Each system vs. the baseline on the same queries: two-sided Wilcoxon signed-rank test,
    with Holm correction across the systems."""
    rows = []
    for name, scores in systems.items():
        other = scores.reindex(baseline.index)
        diff = (other - baseline).to_numpy()
        # The test is undefined when every difference is zero.
        p_w = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(other, baseline).pvalue)
        rows.append({
            "system": name,
            "baseline_mean": baseline.mean(),
            "system_mean": other.mean(),
            "delta": other.mean() - baseline.mean(),
            "wins": int((diff > 1e-12).sum()),
            "ties": int((np.abs(diff) <= 1e-12).sum()),
            "losses": int((diff < -1e-12).sum()),
            "wilcoxon_p": p_w,
        })
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["wilcoxon_p"].tolist())
    return table
