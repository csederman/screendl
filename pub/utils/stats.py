"""Stats utils for ScreenDL eval."""

from __future__ import annotations

import typing as t
import numpy as np

from scipy import stats


def combine_pvalues_cauchy(
    pvals: t.Iterable[float], weights: t.Iterable[float] | None = None
) -> float:
    """Combines p-values using the Cauchy combination method."""
    p = np.asarray(pvals)
    K = len(p)

    if weights is None:
        w = np.ones(K) / K
    else:
        w = np.asarray(weights)
        w = w / w.sum()

    t = np.tan((0.5 - p) * np.pi)
    T = np.sum(w * t)
    p_comb = 0.5 - np.arctan(T) / np.pi

    return np.clip(p_comb, 0, 1)
