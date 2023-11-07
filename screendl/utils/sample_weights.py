"""Generate sample weights for drug responses observations."""

from __future__ import annotations

import numpy as np
import typing as t

from scipy import stats

if t.TYPE_CHECKING:
    from cdrpy.data import Dataset


def generate_dense_weights(
    ds: Dataset,
    alpha: float = 0.5,
    epsilon: float = 1e-4,
    pdf: t.Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Generate dense weights for drug response observations.

    Parameters
    ----------
        ds: The Dataset object to compute sample weights.
        alpha: Weight scaling factor. Higher values of alpha assign higher
            weights to rare samples. Setting alpha = 0.0 corresponds to
            standard training.
        epsilon: The minimum weight for individual samples.
        pdf: The probability density function to use. If `None`, a guassian kde
            is fit to generate the pdf.

    Returns
    -------
        A numpy array containing the sample weights.

    References
    ----------
        .. [1] Steininger, M., Kobs, K., Davidson, P. et al. "Density-based
               weighting for imbalanced regression", Mach Learn 110, 2021.

    """
    Y = ds.labels

    if pdf is None:
        pdf = stats.gaussian_kde(Y)

    Z = pdf(Y)
    Z_std = (Z - Z.min()) / (Z.max() - Z.min())
    weights = np.clip((1 - (alpha * Z_std)), a_min=epsilon, a_max=None)
    scaled_weights = weights / weights.mean()

    return scaled_weights
