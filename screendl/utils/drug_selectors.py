""""""

from __future__ import annotations

import random

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.distance as scd
import typing as t

from abc import ABC, abstractmethod

from cdrpy.data.datasets import Dataset
from cdrpy.core.random import _seeded_state

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_selection import SelectKBest


__all__ = [
    "RandomDrugSelector",
    "KMeansDrugSelector",
    "PrincipalDrugSelector",
    "MeanResponseSelector",
]


def get_response_matrix(D: Dataset, impute: bool = True) -> pd.DataFrame:
    """Converts dataset observations into a drug response matrix."""
    M = D.obs.pivot(index="drug_id", columns="cell_id", values="label")
    if impute:
        M[:] = KNNImputer(n_neighbors=3).fit_transform(M)
    return M


class DrugSelectorBase(ABC):
    """Base class for drug selectors."""

    @abstractmethod
    def select(
        self,
        n: int,
        choices: list[str] | None = None,
    ) -> list[str]:
        """"""
        ...


class RandomDrugSelector(DrugSelectorBase):
    """Random drug selection.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Optional seed for random number generation.
    """

    def __init__(self, D: Dataset, seed: int | float | None = None) -> None:
        self.dataset = D
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, _ = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> list[str]:
        """Samples random drugs.

        Parameters
        ----------
            n: The number of drugs to select.
            choices: The drugs to choose from.

        Returns
        -------
            A list of selected drugs.
        """
        M = self.response_mat
        if choices is not None:
            M = M[M.index.isin(choices)]

        return self._rs.sample(list(M.index), n)


class MeanResponseSelector(DrugSelectorBase):
    """Selects the drugs most predictive of a cell line's mean response.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Ignored, exists for compatability.
    """

    def __init__(self, D: Dataset, seed: int | float | None = None) -> None:
        self.dataset = D
        self.response_mat = get_response_matrix(self.dataset)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> list[str]:
        """Samples random drugs.

        Parameters
        ----------
            n: The number of drugs to select.
            choices: The drugs to choose from.

        Returns
        -------
            A list of selected drugs.
        """
        M = self.response_mat
        if choices is not None:
            M = M[M.index.isin(choices)]

        X = M.transform(stats.zscore, axis=1).T
        y = X.mean(axis=1)

        skb = SelectKBest(k=n).fit(X, y)

        return list(skb.get_feature_names_out())


class KMeansDrugSelector(DrugSelectorBase):
    """Selects drugs using K-Means clustering.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Optional seed for random number generation.
    """

    def __init__(self, D: Dataset, seed: int | float | None = None) -> None:
        self.dataset = D
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, self._np_rs = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> list[str]:
        """Selects the specified number of drugs from the response matrix.

        Parameters
        ----------
            n: The number of drugs to select.
            choices: The drugs to choose from.

        Returns
        -------
            A list of selected drugs.
        """
        M = self.response_mat
        if choices is not None:
            M = M[M.index.isin(choices)]

        km = KMeans(n, n_init="auto", random_state=self._np_rs).fit(M)

        selected_drugs = []
        for i, center in enumerate(km.cluster_centers_):
            M_c = M.iloc[np.argwhere(km.labels_ == i).reshape(-1)]
            dists = euclidean_distances(M_c, np.expand_dims(center, 0))

            idx_min = np.argmin(dists.reshape(-1))
            selected_drugs.append(M_c.index[idx_min])

        return selected_drugs


class PrincipalDrugSelector(DrugSelectorBase):
    """Drug selection using Principal Feature Analysis.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Optional seed for random number generation.

    References
    ----------
        .. [1] Lu, Y., Cohen, I., Zhou, X.S., Tian, Q. "Feature
               selection using principal feature analysis", Proceedings of the
               15th ACM International Conference on Multimedia, 301-304, 2007.
    """

    def __init__(self, D: Dataset, seed: int | float | None = None) -> None:
        self.dataset = D
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, self._np_rs = _seeded_state(seed)

    def select(
        self,
        n: int,
        q: int | None = None,
        choices: t.Iterable[str] | None = None,
    ) -> list[str]:
        """Selects drugs using Principle Feature Analysis.

        Parameters
        ----------
            n: The number of drugs to select.
            q: The number of principal components (must be less than `n`).
            choices: The drugs to choose from.
            random_state: Optional random state for K-Means and PCA.

        Returns
        -------
            A list of selected drugs.
        """
        if q is None:
            q = n - 1
        elif n <= q:
            raise ValueError(f"`q` must be less than `n`.")

        M = self.response_mat
        if choices is not None:
            M = M[M.index.isin(choices)]

        X = M.T
        X_std = StandardScaler().fit_transform(X)
        corr_matrix = np.corrcoef(X_std, rowvar=False)

        pca = PCA(q, random_state=self._np_rs)
        pca.fit(corr_matrix)
        A = pca.components_.T

        # TODO: add min explained variance threshold before restart

        km = KMeans(n, n_init="auto", random_state=self._np_rs)
        _ = km.fit(A)

        D = scd.cdist(km.cluster_centers_, A)
        inds = [np.argmin(x) for x in D]

        return list(M.index[inds])


class ShavingDrugSelector(DrugSelectorBase):
    """Drug selection using Drug Shaving.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Optional seed for random number generation.

    References
    ----------
        .. [1] Hastie T et al., "'Gene shaving' as a method for identifying
               distinct sets of genes with similar expression patterns", Genome
               Biol., 2000, doi: 10.1186/gb-2000-1-2-research0003.
    """

    def __init__(self, D: Dataset, seed: int | float | None = None) -> None:
        self.dataset = D
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, self._np_rs = _seeded_state(seed)
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        n: int,
        alpha: float = 0.1,
        choices: t.Iterable[str] | None = None,
        min_cluster_size: int = 2,
        min_gap_threshold: float = 0.0,
        n_random_iters: int = 10,
    ) -> list[str]:
        """Selects drugs using Drug Shaving.

        Parameters
        ----------
            n: The number of drugs to select.
            alpha: Proporation of drugs to shave at each iteration.
            choices: The drugs to choose from.
            min_cluster_size: Minimum cluster size for the shaving algorithm.

        Returns
        -------
            A list of selected drugs.
        """
        M = self.response_mat
        if choices is not None:
            M = M[M.index.isin(choices)]

        X = M.T
        X -= np.mean(X, axis=0)
        X_t = X.T

        selected_drugs = []
        for _ in range(n):
            cluster_drugs = self._get_cluster_drugs(
                X_t, alpha, min_cluster_size, min_gap_threshold, n_random_iters
            )

            if cluster_drugs is None:
                # restart with a fresh matrix minus the selected drugs
                X = M[~M.index.isin(selected_drug)].T
                X -= np.mean(X, axis=0)
                X_t = X.T

                cluster_drugs = self._get_cluster_drugs(
                    X_t,
                    alpha,
                    min_cluster_size,
                    min_gap_threshold,
                    n_random_iters,
                )

            # FIXME: select a drug first and then orthogonalize w.r.t. the
            #   selected drug (need to fix orthogonalize to take the mean)
            X_clust = X_t.loc[cluster_drugs]
            X_t = self._orthogonalize(X_t, X_clust)

            selected_drug = self._sample_from_cluster(X_clust)
            selected_drugs.append(selected_drug)

            X_t = X_t[X_t.index != selected_drug]

        return selected_drugs

    def _get_cluster_drugs(
        self,
        S: pd.DataFrame,
        alpha: float,
        min_cluster_size: int,
        min_gap_threshold: float,
        n_random_iters: int,
    ) -> list[str] | None:
        """Performs a single iteration of Drug Shaving."""
        pca = PCA(1, random_state=self._np_rs)

        # FIXME: this initialization is a hack to make sure we don't run out of drugs
        current_drugs = None
        current_gap = min_gap_threshold

        while S.shape[0] > min_cluster_size:
            _ = pca.fit(S)
            inner_prods = np.inner(pca.components_[0], S)

            idx_array = np.argsort(np.abs(inner_prods))
            clust_idx = idx_array[int(np.ceil(alpha * len(idx_array))) :]

            S = S.iloc[clust_idx]

            clust_r2 = self._compute_explained_variance(S)
            perm_r2s = []
            for _ in range(n_random_iters):
                S_perm = self._rng.permuted(S, axis=1)
                perm_r2 = self._compute_explained_variance(S_perm)
                perm_r2s.append(perm_r2)

            gap = clust_r2 - np.mean(perm_r2s)

            if gap > current_gap:
                current_drugs = list(S.index)
                current_gap = gap

        return current_drugs

    @staticmethod
    def _sample_from_cluster(X_clust: pd.DataFrame) -> str:
        """Samples a single drug from the cluster of drugs."""
        X_mean = np.mean(X_clust, axis=0)
        dists = scd.cdist(X_clust.values, X_mean.values[None, :])
        return X_clust.index[np.argmin(dists)]

    @staticmethod
    def _compute_explained_variance(X: pd.DataFrame) -> float:
        """Computes the explained variance of a cluster.

        Parameters
        ----------
            X: Array-like of shape (n_drugs, n_samples)

        Returns
        -------
            The explained variance of the cluster in the range of (0, 1).
        """
        V_w = np.mean(np.var(X, axis=0))
        V_b = np.var(np.mean(X, axis=0))
        return (V_b / V_w) / (1 + V_b / V_w)

    @staticmethod
    def _orthogonalize(X: pd.DataFrame, X_clust: pd.DataFrame) -> pd.DataFrame:
        """Orthogonalizes X with respect to the cluster mean."""
        X_clust_mean = np.mean(X_clust, axis=0)

        def func(Q: npt.ArrayLike[float]) -> np.ndarray:
            x = np.dot(Q, X_clust_mean)
            x /= np.dot(X_clust_mean, X_clust_mean)
            return Q - x * X_clust_mean

        return pd.DataFrame(
            np.apply_along_axis(func, 1, X), index=X.index, columns=X.columns
        )
