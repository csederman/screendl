""""""

from __future__ import annotations

import itertools
import random

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.distance as scd
import typing as t

from abc import ABC, abstractmethod
from collections import defaultdict

from cdrpy.data.datasets import Dataset
from cdrpy.core.random import _seeded_state

from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_selection import SelectKBest


__all__ = [
    "RandomDrugSelector",
    "KMeansDrugSelector",
    "AgglomerativeDrugSelector",
    "PrincipalDrugSelector",
    "MeanResponseSelector",
    "UniformDrugSelector",
    "DrugSelectorType",
]


Seed = t.Union[int, float, None]


def get_response_matrix(D: Dataset, impute: bool = True) -> pd.DataFrame:
    """Converts dataset observations into a drug response matrix."""
    M = D.obs.pivot_table(index="drug_id", columns="cell_id", values="label")
    if impute:
        M[:] = KNNImputer(n_neighbors=3).fit_transform(M)
    return M


class DrugSelectorBase(ABC):
    """Base class for drug selectors."""

    @abstractmethod
    def select(
        self,
        n: int,
        choices: t.List[str] | None = None,
    ) -> t.List[str]:
        """"""
        ...


class RandomDrugSelector(DrugSelectorBase):
    """Random drug selection.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Optional seed for random number generation.
    """

    def __init__(self, D: Dataset, seed: Seed = None, name: str | None = None) -> None:
        self.dataset = D
        self.name = name
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, _ = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> t.List[str]:
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


class UniformDrugSelector(DrugSelectorBase):
    """Uniform drug selection along metadata fields."""

    def __init__(self, D: Dataset, seed: Seed = None, name: str | None = None) -> None:
        if D.drug_meta is None:
            raise ValueError("drug selector requires drug metadata")
        self.dataset = D
        self.name = name
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, _ = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
        field: str = "target_pathway",
    ) -> t.List[str]:
        """Uniformly sample drugs along metadata groupings."""
        if not field in self.dataset.drug_meta.columns:
            raise ValueError("field must be a valid metadata column")

        drug_meta = self.dataset.drug_meta.copy()
        if choices is not None:
            drug_meta = drug_meta[drug_meta.index.isin(choices)]

        drug_groups = (
            drug_meta.rename_axis(index="drug_id")
            .reset_index()
            .groupby(field)["drug_id"]
            .apply(list)
            .to_dict()
        )

        sorted_groups = sorted(
            drug_groups, key=lambda k: len(drug_groups[k]), reverse=True
        )

        selected_drugs = []
        for key in itertools.cycle(sorted_groups):
            drugs_for_group: list = drug_groups[key]
            if not drugs_for_group:
                # skip exhausted groups
                continue
            selected_drug = self._rs.choice(drugs_for_group)
            selected_drugs.append(selected_drug)
            drugs_for_group.remove(selected_drug)
            if len(selected_drugs) == n:
                break

        return selected_drugs


class MeanResponseSelector(DrugSelectorBase):
    """Selects the drugs most predictive of a cell line's mean response.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Ignored, exists for compatability.
    """

    def __init__(self, D: Dataset, seed: Seed = None, name: str | None = None) -> None:
        self.dataset = D
        self.name = name
        self.response_mat = get_response_matrix(self.dataset)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> t.List[str]:
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

    def __init__(self, D: Dataset, seed: Seed = None, name: str | None = None) -> None:
        self.dataset = D
        self.name = name
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, self._np_rs = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> t.List[str]:
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


class AgglomerativeDrugSelector(DrugSelectorBase):
    """Selects drugs using agglomerative clustering.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Optional seed for random number generation.
    """

    def __init__(self, D: Dataset, seed: Seed = None, name: str | None = None) -> None:
        self.dataset = D
        self.name = name
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, self._np_rs = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> t.List[str]:
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

        agg = AgglomerativeClustering(n).fit(M)

        selected_drugs = []
        for clust in np.unique(agg.labels_):
            M_clust = M.iloc[np.argwhere(agg.labels_ == clust).flatten()]
            center = np.expand_dims(M_clust.mean(axis=0), 0)
            dists = euclidean_distances(M_clust, center)

            idx_min = np.argmin(dists.reshape(-1))
            selected_drugs.append(M_clust.index[idx_min])

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

    def __init__(self, D: Dataset, seed: Seed = None, name: str | None = None) -> None:
        self.dataset = D
        self.name = name
        self.response_mat = get_response_matrix(self.dataset)
        self._rs, self._np_rs = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
    ) -> t.List[str]:
        """Selects drugs using Principle Feature Analysis.

        Parameters
        ----------
            n: The number of drugs to select.
            choices: The drugs to choose from.
            random_state: Optional random state for K-Means and PCA.

        Returns
        -------
            A list of selected drugs.
        """
        q = n - 1  # number of principal components

        M = self.response_mat
        if choices is not None:
            M = M[M.index.isin(choices)]

        X = M.T
        X_std = StandardScaler().fit_transform(X)

        svd = PCA(q, random_state=self._np_rs).fit(X_std)
        A = svd.components_.T

        kmeans = KMeans(n_clusters=n, random_state=self._np_rs, n_init="auto")
        _ = kmeans.fit(A)
        clusters = kmeans.predict(A)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]

        return list(M.index[indices_])

    # def select(
    #     self,
    #     n: int,
    #     q: int | None = None,
    #     choices: t.Iterable[str] | None = None,
    # ) -> t.List[str]:
    #     """Selects drugs using Principle Feature Analysis.

    #     Parameters
    #     ----------
    #         n: The number of drugs to select.
    #         q: The number of principal components (must be less than `n`).
    #         choices: The drugs to choose from.
    #         random_state: Optional random state for K-Means and PCA.

    #     Returns
    #     -------
    #         A list of selected drugs.
    #     """
    #     if q is None:
    #         q = n - 1
    #     elif n <= q:
    #         raise ValueError(f"`q` must be less than `n`.")

    #     M = self.response_mat
    #     if choices is not None:
    #         M = M[M.index.isin(choices)]

    #     X = M.T
    #     X_std = StandardScaler().fit_transform(X)
    #     corr_matrix = np.corrcoef(X_std, rowvar=False)

    #     pca = PCA(q, random_state=self._np_rs)
    #     pca.fit(corr_matrix)
    #     A = pca.components_.T

    #     # TODO: add min explained variance threshold before restart

    #     km = KMeans(n, n_init="auto", random_state=self._np_rs)
    #     _ = km.fit(A)

    #     D = scd.cdist(km.cluster_centers_, A)
    #     inds = [np.argmin(x) for x in D]

    #     return list(M.index[inds])


DrugSelectorType = t.Union[
    t.Type[RandomDrugSelector],
    t.Type[KMeansDrugSelector],
    t.Type[AgglomerativeDrugSelector],
    t.Type[PrincipalDrugSelector],
    t.Type[MeanResponseSelector],
    t.Type[UniformDrugSelector],
]


SELECTORS: t.Dict[str, DrugSelectorType] = {
    "agglomerative": AgglomerativeDrugSelector,
    "kmeans": KMeansDrugSelector,
    "principal": PrincipalDrugSelector,
    "random": RandomDrugSelector,
    "uniform": UniformDrugSelector,
}
