""""""

from __future__ import annotations

import itertools
import warnings
import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from cdrpy.datasets import Dataset
from cdrpy.core.random import _seeded_state

__all__ = [
    "SELECTORS",
    "DrugSelectorType",
    "AgglomerativeDrugSelector",
    "KMeansDrugSelector",
    "PrincipalDrugSelector",
    "RandomDrugSelector",
    "UniformDrugSelector",
    "get_response_matrix",
]


Seed = t.Union[int, float, None]


def get_response_matrix(
    D: Dataset,
    impute: bool = True,
    na_threshold: float | None = None,
    n_neighbors: int = 3,
) -> pd.DataFrame:
    """Converts dataset observations into a drug response matrix."""
    M = D.obs.pivot_table(index="cell_id", columns="drug_id", values="label")

    # drop drugs not screened in at least a given faction of cell lines
    if na_threshold is not None:
        M = M.dropna(thresh=np.floor(na_threshold * M.shape[0]), axis=1)

    if impute:
        M.loc[:, :] = KNNImputer(
            n_neighbors=n_neighbors, weights="distance"
        ).fit_transform(M)

    return M.T


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

    def __init__(
        self,
        D: Dataset,
        na_threshold: float = 0.9,
        seed: Seed = None,
        name: str | None = None,
    ) -> None:
        self.D = D
        self.na_threshold = na_threshold
        self.name = name
        self.rmat = get_response_matrix(self.D, na_threshold=self.na_threshold)
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
        M = self.rmat
        if choices is not None:
            M = M[M.index.isin(choices)]

        # return all drugs if fewer than choices
        if M.shape[0] <= n:
            warnings.warn(f"Fewer than {n} drugs available for sample.")
            return list(M.index)

        return self._rs.sample(list(M.index), n)


class UniformDrugSelector(DrugSelectorBase):
    """Uniform drug selection along metadata fields."""

    def __init__(
        self,
        D: Dataset,
        na_threshold: float = 0.9,
        seed: Seed = None,
        name: str | None = None,
    ) -> None:
        if D.drug_meta is None:
            raise ValueError("drug selector requires drug metadata")
        self.D = D
        self.na_threshold = na_threshold
        self.name = name
        self.rmat = get_response_matrix(self.D, na_threshold=self.na_threshold)
        self._rs, _ = _seeded_state(seed)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
        field: str = "target_pathway",
    ) -> t.List[str]:
        """Uniformly sample drugs along metadata groupings."""
        if not field in self.D.drug_meta.columns:
            raise ValueError("field must be a valid metadata column")

        drug_meta = self.D.drug_meta.copy()
        if choices is not None:
            drug_meta = drug_meta[drug_meta.index.isin(choices)]

        # return all drugs if fewer than choices
        if drug_meta.shape[0] <= n:
            warnings.warn(f"Fewer than {n} drugs available for sample.")
            return list(drug_meta.index)

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


class KMeansDrugSelector(DrugSelectorBase):
    """Selects drugs using K-Means clustering.

    Parameters
    ----------
        D: The `Dataset` object to select from.
        seed: Optional seed for random number generation.
    """

    def __init__(
        self,
        D: Dataset,
        na_threshold: float = 0.9,
        seed: Seed = None,
        name: str | None = None,
    ) -> None:
        self.D = D
        self.na_threshold = na_threshold
        self.name = name
        self.rmat = get_response_matrix(self.D, na_threshold=self.na_threshold)
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
        M = self.rmat
        if choices is not None:
            M = M[M.index.isin(choices)]

        # return all drugs if fewer than choices
        if M.shape[0] <= n:
            warnings.warn(f"Fewer than {n} drugs available for sample.")
            return list(M.index)

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

    def __init__(
        self,
        D: Dataset,
        na_threshold: float = 0.9,
        seed: Seed = None,
        name: str | None = None,
    ) -> None:
        self.D = D
        self.na_threshold = na_threshold
        self.name = name
        self.rmat = get_response_matrix(self.D, na_threshold=self.na_threshold)
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
        M = self.rmat
        if choices is not None:
            M = M[M.index.isin(choices)]

        # return all drugs if fewer than choices
        if M.shape[0] <= n:
            warnings.warn(f"Fewer than {n} drugs available for sample.")
            return list(M.index)

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

    def __init__(
        self,
        D: Dataset,
        na_threshold: float = 0.9,
        seed: Seed = None,
        name: str | None = None,
    ) -> None:
        self.D = D
        self.na_threshold = na_threshold
        self.name = name
        self.rmat = get_response_matrix(self.D, na_threshold=self.na_threshold)
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

        M = self.rmat
        if choices is not None:
            M = M[M.index.isin(choices)]

        # return all drugs if fewer than choices
        if M.shape[0] <= n:
            warnings.warn(f"Fewer than {n} drugs available for sample.")
            return list(M.index)

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


DrugSelectorType = t.Union[
    t.Type[AgglomerativeDrugSelector],
    t.Type[KMeansDrugSelector],
    t.Type[PrincipalDrugSelector],
    t.Type[RandomDrugSelector],
    t.Type[UniformDrugSelector],
]


SELECTORS: t.Dict[str, DrugSelectorType] = {
    "agglomerative": AgglomerativeDrugSelector,
    "kmeans": KMeansDrugSelector,
    "principal": PrincipalDrugSelector,
    "random": RandomDrugSelector,
    "uniform": UniformDrugSelector,
}
