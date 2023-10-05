""""""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

from abc import ABC, abstractmethod

from cdrpy.data.datasets import Dataset
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


__all__ = [
    "KMeansDrugSelector",
]


def get_response_matrix(D: Dataset, impute_nans: bool = True) -> pd.DataFrame:
    """Converts dataset observations into a drug response matrix."""
    M = D.obs.pivot(index="drug_id", columns="cell_id", values="label")
    if impute_nans:
        M[:] = KNNImputer(n_neighbors=3).fit_transform(M)
    return M


class DrugSelectorBase:
    """Base class for drug selectors."""

    @abstractmethod
    def select(
        self,
        n: int,
        choices: list[str] | None = None,
        random_state: t.Any = None,
    ) -> list[str]:
        """"""
        ...


class KMeansDrugSelector(DrugSelectorBase):
    """"""

    def __init__(self, D: Dataset) -> None:
        self.dataset = D
        self.response_mat = get_response_matrix(self.dataset)

    def select(
        self,
        n: int,
        choices: t.Iterable[str] | None = None,
        random_state: t.Any = None,
    ) -> list[str]:
        """Selects the specified number of drugs from the response matrix."""
        M = self.response_mat
        if choices is not None:
            M = M[M.index.isin(choices)]

        km = KMeans(n, n_init="auto", random_state=random_state).fit(M)

        selected_drugs = []
        for i, center in enumerate(km.cluster_centers_):
            M_c = M.iloc[np.argwhere(km.labels_ == i).reshape(-1)]
            dists = euclidean_distances(M_c, np.expand_dims(center, 0))

            idx_min = np.argmin(dists.reshape(-1))
            selected_drugs.append(M_c.index[idx_min])

        return selected_drugs
