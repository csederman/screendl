"""Data utilities."""

from __future__ import annotations

import itertools

import pandas as pd
import typing as t

from cdrpy.datasets import Dataset


def expand_dataset(D: Dataset, cell_ids: t.List[str], drug_ids: t.List[str]) -> Dataset:
    """Expands the dataset to include all combinations of cell and drug IDs."""
    pairs = list(itertools.product(cell_ids, drug_ids))
    obs = pd.DataFrame(pairs, columns=["cell_id", "drug_id"])
    obs["id"] = range(len(obs))
    # obs["label"] = 1  # dummy label

    obs = obs.merge(
        D.obs[["cell_id", "drug_id", "label"]],
        on=["cell_id", "drug_id"],
        how="left",
    )

    return Dataset(
        obs[["id", "cell_id", "drug_id", "label"]],
        cell_encoders=D.cell_encoders,
        drug_encoders=D.drug_encoders,
        cell_meta=D.cell_meta,
        drug_meta=D.drug_meta,
        name="expanded_dataset",
    )
