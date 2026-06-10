"""Data utilities."""

from __future__ import annotations

import itertools

import pandas as pd
import typing as t

from omegaconf import DictConfig, ListConfig

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


def get_id_prefix(pdmc_id: str) -> str:
    """Gets matching prefixes to avoid data leak."""
    if pdmc_id.startswith("HCI"):
        return pdmc_id[:6]
    elif pdmc_id.startswith("BCM"):
        return pdmc_id[:7]
    elif pdmc_id.startswith("TOW"):
        return pdmc_id[:5]
    else:
        return pdmc_id


def loo_split_generator(
    D: Dataset,
) -> t.Generator[t.Tuple[Dataset, Dataset], None, None]:
    """Generates leave-one-pdmc-out splits for model evaluation.

    Parameters
    ----------
    D : Dataset
        The PDMC dataset to split.

    Returns
    -------
    t.Generator[t.Tuple[Dataset, Dataset], None, None]
        Generator of (train, test) dataset tuples

    Yields
    ------
    Iterator[t.Generator[t.Tuple[Dataset, Dataset], None, None]]
        Iterator of (train, test) dataset tuples
    """
    pdmc_ids = set(D.cell_ids)
    for test_id in pdmc_ids:
        id_prefix = get_id_prefix(test_id)
        train_ids = [x for x in pdmc_ids if not str(x).startswith(id_prefix)]
        train_ds = D.select_cells(train_ids, name="train")
        test_ds = D.select_cells([test_id], name="test")
        yield train_ds, test_ds


def apply_pdmc_drug_filters(
    cfg: DictConfig, pdmc_ds: Dataset, cell_ds: Dataset | t.Iterable[Dataset]
) -> Dataset:
    """Filters drugs based on configured parameters."""
    if isinstance(cell_ds, Dataset):
        keep_drugs = set(cell_ds.drug_ids)
    elif isinstance(cell_ds, list):
        keep_drugs = set()
        for D in cell_ds:
            keep_drugs = keep_drugs.union(D.drug_ids)

    if cfg.experiment.keep_pdmc_only_drugs is not None:
        if cfg.experiment.keep_pdmc_only_drugs == "all":
            keep_drugs = keep_drugs.union(pdmc_ds.drug_ids)
        elif isinstance(cfg.experiment.keep_pdmc_only_drugs, ListConfig):
            keep_drugs = keep_drugs.union(cfg.experiment.keep_pdmc_only_drugs)
        else:
            raise TypeError(
                "Unsupported parameter type (experiment.keep_pdmc_only_drugs)"
            )

    if cfg.experiment.min_pdmcs_per_drug is not None:
        pdmcs_per_drug = pdmc_ds.obs["drug_id"].value_counts()
        drugs_with_min_pdmcs = pdmcs_per_drug[
            pdmcs_per_drug >= cfg.experiment.min_pdmcs_per_drug
        ]
        keep_drugs = keep_drugs.intersection(drugs_with_min_pdmcs.index)

    return pdmc_ds.select_drugs(keep_drugs, name=pdmc_ds.name)


def copy_dataset(D: Dataset) -> Dataset:
    """Creates a deep copy of the Dataset."""
    obs = D.obs.copy(deep=True)

    cell_encoders = None
    if D.cell_encoders is not None:
        cell_encoders = {k: v.copy() for k, v in D.cell_encoders.items()}

    drug_encoders = None
    if D.drug_encoders is not None:
        drug_encoders = {k: v.copy() for k, v in D.drug_encoders.items()}

    cell_meta = None
    if D.cell_meta is not None:
        cell_meta = D.cell_meta.copy(deep=True)

    drug_meta = None
    if D.drug_meta is not None:
        drug_meta = D.drug_meta.copy(deep=True)

    return Dataset(
        obs,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
        cell_meta=cell_meta,
        drug_meta=drug_meta,
        transforms=None,
        encode_drugs_first=D.encode_drugs_first,
        name=D.name,
        desc=D.desc,
    )
