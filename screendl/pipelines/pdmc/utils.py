"""PDMC pipeline utilities."""

from __future__ import annotations

import logging
import typing as t

from cdrpy.datasets import Dataset

from screendl.screenahead import DrugSelectorType

log = logging.getLogger(__name__)


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
    pdmc_ids = sorted(list(set(D.cell_ids)))
    for test_id in pdmc_ids:
        if ":AUG" in test_id:
            continue
        id_prefix = get_id_prefix(test_id)
        train_ids = [x for x in pdmc_ids if not str(x).startswith(id_prefix)]
        train_ds = D.select_cells(train_ids, name="train")
        test_ds = D.select_cells([test_id], name="test")
        yield train_ds, test_ds


def get_screenahead_split(
    dataset: Dataset,
    drug_selector: DrugSelectorType,
    num_drugs: int,
    exclude_drugs: t.List[str] | None = None,
) -> t.Tuple[Dataset, Dataset]:
    """"""
    drug_choices = set(dataset.drug_ids)
    if exclude_drugs is not None:
        drug_choices = set(x for x in drug_choices if x not in exclude_drugs)
    screen_drugs = drug_selector.select(num_drugs, choices=drug_choices)
    holdout_drugs = drug_choices.difference(screen_drugs)

    screen_ds = dataset.select_drugs(screen_drugs, name="this_screen")
    holdout_ds = dataset.select_drugs(holdout_drugs, name="this_holdout")

    return screen_ds, holdout_ds


def get_screenahead_dataset(
    dataset: Dataset,
    mode: t.Literal["screen-all", "screen-selected", "screen-non-pdx"],
    drug_selector: DrugSelectorType,
    num_drugs: int,
    exclude_drugs: t.List[str] | None = None,
) -> Dataset:
    """"""
    if mode == "screen-all":
        screen_ds = dataset
    elif mode == "screen-selected":
        screen_ds, _ = get_screenahead_split(
            dataset, drug_selector=drug_selector, num_drugs=num_drugs
        )
    elif mode == "screen-non-pdx":
        screen_ds, _ = get_screenahead_split(
            dataset,
            drug_selector=drug_selector,
            num_drugs=num_drugs,
            exclude_drugs=exclude_drugs,
        )
    else:
        raise ValueError(f"Invalid screenahead mode (got {mode})")

    return screen_ds
