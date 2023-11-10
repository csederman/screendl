"""Utilities for generating train/validation/test splits."""

from __future__ import annotations

import pickle

import pandas as pd
import typing as t

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold


SplitType = t.Tuple[t.Dict[str, t.List[str]]]


def kfold_split_generator(
    ids: pd.Series,
    groups: pd.Series,
    n_splits: int = 10,
    random_state: t.Any | None = None,
    include_validation_set: bool = True,
) -> t.Generator[SplitType, None, None]:
    """Generates stratified k-fold splits for the specified ids and groups."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(ids, groups):
        train_ids, test_ids = ids.iloc[train_idx], ids.iloc[test_idx]

        val_ids = []
        if include_validation_set:
            train_ids, val_ids = train_test_split(
                train_ids,
                test_size=0.11,
                stratify=groups.iloc[train_idx],
                random_state=random_state,
            )
            val_ids = list(val_ids)

        train_ids = list(train_ids)
        test_ids = list(test_ids)

        yield {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }


def generate_mixed_splits(
    out_dir: Path,
    label_df: pd.DataFrame,
    n_splits: int = 10,
    seed: t.Any | None = None,
) -> None:
    """Generates mixed train/val/test splits stratified by `cell_id`.

    Parameters
    ----------
        out_dir: Path to the output directory.
        label_df: A `pd.DataFrame` instance containing the labels.
        n_splits: The number of splits (folds) to generate.
        random_state: Optional random seed.
    """
    out_dir.mkdir(exist_ok=True)

    ids = label_df["id"]
    groups = label_df["cell_id"]

    split_gen = kfold_split_generator(ids, groups, n_splits=n_splits, random_state=seed)

    for i, split_dict in enumerate(split_gen, 1):
        with open(out_dir / f"fold_{i}.pkl", "wb") as fh:
            pickle.dump(split_dict, fh)


def generate_tumor_blind_splits(
    out_dir: Path,
    label_df: pd.DataFrame,
    cell_meta: pd.DataFrame,
    n_splits: int = 10,
    seed: t.Any | None = None,
) -> None:
    """Generates tumor blind train/val/test splits.

    Parameters
    ----------
        out_dir: Path to the output directory.
        label_df: A `pd.DataFrame` instance containing the labels.
        cell_meta: A `pd.DataFrame` instance containing cell metadata.
        n_splits: The number of splits (folds) to generate.
        random_state: Optional random seed.
    """
    out_dir.mkdir(exist_ok=True)

    ids = cell_meta["model_id"]
    groups = cell_meta["cancer_type"]

    split_gen = kfold_split_generator(ids, groups, n_splits=n_splits, random_state=seed)

    for i, split in enumerate(split_gen, 1):
        train_cells = split["train"]
        val_cells = split["val"]
        test_cells = split["test"]

        train_ids = label_df[label_df["cell_id"].isin(train_cells)]["id"]
        val_ids = label_df[label_df["cell_id"].isin(val_cells)]["id"]
        test_ids = label_df[label_df["cell_id"].isin(test_cells)]["id"]

        split_dict = {
            "train": list(train_ids),
            "val": list(val_ids),
            "test": list(test_ids),
        }

        with open(out_dir / f"fold_{i}.pkl", "wb") as fh:
            pickle.dump(split_dict, fh)
