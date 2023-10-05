"""Utilities for generating train/validation/test splits."""

from __future__ import annotations

import pandas as pd
import typing as t

from sklearn.model_selection import train_test_split, StratifiedKFold


def kfold_tumor_blind_split(
    ids: pd.Series,
    groups: pd.Series,
    n_splits: int = 10,
    random_state: t.Any | None = None,
) -> t.Generator[tuple[dict[str, list[str]]], None, None]:
    """"""
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for train_idx, test_idx in skf.split(ids, groups):
        train_ids, test_ids = ids.iloc[train_idx], ids.iloc[test_idx]
        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.11,
            stratify=groups.iloc[train_idx],
            random_state=random_state,
        )
        yield {
            "train": list(train_ids),
            "val": list(val_ids),
            "test": list(test_ids),
        }
