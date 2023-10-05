#!/usr/bin/env python
"""Generates train/validation/test splits for model evaluation."""

from __future__ import annotations

import click
import pickle

import pandas as pd
import typing as t

from pathlib import Path

from utils.splits import kfold_tumor_blind_split


def make_tumor_blind_folds(
    labels_df: pd.DataFrame,
    cell_info_df: pd.DataFrame,
    output_dir: str | Path,
    n_splits: int = 10,
    id_col: str = "model_id",
    group_col: str = "cancer_type",
    random_state: t.Any | None = None,
) -> None:
    """"""
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    split_generator = kfold_tumor_blind_split(
        ids=cell_info_df[id_col],
        groups=cell_info_df[group_col],
        n_splits=n_splits,
        random_state=random_state,
    )
    for i, split in enumerate(split_generator, 1):
        train_ids = labels_df[labels_df["cell_id"].isin(split["train"])]["id"]
        val_ids = labels_df[labels_df["cell_id"].isin(split["val"])]["id"]
        test_ids = labels_df[labels_df["cell_id"].isin(split["test"])]["id"]

        split_ids = {
            "train": list(train_ids),
            "val": list(val_ids),
            "test": list(test_ids),
        }

        with open(output_dir / f"fold_{i}.pkl", "wb") as fh:
            pickle.dump(split_ids, fh)


@click.command()
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory where outputs should be saved.",
)
@click.option(
    "--labels-path",
    type=str,
    required=True,
    help="Path to labels .csv file.",
)
@click.option(
    "--cell-info-path",
    type=str,
    required=True,
    help="Path to cell annotations .csv file.",
)
def cli(output_dir: str, labels_path: str, cell_info_path: str) -> None:
    """"""
    output_dir: Path = Path(output_dir) / "tumor_blind"

    labels_df = pd.read_csv(labels_path)
    cell_info_df = pd.read_csv(cell_info_path)

    make_tumor_blind_folds(
        labels_df, cell_info_df, output_dir, n_splits=10, random_state=1771
    )


if __name__ == "__main__":
    cli()
