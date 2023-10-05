#!/usr/bin/env python
"""Generates target labels."""

from __future__ import annotations

import click

import pandas as pd
import numpy as np

from pathlib import Path


def make_labels(
    resp_df: pd.DataFrame, cell_id_col: str, drug_id_col: str, label_col: str
) -> pd.DataFrame:
    """Creates the drug response labels.

    Parameters
    ----------
        resp_df:
        cell_id_col:
        drug_id_col:
        label_col:

    Returns
    -------

    """
    required_cols = [cell_id_col, drug_id_col, label_col]
    assert all(x in resp_df.columns for x in required_cols)

    nan_values = [np.inf, -np.inf, np.nan]
    resp_df = resp_df[~resp_df[label_col].isin(nan_values)]
    resp_df = resp_df.dropna(subset=label_col)

    resp_df = resp_df.reset_index(drop=True)
    resp_df["id"] = resp_df.index

    return resp_df[["id", *required_cols]].rename(
        columns={
            cell_id_col: "cell_id",
            drug_id_col: "drug_id",
            label_col: "label",
        }
    )


@click.command()
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory where outputs should be saved.",
)
@click.option(
    "--drug-resp-path",
    type=str,
    required=True,
    help="Path to drug response .csv file.",
)
def cli(output_dir: str, drug_resp_path: str) -> None:
    """Creates the drug response labels."""
    output_dir: Path = Path(output_dir)

    resp_df = pd.read_csv(drug_resp_path)

    label_df = make_labels(
        resp_df,
        cell_id_col="SANGER_MODEL_ID",
        drug_id_col="DRUG_NAME",
        label_col="LN_IC50",
    )

    label_df.to_csv(output_dir / "LabelsLogIC50.csv", index=False)


if __name__ == "__main__":
    cli()
