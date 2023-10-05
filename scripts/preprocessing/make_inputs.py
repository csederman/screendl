#!/usr/bin/env python
""""""

from __future__ import annotations

import click
import functools

import pandas as pd

from utils.models import (
    deepcdr_command,
    dualgcn_command,
    hidra_command,
    screendl_command,
)


class LazyDataStore:
    """Container for holding input data."""

    def __init__(
        self,
        cell_exp_path: str,
        cell_cnv_path: str,
        cell_mut_path: str,
        cell_info_path: str,
        drug_info_path: str,
        drug_resp_path: str,
    ) -> None:
        self.cell_exp_path = cell_exp_path
        self.cell_cnv_path = cell_cnv_path
        self.cell_mut_path = cell_mut_path
        self.cell_info_path = cell_info_path
        self.drug_info_path = drug_info_path
        self.drug_resp_path = drug_resp_path

    @functools.cached_property
    def cell_exp(self) -> pd.DataFrame:
        return pd.read_csv(self.cell_exp_path, index_col=0)

    @functools.cached_property
    def cell_cnv(self) -> pd.DataFrame:
        return pd.read_csv(self.cell_cnv_path, index_col=0)

    @functools.cached_property
    def cell_mut(self) -> pd.DataFrame:
        return pd.read_csv(self.cell_mut_path)

    @functools.cached_property
    def cell_info(self) -> pd.DataFrame:
        return pd.read_csv(self.cell_info_path)

    @functools.cached_property
    def drug_info(self) -> pd.DataFrame:
        return pd.read_csv(self.drug_info_path)

    @functools.cached_property
    def drug_resp(self) -> pd.DataFrame:
        return pd.read_csv(self.drug_resp_path)


@click.group()
@click.option(
    "--exp-path",
    type=str,
    required=True,
    help="Path to expression .csv file.",
)
@click.option(
    "--cnv-path",
    type=str,
    required=True,
    help="Path to copy number .csv file.",
)
@click.option(
    "--mut-path",
    type=str,
    required=True,
    help="Path to mutation .csv file.",
)
@click.option(
    "--cell-info-path",
    type=str,
    required=True,
    help="Path to cell annotations .csv file.",
)
@click.option(
    "--drug-info-path",
    type=str,
    required=True,
    help="Path to drug annotations .csv file.",
)
@click.option(
    "--drug-resp-path",
    type=str,
    required=True,
    help="Path to drug response .csv file.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    exp_path: str,
    cnv_path: str,
    mut_path: str,
    cell_info_path: str,
    drug_info_path: str,
    drug_resp_path: str,
) -> None:
    """"""
    ctx.obj = LazyDataStore(
        exp_path,
        cnv_path,
        mut_path,
        cell_info_path,
        drug_info_path,
        drug_resp_path,
    )


cli.add_command(deepcdr_command, "deepcdr")
cli.add_command(dualgcn_command, "dualgcn")
cli.add_command(hidra_command, "hidra")
cli.add_command(screendl_command, "screendl")


if __name__ == "__main__":
    cli()
