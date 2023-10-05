"""Utilities for dataset preprocessing and harmonization."""

from __future__ import annotations

import pandas as pd

from .cmp import load_cmp_data, harmonize_cmp_data
from .depmap import load_oncotree_annotations
from .gdsc import load_gdsc_data, harmonize_gdsc_data
from .pubchem import fetch_pubchem_properties


__all__ = [
    "load_cmp_data",
    "harmonize_cmp_data",
    "load_gdsc_data",
    "harmonize_gdsc_data",
    "harmonize_gdsc_and_cmp_data",
    "fetch_pubchem_properties",
    "load_oncotree_annotations",
]


def harmonize_gdsc_and_cmp_data(
    exp_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    mut_df_pos: pd.DataFrame,
    cell_info_df: pd.DataFrame,
    resp_df: pd.DataFrame,
    drug_info_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Harmonizes the GDSCv2 and Cell Model Passports dataset."""
    cmp_cells = set(cell_info_df["model_id"])
    gdsc_cells = set(resp_df["SANGER_MODEL_ID"])

    common_cells = sorted(list(set.intersection(cmp_cells, gdsc_cells)))

    # filter the Cell Model Passports data
    exp_df = exp_df.loc[common_cells]
    cnv_df = cnv_df.loc[common_cells]
    mut_df = mut_df[mut_df["model_id"].isin(common_cells)]
    mut_df_pos = mut_df_pos[mut_df_pos["model_id"].isin(common_cells)]
    cell_info_df = cell_info_df[cell_info_df["model_id"].isin(common_cells)]

    # filter the GDSC data
    resp_df = resp_df[resp_df["SANGER_MODEL_ID"].isin(common_cells)]

    return (
        exp_df,
        cnv_df,
        mut_df,
        mut_df_pos,
        cell_info_df,
        resp_df,
        drug_info_df,
    )
