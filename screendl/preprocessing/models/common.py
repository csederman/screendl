"""Core feature extraction utilities."""

from __future__ import annotations

import pandas as pd


def compute_copy_number_ratios(
    cnv_df: pd.DataFrame, cell_info_df: pd.DataFrame
) -> pd.DataFrame:
    """Computes copy number ratios from absolute copy number and ploidy."""
    cell_to_ploidy = dict(
        zip(cell_info_df["model_id"], cell_info_df["ploidy_wes"])
    )
    ploidies = cnv_df.index.map((cell_to_ploidy))
    return cnv_df.apply(lambda c: c / ploidies)
