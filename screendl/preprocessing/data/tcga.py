"""Preprocessing utilities for TCGA data."""


from __future__ import annotations

import pandas as pd

from pathlib import Path


def load_tcga_data(
    exp_path: str | Path,
    resp_path: str | Path,
    meta_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the raw TCGA data."""
    exp_df = pd.read_csv(exp_path, index_col=0)
    resp_df = pd.read_csv(resp_path)
    pt_meta = pd.read_csv(meta_path)

    return exp_df, resp_df, pt_meta


def harmonize_tcga_data(
    exp_df: pd.DataFrame,
    resp_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    min_samples_per_drug: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Harmonizes the raw TCGA data."""
    common_samples = exp_df.index.intersection(resp_df["sample_id"])
    common_samples = common_samples.intersection(meta_df["sample_id"])
    exp_df = exp_df.loc[common_samples].sort_index()
    resp_df = resp_df[resp_df["sample_id"].isin(common_samples)]
    meta_df = meta_df[meta_df["sample_id"].isin(common_samples)]

    resp_df = resp_df.rename(columns={"sample_id": "model_id"})
    meta_df = meta_df.rename(columns={"sample_id": "model_id"})

    if min_samples_per_drug is not None:
        drug_counts = resp_df["drug_name"].value_counts()
        keep_drugs = drug_counts[drug_counts >= min_samples_per_drug].index
        resp_df = resp_df[resp_df["drug_name"].isin(keep_drugs)]

    return exp_df, resp_df, meta_df
