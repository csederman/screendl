"""Preprocessing utilities for HCI data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

from pathlib import Path


def load_hci_data(
    exp_path: str | Path,
    resp_path: str | Path,
    pdmc_meta_path: str | Path,
    drug_meta_path: str | Path,
    mut_path: str | Path | None = None,
) -> t.Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None
]:
    """Loads the raw HCI PDMC data."""
    exp_df = pd.read_csv(exp_path, index_col=0)
    mut_df = None if mut_path is None else pd.read_csv(mut_path)

    resp_df = pd.read_csv(resp_path)

    pdmc_meta = pd.read_csv(pdmc_meta_path)
    drug_meta = pd.read_csv(drug_meta_path, dtype={"pubchem_id": str})

    return exp_df, resp_df, pdmc_meta, drug_meta, mut_df


def harmonize_hci_data(
    exp_df: pd.DataFrame,
    resp_df: pd.DataFrame,
    pdmc_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
    model_types: t.List[str],
    mut_df: pd.DataFrame | None = None,
    min_samples_per_drug: int | None = None,
) -> t.Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None
]:
    """Cleans and harmonizes the raw HCI data."""

    # only include drugs with PubCHEM ids
    drug_meta = drug_meta.dropna(subset="pubchem_id").drop_duplicates()

    resp_df["ln_ic50"] = np.log(resp_df["IC50"])
    resp_df = resp_df[resp_df["drug_name"].isin(drug_meta["drug_name"])]

    # only consider specified model types (e.g. PDX vs PDO)
    pdmc_meta = pdmc_meta[pdmc_meta["model_type"].isin(model_types)]

    if mut_df is not None:
        pdmc_meta = pdmc_meta[pdmc_meta["has_matching_wes"] == True]

    common_models = set(resp_df["model_id"])
    common_models = common_models.intersection(pdmc_meta["model_id"])
    common_models = sorted(list(common_models))

    pdmc_meta = pdmc_meta[pdmc_meta["model_id"].isin(common_models)]
    pdmc_meta = pdmc_meta.drop_duplicates(subset="model_id")

    mapper = dict(zip(pdmc_meta["sample_id_rna"], pdmc_meta["model_id"]))
    exp_df = exp_df[exp_df.index.isin(mapper)]
    exp_df.index = exp_df.index.map(mapper)

    if mut_df is not None:
        mapper = dict(zip(pdmc_meta["sample_id_wes"], pdmc_meta["model_id"]))
        mut_df["model_id"] = mut_df["sample_barcode"].map(mapper)
        mut_df = mut_df[mut_df["sample_barcode"].isin(mapper)]
        mut_df = mut_df.drop(columns="sample_barcode")

    resp_df = resp_df[resp_df["model_id"].isin(common_models)]
    resp_df = resp_df.drop_duplicates(subset=["model_id", "drug_name"])

    if min_samples_per_drug is not None:
        drug_counts = resp_df["drug_name"].value_counts()
        keep_drugs = drug_counts[drug_counts >= min_samples_per_drug].index
        resp_df = resp_df[resp_df["drug_name"].isin(keep_drugs)]

    drug_meta = drug_meta[drug_meta["drug_name"].isin(resp_df["drug_name"])]

    return exp_df, resp_df, pdmc_meta, drug_meta, mut_df
