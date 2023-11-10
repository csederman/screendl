"""Preprocessing functionality for the Genomics of GDSC dataset."""

from __future__ import annotations

import pandas as pd
import typing as t

from pathlib import Path


def load_gdsc_data(
    resp_path: str | Path, meta_path: str | Path
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the raw GDSCv2 data.

    Parameters
    ----------
        resp_path: Path to the raw drug response data.
        meta_path: Path to the raw drug annotations.

    Returns
    -------
        A tuple of (resp_df, meta_df) `pd.DataFrame` instances.
    """
    resp_df = pd.read_excel(resp_path)
    meta_df = pd.read_csv(meta_path)

    return resp_df, meta_df


def harmonize_gdsc_data(
    resp_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Harmonizes the GDSCv2 data.

    Parameters
    ----------
        resp_df: The raw drug response data.
        meta_df: The raw drug annotations.

    Returns
    -------
        A tuple of (resp_df, meta_df) `pd.DataFrame` instances.
    """
    screened_drugs = set(resp_df["DRUG_ID"])

    # only consider GDSCv2 drugs with screening data
    meta_df = meta_df[meta_df["Datasets"] == "GDSC2"]
    meta_df = meta_df[meta_df["Drug Id"].isin(screened_drugs)]

    # filter drugs without PubCHEM IDs and remove duplicates
    meta_df = meta_df.dropna(subset="PubCHEM")
    meta_df = meta_df.drop_duplicates(subset="Name")

    # check for valid PubCHEM ids
    invalid_pchem = ["several", "none", "None", None]
    meta_df = meta_df[~meta_df["PubCHEM"].isin(invalid_pchem)]

    # select the first PubCHEM id when there are multiple
    meta_df["PubCHEM"] = meta_df["PubCHEM"].map(lambda x: str(x).split(",")[0])
    resp_df = resp_df[resp_df["DRUG_ID"].isin(meta_df["Drug Id"])]

    # cleanup column names for downstream harmonization
    meta_col_mapper = {
        "Name": "drug_name",
        "PubCHEM": "pubchem_id",
        "Targets": "targets",
        "Target pathway": "target_pathway",
    }
    meta_df = meta_df.rename(columns=meta_col_mapper)

    resp_col_mapper = {
        "DRUG_NAME": "drug_name",
        "SANGER_MODEL_ID": "model_id",
        "LN_IC50": "ln_ic50",
    }
    resp_df = resp_df.rename(columns=resp_col_mapper)

    return resp_df, meta_df
