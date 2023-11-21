"""Preprocessing functionality for the Genomics of GDSC dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

from pathlib import Path
from cdrpy.core.utils import is_real_iterable


DRUG_INFO_COLUMN_MAPPER = {
    "Drug Id": "gdsc_drug_id",
    "Name": "drug_name",
    "Datasets": "dataset",
    "PubCHEM": "pubchem_id",
    "Targets": "targets",
    "Target pathway": "target_pathway",
}

DRUG_RESP_COLUMN_MAPPER = {
    "DATASET": "dataset",
    "DRUG_ID": "gdsc_drug_id",
    "DRUG_NAME": "drug_name",
    "SANGER_MODEL_ID": "model_id",
    "LN_IC50": "ln_ic50",
}

INVALID_PUBCHEM_IDS = ("several", "none", "None")
INVALID_RESPONSE_VALUES = (np.nan, np.inf, -np.inf)


def _fix_pubchem_id(id_: t.Any) -> str | None:
    """Fixes GDSC PubCHEM ids."""
    if not isinstance(id_, str) or id_ in INVALID_PUBCHEM_IDS:
        return None
    return id_.split(",")[0]


def load_gdsc_data(
    resp_path: str | Path | t.Iterable[str | Path], meta_path: str | Path
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the raw GDSCv2 data.

    Parameters
    ----------
        resp_path: Path (or a list of paths) to the raw drug response data.
        meta_path: Path to the raw drug annotations.

    Returns
    -------
        A tuple of (resp_df, meta_df) `pd.DataFrame` instances.
    """
    if not is_real_iterable(resp_path):
        resp_path = [resp_path]

    resp_df: pd.DataFrame = pd.concat(map(pd.read_excel, resp_path))
    meta_df = pd.read_csv(meta_path, dtype={"PubCHEM": str, "Drug Id": int})

    # cleanup column names
    resp_df = resp_df.rename(columns=DRUG_RESP_COLUMN_MAPPER)
    meta_df = meta_df.rename(columns=DRUG_INFO_COLUMN_MAPPER)

    return resp_df, meta_df


def harmonize_gdsc_data(
    resp_df: pd.DataFrame, meta_df: pd.DataFrame
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
    screened_drugs = set(resp_df["gdsc_drug_id"])

    meta_df["pubchem_id"] = meta_df["pubchem_id"].map(_fix_pubchem_id)
    meta_df = meta_df[meta_df["gdsc_drug_id"].isin(screened_drugs)]

    # remove duplicate entries (prefer GDSC2 if available)
    meta_df = (
        meta_df.dropna(subset="pubchem_id")
        .sort_values(["pubchem_id", "dataset", "number of cell lines"])
        .drop_duplicates("pubchem_id", keep="last")
        .sort_values(["drug_name", "dataset", "number of cell lines"])
        .drop_duplicates("drug_name", keep="last")
    )

    # filter the responses by the meta data
    resp_df = resp_df.merge(
        meta_df[["dataset", "gdsc_drug_id"]],
        on=["dataset", "gdsc_drug_id"],
        how="inner",
    )

    return resp_df, meta_df
