"""

"""

from __future__ import annotations

import pandas as pd

from pathlib import Path


def load_gdsc_data(
    resp_path: str | Path, drug_info_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the raw GDSCv2 data.

    Parameters
    ----------
        resp_path: Path to the raw drug response data.
        drug_info_path: Path to the raw drug annotations.

    Returns
    -------
        A tuple of (resp_df, drug_info_df) `pd.DataFrame` instances.
    """
    resp_df = pd.read_excel(resp_path)
    drug_info_df = pd.read_csv(drug_info_path)

    return resp_df, drug_info_df


def harmonize_gdsc_data(
    resp_df: pd.DataFrame,
    drug_info_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Harmonizes the GDSCv2 data.

    Parameters
    ----------
        resp_df: The raw drug response data.
        drug_info_df: The raw drug annotations.

    Returns
    -------
        A tuple of (resp_df, drug_info_df) `pd.DataFrame` instances.
    """
    screened_drugs = set(resp_df["DRUG_ID"])

    # only consider GDSCv2 drugs with screening data
    drug_info_df = drug_info_df[drug_info_df["Datasets"] == "GDSC2"]
    drug_info_df = drug_info_df[drug_info_df["Drug Id"].isin(screened_drugs)]

    # filter drugs without PubCHEM IDs and remove duplicates
    drug_info_df = drug_info_df.dropna(subset="PubCHEM")
    drug_info_df = drug_info_df.drop_duplicates(subset="Name")

    # check for valid PubCHEM ids
    invalid_pchem = ["several", "none", "None", None]
    drug_info_df = drug_info_df[~drug_info_df["PubCHEM"].isin(invalid_pchem)]

    # select the first PubCHEM id when there are multiple
    drug_info_df["PubCHEM"] = drug_info_df["PubCHEM"].map(
        lambda x: str(x).split(",")[0]
    )

    resp_df = resp_df[resp_df["DRUG_ID"].isin(drug_info_df["Drug Id"])]

    return resp_df, drug_info_df
