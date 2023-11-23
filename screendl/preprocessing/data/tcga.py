"""Preprocessing utilities for TCGA data."""


from __future__ import annotations

import pandas as pd
import pandas._typing as pdt
import typing as t

from dataclasses import dataclass

from ..utils import filter_by_value_counts


FilePathOrBuff = t.Union[pdt.FilePath, pdt.ReadCsvBuffer[bytes], pdt.ReadCsvBuffer[str]]


@dataclass
class TCGAData:
    """Container for TCGA data sources."""

    resp: pd.DataFrame
    cell_meta: pd.DataFrame
    exp: pd.DataFrame
    drug_meta: pd.DataFrame | None = None


def load_tcga_data(
    exp_path: FilePathOrBuff,
    resp_path: FilePathOrBuff,
    meta_path: FilePathOrBuff,
) -> TCGAData:
    """Loads the raw TCGA data."""
    resp_data = pd.read_csv(resp_path)
    cell_meta = pd.read_csv(meta_path)
    exp_data = pd.read_csv(exp_path, index_col=0)
    return TCGAData(resp_data, cell_meta, exp_data)


def clean_tcga_data(
    data: TCGAData,
    min_samples_per_drug: int | None = None,
) -> TCGAData:
    """Harmonizes the raw TCGA data."""
    common_samples = data.exp.index.intersection(data.resp["sample_id"])
    common_samples = common_samples.intersection(data.cell_meta["sample_id"])

    data.exp = data.exp.loc[common_samples].sort_index()
    data.resp = data.resp[data.resp["sample_id"].isin(common_samples)]
    data.cell_meta = data.cell_meta[data.cell_meta["sample_id"].isin(common_samples)]

    data.resp = data.resp.rename(columns={"sample_id": "model_id"})
    data.cell_meta = data.cell_meta.rename(columns={"sample_id": "model_id"})

    if min_samples_per_drug is not None:
        data.resp = filter_by_value_counts(data.resp, "drug_name", min_samples_per_drug)

    return data


def load_and_clean_tcga_data(
    exp_path: FilePathOrBuff,
    resp_path: FilePathOrBuff,
    meta_path: FilePathOrBuff,
    min_samples_per_drug: int | None = None,
) -> TCGAData:
    """Loads and cleans the raw TCGA data."""
    tcga_data = load_tcga_data(exp_path, resp_path, meta_path)
    tcga_data = clean_tcga_data(tcga_data, min_samples_per_drug)
    return tcga_data
