"""Preprocessing functionality for the Genomics of GDSC dataset."""

from __future__ import annotations

import io
import requests

import numpy as np
import pandas as pd
import pandas._typing as pdt
import typing as t

from dataclasses import dataclass
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


FilePathOrBuff = t.Union[pdt.FilePath, pdt.ReadCsvBuffer[bytes], pdt.ReadCsvBuffer[str]]


@dataclass(repr=False)
class GDSCData:
    """Container for GDSC data sources."""

    resp: pd.DataFrame
    meta: pd.DataFrame


def _fix_pubchem_id(id_: t.Any) -> str | None:
    """Fix GDSC PubCHEM identifiers."""
    if not isinstance(id_, str) or id_ in INVALID_PUBCHEM_IDS:
        return None
    return id_.split(",")[0]


def fetch_gdsc_drug_info() -> io.StringIO:
    """Downloads GDSC drug annotations."""
    url = "https://www.cancerrxgene.org/api/compounds"
    with requests.Session() as s:
        response = s.get(url, params={"list": "all", "export": "csv"})
    return io.StringIO(response.content.decode())


def load_gdsc_data(
    resp_path: FilePathOrBuff | t.Iterable[FilePathOrBuff],
    meta_path: FilePathOrBuff,
) -> GDSCData:
    """Loads the raw GDSCv2 data.

    Parameters
    ----------
        resp_path: Path (or an iterable of paths) to the raw drug response data.
        meta_path: Path to the raw drug annotations.

    Returns
    -------
        A GDSCData object.
    """
    if not is_real_iterable(resp_path):
        resp_path = [resp_path]

    resp_data: pd.DataFrame = pd.concat(map(pd.read_excel, resp_path))
    meta_data = pd.read_csv(meta_path, dtype={"PubCHEM": str, "Drug Id": int})

    # cleanup column names
    meta_data.columns = [c.strip() for c in meta_data.columns]
    meta_data = meta_data.rename(columns=DRUG_INFO_COLUMN_MAPPER)
    resp_data = resp_data.rename(columns=DRUG_RESP_COLUMN_MAPPER)

    return GDSCData(resp_data, meta_data)


def clean_gdsc_data(data: GDSCData) -> GDSCData:
    """Harmonizes the GDSCv2 data.

    Parameters
    ----------
        gdsc_data: The raw GDSCData object.

    Returns
    -------
        The harmonized GDSCData object.
    """
    screened_drugs = set(data.resp["gdsc_drug_id"])

    data.meta["pubchem_id"] = data.meta["pubchem_id"].map(_fix_pubchem_id)
    data.meta = data.meta[data.meta["gdsc_drug_id"].isin(screened_drugs)]

    # remove duplicate entries (prefer GDSC2 if available)
    data.meta = (
        data.meta.dropna(subset="pubchem_id")
        .sort_values(["pubchem_id", "dataset", "number of cell lines"])
        .drop_duplicates("pubchem_id", keep="last")
        .sort_values(["drug_name", "dataset", "number of cell lines"])
        .drop_duplicates("drug_name", keep="last")
    )

    # filter the responses by the meta data
    data.resp = data.resp.merge(
        data.meta[["dataset", "gdsc_drug_id"]],
        on=["dataset", "gdsc_drug_id"],
        how="inner",
    )

    return data


def load_and_clean_gdsc_data(
    resp_path: FilePathOrBuff | t.Iterable[FilePathOrBuff],
    meta_path: FilePathOrBuff,
) -> GDSCData:
    """Loads and cleans the GDSC data.

    Parameters
    ----------
        resp_path: Path (or an iterable of paths) to the raw drug response data.
        meta_path: Path to the raw drug annotations.

    Returns
    -------
        The cleaned GDSCData object.
    """
    gdsc_data = load_gdsc_data(resp_path, meta_path)
    gdsc_data = clean_gdsc_data(gdsc_data)
    return gdsc_data
