"""Preprocessing utilities for HCI data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas._typing as pdt
import typing as t

from dataclasses import dataclass
from pathlib import Path

from ..utils import intersect_columns, filter_by_value_counts

StrOrPath = t.Union[str, Path]
FilePathOrBuff = t.Union[pdt.FilePath, pdt.ReadCsvBuffer[bytes], pdt.ReadCsvBuffer[str]]


DEFAULT_MODEL_TYPES = ["pdx"]


@dataclass(repr=False)
class HCIData:
    """Container for HCI data sources."""

    resp: pd.DataFrame
    cell_meta: pd.DataFrame
    drug_meta: pd.DataFrame
    exp: pd.DataFrame
    mut: pd.DataFrame | None = None

    # TODO: add save method
    # TODO: create a DEFAULT_PROCESSED_FILE_NAMES dict like:
    #   {
    #       "meta_samples": "MetaSampleAnnotations.csv",
    #       "meta_drugs": "MetaDrugAnnotations.csv",
    #       "omics_exp": "OmicsGeneExpression.csv",
    #       ...
    #   }
    #   -> I can then use this dict to grab the file names


def load_hci_data(
    exp_path: FilePathOrBuff,
    resp_path: FilePathOrBuff,
    pdmc_meta_path: FilePathOrBuff,
    drug_meta_path: FilePathOrBuff,
    mut_path: FilePathOrBuff | None = None,
) -> HCIData:
    """Loads the raw HCI PDMC data."""
    cell_meta = pd.read_csv(pdmc_meta_path)
    drug_meta = pd.read_csv(drug_meta_path, dtype={"pubchem_id": str})

    resp_data = pd.read_csv(resp_path)

    exp_data = pd.read_csv(exp_path, index_col=0)
    mut_data = None if mut_path is None else pd.read_csv(mut_path)

    return HCIData(resp_data, cell_meta, drug_meta, exp_data, mut_data)


def clean_hci_data(
    data: HCIData,
    model_types: t.List[str] | None = None,
    min_samples_per_drug: int | None = None,
    gr_metric: str = "IC50",
    log_transform: bool = True,
) -> HCIData:
    """Cleans and harmonizes the raw HCI data."""
    resp = data.resp
    exp = data.exp
    mut = data.mut
    drug_meta = data.drug_meta
    cell_meta = data.cell_meta

    if model_types is None:
        model_types = DEFAULT_MODEL_TYPES

    # only include drugs with PubCHEM ids
    drug_meta = drug_meta.dropna(subset="pubchem_id").drop_duplicates()

    resp_labels = resp[gr_metric]
    if log_transform:
        resp_labels = np.log(resp_labels)
    resp["label"] = resp_labels

    resp = resp[resp["drug_name"].isin(drug_meta["drug_name"])]

    if mut is not None:
        cell_meta = cell_meta[cell_meta["has_matching_wes"] == True]

    # only consider specified model types (e.g. PDX vs PDO)
    cell_meta = cell_meta[cell_meta["model_type"].isin(model_types)].copy()

    # remove duplicates, choosing the prefered model type
    if len(model_types) > 1:
        cell_meta["model_type"] = pd.Categorical(
            cell_meta["model_type"], model_types, ordered=True
        )
        cell_meta = cell_meta.sort_values(["model_id", "model_type"])
        cell_meta = cell_meta.drop_duplicates("model_id")

    common_samples = intersect_columns(resp, cell_meta, "model_id")
    common_samples = sorted(list(common_samples))

    cell_meta = cell_meta[cell_meta["model_id"].isin(common_samples)]
    cell_meta = cell_meta.drop_duplicates("model_id")

    mapper = dict(zip(cell_meta["sample_id_rna"], cell_meta["model_id"]))
    exp = exp[exp.index.isin(mapper)]
    exp.index = exp.index.map(mapper)

    if mut is not None:
        mapper = dict(zip(cell_meta["sample_id_wes"], cell_meta["model_id"]))
        mut["model_id"] = mut["sample_barcode"].map(mapper)
        mut = mut[mut["sample_barcode"].isin(mapper)].drop(columns="sample_barcode")

    resp = resp[resp["model_id"].isin(common_samples)]
    resp = resp.drop_duplicates(["model_id", "drug_name"])

    if min_samples_per_drug is not None:
        resp = filter_by_value_counts(resp, "drug_name", min_samples_per_drug)

    drug_meta = drug_meta[drug_meta["drug_name"].isin(resp["drug_name"])]

    return HCIData(resp, cell_meta, drug_meta, exp, mut)


def load_and_clean_hci_data(
    exp_path: FilePathOrBuff,
    resp_path: FilePathOrBuff,
    pdmc_meta_path: FilePathOrBuff,
    drug_meta_path: FilePathOrBuff,
    mut_path: FilePathOrBuff | None = None,
    model_types: t.List[str] | None = None,
    min_samples_per_drug: int | None = None,
    gr_metric: str = "IC50",
    log_transform: bool = True,
) -> HCIData:
    """Loads and cleans the raw HCI data."""
    hci_data = load_hci_data(
        exp_path, resp_path, pdmc_meta_path, drug_meta_path, mut_path
    )
    hci_data = clean_hci_data(
        hci_data,
        model_types,
        min_samples_per_drug,
        gr_metric=gr_metric,
        log_transform=log_transform,
    )
    return hci_data
