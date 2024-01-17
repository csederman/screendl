"""Utilities for dataset preprocessing and harmonization."""

from __future__ import annotations

import typing as t

from .cmp import CMPData
from .cmp import load_cmp_data
from .cmp import clean_cmp_data
from .cmp import load_and_clean_cmp_data
from .gdsc import GDSCData
from .gdsc import load_gdsc_data
from .gdsc import clean_gdsc_data
from .gdsc import load_and_clean_gdsc_data
from .hci import HCIData
from .hci import load_hci_data
from .hci import clean_hci_data
from .hci import load_and_clean_hci_data
from .tcga import TCGAData
from .tcga import load_tcga_data
from .tcga import clean_tcga_data
from .tcga import load_and_clean_tcga_data
from .pubchem import fetch_pubchem_properties

from ..utils import intersect_columns


__all__ = [
    "CMPData",
    "GDSCData",
    "HCIData",
    "load_cmp_data",
    "clean_cmp_data",
    "load_and_clean_cmp_data",
    "load_gdsc_data",
    "clean_gdsc_data",
    "load_and_clean_gdsc_data",
    "load_hci_data",
    "clean_hci_data",
    "load_and_clean_hci_data",
    "load_tcga_data",
    "clean_tcga_data",
    "load_and_clean_tcga_data",
    "harmonize_cmp_gdsc_data",
    "harmonize_cmp_gdsc_hci_data",
    "harmonize_cmp_gdsc_tcga_data",
    "fetch_pubchem_properties",
]


def harmonize_cmp_gdsc_data(
    cmp_data: CMPData, gdsc_data: GDSCData
) -> t.Tuple[CMPData, GDSCData]:
    """Harmonizes the GDSC and Cell Model Passports data.

    Parameters
    ----------
        cmp_data: A harmonized CMPData object.
        gdsc_data: The harmonized GDSCData object.

    Returns
    -------
        A tuple (cmp_data, gdsc_data) of harmonized data objects.
    """
    common_samples = intersect_columns(cmp_data.meta, gdsc_data.resp, "model_id")
    common_samples = sorted(list(common_samples))

    cmp_data.exp = cmp_data.exp.loc[common_samples]
    cmp_data.meta = cmp_data.meta[cmp_data.meta["model_id"].isin(common_samples)]

    if cmp_data.cnv is not None:
        cmp_data.cnv = cmp_data.cnv.loc[common_samples]

    if cmp_data.mut is not None:
        cmp_data.mut = cmp_data.mut[cmp_data.mut["model_id"].isin(common_samples)]

    gdsc_data.resp = gdsc_data.resp[gdsc_data.resp["model_id"].isin(common_samples)]

    return cmp_data, gdsc_data


def harmonize_cmp_gdsc_hci_data(
    cmp_data: CMPData,
    hci_data: HCIData,
    gdsc_data: GDSCData,
    include_all_hci_drugs: bool = False,
) -> t.Tuple[CMPData, HCIData, GDSCData]:
    """Harmonizes the Cell Model Passports, GDSC and HCI data."""

    # 1. harmonize the omics data
    cmp_data.meta["model_type"] = "cell line"
    cmp_data.meta["domain"] = "CELL"
    hci_data.cell_meta["domain"] = "PDMC"

    cols = ["model_id", "cancer_type", "model_type", "domain"]
    cmp_data.meta = cmp_data.meta[cols]
    hci_data.cell_meta = hci_data.cell_meta[cols]

    cmp_exp_genes = cmp_data.exp.columns
    hci_exp_genes = hci_data.exp.columns
    common_genes = cmp_exp_genes.intersection(hci_exp_genes).sort_values()
    cmp_data.exp = cmp_data.exp[common_genes]
    hci_data.exp = hci_data.exp[common_genes]

    if cmp_data.mut is not None and hci_data.mut is not None:
        cmp_mut_genes = set(cmp_data.mut["gene_symbol"])
        hci_mut_genes = set(hci_data.mut["gene_symbol"])
        common_genes = cmp_mut_genes.intersection(hci_mut_genes)

        cols = ["model_id", "gene_symbol", "chrom", "pos", "protein_mutation"]
        cmp_mut_data = cmp_data.mut[cmp_data.mut["gene_symbol"].isin(common_genes)]
        hci_mut_data = hci_data.mut[hci_data.mut["gene_symbol"].isin(common_genes)]
        cmp_data.mut = cmp_mut_data[cols]
        hci_data.mut = hci_mut_data[cols]

    # 2. harmonize the drug response data
    mapper = dict(zip(gdsc_data.meta["pubchem_id"], gdsc_data.meta["drug_name"]))
    func = lambda r: mapper.get(r["pubchem_id"], r["drug_name"])

    old_names = hci_data.drug_meta["drug_name"].copy()
    hci_data.drug_meta["drug_name"] = hci_data.drug_meta.apply(func, axis=1)

    mapper = dict(zip(old_names, hci_data.drug_meta["drug_name"]))
    hci_data.resp["drug_name"] = hci_data.resp["drug_name"].map(mapper)

    if not include_all_hci_drugs:
        hci_data.drug_meta = hci_data.drug_meta[
            hci_data.drug_meta["pubchem_id"].isin(gdsc_data.meta["pubchem_id"])
        ]
        hci_data.resp = hci_data.resp[
            hci_data.resp["drug_name"].isin(gdsc_data.meta["drug_name"])
        ]

    gdsc_data.resp = gdsc_data.resp[["model_id", "drug_name", "label"]]
    hci_data.resp = hci_data.resp[["model_id", "drug_name", "label"]]

    gdsc_data.meta = gdsc_data.meta[["drug_name", "pubchem_id"]]
    hci_data.drug_meta = hci_data.drug_meta[["drug_name", "pubchem_id"]]

    return cmp_data, hci_data, gdsc_data


def harmonize_cmp_gdsc_tcga_data(
    cmp_data: CMPData, tcga_data: TCGAData, gdsc_data: GDSCData
) -> t.Tuple[CMPData, TCGAData, GDSCData]:
    """Harmonizes the Cell Model Passports, GDSC and TCGA data."""

    cmp_data.meta["domain"] = "CELL"
    tcga_data.cell_meta["domain"] = "PATIENT"

    cols = ["model_id", "cancer_type", "domain"]
    cmp_data.meta = cmp_data.meta[cols]
    tcga_data.cell_meta = tcga_data.cell_meta[cols]

    common_genes = cmp_data.exp.columns.intersection(tcga_data.exp.columns)
    cmp_data.exp = cmp_data.exp[common_genes.sort_values()]
    tcga_data.exp = tcga_data.exp[common_genes.sort_values()]

    tcga_drugs = set(tcga_data.resp["drug_name"])
    gdsc_drugs = set(gdsc_data.meta["drug_name"])
    common_drugs = set.intersection(tcga_drugs, gdsc_drugs)

    tcga_data.resp = tcga_data.resp[tcga_data.resp["drug_name"].isin(common_drugs)]
    tcga_drug_meta = gdsc_data.meta[gdsc_data.meta["drug_name"].isin(common_drugs)]
    tcga_data.drug_meta = tcga_drug_meta.copy()

    return cmp_data, tcga_data, gdsc_data
