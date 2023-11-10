"""Utilities for dataset preprocessing and harmonization."""

from __future__ import annotations

import pandas as pd
import typing as t

from .cmp import load_cmp_data, harmonize_cmp_data
from .gdsc import load_gdsc_data, harmonize_gdsc_data
from .hci import load_hci_data, harmonize_hci_data
from .pubchem import fetch_pubchem_properties
from .tcga import load_tcga_data, harmonize_tcga_data


__all__ = [
    "load_cmp_data",
    "harmonize_cmp_data",
    "load_gdsc_data",
    "harmonize_gdsc_data",
    "load_hci_data",
    "harmonize_hci_data",
    "load_tcga_data",
    "harmonize_tcga_data",
    "harmonize_cmp_gdsc_data",
    "harmonize_cmp_gdsc_hci_data",
    "harmonize_cmp_gdsc_tcga_data",
    "fetch_pubchem_properties",
]


def harmonize_cmp_gdsc_data(
    exp_df: pd.DataFrame,
    resp_df: pd.DataFrame,
    cell_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
    cnv_df: pd.DataFrame | None = None,
    mut_df: pd.DataFrame | None = None,
) -> t.Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """Harmonizes the GDSCv2 and Cell Model Passports dataset.

    This function is designed to be run following initial loading and
    harmonization of the individual dataset components as part of the
    preprocessing pipeline.

    Parameters
    ----------
        exp_df: The harmonized Cell Model Passports expression data.
        cell_meta: The harmonized Cell Model Passports cell line meta data.
        resp_df: The harmonized GDSC drug response data.
        drug_meta: The harmonized GDSC drug meta data.
        cnv_df: The harmonized Cell Model Passports copy number data.
        mut_df: The harmonized Cell Model Passports somatic MAF data.

    Returns
    -------
        A (exp, resp, cell_meta, drug_meta, cnv, mut) tuple of harmonized
            `pandas.DataFrame` instances.
    """
    cmp_cells = set(cell_meta["model_id"])
    gdsc_cells = set(resp_df["model_id"])
    common_cells = sorted(list(set.intersection(cmp_cells, gdsc_cells)))

    # filter the Cell Model Passports data
    exp_df = exp_df.loc[common_cells]
    cell_meta = cell_meta[cell_meta["model_id"].isin(common_cells)]

    if cnv_df is not None:
        cnv_df = cnv_df.loc[common_cells]

    if mut_df is not None:
        mut_df = mut_df[mut_df["model_id"].isin(common_cells)]

    # filter the GDSC data
    resp_df = resp_df[resp_df["model_id"].isin(common_cells)]

    return (
        exp_df,
        resp_df,
        cell_meta,
        drug_meta,
        cnv_df,
        mut_df,
    )


def harmonize_cmp_gdsc_hci_data(
    cmp_exp: pd.DataFrame,
    hci_exp: pd.DataFrame,
    gdsc_resp: pd.DataFrame,
    hci_resp: pd.DataFrame,
    cmp_cell_meta: pd.DataFrame,
    hci_pdmc_meta: pd.DataFrame,
    gdsc_drug_meta: pd.DataFrame,
    hci_drug_meta: pd.DataFrame,
    cmp_mut: pd.DataFrame | None = None,
    hci_mut: pd.DataFrame | None = None,
    include_all_hci_drugs: bool = False,
) -> t.Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None
]:
    """Harmonizes the Cell Model Passports, GDSC and HCI data."""

    # 1. harmonize the omics data
    cmp_cell_meta["model_type"] = "cell line"
    cmp_cell_meta["domain"] = "CELL"
    hci_pdmc_meta["domain"] = "PDMC"

    cols = ["model_id", "cancer_type", "model_type", "domain"]
    sample_meta = pd.concat([cmp_cell_meta[cols], hci_pdmc_meta[cols]])

    common_genes = cmp_exp.columns.intersection(hci_exp.columns).sort_values()
    exp_df = pd.concat([cmp_exp[common_genes], hci_exp[common_genes]])

    mut_df = None
    if all(x is not None for x in [cmp_mut, hci_mut]):
        cmp_mut_genes = set(cmp_mut["gene_symbol"])
        hci_mut_genes = set(hci_mut["gene_symbol"])
        common_genes = cmp_mut_genes.intersection(hci_mut_genes)

        cmp_mut = cmp_mut[cmp_mut["gene_symbol"].isin(common_genes)]
        hci_mut = hci_mut[hci_mut["gene_symbol"].isin(common_genes)]

        cols = ["model_id", "gene_symbol", "chrom", "pos", "protein_mutation"]
        mut_df = pd.concat([cmp_mut[cols], hci_mut[cols]])

    # 2. harmonize the drug response data
    mapper = dict(zip(gdsc_drug_meta.pubchem_id, gdsc_drug_meta.drug_name))
    func = lambda r: mapper.get(r["pubchem_id"], r["drug_name"])

    old_names = hci_drug_meta["drug_name"].copy()
    hci_drug_meta["drug_name"] = hci_drug_meta.apply(func, axis=1)

    mapper = dict(zip(old_names, hci_drug_meta["drug_name"]))
    hci_resp["drug_name"] = hci_resp["drug_name"].map(mapper)

    if not include_all_hci_drugs:
        hci_drug_meta = hci_drug_meta[
            hci_drug_meta.pubchem_id.isin(gdsc_drug_meta["pubchem_id"])
        ]
        hci_resp = hci_resp[hci_resp["drug_name"].isin(gdsc_drug_meta["drug_name"])]

    cols = ["model_id", "drug_name", "ln_ic50"]
    resp_df = pd.concat([gdsc_resp[cols], hci_resp[cols]])

    cols = ["drug_name", "pubchem_id"]
    drug_meta = pd.concat([gdsc_drug_meta[cols], hci_drug_meta[cols]])
    drug_meta = drug_meta.drop_duplicates()

    return exp_df, resp_df, sample_meta, drug_meta, mut_df


def harmonize_cmp_gdsc_tcga_data(
    cmp_exp: pd.DataFrame,
    tcga_exp: pd.DataFrame,
    gdsc_resp: pd.DataFrame,
    tcga_resp: pd.DataFrame,
    cmp_sample_meta: pd.DataFrame,
    tcga_sample_meta: pd.DataFrame,
    gdsc_drug_meta: pd.DataFrame,
) -> t.Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Harmonizes the Cell Model Passports, GDSC and TCGA data."""
    cmp_sample_meta["domain"] = "CELL"
    tcga_sample_meta["domain"] = "PATIENT"

    cols = ["model_id", "cancer_type", "domain"]
    cmp_sample_meta = cmp_sample_meta[cols]
    tcga_sample_meta = tcga_sample_meta[cols]

    common_genes = cmp_exp.columns.intersection(tcga_exp.columns)
    cmp_exp = cmp_exp[common_genes.sort_values()]
    tcga_exp = tcga_exp[common_genes.sort_values()]

    tcga_drugs = set(tcga_resp["drug_name"])
    common_drugs = tcga_drugs.intersection(gdsc_drug_meta["drug_name"])
    tcga_resp = tcga_resp[tcga_resp["drug_name"].isin(common_drugs)]

    tcga_drug_meta = gdsc_drug_meta[
        gdsc_drug_meta["drug_name"].isin(common_drugs)
    ].copy()

    return (
        cmp_exp,
        tcga_exp,
        gdsc_resp,
        tcga_resp,
        cmp_sample_meta,
        tcga_sample_meta,
        gdsc_drug_meta,
        tcga_drug_meta,
    )
