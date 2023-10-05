#!/usr/bin/env python
"""Cleans and harmonizes data sources for downstream feature extraction."""

from __future__ import annotations

import click

from pathlib import Path
from types import SimpleNamespace

from utils.datasets import (
    load_cmp_data,
    load_gdsc_data,
    load_oncotree_annotations,
    harmonize_cmp_data,
    harmonize_gdsc_data,
    harmonize_gdsc_and_cmp_data,
    fetch_pubchem_properties,
)


DATASTORE_ROOT = Path("/scratch/ucgd/lustre-work/marth/u0871891/datastore")
cmp_dir = DATASTORE_ROOT / "raw/CellModelPassports"
gdsc_dir = DATASTORE_ROOT / "raw/GDSC"
depmap_dir = DATASTORE_ROOT / "raw/DepMap"


PATHS = SimpleNamespace(
    cmp_cell_info=cmp_dir / "model_list_20230608.csv",
    cmp_gene_info=cmp_dir / "gene_identifiers_20191101.csv",
    cmp_vcf_dir=cmp_dir / "mutations_wes_vcf_20221010",
    cmp_mut=cmp_dir / "mutations_all_20230202.csv",
    cmp_exp=cmp_dir / "rnaseq_tpm_20220624.csv",
    cmp_cnv=cmp_dir / "WES_pureCN_CNV_genes_total_copy_number_20221213.csv",
    dmap_cell_info=depmap_dir / "Model.csv",
    gdsc_drug_info=gdsc_dir / "drug_list_2023_06_23.csv",
    gdsc_resp=gdsc_dir / "GDSC2_fitted_dose_response_24Jul22.xlsx",
    output_dir=DATASTORE_ROOT / "datasets/CellModelPassportsGDSCv2",
)


def main() -> None:
    """"""

    # load the raw Cell Model Passports omics data
    click.echo("Loading Cell Model Passports data...")
    cmp_exp, cmp_cnv, cmp_mut, cmp_mut_pos, cmp_cell_info = load_cmp_data(
        PATHS.cmp_exp,
        PATHS.cmp_cnv,
        PATHS.cmp_mut,
        PATHS.cmp_vcf_dir,
        PATHS.cmp_cell_info,
    )

    # add in oncotree annotations for downstream feature extraction
    cell_to_ot = load_oncotree_annotations(PATHS.dmap_cell_info)
    cmp_cell_info["oncotree_code"] = cmp_cell_info["model_id"].map(cell_to_ot)

    # harmonize the raw Cell Model Passports omics data
    click.echo("Harmonizing Cell Model Passports data...")
    cmp_exp, cmp_cnv, cmp_mut, cmp_mut_pos, cmp_cell_info = harmonize_cmp_data(
        cmp_exp,
        cmp_cnv,
        cmp_mut,
        cmp_mut_pos,
        cmp_cell_info,
        min_cells_per_cancer_type=20,
        required_info_columns=["ploidy_wes", "cancer_type", "oncotree_code"],
        cancer_type_blacklist=["Non-Cancerous", "Unknown"],
    )

    # load the raw GDSC drug response data
    click.echo("Loading GDSC response data...")
    gdsc_resp, gdsc_drug_info = load_gdsc_data(
        PATHS.gdsc_resp, PATHS.gdsc_drug_info
    )

    # harmonize the raw GDSC drug response data
    click.echo("Harmonizing GDSC response data...")
    gdsc_resp, gdsc_drug_info = harmonize_gdsc_data(gdsc_resp, gdsc_drug_info)

    click.echo("Harmonizing GDSC and Cell Model Passports...")
    # harmonize the omics and drug response data
    (
        cmp_exp,
        cmp_cnv,
        cmp_mut,
        cmp_mut_pos,
        cmp_cell_info,
        gdsc_resp,
        gdsc_drug_info,
    ) = harmonize_gdsc_and_cmp_data(
        exp_df=cmp_exp,
        cnv_df=cmp_cnv,
        mut_df=cmp_mut,
        mut_df_pos=cmp_mut_pos,
        cell_info_df=cmp_cell_info,
        resp_df=gdsc_resp,
        drug_info_df=gdsc_drug_info,
    )

    # query PubCHEM annotations
    pubchem_cids = list(gdsc_drug_info["PubCHEM"])
    pubchem_annots = fetch_pubchem_properties(pubchem_cids)
    pubchem_annots["CID"] = pubchem_annots["CID"].astype(str)

    # merge in the PubCHEM annotations
    gdsc_drug_info = gdsc_drug_info.merge(
        pubchem_annots, left_on="PubCHEM", right_on="CID"
    )

    # save the harmonized omics files
    click.echo("Saving outputs...")
    cmp_exp.to_csv(PATHS.output_dir / "OmicsGeneExpressionTPM.csv")
    cmp_cnv.to_csv(PATHS.output_dir / "OmicsTotalCopyNumber.csv")
    cmp_mut.to_csv(PATHS.output_dir / "OmicsSomaticMutations.csv", index=False)
    cmp_mut_pos.to_csv(
        PATHS.output_dir / "OmicsSomaticMutationsWithPosition.csv", index=False
    )
    cmp_cell_info.to_csv(
        PATHS.output_dir / "CellLineAnnotations.csv", index=False
    )

    # save the harmonized drug response data
    gdsc_resp.to_csv(PATHS.output_dir / "ScreenDoseResponse.csv", index=False)
    gdsc_drug_info.to_csv(
        PATHS.output_dir / "DrugAnnotations.csv", index=False
    )


if __name__ == "__main__":
    main()
