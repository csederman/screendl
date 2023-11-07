#!/usr/bin/env python
"""Cleans and harmonizes data sources for downstream feature extraction."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import click
import logging
import pickle

import pandas as pd
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from screendl.preprocessing.data import (
    load_cmp_data,
    load_gdsc_data,
    load_hci_data,
    harmonize_cmp_data,
    harmonize_gdsc_data,
    harmonize_hci_data,
    harmonize_cmp_gdsc_data,
    harmonize_cmp_gdsc_hci_data,
    fetch_pubchem_properties,
)
from screendl.preprocessing.models import (
    generate_and_save_deepcdr_inputs,
    generate_and_save_hidra_inputs,
    generate_and_save_screendl_inputs,
)
from screendl.preprocessing.splits import kfold_split_generator


logging.basicConfig(
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S,%03d",
    handlers=[logging.StreamHandler(sys.stdout)],
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def make_dataset(
    cfg: DictConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
]:
    """Cleans and harmonizes the data sources."""
    paths = cfg.dataset.paths
    params = cfg.dataset.params

    log.info("Loading Cell Model Passports data...")
    cmp_cell_exp, cmp_cell_meta, _, cmp_cell_mut = load_cmp_data(
        exp_path=paths.cmp.exp,
        meta_path=paths.cmp.meta,
        vcf_dir=paths.cmp.vcf,
    )

    log.info("Harmonizing Cell Model Passports data...")
    cmp_cell_exp, cmp_cell_meta, _, cmp_cell_mut = harmonize_cmp_data(
        exp_df=cmp_cell_exp,
        meta_df=cmp_cell_meta,
        mut_df=cmp_cell_mut,
        min_cells_per_cancer_type=params.cmp.min_cells_per_cancer_type,
        required_info_columns=params.cmp.required_info_columns,
        cancer_type_blacklist=params.cmp.cancer_type_blacklist,
    )

    log.info("Loading GDSC response data...")
    gdsc_drug_resp, gdsc_drug_meta = load_gdsc_data(
        resp_path=paths.gdsc.resp, meta_path=paths.gdsc.meta
    )

    log.info("Harmonizing GDSC response data...")
    gdsc_drug_resp, gdsc_drug_meta = harmonize_gdsc_data(
        resp_df=gdsc_drug_resp, meta_df=gdsc_drug_meta
    )

    log.info("Harmonizing GDSC and Cell Model Passports...")
    (
        cmp_cell_exp,
        gdsc_drug_resp,
        cmp_cell_meta,
        gdsc_drug_meta,
        _,
        cmp_mut_df,
    ) = harmonize_cmp_gdsc_data(
        exp_df=cmp_cell_exp,
        resp_df=gdsc_drug_resp,
        cell_meta=cmp_cell_meta,
        drug_meta=gdsc_drug_meta,
        mut_df=cmp_cell_mut,
    )

    log.info("Loading HCI PDMC data...")
    (
        hci_pdmc_exp,
        hci_drug_resp,
        hci_pdmc_meta,
        hci_drug_meta,
        hci_pdmc_mut,
    ) = load_hci_data(
        exp_path=paths.hci.exp,
        resp_path=paths.hci.resp,
        pdmc_meta_path=paths.hci.pdmc_meta,
        drug_meta_path=paths.hci.drug_meta,
        mut_path=paths.hci.mut,
    )

    log.info("Harmonizing HCI PDMC data...")
    (
        hci_pdmc_exp,
        hci_drug_resp,
        hci_pdmc_meta,
        hci_drug_meta,
        hci_pdmc_mut,
    ) = harmonize_hci_data(
        exp_df=hci_pdmc_exp,
        resp_df=hci_drug_resp,
        pdmc_meta=hci_pdmc_meta,
        drug_meta=hci_drug_meta,
        model_types=params.hci.pdmc_model_types,
        mut_df=hci_pdmc_mut,
        min_samples_per_drug=params.hci.min_samples_per_drug,
    )

    log.info("Harmonizing GDSC, Cell Model Passports, and HCI data...")
    (
        exp_df,
        resp_df,
        sample_meta,
        drug_meta,
        mut_df,
    ) = harmonize_cmp_gdsc_hci_data(
        cmp_exp=cmp_cell_exp,
        hci_exp=hci_pdmc_exp,
        gdsc_resp=gdsc_drug_resp,
        hci_resp=hci_drug_resp,
        cmp_cell_meta=cmp_cell_meta,
        hci_pdmc_meta=hci_pdmc_meta,
        gdsc_drug_meta=gdsc_drug_meta,
        hci_drug_meta=hci_drug_meta,
        cmp_mut=cmp_cell_mut,
        hci_mut=hci_pdmc_mut,
        include_all_hci_drugs=params.hci.include_all_hci_drugs,
    )

    # query PubCHEM annotations
    pubchem_cids = list(drug_meta["pubchem_id"])
    pubchem_annots = fetch_pubchem_properties(pubchem_cids)
    pubchem_annots["CID"] = pubchem_annots["CID"].astype(str)
    pubchem_annots = pubchem_annots.rename(
        columns={"CanonicalSMILES": "smiles"}
    )

    # merge in the PubCHEM annotations
    drug_meta = drug_meta.merge(
        pubchem_annots, left_on="pubchem_id", right_on="CID"
    )

    if cfg.dataset.save:
        log.info("Saving dataset...")

        out_dir = Path(cfg.dataset.dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        exp_df.to_csv(out_dir / "OmicsGeneExpressionTPM.csv")
        resp_df.to_csv(out_dir / "ScreenDoseResponse.csv", index=False)
        sample_meta.to_csv(out_dir / "CellLineAnnotations.csv", index=False)
        drug_meta.to_csv(out_dir / "DrugAnnotations.csv", index=False)

        if mut_df is not None:
            mut_df.to_csv(out_dir / "OmicsSomaticMutations.csv", index=False)

    return exp_df, resp_df, sample_meta, drug_meta, mut_df


def make_labels(cfg: DictConfig, resp_df: pd.DataFrame) -> pd.DataFrame:
    """Creates the target labels."""
    nan_values = [np.inf, -np.inf, np.nan]
    resp_df = resp_df[~resp_df["ln_ic50"].isin(nan_values)]
    resp_df = resp_df.dropna(subset="ln_ic50")
    resp_df["id"] = range(resp_df.shape[0])

    resp_df = resp_df[["id", "model_id", "drug_name", "ln_ic50"]]
    resp_df = resp_df.rename(
        columns={
            "model_id": "cell_id",
            "drug_name": "drug_id",
            "ln_ic50": "label",
        }
    )

    out_dir = Path(cfg.inputs.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    resp_df.to_csv(out_dir / "LabelsLogIC50.csv", index=False)

    return resp_df


def make_splits(
    cfg: DictConfig, resp_df: pd.DataFrame, sample_meta: pd.DataFrame
) -> None:
    """Creates train/val/test splits."""
    params = cfg.splits.params

    out_dir = Path(cfg.splits.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    cell_sample_meta = sample_meta[sample_meta["domain"] == "CELL"]
    pdmc_sample_ids = sample_meta[sample_meta["domain"] == "PDMC"]["model_id"]

    cell_sample_ids = cell_sample_meta["model_id"]
    cell_sample_groups = cell_sample_meta["cancer_type"]

    split_gen = kfold_split_generator(
        cell_sample_ids,
        cell_sample_groups,
        n_splits=params.n_splits,
        random_state=params.seed,
        include_validation_set=False,
    )

    for i, split in enumerate(split_gen, 1):
        train_cells = split["train"]
        val_cells = split["test"]
        test_cells = pdmc_sample_ids

        train_ids = resp_df[resp_df["cell_id"].isin(train_cells)]["id"]
        val_ids = resp_df[resp_df["cell_id"].isin(val_cells)]["id"]
        test_ids = resp_df[resp_df["cell_id"].isin(test_cells)]["id"]

        split_dict = {
            "train": list(train_ids),
            "val": list(val_ids),
            "test": list(test_ids),
        }

        with open(out_dir / f"fold_{i}.pkl", "wb") as fh:
            pickle.dump(split_dict, fh)


def make_meta(
    cfg: DictConfig, sample_meta: pd.DataFrame, drug_meta: pd.DataFrame
) -> None:
    """Creates metadata inputs."""
    out_root = Path(cfg.inputs.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    # generate metadata inputs
    cols = ["model_id", "cancer_type", "model_type", "domain"]
    sample_meta_feat = sample_meta[cols].set_index("model_id")
    sample_meta_feat = sample_meta_feat.rename_axis(index="cell_id")

    cols = ["drug_name", "pubchem_id", "smiles"]
    drug_meta_feat = drug_meta[cols].set_index("drug_name")
    drug_meta_feat = drug_meta_feat.rename_axis(index="drug_id")

    sample_meta_feat.to_csv(out_root / "MetaSampleAnnotations.csv")
    drug_meta_feat.to_csv(out_root / "MetaDrugAnnotations.csv")


def make_inputs(
    cfg: DictConfig,
    exp_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
    mut_df: pd.DataFrame | None = None,
) -> None:
    """Creates model inputs."""
    out_root = Path(cfg.inputs.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    if "DeepCDR" in cfg.inputs.include:
        if mut_df is None:
            log.warning("Skipping DeepCDR inputs (no mutation data provided)")
        else:
            log.info("Generating DeepCDR inputs...")
            generate_and_save_deepcdr_inputs(
                cfg.inputs.deepcdr,
                out_dir=out_root / "DeepCDR",
                exp_df=exp_df,
                mut_df=mut_df,
                drug_meta=drug_meta,
            )

    if "HiDRA" in cfg.inputs.include:
        log.info("Generating HiDRA inputs...")
        generate_and_save_hidra_inputs(
            cfg.inputs.hidra,
            out_dir=out_root / "HiDRA",
            exp_df=exp_df,
            drug_meta=drug_meta,
        )

    if "ScreenDL" in cfg.inputs.include:
        log.info("Generating ScreenDL inputs...")
        generate_and_save_screendl_inputs(
            cfg.inputs.screendl,
            out_dir=out_root / "ScreenDL",
            cell_meta=sample_meta,
            drug_meta=drug_meta,
            exp_df=exp_df,
            mut_df=mut_df,
        )


@click.command()
@click.option(
    "--config-path",
    "-c",
    type=str,
    required=True,
    help="Path to config file.",
)
def cli(config_path: str) -> None:
    """Creates the dataset and all inputs."""
    cfg = OmegaConf.load(config_path)

    log.info("Building the dataset...")
    exp_df, resp_df, sample_meta, drug_meta, mut_df = make_dataset(cfg)

    log.info("Generating labels...")
    resp_df = make_labels(cfg, resp_df)

    log.info("Generating folds...")
    make_splits(cfg, resp_df, sample_meta)

    log.info("Generating meta data...")
    make_meta(cfg, sample_meta, drug_meta)

    log.info("Generating model inputs...")
    make_inputs(cfg, exp_df, sample_meta, drug_meta, mut_df)


if __name__ == "__main__":
    cli()
