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
    load_tcga_data,
    harmonize_cmp_data,
    harmonize_gdsc_data,
    harmonize_tcga_data,
    harmonize_cmp_gdsc_data,
    harmonize_cmp_gdsc_tcga_data,
    fetch_pubchem_properties,
)
from screendl.preprocessing.models import generate_and_save_screendl_inputs
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
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Cleans and harmonizes the data sources."""
    paths = cfg.dataset.paths
    params = cfg.dataset.params

    log.info("Loading Cell Model Passports data...")
    cmp_cell_exp, cmp_cell_meta, _, _ = load_cmp_data(
        exp_path=paths.cmp.exp,
        meta_path=paths.cmp.meta,
    )

    log.info("Harmonizing Cell Model Passports data...")
    cmp_cell_exp, cmp_cell_meta, _, cmp_cell_mut = harmonize_cmp_data(
        exp_df=cmp_cell_exp,
        meta_df=cmp_cell_meta,
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

    log.info("Loading TCGA data...")
    tcga_pt_exp, tcga_drug_resp, tcga_pt_meta = load_tcga_data(
        exp_path=paths.tcga.exp,
        resp_path=paths.tcga.resp,
        meta_path=paths.tcga.meta,
    )

    log.info("Harmonizing TCGA data...")
    tcga_pt_exp, tcga_drug_resp, tcga_pt_meta = harmonize_tcga_data(
        exp_df=tcga_pt_exp,
        resp_df=tcga_drug_resp,
        meta_df=tcga_pt_meta,
        min_samples_per_drug=params.tcga.min_samples_per_drug,
    )

    log.info("Harmonizing GDSC and Cell Model Passports...")
    (
        cmp_cell_exp,
        gdsc_drug_resp,
        cmp_cell_meta,
        gdsc_drug_meta,
        _,
        _,
    ) = harmonize_cmp_gdsc_data(
        exp_df=cmp_cell_exp,
        resp_df=gdsc_drug_resp,
        cell_meta=cmp_cell_meta,
        drug_meta=gdsc_drug_meta,
    )

    log.info("Harmonizing GDSC, Cell Model Passports, and TCGA...")
    (
        cmp_cell_exp,
        tcga_pt_exp,
        gdsc_drug_resp,
        tcga_drug_resp,
        cmp_cell_meta,
        tcga_pt_meta,
        gdsc_drug_meta,
        tcga_drug_meta,
    ) = harmonize_cmp_gdsc_tcga_data(
        cmp_exp=cmp_cell_exp,
        tcga_exp=tcga_pt_exp,
        gdsc_resp=gdsc_drug_resp,
        tcga_resp=tcga_drug_resp,
        cmp_sample_meta=cmp_cell_meta,
        tcga_sample_meta=tcga_pt_meta,
        gdsc_drug_meta=gdsc_drug_meta,
    )

    # query PubCHEM annotations
    pubchem_cids = set(gdsc_drug_meta["pubchem_id"])
    pubchem_cids = list(pubchem_cids.union(tcga_drug_meta["pubchem_id"]))

    pubchem_annots = fetch_pubchem_properties(pubchem_cids, paths.pubchem.cache)
    pubchem_annots["CID"] = pubchem_annots["CID"].astype(str)
    pubchem_annots = pubchem_annots.rename(columns={"CanonicalSMILES": "smiles"})

    # merge in the PubCHEM annotations
    gdsc_drug_meta = gdsc_drug_meta.merge(
        pubchem_annots, left_on="pubchem_id", right_on="CID"
    )
    tcga_drug_meta = tcga_drug_meta.merge(
        pubchem_annots, left_on="pubchem_id", right_on="CID"
    )

    if cfg.dataset.save:
        log.info("Saving dataset...")

        cell_out_dir = Path(cfg.dataset.dir) / "cell"
        cell_out_dir.mkdir(exist_ok=True, parents=True)

        cmp_cell_exp.to_csv(cell_out_dir / "OmicsGeneExpressionTPM.csv")
        cmp_cell_meta.to_csv(cell_out_dir / "SampleAnnotations.csv", index=False)

        gdsc_drug_resp.to_csv(cell_out_dir / "ScreenDoseResponse.csv", index=False)
        gdsc_drug_meta.to_csv(cell_out_dir / "DrugAnnotations.csv", index=False)

        pt_out_dir = Path(cfg.dataset.dir) / "patient"
        pt_out_dir.mkdir(exist_ok=True, parents=True)

        tcga_pt_exp.to_csv(pt_out_dir / "OmicsGeneExpressionTPM.csv")
        tcga_pt_meta.to_csv(pt_out_dir / "SampleAnnotations.csv", index=False)
        tcga_drug_resp.to_csv(pt_out_dir / "ScreenDoseResponse.csv", index=False)
        tcga_drug_meta.to_csv(pt_out_dir / "DrugAnnotations.csv", index=False)

    return (
        cmp_cell_exp,
        tcga_pt_exp,
        gdsc_drug_resp,
        tcga_drug_resp,
        cmp_cell_meta,
        tcga_pt_meta,
        gdsc_drug_meta,
        tcga_drug_meta,
    )


def make_labels(
    cfg: DictConfig, cell_resp_df: pd.DataFrame, pt_resp_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates the target labels."""
    nan_values = [np.inf, -np.inf, np.nan]
    cell_resp_df = cell_resp_df[~cell_resp_df["ln_ic50"].isin(nan_values)]
    cell_resp_df = cell_resp_df.dropna(subset="ln_ic50")
    cell_resp_df["id"] = range(cell_resp_df.shape[0])

    cell_resp_df = cell_resp_df[["id", "model_id", "drug_name", "ln_ic50"]]
    cell_resp_df = cell_resp_df.rename(
        columns={
            "model_id": "cell_id",
            "drug_name": "drug_id",
            "ln_ic50": "label",
        }
    )

    cell_out_dir = Path(cfg.inputs.dir) / "cell"
    cell_out_dir.mkdir(exist_ok=True, parents=True)

    cell_resp_df.to_csv(cell_out_dir / "LabelsLogIC50.csv", index=False)

    pt_resp_df["id"] = range(pt_resp_df.shape[0])
    pt_resp_df = pt_resp_df[["id", "model_id", "drug_name", "binary_response"]]
    pt_resp_df = pt_resp_df.rename(
        columns={
            "model_id": "cell_id",
            "drug_name": "drug_id",
            "binary_response": "label",
        }
    )

    pt_out_dir = Path(cfg.inputs.dir) / "patient"
    pt_out_dir.mkdir(exist_ok=True, parents=True)

    pt_resp_df.to_csv(pt_out_dir / "LabelsBinaryResponse.csv", index=False)

    return cell_resp_df, pt_resp_df


def make_splits(
    cfg: DictConfig,
    cell_resp_df: pd.DataFrame,
    cell_sample_meta: pd.DataFrame,
) -> None:
    """Creates train/val/test splits."""
    params = cfg.splits.params

    out_dir = Path(cfg.splits.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

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

        train_ids = cell_resp_df[cell_resp_df["cell_id"].isin(train_cells)]["id"]
        val_ids = cell_resp_df[cell_resp_df["cell_id"].isin(val_cells)]["id"]

        split_dict = {"train": list(train_ids), "val": list(val_ids)}

        with open(out_dir / f"fold_{i}.pkl", "wb") as fh:
            pickle.dump(split_dict, fh)


def make_meta(
    cfg: DictConfig,
    cell_sample_meta: pd.DataFrame,
    cell_drug_meta: pd.DataFrame,
    pt_sample_meta: pd.DataFrame,
    pt_drug_meta: pd.DataFrame,
) -> None:
    """Creates metadata inputs."""
    cell_out_dir = Path(cfg.inputs.dir) / "cell"
    cell_out_dir.mkdir(exist_ok=True, parents=True)

    pt_out_dir = Path(cfg.inputs.dir) / "patient"
    pt_out_dir.mkdir(exist_ok=True, parents=True)

    # generate metadata inputs
    cols = ["model_id", "cancer_type", "domain"]
    cell_sample_meta_feat = cell_sample_meta[cols].set_index("model_id")
    cell_sample_meta_feat = cell_sample_meta_feat.rename_axis(index="cell_id")
    cell_sample_meta.to_csv(cell_out_dir / "MetaSampleAnnotations.csv")

    pt_sample_meta_feat = pt_sample_meta[cols].set_index("model_id")
    pt_sample_meta_feat = pt_sample_meta_feat.rename_axis(index="cell_id")
    pt_sample_meta.to_csv(pt_out_dir / "MetaSampleAnnotations.csv")

    cols = ["drug_name", "pubchem_id", "smiles"]
    cell_drug_meta_feat = cell_drug_meta[cols].set_index("drug_name")
    cell_drug_meta_feat = cell_drug_meta_feat.rename_axis(index="drug_id")
    cell_drug_meta_feat.to_csv(cell_out_dir / "MetaDrugAnnotations.csv")

    pt_drug_meta_feat = pt_drug_meta[cols].set_index("drug_name")
    pt_drug_meta_feat = pt_drug_meta_feat.rename_axis(index="drug_id")
    pt_drug_meta_feat.to_csv(pt_out_dir / "MetaDrugAnnotations.csv")


def make_inputs(
    cfg: DictConfig,
    cell_exp_df: pd.DataFrame,
    cell_sample_meta: pd.DataFrame,
    cell_drug_meta: pd.DataFrame,
    pt_exp_df: pd.DataFrame,
    pt_sample_meta: pd.DataFrame,
    pt_drug_meta: pd.DataFrame,
) -> None:
    """Creates model inputs."""
    log.info("Generating ScreenDL inputs...")

    cell_out_root = Path(cfg.inputs.dir) / "cell"
    cell_out_root.mkdir(exist_ok=True, parents=True)

    generate_and_save_screendl_inputs(
        cfg.inputs.screendl,
        out_dir=cell_out_root / "ScreenDL",
        cell_meta=cell_sample_meta,
        drug_meta=cell_drug_meta,
        exp_df=cell_exp_df,
    )

    pt_out_root = Path(cfg.inputs.dir) / "patient"
    pt_out_root.mkdir(exist_ok=True, parents=True)

    generate_and_save_screendl_inputs(
        cfg.inputs.screendl,
        out_dir=pt_out_root / "ScreenDL",
        cell_meta=pt_sample_meta,
        drug_meta=pt_drug_meta,
        exp_df=pt_exp_df,
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
    (
        cell_exp,
        pt_exp,
        cell_resp,
        pt_resp,
        cell_sample_meta,
        pt_sample_meta,
        cell_drug_meta,
        pt_drug_meta,
    ) = make_dataset(cfg)

    log.info("Generating labels...")
    cell_resp_df, pt_resp_df = make_labels(
        cfg, cell_resp_df=cell_resp, pt_resp_df=pt_resp
    )

    log.info("Generating folds...")
    make_splits(cfg, cell_resp_df=cell_resp_df, cell_sample_meta=cell_sample_meta)

    log.info("Generating meta data...")
    make_meta(
        cfg,
        cell_sample_meta=cell_sample_meta,
        cell_drug_meta=cell_drug_meta,
        pt_sample_meta=pt_sample_meta,
        pt_drug_meta=pt_drug_meta,
    )

    log.info("Generating model inputs...")
    make_inputs(
        cfg,
        cell_exp_df=cell_exp,
        cell_sample_meta=cell_sample_meta,
        cell_drug_meta=cell_drug_meta,
        pt_exp_df=pt_exp,
        pt_sample_meta=pt_sample_meta,
        pt_drug_meta=pt_drug_meta,
    )


if __name__ == "__main__":
    cli()
