#!/usr/bin/env python
"""Cleans and harmonizes data sources for downstream feature extraction."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import click
import logging
import pickle

import numpy as np
import pandas as pd
import typing as t

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from screendl.preprocessing.data import cmp, gdsc, tcga
from screendl.preprocessing.data import (
    harmonize_cmp_gdsc_data,
    harmonize_cmp_gdsc_tcga_data,
    fetch_pubchem_properties,
)
from screendl.preprocessing.models import generate_and_save_screendl_inputs
from screendl.preprocessing.splits import kfold_split_generator


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def make_dataset(cfg: DictConfig) -> t.Tuple[cmp.CMPData, gdsc.GDSCData, tcga.TCGAData]:
    """Cleans and harmonizes the data sources."""
    paths = cfg.dataset.paths
    params = cfg.dataset.params

    log.info("Loading Cell Model Passports data...")
    cmp_data = cmp.load_and_clean_cmp_data(
        exp_path=paths.cmp.exp,
        meta_path=paths.cmp.meta,
        min_cells_per_cancer_type=params.cmp.min_cells_per_cancer_type,
        required_info_columns=params.cmp.required_info_columns,
        cancer_type_blacklist=params.cmp.cancer_type_blacklist,
    )

    log.info("Loading GDSC response data...")
    gdsc_data = gdsc.load_and_clean_gdsc_data(paths.gdsc.resp, paths.gdsc.meta)

    log.info("Loading TCGA data...")
    tcga_data = tcga.load_and_clean_tcga_data(
        paths.tcga.exp,
        paths.tcga.resp,
        paths.tcga.meta,
        min_samples_per_drug=params.tcga.min_samples_per_drug,
    )

    log.info("Harmonizing GDSC and Cell Model Passports...")
    cmp_data, gdsc_data = harmonize_cmp_gdsc_data(cmp_data, gdsc_data)

    log.info("Harmonizing GDSC, Cell Model Passports, and TCGA...")
    cmp_data, tcga_data, gdsc_data = harmonize_cmp_gdsc_tcga_data(
        cmp_data, tcga_data, gdsc_data
    )

    # query PubCHEM annotations
    pchem_ids = set(gdsc_data.meta["pubchem_id"])
    pchem_ids = list(pchem_ids.union(tcga_data.drug_meta["pubchem_id"]))
    pchem_props = fetch_pubchem_properties(pchem_ids, paths.pubchem.cache)
    pchem_props["CID"] = pchem_props["CID"].astype(str)
    pchem_props = pchem_props.rename(columns={"CanonicalSMILES": "smiles"})

    gdsc_data.meta = gdsc_data.meta.merge(
        pchem_props, left_on="pubchem_id", right_on="CID"
    )
    tcga_data.drug_meta = tcga_data.drug_meta.merge(
        pchem_props, left_on="pubchem_id", right_on="CID"
    )

    if cfg.dataset.save:
        log.info("Saving dataset...")

        out_root = Path(cfg.dataset.dir)

        # save the cell line data
        cell_subdir = out_root / "cell"
        cell_subdir.mkdir(exist_ok=True, parents=True)

        gdsc_data.meta.to_csv(cell_subdir / "DrugAnnotations.csv", index=False)
        gdsc_data.resp.to_csv(cell_subdir / "ScreenDoseResponse.csv", index=False)
        cmp_data.meta.to_csv(cell_subdir / "SampleAnnotations.csv", index=False)
        cmp_data.exp.to_csv(cell_subdir / "OmicsGeneExpressionTPM.csv")

        # save the patient data
        pt_subdir = out_root / "patient"
        pt_subdir.mkdir(exist_ok=True, parents=True)

        tcga_data.drug_meta.to_csv(pt_subdir / "DrugAnnotations.csv", index=False)
        tcga_data.resp.to_csv(pt_subdir / "ScreenDoseResponse.csv", index=False)
        tcga_data.cell_meta.to_csv(pt_subdir / "SampleAnnotations.csv", index=False)
        tcga_data.exp.to_csv(pt_subdir / "OmicsGeneExpressionTPM.csv")

    return cmp_data, gdsc_data, tcga_data


def make_labels(
    cfg: DictConfig, cell_resp_data: pd.DataFrame, pt_resp_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates the target labels."""
    out_root = Path(cfg.inputs.dir)

    # create cell line labels
    col_map = {"model_id": "cell_id", "drug_name": "drug_id", "ln_ic50": "label"}
    cell_resp_data_valid = cell_resp_data[
        ~cell_resp_data["ln_ic50"].isin(gdsc.INVALID_RESPONSE_VALUES)
    ]
    cell_resp_data_valid["id"] = range(cell_resp_data_valid.shape[0])
    cell_resp_data_valid = cell_resp_data_valid[["id", *col_map]].rename(
        columns=col_map
    )

    cell_subdir = out_root / "cell"
    cell_subdir.mkdir(exist_ok=True, parents=True)
    cell_resp_data_valid.to_csv(cell_subdir / "LabelsLogIC50.csv", index=False)

    # save the patient data
    col_map = {
        "model_id": "cell_id",
        "drug_name": "drug_id",
        "binary_response": "label",
    }
    pt_resp_data_valid = pt_resp_data.dropna(subset="binary_response")
    pt_resp_data_valid["id"] = range(pt_resp_data_valid.shape[0])
    pt_resp_data_valid = pt_resp_data_valid[["id", *col_map]].rename(columns=col_map)

    pt_subdir = out_root / "patient"
    pt_subdir.mkdir(exist_ok=True, parents=True)
    pt_resp_data_valid.to_csv(pt_subdir / "LabelsBinaryResponse.csv", index=False)

    return cell_resp_data_valid, pt_resp_data_valid


def make_splits(
    cfg: DictConfig,
    cl_resp_data: pd.DataFrame,
    cl_sample_meta: pd.DataFrame,
) -> None:
    """Creates train/val/test splits."""
    params = cfg.splits.params

    out_dir = Path(cfg.splits.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    cl_sample_ids = cl_sample_meta["cell_id"]
    cl_sample_groups = cl_sample_meta["cancer_type"]

    split_gen = kfold_split_generator(
        cl_sample_ids,
        cl_sample_groups,
        n_splits=params.n_splits,
        random_state=params.seed,
        include_validation_set=False,
    )

    for i, split in enumerate(split_gen, 1):
        train_samples = split["train"]
        val_samples = split["test"]

        train_ids = cl_resp_data[cl_resp_data["cell_id"].isin(train_samples)]["id"]
        val_ids = cl_resp_data[cl_resp_data["cell_id"].isin(val_samples)]["id"]

        split_dict = {"train": list(train_ids), "val": list(val_ids)}

        with open(out_dir / f"fold_{i}.pkl", "wb") as fh:
            pickle.dump(split_dict, fh)


def make_meta(
    cfg: DictConfig,
    cl_sample_meta: pd.DataFrame,
    cl_drug_meta: pd.DataFrame,
    pt_sample_meta: pd.DataFrame,
    pt_drug_meta: pd.DataFrame,
) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates metadata inputs."""
    out_root = Path(cfg.inputs.dir)

    cl_subdir = out_root / "cell"
    cl_subdir.mkdir(exist_ok=True, parents=True)

    pt_subdir = out_root / "patient"
    pt_subdir.mkdir(exist_ok=True, parents=True)

    # generate metadata inputs
    cols = ["model_id", "cancer_type", "domain"]
    cl_sample_meta = cl_sample_meta[cols].set_index("model_id")
    cl_sample_meta = cl_sample_meta.rename_axis(index="cell_id")
    cl_sample_meta.to_csv(cl_subdir / "MetaSampleAnnotations.csv")

    pt_sample_meta = pt_sample_meta[cols].set_index("model_id")
    pt_sample_meta = pt_sample_meta.rename_axis(index="cell_id")
    pt_sample_meta.to_csv(pt_subdir / "MetaSampleAnnotations.csv")

    cols = ["drug_name", "pubchem_id", "smiles"]
    cl_drug_meta = cl_drug_meta[cols].set_index("drug_name")
    cl_drug_meta = cl_drug_meta.rename_axis(index="drug_id")
    cl_drug_meta.to_csv(cl_subdir / "MetaDrugAnnotations.csv")

    pt_drug_meta = pt_drug_meta[cols].set_index("drug_name")
    pt_drug_meta = pt_drug_meta.rename_axis(index="drug_id")
    pt_drug_meta.to_csv(pt_subdir / "MetaDrugAnnotations.csv")

    return cl_sample_meta, cl_drug_meta, pt_sample_meta, pt_drug_meta


def make_inputs(
    cfg: DictConfig,
    cl_exp_data: pd.DataFrame,
    cl_sample_meta: pd.DataFrame,
    cl_drug_meta: pd.DataFrame,
    pt_exp_data: pd.DataFrame,
    pt_sample_meta: pd.DataFrame,
    pt_drug_meta: pd.DataFrame,
) -> None:
    """Creates model inputs."""
    log.info("Generating ScreenDL inputs...")

    cell_root = Path(cfg.inputs.dir) / "cell"
    cell_root.mkdir(exist_ok=True, parents=True)

    generate_and_save_screendl_inputs(
        cfg.inputs.screendl,
        out_dir=cell_root / "ScreenDL",
        cell_meta=cl_sample_meta,
        drug_meta=cl_drug_meta,
        exp_df=cl_exp_data,
    )

    pt_root = Path(cfg.inputs.dir) / "patient"
    pt_root.mkdir(exist_ok=True, parents=True)

    generate_and_save_screendl_inputs(
        cfg.inputs.screendl,
        out_dir=pt_root / "ScreenDL",
        cell_meta=pt_sample_meta,
        drug_meta=pt_drug_meta,
        exp_df=pt_exp_data,
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
    cmp_data, gdsc_data, tcga_data = make_dataset(cfg)

    log.info("Generating model inputs...")
    make_inputs(
        cfg,
        cl_exp_data=cmp_data.exp,
        cl_sample_meta=cmp_data.meta,
        cl_drug_meta=gdsc_data.meta,
        pt_exp_data=tcga_data.exp,
        pt_sample_meta=tcga_data.cell_meta,
        pt_drug_meta=tcga_data.drug_meta,
    )

    log.info("Generating labels...")
    cl_labels, _ = make_labels(cfg, gdsc_data.resp, tcga_data.resp)

    log.info("Generating meta data...")
    cl_sample_meta, *_ = make_meta(
        cfg,
        cl_sample_meta=cmp_data.meta,
        cl_drug_meta=gdsc_data.meta,
        pt_sample_meta=tcga_data.cell_meta,
        pt_drug_meta=tcga_data.drug_meta,
    )

    log.info("Generating folds...")
    make_splits(
        cfg, cl_resp_data=cl_labels, cl_sample_meta=cl_sample_meta.reset_index()
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S,%03d",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cli()
