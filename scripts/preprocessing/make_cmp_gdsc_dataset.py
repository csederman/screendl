#!/usr/bin/env python
"""Cleans and harmonizes data sources for downstream feature extraction."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import click
import logging

import pandas as pd
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from screendl.preprocessing.splits import (
    generate_mixed_splits,
    generate_tumor_blind_splits,
    generate_tumor_type_blind_splits,
)
from screendl.preprocessing.data import (
    load_cmp_data,
    load_gdsc_data,
    harmonize_cmp_data,
    harmonize_gdsc_data,
    harmonize_cmp_gdsc_data,
    fetch_pubchem_properties,
)
from screendl.preprocessing.models import (
    generate_and_save_deepcdr_inputs,
    generate_and_save_dualgcn_inputs,
    generate_and_save_hidra_inputs,
    generate_and_save_screendl_inputs,
)

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
]:
    """Cleans and harmonizes the data sources."""
    paths = cfg.dataset.paths
    params = cfg.dataset.params

    log.info("Loading Cell Model Passports data...")
    exp_df, cell_meta, cnv_df, mut_df = load_cmp_data(
        exp_path=paths.cell_exp,
        meta_path=paths.cell_info,
        cnv_path=paths.cell_cnv,
        vcf_dir=paths.cell_vcf,
    )

    log.info("Harmonizing Cell Model Passports data...")
    exp_df, cell_meta, cnv_df, mut_df = harmonize_cmp_data(
        exp_df=exp_df,
        meta_df=cell_meta,
        cnv_df=cnv_df,
        mut_df=mut_df,
        min_cells_per_cancer_type=params.min_cells_per_cancer_type,
        required_info_columns=params.required_info_columns,
        cancer_type_blacklist=params.cancer_type_blacklist,
    )

    log.info("Loading GDSC response data...")
    resp_df, drug_meta = load_gdsc_data(
        resp_path=paths.drug_resp, meta_path=paths.drug_info
    )

    log.info("Harmonizing GDSC response data...")
    resp_df, drug_meta = harmonize_gdsc_data(resp_df=resp_df, meta_df=drug_meta)

    log.info("Harmonizing GDSC and Cell Model Passports...")
    (
        exp_df,
        resp_df,
        cell_meta,
        drug_meta,
        cnv_df,
        mut_df,
    ) = harmonize_cmp_gdsc_data(
        exp_df=exp_df,
        resp_df=resp_df,
        cell_meta=cell_meta,
        drug_meta=drug_meta,
        cnv_df=cnv_df,
        mut_df=mut_df,
    )

    # query PubCHEM annotations
    pubchem_cids = list(drug_meta["pubchem_id"])
    pubchem_annots = fetch_pubchem_properties(pubchem_cids)
    pubchem_annots["CID"] = pubchem_annots["CID"].astype(str)
    pubchem_annots = pubchem_annots.rename(columns={"CanonicalSMILES": "smiles"})

    # merge in the PubCHEM annotations
    drug_meta = drug_meta.merge(pubchem_annots, left_on="pubchem_id", right_on="CID")

    if cfg.dataset.save:
        log.info("Saving dataset...")

        out_dir = Path(cfg.dataset.dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        exp_df.to_csv(out_dir / "OmicsGeneExpressionTPM.csv")
        resp_df.to_csv(out_dir / "ScreenDoseResponse.csv", index=False)
        cell_meta.to_csv(out_dir / "CellLineAnnotations.csv", index=False)
        drug_meta.to_csv(out_dir / "DrugAnnotations.csv", index=False)
        cnv_df.to_csv(out_dir / "OmicsTotalCopyNumber.csv")
        mut_df.to_csv(out_dir / "OmicsSomaticMutations.csv", index=False)

    return exp_df, cnv_df, mut_df, cell_meta, resp_df, drug_meta


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
    cfg: DictConfig, resp_df: pd.DataFrame, cell_meta: pd.DataFrame
) -> None:
    """Generates the tumor blind train/test splits."""
    params = cfg.splits.params

    out_root = Path(cfg.splits.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    if "mixed" in cfg.splits.types:
        generate_mixed_splits(
            out_dir=out_root / "mixed",
            label_df=resp_df,
            n_splits=params.n_splits,
            seed=params.seed,
        )

    if "tumor_blind" in cfg.splits.types:
        generate_tumor_blind_splits(
            out_dir=out_root / "tumor_blind",
            label_df=resp_df,
            cell_meta=cell_meta,
            n_splits=params.n_splits,
            seed=params.seed,
        )

    if "tumor_type_blind" in cfg.splits.types:
        generate_tumor_type_blind_splits(
            out_dir=out_root / "tumor_type_blind",
            label_df=resp_df,
            cell_meta=cell_meta,
            seed=params.seed,
        )


def make_meta(
    cfg: DictConfig, cell_meta: pd.DataFrame, drug_meta: pd.DataFrame
) -> None:
    """Creates metadata inputs."""
    out_root = Path(cfg.inputs.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    # generate metadata inputs
    cols = ["model_id", "tissue", "cancer_type", "cancer_type_detail"]
    cell_meta_feat = cell_meta[cols].set_index("model_id")
    cell_meta_feat = cell_meta_feat.rename_axis(index="cell_id")

    cols = ["drug_name", "targets", "target_pathway", "pubchem_id", "smiles"]
    drug_meta_feat = drug_meta[cols].set_index("drug_name")
    drug_meta_feat = drug_meta_feat.rename_axis(index="drug_id")

    cell_meta_feat.to_csv(out_root / "MetaCellAnnotations.csv")
    drug_meta_feat.to_csv(out_root / "MetaDrugAnnotations.csv")


def make_inputs(
    cfg: DictConfig,
    exp_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    cell_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
) -> None:
    """Creates model inputs."""
    out_root = Path(cfg.inputs.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    if "DeepCDR" in cfg.inputs.include:
        log.info("Generating DeepCDR inputs...")
        generate_and_save_deepcdr_inputs(
            cfg.inputs.deepcdr,
            out_dir=out_root / "DeepCDR",
            exp_df=exp_df,
            mut_df=mut_df,
            drug_meta=drug_meta,
        )

    if "DualGCN" in cfg.inputs.include:
        log.info("Generating DualGCN inputs...")
        generate_and_save_dualgcn_inputs(
            cfg.inputs.dualgcn,
            out_dir=out_root / "DualGCN",
            exp_df=exp_df,
            cnv_df=cnv_df,
            cell_meta=cell_meta,
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
            cell_meta=cell_meta,
            drug_meta=drug_meta,
            exp_df=exp_df,
            cnv_df=cnv_df,
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
    exp_df, cnv_df, mut_df, cell_meta, resp_df, drug_meta = make_dataset(cfg)

    log.info("Generating labels...")
    resp_df = make_labels(cfg, resp_df)

    log.info("Generating folds...")
    make_splits(cfg, resp_df, cell_meta)

    log.info("Generating meta data...")
    make_meta(cfg, cell_meta, drug_meta)

    log.info("Generating model inputs...")
    make_inputs(cfg, exp_df, cnv_df, mut_df, cell_meta, drug_meta)


if __name__ == "__main__":
    cli()
