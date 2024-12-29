#!/usr/bin/env python
"""Cleans and harmonizes data sources for downstream feature extraction."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import click
import logging

import numpy as np
import pandas as pd
import typing as t

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from screendl.preprocessing import splits, models
from screendl.preprocessing.data import gdsc, cmp, pubchem
from screendl.preprocessing.data import harmonize_cmp_gdsc_data


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def make_dataset(cfg: DictConfig) -> t.Tuple[cmp.CMPData, gdsc.GDSCData]:
    """Cleans and harmonizes the data sources."""
    paths = cfg.dataset.paths
    params = cfg.dataset.params

    log.info("Loading Cell Model Passports data...")
    cmp_data = cmp.load_and_clean_cmp_data(
        exp_path=paths.cell_exp,
        meta_path=paths.cell_info,
        cnv_path=paths.cell_cnv,
        vcf_dir=paths.cell_vcf,
        min_cells_per_cancer_type=params.cmp.min_cells_per_cancer_type,
        required_info_columns=params.cmp.required_info_columns,
        cancer_type_blacklist=params.cmp.cancer_type_blacklist,
    )

    log.info("Loading GDSC data...")
    gdsc_data = gdsc.load_and_clean_gdsc_data(
        resp_path=paths.drug_resp,
        meta_path=paths.drug_info,
        gr_metric=params.gdsc.gr_metric,
        log_transform=params.gdsc.log_transform,
    )

    log.info("Harmonizing GDSC and Cell Model Passports...")
    cmp_data, gdsc_data = harmonize_cmp_gdsc_data(cmp_data, gdsc_data)

    # log transform the TPM values
    cmp_data.exp = np.log2(cmp_data.exp + 1)

    # query PubCHEM annotations
    pchem_ids = list(gdsc_data.meta["pubchem_id"])
    pchem_props = pubchem.fetch_pubchem_properties(pchem_ids, paths.pubchem_cache)
    pchem_props = pchem_props.rename(columns={"CanonicalSMILES": "smiles"})
    pchem_props["CID"] = pchem_props["CID"].astype(str)

    gdsc_data.meta = gdsc_data.meta.merge(
        pchem_props, left_on="pubchem_id", right_on="CID"
    )

    if cfg.dataset.save:
        log.info("Saving dataset...")

        out_dir = Path(cfg.dataset.dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        gdsc_data.meta.to_csv(out_dir / "DrugAnnotations.csv", index=False)
        gdsc_data.resp.to_csv(out_dir / "ScreenDoseResponse.csv", index=False)

        cmp_data.meta.to_csv(out_dir / "CellLineAnnotations.csv", index=False)
        cmp_data.exp.to_csv(out_dir / "OmicsGeneExpressionTPM.csv")
        if cmp_data.cnv is not None:
            cmp_data.cnv.to_csv(out_dir / "OmicsTotalCopyNumber.csv")
        if cmp_data.mut is not None:
            cmp_data.mut.to_csv(out_dir / "OmicsSomaticMutations.csv", index=False)

    return cmp_data, gdsc_data


def make_labels(cfg: DictConfig, resp_data: pd.DataFrame) -> pd.DataFrame:
    """Creates the target labels."""
    out_dir = Path(cfg.inputs.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    col_map = {"model_id": "cell_id", "drug_name": "drug_id"}

    is_valid = ~resp_data["label"].isin(gdsc.INVALID_RESPONSE_VALUES)
    resp_data_valid = resp_data[is_valid].copy()

    resp_data_valid["id"] = range(resp_data_valid.shape[0])
    resp_data_valid = resp_data_valid[["id", *col_map, "label"]].rename(columns=col_map)

    # save the results
    resp_data_valid.to_csv(out_dir / "LabelsLogIC50.csv", index=False)

    return resp_data_valid


def make_splits(cfg: DictConfig, resp_df: pd.DataFrame, cell_meta: pd.DataFrame) -> None:
    """Generates the tumor blind train/test splits."""
    params = cfg.splits.params

    out_root = Path(cfg.splits.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    if "mixed" in cfg.splits.types:
        splits.generate_mixed_splits(
            out_dir=out_root / "mixed",
            label_df=resp_df,
            n_splits=params.n_splits,
            seed=params.seed,
        )

    if "tumor_blind" in cfg.splits.types:
        splits.generate_tumor_blind_splits(
            out_dir=out_root / "tumor_blind",
            label_df=resp_df,
            cell_meta=cell_meta,
            n_splits=params.n_splits,
            seed=params.seed,
        )

    if "tumor_type_blind" in cfg.splits.types:
        splits.generate_tumor_type_blind_splits(
            out_dir=out_root / "tumor_type_blind",
            label_df=resp_df,
            cell_meta=cell_meta,
            seed=params.seed,
        )


def make_meta(
    cfg: DictConfig, cell_meta: pd.DataFrame, drug_meta: pd.DataFrame
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates metadata inputs."""
    out_dir = Path(cfg.inputs.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    cols = ["model_id", "tissue", "cancer_type", "cancer_type_detail"]
    cell_meta = cell_meta[cols].set_index("model_id").rename_axis(index="cell_id")

    cols = ["drug_name", "targets", "target_pathway", "pubchem_id", "smiles"]
    drug_meta = drug_meta[cols].set_index("drug_name").rename_axis(index="drug_id")

    cell_meta.to_csv(out_dir / "MetaCellAnnotations.csv")
    drug_meta.to_csv(out_dir / "MetaDrugAnnotations.csv")

    return cell_meta, drug_meta


def make_inputs(
    cfg: DictConfig,
    cmp_data: cmp.CMPData,
    gdsc_data: gdsc.GDSCData,
) -> None:
    """Creates model inputs."""
    out_root = Path(cfg.inputs.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    if "DeepCDR" in cfg.inputs.include:
        log.info("Generating DeepCDR inputs...")
        models.generate_and_save_deepcdr_inputs(
            cfg.inputs.deepcdr,
            out_dir=out_root / "DeepCDR",
            exp_df=cmp_data.exp,
            mut_df=cmp_data.mut,
            drug_meta=gdsc_data.meta,
        )

    if "DualGCN" in cfg.inputs.include:
        log.info("Generating DualGCN inputs...")
        models.generate_and_save_dualgcn_inputs(
            cfg.inputs.dualgcn,
            out_dir=out_root / "DualGCN",
            exp_df=cmp_data.exp,
            cnv_df=cmp_data.cnv,
            cell_meta=cmp_data.meta,
            drug_meta=gdsc_data.meta,
        )

    if "HiDRA" in cfg.inputs.include:
        log.info("Generating HiDRA inputs...")
        models.generate_and_save_hidra_inputs(
            cfg.inputs.hidra,
            out_dir=out_root / "HiDRA",
            exp_df=cmp_data.exp,
            drug_meta=gdsc_data.meta,
        )

    if "ScreenDL" in cfg.inputs.include:
        log.info("Generating ScreenDL inputs...")
        models.generate_and_save_screendl_inputs(
            cfg.inputs.screendl,
            out_dir=out_root / "ScreenDL",
            cell_meta=cmp_data.meta,
            drug_meta=gdsc_data.meta,
            exp_df=cmp_data.exp,
            cnv_df=cmp_data.cnv,
            mut_df=cmp_data.mut,
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
    cmp_data, gdsc_data = make_dataset(cfg)

    log.info("Generating model inputs...")
    make_inputs(cfg, cmp_data, gdsc_data)

    log.info("Generating labels...")
    labels = make_labels(cfg, gdsc_data.resp)

    log.info("Generating meta data...")
    cell_meta, drug_meta = make_meta(cfg, cmp_data.meta, gdsc_data.meta)

    log.info("Generating folds...")
    make_splits(cfg, labels, cell_meta.reset_index())


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S,%03d",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cli()
