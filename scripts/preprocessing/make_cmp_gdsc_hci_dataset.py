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
import typing as t

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from screendl.preprocessing import splits, models
from screendl.preprocessing.data import gdsc, cmp, hci, pubchem
from screendl.preprocessing.data import (
    harmonize_cmp_gdsc_data,
    harmonize_cmp_gdsc_hci_data,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def make_dataset(cfg: DictConfig) -> t.Tuple[cmp.CMPData, gdsc.GDSCData, hci.HCIData]:
    """Cleans and harmonizes the data sources."""
    paths = cfg.dataset.paths
    params = cfg.dataset.params

    log.info("Loading Cell Model Passports data...")
    cmp_data = cmp.load_and_clean_cmp_data(
        exp_path=paths.cmp.exp,
        meta_path=paths.cmp.meta,
        vcf_dir=paths.cmp.vcf,
        min_cells_per_cancer_type=params.cmp.min_cells_per_cancer_type,
        required_info_columns=params.cmp.required_info_columns,
        cancer_type_blacklist=params.cmp.cancer_type_blacklist,
    )

    log.info("Loading GDSC response data...")
    gdsc_data = gdsc.load_and_clean_gdsc_data(paths.gdsc.resp, paths.gdsc.meta)

    log.info("Harmonizing GDSC and Cell Model Passports...")
    cmp_data, gdsc_data = harmonize_cmp_gdsc_data(cmp_data, gdsc_data)

    log.info("Loading HCI PDMC data...")
    hci_data = hci.load_and_clean_hci_data(
        exp_path=paths.hci.exp,
        resp_path=paths.hci.resp,
        pdmc_meta_path=paths.hci.pdmc_meta,
        drug_meta_path=paths.hci.drug_meta,
        mut_path=paths.hci.mut,
        model_types=params.hci.pdmc_model_types,
        min_samples_per_drug=params.hci.min_samples_per_drug,
    )

    log.info("Harmonizing GDSC, Cell Model Passports, and HCI data...")
    cmp_data, hci_data, gdsc_data = harmonize_cmp_gdsc_hci_data(
        cmp_data,
        hci_data,
        gdsc_data,
        include_all_hci_drugs=params.hci.include_all_hci_drugs,
    )

    # query PubCHEM annotations
    pchem_ids = set(gdsc_data.meta["pubchem_id"])
    pchem_ids = list(pchem_ids.union(hci_data.drug_meta["pubchem_id"]))
    pchem_props = pubchem.fetch_pubchem_properties(pchem_ids, paths.pubchem.cache)
    pchem_props = pchem_props.rename(columns={"CanonicalSMILES": "smiles"})
    pchem_props["CID"] = pchem_props["CID"].astype(str)

    gdsc_data.meta = gdsc_data.meta.merge(
        pchem_props, left_on="pubchem_id", right_on="CID"
    )
    hci_data.drug_meta = hci_data.drug_meta.merge(
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

        cmp_data.meta.to_csv(cell_subdir / "CellLineAnnotations.csv", index=False)
        cmp_data.exp.to_csv(cell_subdir / "OmicsGeneExpressionTPM.csv")
        if cmp_data.mut is not None:
            cmp_data.mut.to_csv(cell_subdir / "OmicsSomaticMutations.csv", index=False)

        # save the pdmc data
        pdmc_subdir = out_root / "pdmc"
        pdmc_subdir.mkdir(exist_ok=True, parents=True)

        hci_data.drug_meta.to_csv(pdmc_subdir / "DrugAnnotations.csv", index=False)
        hci_data.cell_meta.to_csv(pdmc_subdir / "CellLineAnnotations.csv", index=False)
        hci_data.resp.to_csv(pdmc_subdir / "ScreenDoseResponse.csv", index=False)
        hci_data.exp.to_csv(pdmc_subdir / "OmicsGeneExpressionTPM.csv")
        if hci_data.mut is not None:
            hci_data.mut.to_csv(pdmc_subdir / "OmicsSomaticMutations.csv", index=False)

    return cmp_data, gdsc_data, hci_data


def make_labels(cfg: DictConfig, resp_data: pd.DataFrame) -> pd.DataFrame:
    """Creates the target labels."""
    out_dir = Path(cfg.inputs.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    col_map = {"model_id": "cell_id", "drug_name": "drug_id", "ln_ic50": "label"}
    resp_data_valid = resp_data[
        ~resp_data["ln_ic50"].isin(gdsc.INVALID_RESPONSE_VALUES)
    ]
    resp_data_valid["id"] = range(resp_data_valid.shape[0])
    resp_data_valid = resp_data_valid[["id", *col_map]].rename(columns=col_map)

    # save the results
    resp_data_valid.to_csv(out_dir / "LabelsLogIC50.csv", index=False)

    return resp_data_valid


def make_splits(
    cfg: DictConfig, resp_df: pd.DataFrame, sample_meta: pd.DataFrame
) -> None:
    """Creates train/val/test splits."""
    params = cfg.splits.params

    out_dir = Path(cfg.splits.dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    cell_sample_meta = sample_meta[sample_meta["domain"] == "CELL"]
    pdmc_sample_ids = sample_meta[sample_meta["domain"] == "PDMC"]["cell_id"]

    cell_sample_ids = cell_sample_meta["cell_id"]
    cell_sample_groups = cell_sample_meta["cancer_type"]

    split_gen = splits.kfold_split_generator(
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
    cfg: DictConfig, cell_meta: pd.DataFrame, drug_meta: pd.DataFrame
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates metadata inputs."""
    out_root = Path(cfg.inputs.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    # generate metadata inputs
    cols = ["model_id", "cancer_type", "model_type", "domain"]
    cell_meta = cell_meta[cols].set_index("model_id").rename_axis(index="cell_id")

    cols = ["drug_name", "pubchem_id", "smiles"]
    drug_meta = drug_meta[cols].set_index("drug_name").rename_axis(index="drug_id")

    cell_meta.to_csv(out_root / "MetaSampleAnnotations.csv")
    drug_meta.to_csv(out_root / "MetaDrugAnnotations.csv")

    return cell_meta, drug_meta


def make_inputs(
    cfg: DictConfig,
    exp_data: pd.DataFrame,
    cell_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
    mut_data: pd.DataFrame | None = None,
) -> None:
    """Creates model inputs."""
    out_root = Path(cfg.inputs.dir)
    out_root.mkdir(exist_ok=True, parents=True)

    if "DeepCDR" in cfg.inputs.include:
        if mut_data is None:
            log.warning("Skipping DeepCDR inputs (no mutation data provided)")
        else:
            log.info("Generating DeepCDR inputs...")
            models.generate_and_save_deepcdr_inputs(
                cfg.inputs.deepcdr,
                out_dir=out_root / "DeepCDR",
                exp_df=exp_data,
                mut_df=mut_data,
                drug_meta=drug_meta,
            )

    if "HiDRA" in cfg.inputs.include:
        log.info("Generating HiDRA inputs...")
        models.generate_and_save_hidra_inputs(
            cfg.inputs.hidra,
            out_dir=out_root / "HiDRA",
            exp_df=exp_data,
            drug_meta=drug_meta,
        )

    if "ScreenDL" in cfg.inputs.include:
        log.info("Generating ScreenDL inputs...")
        models.generate_and_save_screendl_inputs(
            cfg.inputs.screendl,
            out_dir=out_root / "ScreenDL",
            cell_meta=cell_meta,
            drug_meta=drug_meta,
            exp_df=exp_data,
            mut_df=mut_data,
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
    cmp_data, gdsc_data, hci_data = make_dataset(cfg)

    # combine the data sources
    resp_data = pd.concat([gdsc_data.resp, hci_data.resp])

    drug_meta = pd.concat([gdsc_data.meta, hci_data.drug_meta]).drop_duplicates()
    cell_meta = pd.concat([cmp_data.meta, hci_data.cell_meta])

    exp_data = pd.concat([cmp_data.exp, hci_data.exp])
    mut_data = None
    if cmp_data.mut is not None and hci_data.mut is not None:
        mut_data = pd.concat([cmp_data.mut, hci_data.mut])

    log.info("Generating model inputs...")
    make_inputs(cfg, exp_data, cell_meta, drug_meta, mut_data)

    log.info("Generating labels...")
    labels = make_labels(cfg, resp_data)

    log.info("Generating meta data...")
    cell_meta, drug_meta = make_meta(cfg, cell_meta, drug_meta)

    log.info("Generating folds...")
    make_splits(cfg, labels, cell_meta.reset_index())


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S,%03d",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cli()
