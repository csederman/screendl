#!/usr/bin/env python
"""Runs transfer learning experiments on the TCGA dataset."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import logging
import pickle

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tqdm import tqdm

from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.metrics import tf_metrics

from screendl import model as screendl
from screendl.utils.evaluation import make_pred_df

if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder


log = logging.getLogger(__name__)


def data_loader(cfg: DictConfig) -> t.Tuple[Dataset, Dataset]:
    """Loads the input datasets."""

    pt_paths = cfg.dataset.sources.patient
    cell_paths = cfg.dataset.sources.cell

    pt_drug_encoders = screendl.load_drug_features(pt_paths.screendl.mol)
    pt_sample_encoders = screendl.load_cell_features(pt_paths.screendl.exp)

    cell_drug_encoders = screendl.load_drug_features(cell_paths.screendl.mol)
    cell_sample_encoders = screendl.load_cell_features(cell_paths.screendl.exp)

    pt_ds = Dataset.from_csv(
        pt_paths.labels,
        name=cfg.dataset.name,
        cell_encoders=pt_sample_encoders,
        drug_encoders=pt_drug_encoders,
    )

    cell_ds = Dataset.from_csv(
        cell_paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_sample_encoders,
        drug_encoders=cell_drug_encoders,
    )

    return cell_ds, pt_ds


def data_splitter(
    cfg: DictConfig, cell_ds: Dataset, pt_ds: Dataset
) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Splits the dataset into train/validation/test sets."""
    split_id = cfg.dataset.split.id
    split_dir = cfg.dataset.split.dir
    split_path = Path(split_dir) / f"fold_{split_id}.pkl"

    with open(split_path, "rb") as fh:
        split = pickle.load(fh)

    return (
        cell_ds.select(split["train"], name="train"),
        cell_ds.select(split["val"], name="val"),
        pt_ds.select(pt_ds.obs["id"], name="test"),
    )


def data_preprocessor(
    cfg: DictConfig,
    train_cell_ds: Dataset,
    val_cell_ds: Dataset,
    test_pt_ds: Dataset,
) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Preprocessing pipeline."""

    # normalize the gene expression data
    cell_exp_enc: PandasEncoder = train_cell_ds.cell_encoders["exp"]


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="xfer_pdmc_config",
)
def run_experiment(cfg: DictConfig) -> None:
    """Runs the HCI PDMC training and evaluation pipeline."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    cell_ds, pt_ds = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    train_cell_ds, val_cell_ds, test_pt_ds = data_splitter(cfg, cell_ds, pt_ds)

    # log.info(f"Preprocessing {dataset_name}...")
    # cell_train_ds, cell_val_ds, pdmc_ds = data_preprocessor(
    #     cfg,
    #     cell_train_ds=cell_train_ds,
    #     cell_val_ds=cell_val_ds,
    #     pdmc_ds=pdmc_ds,
    # )

    # log.info(f"Building {model_name}...")
    # base_model = base_model_builder(cfg, train_dataset=cell_train_ds)

    # log.info(f"Pretraining {model_name}...")
    # base_model = base_model_trainer(
    #     cfg,
    #     model=base_model,
    #     train_dataset=cell_train_ds,
    #     val_dataset=cell_val_ds,
    # )

    # log.info(f"Configuring {model_name} for transfer learning...")
    # xfer_model = xfer_model_builder(cfg, base_model=base_model)

    # log.info(f"Running transfer learning loop...")
    # xfer_model, xfer_weights, xfer_pred_df = xfer_model_trainer(
    #     cfg, xfer_model=xfer_model, pdmc_ds=pdmc_ds
    # )

    # log.info(f"Configuring {model_name} for ScreenAhead...")
    # sa_model = screenahead_model_builder(cfg, model=xfer_model)

    # log.info(f"Running screenahead loop...")
    # sa_model, _, sa_pred_df = screenahead_model_trainer(
    #     cfg,
    #     sa_model=sa_model,
    #     cell_ds=cell_train_ds,
    #     pdmc_ds=pdmc_ds,
    #     xfer_weights=xfer_weights,
    # )

    # pred_df = pd.concat([xfer_pred_df, sa_pred_df])
    # pred_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    run_experiment()
