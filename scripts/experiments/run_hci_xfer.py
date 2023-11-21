#!/usr/bin/env python
"""Runs transfer learning experiments on the HCI dataset."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import logging

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tqdm import tqdm

from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.metrics import tf_metrics

from screendl.utils.evaluation import make_pred_df
from screendl.pipelines.basic.screendl import (
    data_loader,
    data_splitter,
    model_builder as base_model_builder,
    model_trainer as base_model_trainer,
)
from screendl.utils.drug_selectors import (
    DrugSelectorBase,
    KMeansDrugSelector,
    PrincipalDrugSelector,
    RandomDrugSelector,
    MeanResponseSelector,
)


if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder


log = logging.getLogger(__name__)


WeightsDict = t.Tuple[str, t.Any]


SELECTORS = {
    "random": RandomDrugSelector,
    "kmeans": KMeansDrugSelector,
    "principal": PrincipalDrugSelector,
    "responsiveness": MeanResponseSelector,
}


def data_preprocessor(
    cfg: DictConfig,
    cell_train_ds: Dataset,
    cell_val_ds: Dataset,
    pdmc_ds: Dataset,
) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Preprocessing pipeline.

    Parameters
    ----------
        cfg:
        cell_train_ds:
        cell_val_ds:
        pdmc_ds:

    Returns
    -------
        A (train, val, test) tuple of processed datasets.
    """

    # normalize the gene expression
    exp_enc: PandasEncoder = cell_train_ds.cell_encoders["exp"]

    train_cell_ids = list(set(cell_train_ds.cell_ids))
    val_cell_ids = list(set(cell_val_ds.cell_ids))
    pdmc_ids = list(set(pdmc_ds.cell_ids))

    x_cell_train = exp_enc.data.loc[train_cell_ids]
    x_cell = exp_enc.data.loc[train_cell_ids + val_cell_ids]
    x_pdmc = exp_enc.data.loc[pdmc_ids]

    # for transfer learning, we normalize independently across domains
    x_cell[:] = StandardScaler().fit(x_cell_train).transform(x_cell)
    x_pdmc[:] = StandardScaler().fit_transform(x_pdmc)

    exp_enc.data = pd.concat([x_cell, x_pdmc])

    # normalize the drug responses
    cell_train_ds, cell_val_ds, _ = normalize_responses(
        cell_train_ds, cell_val_ds, norm_method="grouped"
    )
    pdmc_ds, *_ = normalize_responses(pdmc_ds, norm_method="grouped")

    return cell_train_ds, cell_val_ds, pdmc_ds


def xfer_model_builder(cfg: DictConfig, base_model: keras.Model) -> keras.Model:
    """Builds the transfer learning model."""
    hparams = cfg.xfer.hyper

    xfer_model = base_model
    for layer in xfer_model.layers:
        prefixes = ("mol", "exp", "ont", "mut", "cnv")
        if layer.name.startswith(prefixes):
            layer.trainable = False

    xfer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hparams.learning_rate),
        loss="mean_squared_error",
        metrics=[tf_metrics.pearson],
    )

    return xfer_model


def xfer_model_trainer(
    cfg: DictConfig, xfer_model: keras.Model, pdmc_ds: Dataset
) -> t.Tuple[keras.Model, WeightsDict, pd.DataFrame]:
    """Runs transfer learning loop for each PDMC."""
    hparams = cfg.xfer.hyper
    batch_size = pdmc_ds.size if hparams.batch_size is None else hparams.batch_size

    base_weights = xfer_model.get_weights()
    pdmc_gen = BatchedResponseGenerator(pdmc_ds, batch_size=batch_size)
    pdmc_ids = sorted(list(set(pdmc_ds.cell_ids)))

    pred_dfs = []
    xfer_weights = {}
    for this_pdmc_id in tqdm(pdmc_ids):
        xfer_model.set_weights(base_weights)

        this_prefix = this_pdmc_id[:6]
        other_pdmc_ids = [x for x in pdmc_ids if not x.startswith(this_prefix)]

        this_ds = pdmc_ds.select_cells([this_pdmc_id], name="this")
        this_seq = pdmc_gen.flow_from_dataset(this_ds)

        other_ds = pdmc_ds.select_cells(other_pdmc_ids, name="other")
        other_seq = pdmc_gen.flow_from_dataset(other_ds, shuffle=True, seed=1441)

        base_preds: np.ndarray = xfer_model.predict(this_seq, verbose=0)
        pred_dfs.append(make_pred_df(this_ds, base_preds, model="base"))

        _ = xfer_model.fit(other_seq, epochs=hparams.epochs, verbose=0)
        xfer_weights[this_pdmc_id] = xfer_model.get_weights()

        xfer_preds: np.ndarray = xfer_model.predict(this_seq, verbose=0)
        pred_dfs.append(make_pred_df(this_ds, xfer_preds, model="xfer"))

    xfer_model.set_weights(base_weights)
    pred_df = pd.concat(pred_dfs)

    return xfer_model, xfer_weights, pred_df


def screenahead_model_builder(cfg: DictConfig, model: keras.Model) -> keras.Model:
    """Configures the model for ScreenAhead."""
    hparams = cfg.screenahead.hyper

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hparams.learning_rate),
        loss="mean_squared_error",
        metrics=[tf_metrics.pearson],
    )

    return model


def screenahead_model_trainer(
    cfg: DictConfig,
    sa_model: keras.Model,
    cell_ds: Dataset,
    pdmc_ds: Dataset,
    xfer_weights: WeightsDict,
) -> t.Tuple[keras.Model, WeightsDict, pd.DataFrame]:
    """Runs the ScreenAhead training loop."""
    opts = cfg.screenahead.opt
    hparams = cfg.screenahead.hyper

    batch_size = pdmc_ds.size if hparams.batch_size is None else hparams.batch_size

    selector_cls: DrugSelectorBase = SELECTORS[opts.selector]
    selector = selector_cls(cell_ds, seed=opts.seed)

    base_weights = sa_model.get_weights()
    pdmc_gen = BatchedResponseGenerator(pdmc_ds, batch_size=batch_size)
    pdmc_ids = sorted(list(set(pdmc_ds.cell_ids)))

    sa_pred_dfs = []
    sa_weights = {}
    for this_pdmc_id in tqdm(pdmc_ids):
        sa_model.set_weights(xfer_weights[this_pdmc_id])

        this_ds = pdmc_ds.select_cells([this_pdmc_id], name="this")

        choices = set(this_ds.drug_ids)
        screen_drugs = selector.select(opts.n_drugs, choices=choices)
        holdout_drugs = choices.difference(screen_drugs)

        this_holdout_ds = this_ds.select_drugs(holdout_drugs)
        this_holdout_seq = pdmc_gen.flow_from_dataset(this_holdout_ds)

        this_screen_ds = this_ds.select_drugs(screen_drugs)
        this_screen_seq = pdmc_gen.flow_from_dataset(
            this_screen_ds, shuffle=True, seed=1441
        )

        _ = sa_model.fit(this_screen_seq, epochs=hparams.epochs, verbose=0)
        sa_weights[this_pdmc_id] = sa_model.get_weights()

        sa_preds: np.ndarray = sa_model.predict(this_holdout_seq, verbose=0)
        sa_pred_dfs.append(make_pred_df(this_holdout_ds, sa_preds, model="screen"))

    sa_model.set_weights(base_weights)
    sa_pred_df = pd.concat(sa_pred_dfs)

    return sa_model, sa_weights, sa_pred_df


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
    dataset = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    cell_train_ds, cell_val_ds, pdmc_ds = data_splitter(cfg, dataset=dataset)

    log.info(f"Preprocessing {dataset_name}...")
    cell_train_ds, cell_val_ds, pdmc_ds = data_preprocessor(
        cfg,
        cell_train_ds=cell_train_ds,
        cell_val_ds=cell_val_ds,
        pdmc_ds=pdmc_ds,
    )

    log.info(f"Building {model_name}...")
    base_model = base_model_builder(cfg, train_dataset=cell_train_ds)

    log.info(f"Pretraining {model_name}...")
    base_model = base_model_trainer(
        cfg,
        model=base_model,
        train_dataset=cell_train_ds,
        val_dataset=cell_val_ds,
    )

    log.info(f"Configuring {model_name} for transfer learning...")
    xfer_model = xfer_model_builder(cfg, base_model=base_model)

    log.info(f"Running transfer learning loop...")
    xfer_model, xfer_weights, xfer_pred_df = xfer_model_trainer(
        cfg, xfer_model=xfer_model, pdmc_ds=pdmc_ds
    )

    log.info(f"Configuring {model_name} for ScreenAhead...")
    sa_model = screenahead_model_builder(cfg, model=xfer_model)

    log.info(f"Running screenahead loop...")
    sa_model, _, sa_pred_df = screenahead_model_trainer(
        cfg,
        sa_model=sa_model,
        cell_ds=cell_train_ds,
        pdmc_ds=pdmc_ds,
        xfer_weights=xfer_weights,
    )

    pred_df = pd.concat([xfer_pred_df, sa_pred_df])
    pred_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    run_experiment()
