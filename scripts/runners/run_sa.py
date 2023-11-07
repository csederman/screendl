#!/usr/bin/env python
"""Run ScreenAhead experiments."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import importlib
import logging
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import typing as t

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from tensorflow import keras
from omegaconf import DictConfig
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.metrics import tf_metrics
from cdrpy.data.datasets import Dataset

from screendl.utils.drug_selectors import (
    DrugSelectorBase,
    KMeansDrugSelector,
    PrincipalDrugSelector,
    RandomDrugSelector,
    MeanResponseSelector,
)


log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}
SELECTORS = {
    "random": RandomDrugSelector,
    "kmeans": KMeansDrugSelector,
    "principal": PrincipalDrugSelector,
    "responsiveness": MeanResponseSelector,
}


def generate_all_predictions(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[Dataset],
    cell_id: str,
) -> pd.DataFrame:
    """Generate predictions for all observations."""
    pred_dfs = []
    for ds in datasets:
        gen = BatchedResponseGenerator(ds, cfg.model.hyper.batch_size)
        seq = gen.flow_from_dataset(ds)
        preds: np.ndarray = model.predict(seq, verbose=0)
        pred_dfs.append(
            pd.DataFrame(
                {
                    "cell_id": ds.cell_ids,
                    "drug_id": ds.drug_ids,
                    "y_true": ds.labels,
                    "y_pred": preds.reshape(-1),
                    "split": ds.name,
                }
            )
        )

    pred_df = pd.concat(pred_dfs)
    pred_df["fold"] = cfg.dataset.split.id
    pred_df["sa_fold"] = cell_id

    return pred_df


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="sa_config",
)
def run_sa(cfg: DictConfig) -> float:
    """"""
    # What I should do here is just use importlib
    if not cfg.model.name in PIPELINES:
        raise ValueError("Unsupported model.")

    module_file = PIPELINES[cfg.model.name]
    module_name = f"screendl.pipelines.basic.{module_file}"
    module = importlib.import_module(module_name)

    model, full_ds, train_ds, val_ds, test_ds = module.run_sa_pipeline(cfg)

    # freeze drug and cell subnetworks
    for layer in model.layers:
        if not layer.name.startswith("shared"):
            layer.trainable = False

    params = cfg.screenahead.hyper
    num_drugs = cfg.screenahead.opt.n_drugs

    # NOTE: some drug selection algorithms require un-normalized responses
    selection_ds = full_ds.select_cells(set(train_ds.cell_ids))
    selector_cls: DrugSelectorBase = SELECTORS[cfg.screenahead.opt.selector]
    selector = selector_cls(selection_ds, seed=cfg.screenahead.opt.seed)

    generator = BatchedResponseGenerator(test_ds, batch_size=params.batch_size)

    base_weights = model.get_weights()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss="mean_squared_error",
        metrics=[tf_metrics.pearson],
    )

    pred_dfs = []
    for cell_id in set(test_ds.cell_ids):
        # restor the weights before each iteration
        model.set_weights(base_weights)

        cell_ds: Dataset = test_ds.select_cells([cell_id])
        choices = set(cell_ds.drug_ids)

        # require at least 20 drugs in the holdout set
        if len(choices) < num_drugs + 20:
            log.warning(
                f"Skipping ScreenAhead for {cell_id} "
                f"(fewer then {num_drugs} drugs screened)."
            )
            continue

        screen_drugs = selector.select(num_drugs, choices=choices)
        screen_ds = cell_ds.select_drugs(screen_drugs)
        holdout_ds = cell_ds.select_drugs(choices.difference(screen_drugs))

        screen_seq = generator.flow_from_dataset(screen_ds, shuffle=True)
        holdout_seq = generator.flow_from_dataset(holdout_ds)

        model.fit(screen_seq, epochs=params.epochs, verbose=0)

        if cfg.screenahead.io.predict_all:
            pred_df = generate_all_predictions(
                cfg, model, [train_ds, val_ds, test_ds], cell_id=cell_id
            )
            pred_dfs.append(pred_df)
        else:
            preds: np.ndarray = model.predict(holdout_seq, verbose=0)
            pred_dfs.append(
                pd.DataFrame(
                    {
                        "cell_id": holdout_ds.cell_ids,
                        "drug_id": holdout_ds.drug_ids,
                        "y_true": holdout_ds.labels,
                        "y_pred": preds.reshape(-1),
                        "fold": cfg.dataset.split.id,
                        "sa_fold": cell_id,
                        "split": "test",
                    }
                )
            )

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions_sa.csv", index=False)


if __name__ == "__main__":
    run_sa()
