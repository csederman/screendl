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

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from tensorflow import keras
from tqdm import tqdm
from omegaconf import DictConfig
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.metrics import tf_metrics

from screendl.utils.drug_selectors import KMeansDrugSelector


log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}


@hydra.main(
    version_base=None, config_path="../../conf", config_name="sa_config"
)
def run_sa(cfg: DictConfig) -> float:
    """"""
    # What I should do here is just use importlib
    if not cfg.model.name in PIPELINES:
        raise ValueError("Unsupported model.")

    module_file = PIPELINES[cfg.model.name]
    module_path = f"pipelines.{module_file}"
    module = importlib.import_module(module_path)

    model, train_ds, _, test_ds = module.run_sa_pipeline(cfg)

    # freeze drug and cell subnetworks
    for layer in model.layers:
        if layer.name.startswith("mol") or layer.name.startswith("exp"):
            layer.trainable = False

    params = cfg.screenahead.hyper
    num_drugs = cfg.screenahead.opt.n_drugs

    generator = BatchedResponseGenerator(test_ds, batch_size=params.batch_size)
    drug_selector = KMeansDrugSelector(train_ds)
    base_weights = model.get_weights()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss="mean_squared_error",
        metrics=[tf_metrics.pearson],
    )

    pred_dfs = []
    for cell_id in tqdm(set(test_ds.cell_ids)):
        # restore weights
        model.set_weights(base_weights)

        obs = test_ds.obs[test_ds.obs["cell_id"] == cell_id]

        choices = set(obs["drug_id"])
        if len(choices) < num_drugs:
            # skip cell lines for which there are fewer than num_drugs screened
            continue

        screen_drugs = drug_selector.select(
            num_drugs, choices=choices, random_state=1771
        )

        screen_obs = obs[obs["drug_id"].isin(screen_drugs)]
        holdout_obs = obs[~obs["drug_id"].isin(screen_drugs)]

        screen_seq = generator.flow(
            list(screen_obs["cell_id"]),
            list(screen_obs["drug_id"]),
            targets=list(screen_obs["label"]),
            shuffle=True,
        )

        holdout_cell_ids = list(holdout_obs["cell_id"])
        holdout_drug_ids = list(holdout_obs["drug_id"])
        holdout_targets = list(holdout_obs["label"])
        holdout_seq = generator.flow(holdout_cell_ids, holdout_drug_ids)

        model.fit(screen_seq, epochs=params.epochs, verbose=0)

        preds: np.ndarray = model.predict(holdout_seq, verbose=0)
        pred_dfs.append(
            pd.DataFrame(
                {
                    "cell_id": holdout_cell_ids,
                    "drug_id": holdout_drug_ids,
                    "y_true": holdout_targets,
                    "y_pred": preds.reshape(-1),
                    "split": "holdout",
                }
            )
        )

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions_sa.csv", index=False)


if __name__ == "__main__":
    run_sa()
