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
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]
import typing as t

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from omegaconf import DictConfig
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.datasets import Dataset

from screendl.utils import model_utils
from screendl.utils import evaluation as eval_utils
from screendl.utils.drug_selectors import SELECTORS


log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}


# def generate_all_predictions(
#     cfg: DictConfig,
#     model: keras.Model,
#     datasets: t.Iterable[Dataset],
#     cell_id: str,
# ) -> pd.DataFrame:
#     """Generate predictions for all observations."""
#     pred_dfs = []
#     for ds in datasets:
#         gen = BatchedResponseGenerator(ds, cfg.model.hyper.batch_size)
#         seq = gen.flow_from_dataset(ds)
#         preds: np.ndarray = model.predict(seq, verbose=0)
#         pred_dfs.append(
#             # make_pred_df(
#             #     ds,
#             #     preds,
#             #     split_group="test",
#             #     model="ScreenDL-SA",
#             #     split_id=cfg.dataset.split.id,
#             #     split_type=cfg.dataset.split.name,
#             #     norm_method=cfg.dataset.preprocess.norm,
#             # )
#             pd.DataFrame(
#                 {
#                     "cell_id": ds.cell_ids,
#                     "drug_id": ds.drug_ids,
#                     "y_true": ds.labels,
#                     "y_pred": preds.reshape(-1),
#                     "split": ds.name,
#                 }
#             )
#         )

#     pred_df = pd.concat(pred_dfs)
#     pred_df["fold"] = cfg.dataset.split.id
#     pred_df["sa_fold"] = cell_id

#     return pred_df


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

    base_model, _, ds_dict = module.run_pipeline(cfg)

    opts = cfg.screenahead.opt
    hparams = cfg.screenahead.hyper

    # NOTE: some drug selection algorithms require un-normalized responses
    train_cell_ids = set(ds_dict["train"].cell_ids)
    selection_ds = ds_dict["full"].select_cells(train_cell_ids)

    test_ds: Dataset = ds_dict["test"]
    test_gen = BatchedResponseGenerator(test_ds, 256)

    base_weights = base_model.get_weights()

    pred_dfs = []
    for cell_id in set(test_ds.cell_ids):
        cell_ds: Dataset = test_ds.select_cells([cell_id])
        choices = set(cell_ds.drug_ids)

        # require at least 1 drug in the holdout set
        if len(choices) <= opts.n_drugs:
            log.warning(
                f"Skipping ScreenAhead for {cell_id} "
                f"(fewer then {opts.n_drugs} drugs screened)."
            )
            continue

        try:
            # try the selection with default na_threshold
            selector = SELECTORS[opts.selector](selection_ds, seed=opts.seed)
            screen_drugs = selector.select(opts.n_drugs, choices=choices)
            screen_ds = cell_ds.select_drugs(screen_drugs)

        except Exception as e:
            print(e)

            selector = SELECTORS[opts.selector](
                selection_ds, seed=opts.seed, na_threshold=0.0
            )
            screen_drugs = selector.select(opts.n_drugs, choices=choices)
            screen_ds = cell_ds.select_drugs(screen_drugs)

        holdout_ds = cell_ds.select_drugs(choices.difference(screen_drugs))
        holdout_seq = test_gen.flow_from_dataset(holdout_ds)

        sa_model = model_utils.fit_screenahead_model(
            base_model,
            screen_ds,
            batch_size=hparams.batch_size,
            epochs=hparams.epochs,
            learning_rate=hparams.learning_rate,
            frozen_layer_prefixes=("mol", "exp", "ont", "mut", "cnv", "mr"),
            training=False,
        )

        preds = sa_model.predict(holdout_seq, verbose=0)
        pred_dfs.append(
            eval_utils.make_pred_df(
                holdout_ds,
                preds,
                split_group="test",
                model="ScreenDL-SA",
                split_id=cfg.dataset.split.id,
                split_type=cfg.dataset.split.name,
                norm_method=cfg.dataset.preprocess.norm,
            )
        )

        # restore the weights before each iteration
        base_model.set_weights(base_weights)
        K.clear_session()

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions_sa.csv", index=False)


if __name__ == "__main__":
    run_sa()
    K.clear_session()
