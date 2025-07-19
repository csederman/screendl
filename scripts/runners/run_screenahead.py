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

from omegaconf import DictConfig
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.datasets import Dataset

from screendl.utils import model_utils
from screendl.utils import evaluation as eval_utils
from screendl.utils.drug_selectors import SELECTORS


log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead",
)
def run_sa(cfg: DictConfig) -> float:
    """"""
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

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

    if cfg.screenahead.io.permute_exp:
        # randomly scramble the gene expression data
        test_ds.cell_encoders["exp"].data = test_ds.cell_encoders["exp"].data.apply(
            np.random.permutation
        )

    pred_dfs = []
    for cell_id in set(test_ds.cell_ids):
        cell_ds: Dataset = test_ds.select_cells([cell_id])
        cell_gen = BatchedResponseGenerator(cell_ds, batch_size=cell_ds.n_drugs)
        choices = set(cell_ds.drug_ids)

        # require at least 1 drug in the holdout set
        if len(choices) <= opts.n_drugs:
            print(cell_id)
            msg = f"Skipping ScreenAhead for {cell_id} (fewer than {opts.n_drugs} drugs)"
            log.warning(msg)
            continue

        selector = SELECTORS[opts.selector](
            selection_ds, seed=opts.seed, na_threshold=None
        )
        screen_drugs = selector.select(opts.n_drugs, choices=choices)
        screen_ds = cell_ds.select_drugs(screen_drugs)

        sa_model = model_utils.fit_screenahead_model(
            base_model,
            screen_ds,
            batch_size=hparams.batch_size,
            epochs=hparams.epochs,
            learning_rate=hparams.learning_rate,
            frozen_layer_prefixes=("mol", "exp", "ont", "mut", "cnv", "mr"),
            training=False,
        )

        pred_df = eval_utils.make_pred_df(
            cell_ds,
            sa_model.predict(cell_gen.flow_from_dataset(cell_ds), verbose=0),
            split_group="test",
            model="ScreenDL-SA",
            split_id=cfg.dataset.split.id,
            split_type=cfg.dataset.split.name,
            norm_method=cfg.dataset.preprocess.norm,
        )
        pred_dfs.append(
            pred_df.assign(was_screened=lambda df: df["drug_id"].isin(screen_drugs))
        )

        # restore the weights before each iteration
        base_model.set_weights(base_weights)
        K.clear_session()

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions_sa.csv", index=False)


if __name__ == "__main__":
    run_sa()
    K.clear_session()
