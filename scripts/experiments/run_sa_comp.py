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
from tensorflow import keras
from tqdm import tqdm

from cdrpy.mapper import BatchedResponseGenerator

from screendl.utils.evaluation import make_pred_df
from screendl.pipelines.basic.screendl import run_pipeline
from screendl.utils.drug_selectors import (
    DrugSelectorType,
    AgglomerativeDrugSelector,
    KMeansDrugSelector,
    PrincipalDrugSelector,
    RandomDrugSelector,
)


log = logging.getLogger(__name__)


SELECTORS: t.Dict[str, DrugSelectorType] = {
    "agglomerative": AgglomerativeDrugSelector,
    "kmeans": KMeansDrugSelector,
    "principal": PrincipalDrugSelector,
    "random": RandomDrugSelector,
}


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="sa_comp_config",
)
def run_experiment(cfg: DictConfig) -> None:
    """Runs ScreenAhead comparative analysis."""

    # run the training pipeline
    model, _, ds_dict = run_pipeline(cfg)

    opts = cfg.screenahead.opt
    hparams = cfg.screenahead.hyper

    # configure the model for ScreenAhead
    layer_prefixes = ("mol", "exp", "ont", "mut", "cnv")
    for layer in model.layers:
        if layer.name.startswith(layer_prefixes):
            layer.trainable = False

    base_weights = model.get_weights()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hparams.learning_rate),
        loss="mean_squared_error",
    )

    train_cell_ids = set(ds_dict["train"].cell_ids)
    selection_ds = ds_dict["full"].select_cells(train_cell_ids)

    test_ds = ds_dict["test"]
    test_gen = BatchedResponseGenerator(test_ds, batch_size=hparams.batch_size)
    test_cell_ids = sorted(list(set(test_ds.cell_ids)))

    pred_dfs = []
    for selector_name in opts.selectors:
        selector = SELECTORS[selector_name](selection_ds, seed=opts.seed)

        for n_drugs in opts.n_drugs:
            param_dict = {
                "model": "ScreenDL-SA",
                "split_id": cfg.dataset.split.id,
                "split_type": cfg.dataset.split.name,
                "norm_method": cfg.dataset.preprocess.norm,
                "selector_type": selector_name,
                "n_drugs": n_drugs,
            }
            skipped_cells = []

            for cell_id in tqdm(test_cell_ids, desc=f"{selector_name}({n_drugs})"):
                # restore model weights
                model.set_weights(base_weights)

                cell_ds = test_ds.select_cells([cell_id])
                drug_choices = set(cell_ds.drug_ids)

                if len(drug_choices) < n_drugs + 5:
                    skipped_cells.append(cell_id)
                    continue

                screen_drugs = selector.select(n_drugs, choices=drug_choices)
                eval_drugs = drug_choices.difference(screen_drugs)

                screen_ds = cell_ds.select_drugs(screen_drugs)
                eval_ds = cell_ds.select_drugs(eval_drugs)

                screen_seq = test_gen.flow_from_dataset(
                    screen_ds, shuffle=True, seed=opts.seed
                )
                eval_seq = test_gen.flow_from_dataset(eval_ds)

                model.fit(screen_seq, epochs=hparams.epochs, verbose=0)

                preds: np.ndarray = model.predict(eval_seq, verbose=0)
                pred_df = make_pred_df(
                    eval_ds, preds, split_group=test_ds.name, **param_dict
                )
                pred_dfs.append(pred_df)

            if len(skipped_cells) > 0:
                log.warning(
                    f"Skipped ScreenAhead for {len(skipped_cells)} sample "
                    f"with fewer then {n_drugs + 5} total drugs screened)."
                )

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions_sa.csv", index=False)


if __name__ == "__main__":
    run_experiment()
