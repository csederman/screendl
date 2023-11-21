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

from omegaconf import DictConfig
from pathlib import Path
from tensorflow import keras
from tqdm import tqdm

from cdrpy.mapper import BatchedResponseGenerator

from screendl.utils.evaluation import make_pred_df
from screendl.pipelines.basic.screendl import run_pipeline
from screendl.utils.drug_selectors import get_response_matrix


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="sa_interp_config",
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

    test_ds = ds_dict["test"]
    test_gen = BatchedResponseGenerator(test_ds, batch_size=hparams.batch_size)
    test_drug_ids = sorted(cfg.experiment.drug_list)
    test_cell_ids = sorted(list(set(test_ds.cell_ids)))

    M = get_response_matrix(ds_dict["train"])
    corr_matrix = M.T.corr().abs()

    results_file = Path("predictions_sa.csv")

    for drug_id in test_drug_ids:
        pred_dfs = []
        for cell_id in tqdm(test_cell_ids, desc=f"{drug_id}"):
            cell_ds = test_ds.select_cells([cell_id])
            choices = list(set(cell_ds.drug_ids))

            if drug_id not in choices:
                # skip cell lines for which this drug was not screened
                continue

            if len(choices) < opts.n_drugs + 5:
                # skip cell lines with too few drugs screened
                continue

            drug_corrs: pd.Series = corr_matrix.loc[choices, drug_id]
            drug_corrs = drug_corrs.sort_values(ascending=False)
            best_drugs = list(drug_corrs.index[1 : opts.n_drugs + 1])
            worst_drugs = list(drug_corrs.index[-opts.n_drugs :])

            eval_ds = cell_ds.select_drugs([drug_id])
            eval_seq = test_gen.flow_from_dataset(eval_ds)

            preds: np.ndarray = model.predict(eval_seq, verbose=0)
            pred_df = make_pred_df(
                eval_ds,
                preds,
                model="ScreenDL",
                n_drugs=0,
                n_best_drugs=0,
                max_pcc=np.nan,
                min_pcc=np.nan,
                std_pcc=np.nan,
                mean_pcc=np.nan,
                median_pcc=np.nan,
            )
            pred_dfs.append(pred_df)

            for i in range(opts.max_best_drugs + 1):
                model.set_weights(base_weights)

                screen_drugs = best_drugs[:i] + worst_drugs[i:]
                assert drug_id not in screen_drugs

                screen_ds = cell_ds.select_drugs(screen_drugs)
                screen_seq = test_gen.flow_from_dataset(
                    screen_ds, shuffle=True, seed=opts.seed
                )

                model.fit(screen_seq, epochs=hparams.epochs, verbose=0)

                preds: np.ndarray = model.predict(eval_seq, verbose=0)
                pred_df = make_pred_df(
                    eval_ds,
                    preds,
                    model="ScreenDL-SA",
                    n_drugs=opts.n_drugs,
                    n_best_drugs=i,
                    max_pcc=drug_corrs.loc[screen_drugs].max(),
                    min_pcc=drug_corrs.loc[screen_drugs].min(),
                    std_pcc=drug_corrs.loc[screen_drugs].std(),
                    mean_pcc=drug_corrs.loc[screen_drugs].mean(),
                    median_pcc=drug_corrs.loc[screen_drugs].median(),
                )
                pred_dfs.append(pred_df)

        drug_pred_df = pd.concat(pred_dfs)
        drug_pred_df.to_csv(
            results_file, index=False, mode="a", header=(not results_file.exists())
        )


if __name__ == "__main__":
    run_experiment()
