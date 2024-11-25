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
import tensorflow as tf
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]
import typing as t

from omegaconf import DictConfig
from pathlib import Path
from tensorflow import keras
from tqdm import tqdm

from cdrpy.datasets import Dataset
from cdrpy.mapper import BatchedResponseGenerator

from screendl.utils import model_utils
from screendl.utils import evaluation as eval_utils
from screendl.pipelines.basic.screendl import run_pipeline
from screendl.utils.drug_selectors import get_response_matrix


if t.TYPE_CHECKING:
    from cdrpy.mapper.sequences import ResponseSequence


log = logging.getLogger(__name__)


def configure_session() -> None:
    """Enable memory growth to avoid mem leak between hydra jobs."""
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def get_predictions(model: keras.Model, batch_seq: ResponseSequence) -> np.ndarray:
    """Generates predictions without triggering tf.function retracing."""
    predictions = []
    for batch_x, *_ in batch_seq:
        batch_preds: tf.Tensor = model(batch_x, training=False)
        predictions.append(batch_preds.numpy().flatten())
    return np.concatenate(predictions)


def run_single_sample(
    cfg: DictConfig,
    model: keras.Model,
    dataset: Dataset,
    corr_matrix: pd.DataFrame,
    drug_choices: t.Iterable[str],
) -> pd.DataFrame:
    """Runs a single cell line sample."""

    opts = cfg.screenahead.opt
    hparams = cfg.screenahead.hyper
    drug_id = cfg.experiment.drug_id

    base_weights = model.get_weights()
    batch_gen = BatchedResponseGenerator(dataset, batch_size=hparams.batch_size)

    drug_corrs: pd.Series = corr_matrix.loc[drug_choices, cfg.experiment.drug_id]
    drug_corrs = drug_corrs.sort_values(ascending=False)
    best_drugs = list(drug_corrs.index[1 : opts.n_drugs + 1])
    worst_drugs = list(drug_corrs.index[-opts.n_drugs :])

    test_ds = dataset.select_drugs([drug_id])
    test_seq = batch_gen.flow_from_dataset(test_ds)

    results = []

    # generate base model predictions
    base_preds: np.ndarray = get_predictions(model, test_seq)
    base_result = eval_utils.make_pred_df(
        test_ds, base_preds, model="ScreenDL", n_drugs=0
    )
    results.append(base_result)

    for i in range(opts.max_best_drugs + 1):
        screen_drugs = best_drugs[:i] + worst_drugs[i:]
        assert drug_id not in screen_drugs

        screen_drug_corrs = drug_corrs.loc[screen_drugs]
        screen_ds = dataset.select_drugs(screen_drugs)
        screen_seq = batch_gen.flow_from_dataset(screen_ds, shuffle=True, seed=1441)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hparams.learning_rate),
            loss="mean_squared_error",
        )
        _ = model.fit(screen_seq, epochs=hparams.epochs, verbose=0)

        run_preds: np.ndarray = get_predictions(model, test_seq)
        run_result = eval_utils.make_pred_df(
            test_ds,
            run_preds,
            model="ScreenDL-SA",
            n_drugs=opts.n_drugs,
            n_best_drugs=i,
            max_pcc=screen_drug_corrs.max(),
            min_pcc=screen_drug_corrs.min(),
            mean_pcc=screen_drug_corrs.mean(),
            mean_pcc_best=screen_drug_corrs.iloc[:i].mean(),
            mean_pcc_worst=screen_drug_corrs.iloc[i:].mean(),
            mean_resp_pred=screen_ds.obs["label"].mean(),
            mean_resp_true=dataset.obs["label"].mean(),
        )
        results.append(run_result)

        model.set_weights(base_weights)

    return pd.concat(results)


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead_related_drugs",
)
def run_experiment(cfg: DictConfig) -> None:
    """Runs ScreenAhead comparative analysis."""
    opts = cfg.screenahead.opt
    drug_id = cfg.experiment.drug_id

    # run the training pipeline
    base_model, _, ds_dict = run_pipeline(cfg)
    base_model.trainable = True

    # configure the ScreenAhead model
    sa_model = model_utils.freeze_layers(
        base_model, prefixes=("mol", "exp", "ont", "mut", "cnv")
    )

    test_ds = ds_dict["test"]
    test_cell_ids = sorted(list(set(test_ds.cell_ids)))

    M = get_response_matrix(ds_dict["train"])
    corr_matrix = M.T.corr().abs()

    results_file = Path("predictions_sa.csv")
    results = []
    for cell_id in tqdm(test_cell_ids, desc=f"{drug_id}"):
        cell_ds = test_ds.select_cells([cell_id])
        drug_choices = list(set(cell_ds.drug_ids))

        if drug_id not in drug_choices:
            # skip cell lines for which this drug was not screened
            continue

        if len(drug_choices) < opts.n_drugs + 5:
            # skip cell lines with too few drugs screened
            continue

        cell_results = run_single_sample(
            cfg,
            model=sa_model,
            dataset=cell_ds,
            corr_matrix=corr_matrix,
            drug_choices=drug_choices,
        )
        results.append(cell_results)

        K.clear_session()

    results = pd.concat(results)
    results.to_csv(
        results_file, index=False, mode="a", header=(not results_file.exists())
    )


if __name__ == "__main__":
    configure_session()
    run_experiment()
    K.clear_session()
