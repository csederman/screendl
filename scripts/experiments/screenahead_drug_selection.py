#!/usr/bin/env python
"""Runs transfer learning experiments on the HCI dataset."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import logging

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

from omegaconf import DictConfig
from pathlib import Path
from tensorflow import keras
from tqdm import tqdm

from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.datasets import Dataset

from screendl.pipelines.basic.screendl import run_pipeline
from screendl.utils import evaluation as eval_utils
from screendl.utils import model_utils
from screendl.utils.drug_selectors import SELECTORS, DrugSelectorType


log = logging.getLogger(__name__)


def configure_session() -> None:
    """Enable memory growth to avoid mem leak between hydra jobs."""
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def run_single_sample(
    cfg: DictConfig,
    model: keras.Model,
    dataset: Dataset,
    selector: DrugSelectorType,
    num_drugs: int,
) -> pd.DataFrame:
    """Runs the experiment for a single set of params."""
    opts = cfg.screenahead.opt
    hparams = cfg.screenahead.hyper

    sa_model = model_utils.configure_screenahead_model(
        model,
        optim=keras.optimizers.Adam(hparams.learning_rate),
        frozen_layer_prefixes=("mol", "exp", "ont", "mut", "cnv"),
        training=False,
    )

    screen_drugs = selector.select(num_drugs, choices=set(dataset.drug_ids))
    screen_ds = dataset.select_drugs(screen_drugs)

    batch_gen = BatchedResponseGenerator(dataset, batch_size=hparams.batch_size)
    batch_seq = batch_gen.flow_from_dataset(dataset)
    screen_seq = batch_gen.flow_from_dataset(screen_ds, shuffle=True, seed=opts.seed)

    _ = sa_model.fit(screen_seq, epochs=hparams.epochs, verbose=0)

    preds = eval_utils.get_predictions(sa_model, batch_seq)
    result = eval_utils.make_pred_df(
        dataset,
        preds,
        n_drugs=num_drugs,
        selector_type=selector.name,
        split_id=cfg.dataset.split.id,
    )
    result["was_screened"] = result["drug_id"].isin(screen_ds.drug_ids)

    return result


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead_drug_selection",
)
def run_experiment(cfg: DictConfig) -> None:
    """Runs ScreenAhead comparative analysis."""
    opts = cfg.screenahead.opt

    model, _, ds_dict = run_pipeline(cfg)
    base_weights = model.get_weights()

    # some selectors require raw resposes
    train_cell_ids = set(ds_dict["train"].cell_ids)
    selection_ds = ds_dict["full"].select_cells(train_cell_ids)

    test_ds = ds_dict["test"]
    test_cell_ids = list(set(test_ds.cell_ids))

    out_file = Path("./predictions_sa.csv")
    for selector_type in opts.selectors:
        selector = SELECTORS[selector_type](
            selection_ds, seed=opts.seed, name=selector_type
        )
        for n_drugs in opts.n_drugs:
            for cell_id in tqdm(test_cell_ids, desc=f"{selector_type}({n_drugs})"):
                cell_ds = test_ds.select_cells([cell_id])

                if cell_ds.n_drugs < n_drugs + 5:
                    continue

                try:
                    result = run_single_sample(cfg, model, cell_ds, selector, n_drugs)
                    result.to_csv(
                        out_file, index=False, mode="a", header=(not out_file.exists())
                    )
                except Exception as e:
                    print(e)

                model.set_weights(base_weights)

        K.clear_session()


if __name__ == "__main__":
    configure_session()
    run_experiment()
    K.clear_session()
