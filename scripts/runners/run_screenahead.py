#!/usr/bin/env python
"""Run ScreenAhead experiments."""

from __future__ import annotations

from screendl.utils.environ import configure_process_env

configure_process_env()

import gc
import importlib
import logging
import random
import typing as t
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from tqdm import tqdm

from cdrpy.datasets import Dataset
from cdrpy.mapper import BatchedResponseGenerator

from screendl.model import utils as model_utils
from screendl.screenahead import SELECTORS
from screendl.utils import evaluation as eval_utils
from screendl.utils.config import safe_lconfig_as_tuple
from screendl.utils.runtime import (
    cleanup_objects,
    configure_runtime,
    log_memory,
    reset_keras_runtime,
)

log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead",
)
def run_sa(cfg: DictConfig) -> float:
    """"""
    seed = cfg.get("seed", cfg.screenahead.opt.seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

        self.D_cell = None
        self.D_screen = None
        self.R_cell = None
        self.screen_drugs = None
        self.M_screen = None

        cleanup_objects()
        model_utils.trim_malloc()


@dataclass
class ExperimentData:
    """Prepared base model and datasets from the basic pipeline."""

    datasets: dict[str, Dataset]
    M_base: t.Any
    W_base: t.Any
    D_select: Dataset
    D_test: Dataset

    def clear(self) -> None:
        """Drop run-level references."""
        model_utils.clear_compiled_model(self.M_base)

        self.M_base = None
        self.W_base = None
        self.datasets = {}
        self.D_select = None
        self.D_test = None

        cleanup_objects()
        model_utils.trim_malloc()


def import_basic_pipeline(cfg: DictConfig):
    """Import the configured basic pipeline module."""
    if cfg.model.name not in PIPELINES:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    module_file = PIPELINES[cfg.model.name]
    module_name = f"screendl.pipelines.basic.{module_file}"
    return importlib.import_module(module_name)


def prepare_experiment(cfg: DictConfig) -> ExperimentData:
    """Run the basic pipeline and prepare reusable base state."""
    module = import_basic_pipeline(cfg)

    log.info("Running base pipeline for %s...", cfg.model.name)
    M_base, _, datasets = module.run_pipeline(cfg)
    W_base = M_base.get_weights()

    train_cell_ids = set(datasets["train"].cell_ids)
    D_select = datasets["full"].select_cells(train_cell_ids)
    D_test: Dataset = datasets["test"]

    if bool(_cfg_get(cfg.screenahead.io, "permute_exp", False)):
        log.warning("Permuting test expression features for ScreenAhead control.")
        D_test.cell_encoders["exp"].data = D_test.cell_encoders["exp"].data.apply(
            np.random.permutation
        )

    return ExperimentData(
        M_base=M_base,
        W_base=W_base,
        datasets=datasets,
        D_select=D_select,
        D_test=D_test,
    )


def select_screen_drugs(
    cfg: DictConfig,
    *,
    D_select: Dataset,
    D_cell: Dataset,
) -> set[str]:
    """Select ScreenAhead drugs for one held-out test cell."""
    opts = cfg.screenahead.opt

    choices = set(D_cell.drug_ids)
    if len(choices) <= opts.n_drugs:
        raise ValueError(
            f"fewer than or equal to {opts.n_drugs} drugs available for "
            f"{D_cell.cell_ids[0]!r}"
        )

    selector = SELECTORS[opts.selector](
        D_select,
        seed=opts.seed,
        na_threshold=_cfg_get(opts, "na_thresh", None),
    )

    return set(selector.select(opts.n_drugs, choices=choices))


def configure_screenahead_for_cell(
    cfg: DictConfig,
    *,
    M_base: t.Any,
    W_base: t.Any,
) -> t.Any:
    """Create a fresh ScreenAhead model initialized from base weights."""
    hparams = cfg.screenahead.hyper

    frozen_prefixes = _cfg_get(
        hparams,
        "frozen_layer_prefixes",
        ("mol", "exp", "ont", "mut", "cnv", "mr"),
    )
    frozen_names = _cfg_get(hparams, "frozen_layer_names", None)

    return model_utils.configure_screenahead_model(
        M_base,
        initial_weights=W_base,
        frozen_layer_prefixes=safe_lconfig_as_tuple(frozen_prefixes),
        frozen_layer_names=safe_lconfig_as_tuple(frozen_names),
        training=False,
    )


def fit_screenahead_for_cell(
    cfg: DictConfig,
    *,
    M_base: t.Any,
    W_base: t.Any,
    D_screen: Dataset,
) -> t.Any:
    """Fit a fresh ScreenAhead response-only model for one cell."""
    hparams = cfg.screenahead.hyper

    M_screen = configure_screenahead_for_cell(
        cfg,
        M_base=M_base,
        W_base=W_base,
    )

    return model_utils.fit_screenahead_model(
        model=M_screen,
        dataset=D_screen,
        batch_size=hparams.batch_size,
        epochs=hparams.epochs,
        learning_rate=hparams.learning_rate,
        weight_decay=_cfg_get(hparams, "weight_decay", None),
        loss="mean_squared_error",
    )


def predict_cell(
    cfg: DictConfig,
    *,
    model: t.Any,
    D_cell: Dataset,
    screen_drugs: set[str],
    cell_id: str,
) -> pd.DataFrame:
    """Predict all drugs for one held-out cell and annotate ScreenAhead metadata."""
    batch_size = int(max(1, D_cell.n_drugs))
    cell_gen = None
    cell_seq = None

    try:
        cell_gen = BatchedResponseGenerator(D_cell, batch_size=batch_size)
        cell_seq = cell_gen.flow_from_dataset(D_cell)
        preds = model.predict(cell_seq, verbose=0)

        R_cell = eval_utils.make_pred_df(
            D_cell,
            preds,
            split_group="test",
            model="ScreenDL-SA",
            split_id=cfg.dataset.split.id,
            split_type=cfg.dataset.split.name,
            norm_method=cfg.dataset.preprocess.norm,
        )

        return R_cell.assign(
            was_screened=lambda df: df["drug_id"].isin(screen_drugs),
            cell_screen_id=cell_id,
        )

    finally:
        del cell_seq
        del cell_gen
        gc.collect()


def run_cell(
    cfg: DictConfig,
    *,
    exp: ExperimentData,
    cell_id: str,
) -> pd.DataFrame | None:
    """Run ScreenAhead for a single held-out test cell."""
    ctx = ScreenAheadContext()

    try:
        ctx.D_cell = exp.D_test.select_cells([cell_id])
        choices = set(ctx.D_cell.drug_ids)

        if len(choices) <= cfg.screenahead.opt.n_drugs:
            log.warning(
                "Skipping ScreenAhead for %s: only %d drugs available <= n_drugs=%d",
                cell_id,
                len(choices),
                cfg.screenahead.opt.n_drugs,
            )
            return None

        ctx.screen_drugs = select_screen_drugs(
            cfg,
            D_select=exp.D_select,
            D_cell=ctx.D_cell,
        )
        ctx.D_screen = ctx.D_cell.select_drugs(ctx.screen_drugs)

        ctx.M_screen = fit_screenahead_for_cell(
            cfg,
            M_base=exp.M_base,
            W_base=exp.W_base,
            D_screen=ctx.D_screen,
        )

        ctx.R_cell = predict_cell(
            cfg,
            model=ctx.M_screen,
            D_cell=ctx.D_cell,
            screen_drugs=ctx.screen_drugs,
            cell_id=str(cell_id),
        )
        return ctx.R_cell

    finally:
        ctx.clear()


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead",
)
def run_sa(cfg: DictConfig) -> None:
    """Run ScreenAhead experiments."""
    configure_runtime()

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    tf.get_logger().setLevel("ERROR")

    log_memory("After TensorFlow runtime setup")

    exp = prepare_experiment(cfg)

    try:
        output_path = Path("predictions_sa.csv")
        if output_path.exists():
            output_path.unlink()

        pred_chunks: list[pd.DataFrame] = []
        test_cell_ids = sorted(set(exp.D_test.cell_ids))

        for cell_id in tqdm(test_cell_ids, desc="Fitting ScreenAhead models"):
            R_cell = run_cell(cfg, exp=exp, cell_id=cell_id)
            if R_cell is None:
                continue

            pred_chunks.append(R_cell)

            model_utils.trim_malloc()

        if not pred_chunks:
            raise RuntimeError("No ScreenAhead predictions were generated.")

        R_cell = pd.concat(pred_chunks, ignore_index=True)
        R_cell.to_csv(output_path, index=False)

    finally:
        exp.clear()
        reset_keras_runtime()
        model_utils.trim_malloc()
        gc.collect()


if __name__ == "__main__":
    run_sa()
