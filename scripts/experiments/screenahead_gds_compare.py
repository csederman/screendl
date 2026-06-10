#!/usr/bin/env python
"""Run ScreenAhead experiments."""

from __future__ import annotations

from screendl.utils.environ import configure_process_env

configure_process_env()

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

from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.datasets import Dataset

from screendl.model import utils as model_utils
from screendl.utils import evaluation as eval_utils
from screendl.screenahead import SELECTORS
from screendl.utils.config import safe_lconfig_as_tuple
from screendl.utils.output import write_predictions_chunk
from screendl.utils.runtime import (
    cleanup_objects,
    configure_runtime,
    log_memory,
    reset_keras_runtime,
)

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}
DEFAULT_FROZEN_LAYER_PREFIXES = ("mol", "exp", "ont", "mut", "cnv", "mr")


def _cfg_get(obj: t.Any, name: str, default: t.Any = None) -> t.Any:
    """Small safe getattr for DictConfig/namespaces."""
    return getattr(obj, name, default) if obj is not None else default


def _screen_frozen_layer_prefixes(cfg: DictConfig) -> tuple[str, ...]:
    """Read ScreenAhead frozen prefixes, preserving original defaults."""
    hp_screen = cfg.screenahead.hyper
    return safe_lconfig_as_tuple(
        _cfg_get(
            hp_screen,
            "frozen_layer_prefixes",
            DEFAULT_FROZEN_LAYER_PREFIXES,
        )
    )


def _screen_frozen_layer_names(cfg: DictConfig) -> tuple[str, ...]:
    """Read ScreenAhead frozen layer names if present."""
    hp_screen = cfg.screenahead.hyper
    return safe_lconfig_as_tuple(_cfg_get(hp_screen, "frozen_layer_names", ()))


@dataclass
class ExperimentData:
    """Prepared datasets and base model state for a run."""

    M_base: t.Any
    W_base: t.Any
    ds_dict: dict[str, Dataset]
    D_selection: Dataset
    D_test: Dataset

    def require_ready(self) -> None:
        """Validate that run-level objects are available."""
        if self.M_base is None:
            raise RuntimeError("Base model has been cleared.")
        if self.W_base is None:
            raise RuntimeError("Base weights have been cleared.")
        if self.ds_dict is None:
            raise RuntimeError("Dataset dictionary has been cleared.")
        if self.D_selection is None:
            raise RuntimeError("Selection dataset has been cleared.")
        if self.D_test is None:
            raise RuntimeError("Test dataset has been cleared.")

    def clear(self) -> None:
        """Drop run-level references."""
        model_utils.clear_compiled_model(self.M_base)

        self.M_base = None
        self.W_base = None
        self.ds_dict = None
        self.D_selection = None
        self.D_test = None

        cleanup_objects()
        model_utils.trim_malloc()


@dataclass
class RunContext:
    """Mutable per-cell state that can be cleared after each cell."""

    D_cell: Dataset | None = None
    D_screen: Dataset | None = None
    D_holdout: Dataset | None = None
    holdout_seq: t.Any = None

    M_screen: t.Any = None
    M_screen_mr: t.Any = None

    R_screen: pd.DataFrame | None = None
    R_screen_mr: pd.DataFrame | None = None

    screen_drugs: set[str] | None = None
    original_screen_obs: pd.DataFrame | None = None

    def clear(self) -> None:
        """Drop per-cell references."""
        model_utils.clear_compiled_model(self.M_screen)
        model_utils.clear_compiled_model(self.M_screen_mr)

        self.D_cell = None
        self.D_screen = None
        self.D_holdout = None
        self.holdout_seq = None

        self.M_screen = None
        self.M_screen_mr = None

        self.R_screen = None
        self.R_screen_mr = None

        self.screen_drugs = None
        self.original_screen_obs = None

        cleanup_objects()
        model_utils.trim_malloc()


def load_pipeline_module(cfg: DictConfig) -> t.Any:
    """Load the configured basic pipeline module."""
    if cfg.model.name not in PIPELINES:
        raise ValueError("Unsupported model.")

    module_file = PIPELINES[cfg.model.name]
    module_name = f"screendl.pipelines.basic.{module_file}"
    return importlib.import_module(module_name)


def prepare_experiment(cfg: DictConfig) -> ExperimentData:
    """Run the base pipeline and prepare datasets for ScreenAhead comparison.

    run_pipeline(cfg) owns model-level functionality, including bilinear
    interactions and optional functional auxiliary training.
    """
    module = load_pipeline_module(cfg)
    M_base, _, ds_dict = module.run_pipeline(cfg)

    # NOTE: some drug selection algorithms require un-normalized responses.
    train_cell_ids = set(ds_dict["train"].cell_ids)
    D_selection = ds_dict["full"].select_cells(train_cell_ids)

    D_test: Dataset = ds_dict["test"]
    W_base = M_base.get_weights()

    return ExperimentData(
        M_base=M_base,
        W_base=W_base,
        ds_dict=ds_dict,
        D_selection=D_selection,
        D_test=D_test,
    )


def select_screen_drugs(
    cfg: DictConfig,
    D_selection: Dataset,
    D_cell: Dataset,
) -> set[str]:
    """Select screened drugs for one test cell."""
    opts = cfg.screenahead.opt
    choices = set(D_cell.drug_ids)

    try:
        selector = SELECTORS[opts.selector](D_selection, seed=opts.seed)
        return set(selector.select(opts.n_drugs, choices=choices))

    except Exception:
        log.exception(
            "Drug selection failed with default selector settings. "
            "Retrying with na_threshold=0.0."
        )

        selector = SELECTORS[opts.selector](
            D_selection,
            seed=opts.seed,
            na_threshold=0.0,
        )
        return set(selector.select(opts.n_drugs, choices=choices))


def prepare_cell_datasets(
    cfg: DictConfig,
    ctx: RunContext,
    exp: ExperimentData,
    test_gen: BatchedResponseGenerator,
    *,
    cell_id: str,
) -> bool:
    """Prepare screened and holdout datasets for one cell.

    Returns False when the cell should be skipped.
    """
    opts = cfg.screenahead.opt

    ctx.D_cell = exp.D_test.select_cells([cell_id])
    choices = set(ctx.D_cell.drug_ids)

    # Require at least one drug in the holdout set.
    if len(choices) <= opts.n_drugs:
        log.warning(
            "Skipping ScreenAhead for %s "
            "(fewer than or equal to %s drugs screened).",
            cell_id,
            opts.n_drugs,
        )
        return False

    ctx.screen_drugs = select_screen_drugs(
        cfg=cfg,
        D_selection=exp.D_selection,
        D_cell=ctx.D_cell,
    )

    ctx.D_screen = ctx.D_cell.select_drugs(ctx.screen_drugs)
    ctx.D_holdout = ctx.D_cell.select_drugs(choices.difference(ctx.screen_drugs))
    ctx.holdout_seq = test_gen.flow_from_dataset(ctx.D_holdout)

    return True


def configure_screenahead_model(
    cfg: DictConfig,
    exp: ExperimentData,
) -> t.Any:
    """Create a fresh ScreenAhead model initialized from post-pipeline weights."""
    exp.M_base.set_weights(exp.W_base)

    M_screen = model_utils.configure_screenahead_model(
        exp.M_base,
        initial_weights=exp.W_base,
        frozen_layer_prefixes=_screen_frozen_layer_prefixes(cfg),
        frozen_layer_names=_screen_frozen_layer_names(cfg),
        training=False,
    )

    if M_screen is exp.M_base:
        log.warning("ScreenAhead model is the same object as base model.")

    return M_screen


def fit_screenahead_model(
    cfg: DictConfig,
    exp: ExperimentData,
    D_screen: Dataset,
) -> t.Any:
    """Fit a fresh ScreenAhead model on screened responses."""
    hparams = cfg.screenahead.hyper

    M_screen = configure_screenahead_model(cfg, exp)

    return model_utils.fit_screenahead_model(
        model=M_screen,
        dataset=D_screen,
        batch_size=hparams.batch_size,
        epochs=hparams.epochs,
        learning_rate=hparams.learning_rate,
        weight_decay=getattr(hparams, "weight_decay", None),
        loss="mean_squared_error",
    )


def make_prediction_df(
    cfg: DictConfig,
    model: t.Any,
    D_holdout: Dataset,
    holdout_seq: t.Any,
    *,
    model_name: str,
) -> pd.DataFrame:
    """Predict holdout responses and format as prediction rows."""
    preds = model.predict(holdout_seq, verbose=0)

    return eval_utils.make_pred_df(
        D_holdout,
        preds,
        split_group="test",
        model=model_name,
        split_id=cfg.dataset.split.id,
        split_type=cfg.dataset.split.name,
        norm_method=cfg.dataset.preprocess.norm,
    )


def run_observed_screenahead(
    cfg: DictConfig,
    ctx: RunContext,
    exp: ExperimentData,
    output_path: Path,
) -> None:
    """Run standard ScreenAhead using observed screened responses."""
    if ctx.D_screen is None:
        raise RuntimeError("Screen dataset has not been prepared.")
    if ctx.D_holdout is None or ctx.holdout_seq is None:
        raise RuntimeError("Holdout dataset has not been prepared.")

    ctx.M_screen = fit_screenahead_model(
        cfg=cfg,
        exp=exp,
        D_screen=ctx.D_screen,
    )

    ctx.R_screen = make_prediction_df(
        cfg=cfg,
        model=ctx.M_screen,
        D_holdout=ctx.D_holdout,
        holdout_seq=ctx.holdout_seq,
        model_name="ScreenDL-SA",
    )
    write_predictions_chunk(ctx.R_screen, output_path)


def run_mean_response_screenahead(
    cfg: DictConfig,
    ctx: RunContext,
    exp: ExperimentData,
    output_path: Path,
) -> None:
    """Run ScreenAhead after replacing selected responses with their mean."""
    if ctx.D_screen is None:
        raise RuntimeError("Screen dataset has not been prepared.")
    if ctx.D_holdout is None or ctx.holdout_seq is None:
        raise RuntimeError("Holdout dataset has not been prepared.")

    try:
        ctx.original_screen_obs = ctx.D_screen.obs
        ctx.D_screen.obs = ctx.D_screen.obs.copy()
        ctx.D_screen.obs["label"] = ctx.D_screen.obs["label"].mean()

        ctx.M_screen_mr = fit_screenahead_model(
            cfg=cfg,
            exp=exp,
            D_screen=ctx.D_screen,
        )

        ctx.R_screen_mr = make_prediction_df(
            cfg=cfg,
            model=ctx.M_screen_mr,
            D_holdout=ctx.D_holdout,
            holdout_seq=ctx.holdout_seq,
            model_name="ScreenDL-SA(MR)",
        )
        write_predictions_chunk(ctx.R_screen_mr, output_path)

    finally:
        if ctx.original_screen_obs is not None:
            ctx.D_screen.obs = ctx.original_screen_obs


def run_cell(
    cfg: DictConfig,
    exp: ExperimentData,
    test_gen: BatchedResponseGenerator,
    output_path: Path,
    *,
    cell_id: str,
) -> None:
    """Run observed and mean-response ScreenAhead for one test cell."""
    ctx = RunContext()
    exp.require_ready()

    try:
        should_run = prepare_cell_datasets(
            cfg=cfg,
            ctx=ctx,
            exp=exp,
            test_gen=test_gen,
            cell_id=cell_id,
        )
        if not should_run:
            return

        run_observed_screenahead(
            cfg=cfg,
            ctx=ctx,
            exp=exp,
            output_path=output_path,
        )

        model_utils.clear_compiled_model(ctx.M_screen)
        ctx.M_screen = None
        exp.M_base.set_weights(exp.W_base)
        cleanup_objects()
        model_utils.trim_malloc()

        run_mean_response_screenahead(
            cfg=cfg,
            ctx=ctx,
            exp=exp,
            output_path=output_path,
        )

    finally:
        exp.M_base.set_weights(exp.W_base)
        ctx.clear()


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead_gds_compare",
)
def run_sa(cfg: DictConfig) -> None:
    """Run ScreenAhead GDS comparison experiments."""
    configure_runtime()
    log_memory("After TensorFlow runtime setup")

    # Suppress TF warnings for retracing.
    tf.get_logger().setLevel("ERROR")

    output_path = Path("predictions_sa.csv")
    if output_path.exists():
        output_path.unlink()

    exp = prepare_experiment(cfg)
    test_gen = BatchedResponseGenerator(exp.D_test, 256)

    try:
        exp.require_ready()

        log.info("Running ScreenAhead GDS comparison...")
        for cell_id in set(exp.D_test.cell_ids):
            log_memory(f"Cell {cell_id} start")

            try:
                run_cell(
                    cfg=cfg,
                    exp=exp,
                    test_gen=test_gen,
                    output_path=output_path,
                    cell_id=str(cell_id),
                )

            except Exception:
                log.exception("Failed ScreenAhead run for cell_id=%s", cell_id)

            model_utils.trim_malloc()
            log_memory(f"Cell {cell_id} end")

    finally:
        del test_gen
        exp.clear()
        reset_keras_runtime()
        model_utils.trim_malloc()


if __name__ == "__main__":
    run_sa()
