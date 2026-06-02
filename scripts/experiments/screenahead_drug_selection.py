#!/usr/bin/env python
"""Runs transfer learning experiments on the HCI dataset."""

from __future__ import annotations

from screendl.utils.environ import configure_process_env

configure_process_env()

import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tensorflow import keras
from tqdm import tqdm

from cdrpy.datasets import Dataset
from cdrpy.mapper import BatchedResponseGenerator

from screendl.model import utils as model_utils
from screendl.pipelines.basic.screendl import run_pipeline
from screendl.screenahead import SELECTORS, DrugSelectorType
from screendl.utils import evaluation as eval_utils
from screendl.utils.config import safe_lconfig_as_tuple
from screendl.utils.output import write_predictions_chunk
from screendl.utils.runtime import (
    cleanup_objects,
    configure_runtime,
    log_memory,
    reset_keras_runtime,
)

log = logging.getLogger(__name__)


def _cfg_get(obj: t.Any, name: str, default: t.Any = None) -> t.Any:
    """Small safe getattr for DictConfig/namespaces."""
    return getattr(obj, name, default) if obj is not None else default


@dataclass
class ExperimentData:
    """Prepared datasets and pretrained model state for the full run."""

    M_base: keras.Model | None
    W_base: t.Any
    ds_dict: dict[str, Dataset] | None
    selection_ds: Dataset | None
    test_ds: Dataset | None
    test_cell_ids: list[str] | None

    def require_ready(self) -> None:
        """Validate that run-level state is available."""
        if self.M_base is None:
            raise RuntimeError("Base model has been cleared.")
        if self.W_base is None:
            raise RuntimeError("Base model weights have been cleared.")
        if self.ds_dict is None:
            raise RuntimeError("Dataset dictionary has been cleared.")
        if self.selection_ds is None:
            raise RuntimeError("Selection dataset has been cleared.")
        if self.test_ds is None:
            raise RuntimeError("Test dataset has been cleared.")
        if self.test_cell_ids is None:
            raise RuntimeError("Test cell IDs have been cleared.")

    def clear(self) -> None:
        """Drop run-level references."""
        model_utils.clear_compiled_model(self.M_base)

        self.M_base = None
        self.W_base = None
        self.ds_dict = None
        self.selection_ds = None
        self.test_ds = None
        self.test_cell_ids = None

        cleanup_objects()
        model_utils.trim_malloc()


@dataclass
class SampleContext:
    """Mutable per-sample state that can be cleared after each sample."""

    M_screen: keras.Model | None = None
    D_cell: Dataset | None = None
    D_screen: Dataset | None = None
    R_pred: pd.DataFrame | None = None

    def clear(self) -> None:
        """Drop per-sample references."""
        model_utils.clear_compiled_model(self.M_screen)

        self.M_screen = None
        self.D_cell = None
        self.D_screen = None
        self.R_pred = None

        cleanup_objects()
        model_utils.trim_malloc()


def prepare_experiment(cfg: DictConfig) -> ExperimentData:
    """Run the base ScreenDL pipeline and prepare datasets for selection tests."""
    log.info("Running base pipeline...")
    M_base, _, ds_dict = run_pipeline(cfg)

    W_base = M_base.get_weights()

    # Some selectors require raw responses from the training cells.
    train_cell_ids = sorted(set(ds_dict["train"].cell_ids))
    selection_ds = ds_dict["full"].select_cells(train_cell_ids)

    test_ds = ds_dict["test"]
    test_cell_ids = sorted(set(test_ds.cell_ids))

    return ExperimentData(
        M_base=M_base,
        W_base=W_base,
        ds_dict=ds_dict,
        selection_ds=selection_ds,
        test_ds=test_ds,
        test_cell_ids=test_cell_ids,
    )


def build_selector(
    cfg: DictConfig,
    selection_ds: Dataset,
    selector_type: str,
) -> DrugSelectorType:
    """Instantiate one drug selector."""
    opts = cfg.screenahead.opt

    selector_kwargs = {
        "seed": opts.seed,
        "name": selector_type,
    }

    # Preserve compatibility with selectors that support na_threshold, without
    # requiring older configs to define it.
    if hasattr(opts, "na_thresh"):
        selector_kwargs["na_threshold"] = opts.na_thresh

    return SELECTORS[selector_type](selection_ds, **selector_kwargs)


def configure_sample_model(
    cfg: DictConfig,
    M_base: keras.Model,
    W_base: t.Any,
) -> keras.Model:
    """Create a fresh ScreenAhead model initialized from base weights."""
    hp = cfg.screenahead.hyper

    frozen_layer_prefixes = safe_lconfig_as_tuple(
        _cfg_get(
            hp,
            "frozen_layer_prefixes",
            ("mol", "exp", "ont", "mut", "cnv", "mr"),
        )
    )
    frozen_layer_names = safe_lconfig_as_tuple(_cfg_get(hp, "frozen_layer_names", ()))

    M_base.set_weights(W_base)

    M_screen = model_utils.configure_screenahead_model(
        M_base,
        initial_weights=W_base,
        frozen_layer_prefixes=frozen_layer_prefixes,
        frozen_layer_names=frozen_layer_names,
        training=False,
    )

    if M_screen is M_base:
        log.warning("ScreenAhead model is the same object as base model.")

    return M_screen


def fit_sample_model(
    cfg: DictConfig,
    M_screen: keras.Model,
    D_screen: Dataset,
) -> keras.Model:
    """Fine-tune one ScreenAhead model on selected drugs."""
    hp = cfg.screenahead.hyper

    return model_utils.fit_screenahead_model(
        model=M_screen,
        dataset=D_screen,
        batch_size=hp.batch_size,
        epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        weight_decay=_cfg_get(hp, "weight_decay", None),
        loss="mean_squared_error",
    )


def make_sample_predictions(
    cfg: DictConfig,
    M_screen: keras.Model,
    D_cell: Dataset,
    D_screen: Dataset,
    *,
    selector_type: str,
    n_drugs: int,
    cell_id: str,
) -> pd.DataFrame:
    """Predict all available drugs for one held-out cell after ScreenAhead."""
    hp = cfg.screenahead.hyper

    # Keep the same prediction path as the original script.
    # This avoids changing semantics while still fixing model-state leakage.
    batch_gen = BatchedResponseGenerator(D_cell, batch_size=hp.batch_size)
    batch_seq = batch_gen.flow_from_dataset(D_cell)

    preds = eval_utils.get_predictions(M_screen, batch_seq)

    result = eval_utils.make_pred_df(
        D_cell,
        preds,
        n_drugs=n_drugs,
        selector_type=selector_type,
        split_id=cfg.dataset.split.id,
    )

    result["cell_id"] = cell_id
    result["was_screened"] = result["drug_id"].isin(D_screen.drug_ids)

    return result


def run_single_sample(
    cfg: DictConfig,
    exp: ExperimentData,
    selector: DrugSelectorType,
    output_path: Path,
    *,
    selector_type: str,
    n_drugs: int,
    cell_id: str,
) -> None:
    """Run one selector/n_drugs/cell sample and write predictions."""
    exp.require_ready()
    assert exp.M_base is not None
    assert exp.W_base is not None
    assert exp.test_ds is not None

    ctx = SampleContext()

    try:
        ctx.D_cell = exp.test_ds.select_cells([cell_id])

        if ctx.D_cell.n_drugs < n_drugs + 5:
            log.info(
                "Skipping cell_id=%s selector=%s n_drugs=%s because only %s drugs are available.",
                cell_id,
                selector_type,
                n_drugs,
                ctx.D_cell.n_drugs,
            )
            return

        screen_drugs = selector.select(
            n_drugs,
            choices=set(ctx.D_cell.drug_ids),
        )
        ctx.D_screen = ctx.D_cell.select_drugs(screen_drugs)

        ctx.M_screen = configure_sample_model(
            cfg=cfg,
            M_base=exp.M_base,
            W_base=exp.W_base,
        )
        ctx.M_screen = fit_sample_model(
            cfg=cfg,
            M_screen=ctx.M_screen,
            D_screen=ctx.D_screen,
        )

        ctx.R_pred = make_sample_predictions(
            cfg=cfg,
            M_screen=ctx.M_screen,
            D_cell=ctx.D_cell,
            D_screen=ctx.D_screen,
            selector_type=selector_type,
            n_drugs=n_drugs,
            cell_id=cell_id,
        )

        write_predictions_chunk(ctx.R_pred, output_path)

    finally:
        ctx.clear()

        # Defensive reset in case configure_screenahead_model returned or mutated
        # the long-lived base object.
        if exp.M_base is not None and exp.W_base is not None:
            exp.M_base.set_weights(exp.W_base)


def run_selector_grid(
    cfg: DictConfig,
    exp: ExperimentData,
    output_path: Path,
) -> None:
    """Run all configured selector/n_drugs/cell combinations."""
    exp.require_ready()
    assert exp.selection_ds is not None
    assert exp.test_cell_ids is not None

    opts = cfg.screenahead.opt

    for selector_type in opts.selectors:
        selector = build_selector(
            cfg=cfg,
            selection_ds=exp.selection_ds,
            selector_type=selector_type,
        )

        for n_drugs in opts.n_drugs:
            desc = f"{selector_type}({n_drugs})"

            for cell_id in tqdm(exp.test_cell_ids, desc=desc):
                try:
                    run_single_sample(
                        cfg=cfg,
                        exp=exp,
                        selector=selector,
                        output_path=output_path,
                        selector_type=selector_type,
                        n_drugs=int(n_drugs),
                        cell_id=str(cell_id),
                    )
                except Exception:
                    log.exception(
                        "Failed sample | selector=%s | n_drugs=%s | cell_id=%s",
                        selector_type,
                        n_drugs,
                        cell_id,
                    )

                model_utils.trim_malloc()


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead_drug_selection",
)
def run(cfg: DictConfig) -> None:
    """Run ScreenAhead drug selection analysis."""
    configure_runtime()
    log_memory("After TensorFlow runtime setup")

    output_path = Path("predictions_sa.csv")
    if output_path.exists():
        output_path.unlink()

    exp = prepare_experiment(cfg)

    try:
        run_selector_grid(
            cfg=cfg,
            exp=exp,
            output_path=output_path,
        )

    finally:
        exp.clear()
        reset_keras_runtime()
        model_utils.trim_malloc()


if __name__ == "__main__":
    run()
