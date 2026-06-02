#!/usr/bin/env python
"""Runs ScreenAhead when adding an increasing number of functionally-related therapies.

Usage
=====
>>> python scripts/experiments/screenahead_related_drugs.py \
        -m \
        dataset.split.id=1,2,3,4,5 \
        experiment.drug_id="5-Fluorouracil","Leflunomide","Epirubicin"
"""

from __future__ import annotations

from screendl.utils.environ import configure_process_env

configure_process_env()

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
from tensorflow import keras
from tqdm import tqdm

from cdrpy.datasets import Dataset
from cdrpy.mapper import BatchedResponseGenerator

from screendl.pipelines.basic.screendl import run_pipeline
from screendl.screenahead import get_response_matrix
from screendl.utils import evaluation as eval_utils
from screendl.model import utils as model_utils
from screendl.utils.config import safe_lconfig_as_tuple
from screendl.utils.output import write_predictions_chunk
from screendl.utils.runtime import (
    cleanup_objects,
    configure_runtime,
    log_memory,
    reset_keras_runtime,
)

if t.TYPE_CHECKING:
    from cdrpy.mapper.sequences import ResponseSequence

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

log = logging.getLogger(__name__)


DEFAULT_FROZEN_LAYER_PREFIXES = ("mol", "exp", "ont", "mut", "cnv", "mr")


def _cfg_get(obj: t.Any, name: str, default: t.Any = None) -> t.Any:
    """Small safe getattr for DictConfig/namespaces."""
    return getattr(obj, name, default) if obj is not None else default


def _screen_frozen_layer_prefixes(cfg: DictConfig) -> tuple[str, ...]:
    """Read ScreenAhead frozen prefixes, preserving newer model defaults."""
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


def get_predictions(model: keras.Model, batch_seq: "ResponseSequence") -> np.ndarray:
    """Generate predictions without triggering tf.function retracing."""
    predictions = []
    for batch_x, *_ in batch_seq:
        batch_preds: tf.Tensor = model(batch_x, training=False)
        predictions.append(batch_preds.numpy().flatten())
    return np.concatenate(predictions)


@dataclass
class ExperimentData:
    """Prepared datasets and base model state for the full run."""

    M_base: keras.Model | None
    W_base: t.Any
    ds_dict: dict[str, Dataset] | None
    D_test: Dataset | None
    test_cell_ids: list[str] | None
    corr_matrix: pd.DataFrame | None

    def require_ready(self) -> None:
        """Validate that run-level state is available."""
        if self.M_base is None:
            raise RuntimeError("Base model has been cleared.")
        if self.W_base is None:
            raise RuntimeError("Base model weights have been cleared.")
        if self.ds_dict is None:
            raise RuntimeError("Dataset dictionary has been cleared.")
        if self.D_test is None:
            raise RuntimeError("Test dataset has been cleared.")
        if self.test_cell_ids is None:
            raise RuntimeError("Test cell IDs have been cleared.")
        if self.corr_matrix is None:
            raise RuntimeError("Drug correlation matrix has been cleared.")

    def clear(self) -> None:
        """Drop run-level references."""
        model_utils.clear_compiled_model(self.M_base)

        self.M_base = None
        self.W_base = None
        self.ds_dict = None
        self.D_test = None
        self.test_cell_ids = None
        self.corr_matrix = None

        cleanup_objects()
        model_utils.trim_malloc()


@dataclass
class SampleContext:
    """Mutable per-cell state that can be cleared after each sample."""

    D_cell: Dataset | None = None
    D_test_drug: Dataset | None = None
    test_seq: t.Any = None

    M_screen: keras.Model | None = None

    base_result: pd.DataFrame | None = None
    screen_result: pd.DataFrame | None = None

    drug_corrs: pd.Series | None = None
    best_drugs: list[str] | None = None
    worst_drugs: list[str] | None = None

    def clear(self) -> None:
        """Drop per-cell references."""
        model_utils.clear_compiled_model(self.M_screen)

        self.D_cell = None
        self.D_test_drug = None
        self.test_seq = None

        self.M_screen = None

        self.base_result = None
        self.screen_result = None

        self.drug_corrs = None
        self.best_drugs = None
        self.worst_drugs = None

        cleanup_objects()
        model_utils.trim_malloc()


def prepare_experiment(cfg: DictConfig) -> ExperimentData:
    """Run the base ScreenDL pipeline and prepare related-drug analysis inputs.

    run_pipeline(cfg) owns model-level functionality, including bilinear
    interactions and optional functional auxiliary training.
    """
    log.info("Running base pipeline...")
    M_base, _, ds_dict = run_pipeline(cfg)

    M_base.trainable = True
    W_base = M_base.get_weights()

    D_test = ds_dict["test"]
    test_cell_ids = sorted(set(D_test.cell_ids))

    # Preserve the original response-matrix/correlation construction.
    response_matrix = get_response_matrix(ds_dict["train"])
    corr_matrix = response_matrix.T.corr().abs()

    return ExperimentData(
        M_base=M_base,
        W_base=W_base,
        ds_dict=ds_dict,
        D_test=D_test,
        test_cell_ids=test_cell_ids,
        corr_matrix=corr_matrix,
    )


def configure_screenahead_model(
    cfg: DictConfig,
    M_base: keras.Model,
    W_base: t.Any,
) -> keras.Model:
    """Create a fresh ScreenAhead model initialized from post-pipeline weights."""
    M_base.set_weights(W_base)

    M_screen = model_utils.configure_screenahead_model(
        M_base,
        initial_weights=W_base,
        frozen_layer_prefixes=_screen_frozen_layer_prefixes(cfg),
        frozen_layer_names=_screen_frozen_layer_names(cfg),
        training=False,
    )

    if M_screen is M_base:
        log.warning("ScreenAhead model is the same object as base model.")

    return M_screen


def fit_screenahead_model(
    cfg: DictConfig,
    M_base: keras.Model,
    W_base: t.Any,
    D_screen: Dataset,
) -> keras.Model:
    """Fit a fresh ScreenAhead model on selected related drugs."""
    hparams = cfg.screenahead.hyper

    M_screen = configure_screenahead_model(
        cfg=cfg,
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


def prepare_related_drug_sets(
    cfg: DictConfig,
    corr_matrix: pd.DataFrame,
    drug_choices: t.Iterable[str],
) -> tuple[pd.Series, list[str], list[str]]:
    """Rank candidate screen drugs by correlation with the target drug."""
    opts = cfg.screenahead.opt
    drug_id = cfg.experiment.drug_id

    valid_choices = [
        drug
        for drug in drug_choices
        if drug in corr_matrix.index and drug in corr_matrix.columns
    ]

    if drug_id not in valid_choices:
        raise ValueError(f"{drug_id} is not available in the correlation matrix.")

    drug_corrs: pd.Series = corr_matrix.loc[valid_choices, drug_id]
    drug_corrs = drug_corrs.drop(index=drug_id).sort_values(ascending=False)

    if len(drug_corrs) < opts.n_drugs:
        raise ValueError(
            f"Not enough correlated drugs for {drug_id}: "
            f"needed {opts.n_drugs}, found {len(drug_corrs)}."
        )

    best_drugs = list(drug_corrs.index[: opts.n_drugs])
    worst_drugs = list(drug_corrs.index[-opts.n_drugs :])

    return drug_corrs, best_drugs, worst_drugs


def make_base_prediction(
    cfg: DictConfig,
    M_base: keras.Model,
    W_base: t.Any,
    D_test_drug: Dataset,
    test_seq: "ResponseSequence",
) -> pd.DataFrame:
    """Generate the base ScreenDL prediction for the target drug."""
    M_base.set_weights(W_base)

    base_preds = get_predictions(M_base, test_seq)

    return eval_utils.make_pred_df(
        D_test_drug,
        base_preds,
        model="ScreenDL",
        n_drugs=0,
        split_id=cfg.dataset.split.id,
    )


def make_screenahead_prediction(
    cfg: DictConfig,
    M_base: keras.Model,
    W_base: t.Any,
    D_cell: Dataset,
    D_test_drug: Dataset,
    test_seq: "ResponseSequence",
    drug_corrs: pd.Series,
    screen_drugs: list[str],
    *,
    n_best_drugs: int,
) -> pd.DataFrame:
    """Run ScreenAhead for one related-drug composition and predict target drug."""
    opts = cfg.screenahead.opt
    hparams = cfg.screenahead.hyper

    D_screen = D_cell.select_drugs(screen_drugs)
    M_screen = None

    try:
        M_screen = fit_screenahead_model(
            cfg=cfg,
            M_base=M_base,
            W_base=W_base,
            D_screen=D_screen,
        )

        run_preds = get_predictions(M_screen, test_seq)
        screen_drug_corrs = drug_corrs.loc[screen_drugs]

        return eval_utils.make_pred_df(
            D_test_drug,
            run_preds,
            model="ScreenDL-SA",
            n_drugs=opts.n_drugs,
            n_best_drugs=n_best_drugs,
            max_pcc=screen_drug_corrs.max(),
            min_pcc=screen_drug_corrs.min(),
            mean_pcc=screen_drug_corrs.mean(),
            mean_pcc_best=screen_drug_corrs.iloc[:n_best_drugs].mean(),
            mean_pcc_worst=screen_drug_corrs.iloc[n_best_drugs:].mean(),
            mean_resp_pred=D_screen.obs["label"].mean(),
            mean_resp_true=D_cell.obs["label"].mean(),
            split_id=cfg.dataset.split.id,
            batch_size=hparams.batch_size,
        )

    finally:
        model_utils.clear_compiled_model(M_screen)
        M_base.set_weights(W_base)
        cleanup_objects()
        model_utils.trim_malloc()


def run_single_sample(
    cfg: DictConfig,
    exp: ExperimentData,
    output_path: Path,
    *,
    cell_id: str,
) -> None:
    """Run one cell-line sample and write predictions."""
    exp.require_ready()

    assert exp.M_base is not None
    assert exp.W_base is not None
    assert exp.D_test is not None
    assert exp.corr_matrix is not None

    opts = cfg.screenahead.opt
    drug_id = cfg.experiment.drug_id

    ctx = SampleContext()

    try:
        ctx.D_cell = exp.D_test.select_cells([cell_id])
        drug_choices = list(set(ctx.D_cell.drug_ids))

        if drug_id not in drug_choices:
            log.info(
                "Skipping sample | reason=target_drug_not_screened | "
                "drug_id=%s | cell_id=%s | n_drug_choices=%d",
                drug_id,
                cell_id,
                len(drug_choices),
            )
            return

        if len(drug_choices) < opts.n_drugs + 5:
            log.info(
                "Skipping sample | reason=too_few_screened_drugs | "
                "drug_id=%s | cell_id=%s | n_drug_choices=%d | required=%d",
                drug_id,
                cell_id,
                len(drug_choices),
                opts.n_drugs + 5,
            )
            return

        batch_gen = BatchedResponseGenerator(
            ctx.D_cell,
            batch_size=cfg.screenahead.hyper.batch_size,
        )

        ctx.D_test_drug = ctx.D_cell.select_drugs([drug_id])
        ctx.test_seq = batch_gen.flow_from_dataset(ctx.D_test_drug)

        ctx.drug_corrs, ctx.best_drugs, ctx.worst_drugs = prepare_related_drug_sets(
            cfg=cfg,
            corr_matrix=exp.corr_matrix,
            drug_choices=drug_choices,
        )

        results: list[pd.DataFrame] = []

        ctx.base_result = make_base_prediction(
            cfg=cfg,
            M_base=exp.M_base,
            W_base=exp.W_base,
            D_test_drug=ctx.D_test_drug,
            test_seq=ctx.test_seq,
        )
        results.append(ctx.base_result)

        max_best_drugs = min(opts.max_best_drugs, opts.n_drugs)

        for n_best_drugs in range(max_best_drugs + 1):
            screen_drugs = (
                ctx.best_drugs[:n_best_drugs] + ctx.worst_drugs[n_best_drugs:]
            )

            if drug_id in screen_drugs:
                raise RuntimeError(f"{drug_id} was included in screen drugs.")

            ctx.screen_result = make_screenahead_prediction(
                cfg=cfg,
                M_base=exp.M_base,
                W_base=exp.W_base,
                D_cell=ctx.D_cell,
                D_test_drug=ctx.D_test_drug,
                test_seq=ctx.test_seq,
                drug_corrs=ctx.drug_corrs,
                screen_drugs=screen_drugs,
                n_best_drugs=n_best_drugs,
            )
            results.append(ctx.screen_result)

        result_df = pd.concat(results, ignore_index=True)
        write_predictions_chunk(result_df, output_path)

        log.info(
            "Wrote sample predictions | drug_id=%s | cell_id=%s | rows=%d | "
            "models=%s",
            drug_id,
            cell_id,
            len(result_df),
            result_df["model"].value_counts(dropna=False).to_dict(),
        )

    finally:
        if exp.M_base is not None and exp.W_base is not None:
            exp.M_base.set_weights(exp.W_base)
        ctx.clear()


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="screenahead_related_drugs",
)
def run_experiment(cfg: DictConfig) -> None:
    """Run ScreenAhead comparative analysis for related screened therapies."""
    configure_runtime()
    log_memory("After TensorFlow runtime setup")

    drug_ids = cfg.experiment.drug_id
    if isinstance(drug_ids, str):
        drug_ids = [drug_ids]
    else:
        drug_ids = list(drug_ids)

    output_path = Path("predictions_sa.csv")
    if output_path.exists():
        output_path.unlink()

    exp = prepare_experiment(cfg)

    try:
        exp.require_ready()
        assert exp.test_cell_ids is not None

        log.info(
            "Running ScreenAhead related-drug analysis | n_drugs=%d | drugs=%s",
            len(drug_ids),
            drug_ids,
        )

        for drug_id in drug_ids:
            cfg.experiment.drug_id = drug_id

            before_size = output_path.stat().st_size if output_path.exists() else 0
            n_failures = 0

            log.info("Running target drug: %s", drug_id)

            for cell_id in tqdm(exp.test_cell_ids, desc=str(drug_id)):
                try:
                    run_single_sample(
                        cfg=cfg,
                        exp=exp,
                        output_path=output_path,
                        cell_id=str(cell_id),
                    )

                except Exception:
                    n_failures += 1
                    log.exception(
                        "Failed related-drug ScreenAhead sample | "
                        "drug_id=%s | cell_id=%s",
                        drug_id,
                        cell_id,
                    )

                model_utils.trim_malloc()

            after_size = output_path.stat().st_size if output_path.exists() else 0
            wrote_any = after_size > before_size

            if wrote_any:
                log.info(
                    "Finished target drug | drug_id=%s | wrote_output=true | "
                    "bytes_before=%d | bytes_after=%d | failures=%d",
                    drug_id,
                    before_size,
                    after_size,
                    n_failures,
                )
            else:
                log.warning(
                    "Skipped target drug | drug_id=%s | wrote_output=false | "
                    "bytes_before=%d | bytes_after=%d | failures=%d",
                    drug_id,
                    before_size,
                    after_size,
                    n_failures,
                )

    finally:
        exp.clear()
        reset_keras_runtime()
        model_utils.trim_malloc()


if __name__ == "__main__":
    run_experiment()
