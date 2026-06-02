"""
Main training loop for ScreenDL.
"""

from __future__ import annotations

import os
import typing as t

from tensorflow import keras

from cdrpy.metrics import tf_metrics
from cdrpy.mapper import BatchedResponseGenerator, FunctionAuxResponseGenerator

if t.TYPE_CHECKING:
    from cdrpy.types import PathLike
    from cdrpy.datasets import Dataset


CallbackList = t.Sequence[keras.callbacks.Callback] | None


def _build_callbacks(
    *,
    log_dir: PathLike | None = None,
    early_stopping: bool = False,
    tensorboard: bool = False,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: str = "min",
    early_stopping_patience: int = 10,
    early_stopping_start_from_epoch: int = 3,
    callbacks: CallbackList = None,
) -> list[keras.callbacks.Callback]:
    """Build callbacks for model fitting."""
    cb: list[keras.callbacks.Callback] = [] if callbacks is None else list(callbacks)

    if early_stopping:
        cb.append(
            keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                mode=early_stopping_mode,
                patience=early_stopping_patience,
                restore_best_weights=True,
                start_from_epoch=early_stopping_start_from_epoch,
                verbose=1,
            )
        )

    if tensorboard:
        if log_dir is None:
            raise ValueError("log_dir must be specified when using tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        cb.append(keras.callbacks.TensorBoard(log_dir, histogram_freq=1))

    return cb


def _has_aux_targets(
    drug_function_targets: t.Any | None,
    cell_function_targets: t.Any | None,
) -> bool:
    """Return whether auxiliary training is enabled."""
    return drug_function_targets is not None or cell_function_targets is not None


def _make_response_sequences(
    train_ds: Dataset,
    val_ds: Dataset | None,
    *,
    batch_size: int,
):
    """Create standard response-only train/validation sequences."""
    train_gen = BatchedResponseGenerator(train_ds, batch_size)
    train_seq = train_gen.flow(
        train_ds.cell_ids,
        train_ds.drug_ids,
        targets=train_ds.labels,
        shuffle=True,
        seed=4114,
    )

    val_seq = None
    if val_ds is not None:
        val_gen = BatchedResponseGenerator(val_ds, batch_size)
        val_seq = val_gen.flow(
            val_ds.cell_ids,
            val_ds.drug_ids,
            targets=val_ds.labels,
            shuffle=False,
        )

    return train_seq, val_seq


def _make_function_aux_sequences(
    train_ds: Dataset,
    val_ds: Dataset | None,
    *,
    batch_size: int,
    drug_function_targets: t.Any | None,
    cell_function_targets: t.Any | None,
    response_output_name: str,
    drug_output_name: str,
    cell_output_name: str,
):
    """Create response + functional auxiliary train/validation sequences."""
    train_gen = FunctionAuxResponseGenerator(train_ds, batch_size)
    train_seq = train_gen.flow_from_dataset(
        train_ds,
        drug_function_targets=drug_function_targets,
        cell_function_targets=cell_function_targets,
        response_output_name=response_output_name,
        drug_output_name=drug_output_name,
        cell_output_name=cell_output_name,
        shuffle=True,
        seed=4114,
    )

    val_seq = None
    if val_ds is not None:
        val_gen = FunctionAuxResponseGenerator(val_ds, batch_size)
        val_seq = val_gen.flow_from_dataset(
            val_ds,
            drug_function_targets=drug_function_targets,
            cell_function_targets=cell_function_targets,
            response_output_name=response_output_name,
            drug_output_name=drug_output_name,
            cell_output_name=cell_output_name,
            shuffle=False,
        )

    return train_seq, val_seq


def train_model(
    model: keras.Model,
    opt: keras.optimizers.Optimizer,
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    batch_size: int = 32,
    epochs: int = 10,
    save_dir: PathLike | None = None,
    log_dir: PathLike | None = None,
    early_stopping: bool = False,
    tensorboard: bool = False,
    *,
    drug_function_targets: t.Any | None = None,
    cell_function_targets: t.Any | None = None,
    response_loss_weight: float = 1.0,
    drug_function_loss_weight: float = 0.01,
    cell_function_loss_weight: float = 0.01,
    response_output_name: str = "response",
    drug_output_name: str = "drug_function",
    cell_output_name: str = "cell_function",
    early_stopping_monitor: str | None = None,
    early_stopping_mode: str | None = None,
    early_stopping_patience: int = 10,
    early_stopping_start_from_epoch: int = 3,
    callbacks: CallbackList = None,
) -> keras.Model:
    """Train a ScreenDL model.

    Existing response-only behavior is unchanged when no auxiliary targets are
    provided. If drug/tumor functional target tables are provided, the model is
    compiled as a multi-output model and trained with output-specific sample
    weights from ``FunctionAuxResponseSequence``. Missing drug/tumor auxiliary
    targets are zero-weighted by the sequence, while the response loss remains
    active.
    """
    use_aux = _has_aux_targets(drug_function_targets, cell_function_targets)

    cb = _build_callbacks(
        log_dir=log_dir,
        early_stopping=early_stopping,
        tensorboard=tensorboard,
        early_stopping_monitor=(
            early_stopping_monitor
            if early_stopping_monitor is not None
            else (f"val_{response_output_name}_loss" if use_aux else "val_loss")
        ),
        early_stopping_mode=(
            early_stopping_mode if early_stopping_mode is not None else "min"
        ),
        early_stopping_patience=early_stopping_patience,
        early_stopping_start_from_epoch=early_stopping_start_from_epoch,
        callbacks=callbacks,
    )

    if use_aux:
        train_seq, val_seq = _make_function_aux_sequences(
            train_ds,
            val_ds,
            batch_size=batch_size,
            drug_function_targets=drug_function_targets,
            cell_function_targets=cell_function_targets,
            response_output_name=response_output_name,
            drug_output_name=drug_output_name,
            cell_output_name=cell_output_name,
        )

        losses: dict[str, t.Any] = {response_output_name: "mean_squared_error"}
        loss_weights: dict[str, float] = {response_output_name: response_loss_weight}
        metrics: dict[str, list[t.Any]] = {
            response_output_name: ["mse", tf_metrics.pearson]
        }

        if drug_function_targets is not None:
            losses[drug_output_name] = "mean_squared_error"
            loss_weights[drug_output_name] = drug_function_loss_weight
            metrics[drug_output_name] = ["mse"]

        if cell_function_targets is not None:
            losses[cell_output_name] = "mean_squared_error"
            loss_weights[cell_output_name] = cell_function_loss_weight
            metrics[cell_output_name] = ["mse"]

        model.compile(
            optimizer=opt,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            jit_compile=False,
        )
    else:
        train_seq, val_seq = _make_response_sequences(
            train_ds,
            val_ds,
            batch_size=batch_size,
        )

        model.compile(
            optimizer=opt,
            loss="mean_squared_error",
            metrics=["mse", tf_metrics.pearson],
            jit_compile=False,
        )

    model.fit(
        train_seq,
        epochs=epochs,
        validation_data=val_seq,
        callbacks=cb,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "ScreenDL-PT.keras"))
        model.save_weights(os.path.join(save_dir, "ScreenDL-PT.weights.h5"))

    return model
