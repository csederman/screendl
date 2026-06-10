#!/usr/bin/env python
"""Runs domain-specific fine-tuning on the Welm breast cancer PDXO dataset."""

from __future__ import annotations

from screendl.utils.environ import configure_process_env

configure_process_env()

import gc
import json
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from tensorflow import keras
from tqdm import tqdm

from cdrpy.datasets import Dataset
from cdrpy.mapper import FunctionAuxResponseGenerator

from screendl import model as screendl
from screendl.data import preprocess_pdmc_screendl_datasets, PreprocessingArtifacts
from screendl.model import utils as model_utils
from screendl.pipelines.basic.screendl import (
    data_loader,
    data_splitter,
    model_builder as base_model_builder,
    model_trainer as base_model_trainer,
)
from screendl.pipelines.pdmc.utils import (
    get_screenahead_split,
    loo_split_generator,
)
from screendl.screenahead import SELECTORS
from screendl.utils import data_utils
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


def _is_enabled(obj: t.Any) -> bool:
    """Return whether a config block has enabled=True."""
    return bool(_cfg_get(obj, "enabled", False))


def _fold_model_dir(
    *,
    fold_i: int,
    heldout_cell_id: str,
    root: str | Path = "models",
) -> Path:
    """Return the model output directory for one held-out PDXO fold."""
    return (
        Path(root)
        / f"fold_{fold_i:03d}__{model_utils.safe_path_token(heldout_cell_id)}"
    )


def save_fold_manifest(
    path: str | Path,
    *,
    fold_i: int,
    heldout_cell_id: str,
    screened_drug_ids: t.Iterable[t.Any] | None = None,
) -> None:
    """Save fold metadata alongside fitted models."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "fold_i": fold_i,
        "heldout_cell_id": heldout_cell_id,
        "screened_drug_ids": (
            None if screened_drug_ids is None else [str(x) for x in screened_drug_ids]
        ),
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def data_preprocessor(
    cfg: DictConfig,
    train_cell_ds: Dataset,
    val_cell_ds: Dataset,
    pdmc_ds: Dataset,
) -> tuple[Dataset, Dataset, Dataset, PreprocessingArtifacts]:
    """Preprocess cell-line and PDXO datasets."""
    return preprocess_pdmc_screendl_datasets(
        train_cell_ds,
        val_cell_ds,
        pdmc_ds,
        exp_norm_method=cfg.dataset.preprocess.norm_exp,
        keep_pdmc_only_drugs=cfg.experiment.keep_pdmc_only_drugs,
        min_pdmcs_per_drug=cfg.experiment.min_pdmcs_per_drug,
        normalize_cnv=True,
        artifact_path="preprocessing.pkl",
    )


def _similarity_matrix_to_targets(
    sim_df: pd.DataFrame,
    *,
    n_components: int,
    prefix: str,
    fill_value: float = 0.0,
    seed: int = 1441,
) -> pd.DataFrame:
    """Convert square similarity matrix to fixed-width PCA targets."""
    common = sim_df.index.intersection(sim_df.columns)
    sim_df = sim_df.loc[common, common]

    columns = [f"{prefix}_{i}" for i in range(n_components)]
    if sim_df.empty:
        return pd.DataFrame(columns=columns, dtype="float32")

    x = sim_df.fillna(fill_value).to_numpy(dtype="float32")
    k = min(n_components, x.shape[0], x.shape[1])

    z = PCA(n_components=k, random_state=seed).fit_transform(x).astype("float32")
    if k < n_components:
        z_padded = pd.DataFrame(
            0.0,
            index=sim_df.index,
            columns=columns,
            dtype="float32",
        )
        z_padded.iloc[:, :k] = z
        return z_padded

    return pd.DataFrame(z, index=sim_df.index, columns=columns)


def make_fold_pdxo_function_aux_targets(
    D_t_pdxo: Dataset,
    *,
    drug_n_components: int,
    cell_n_components: int,
    cell_col: str = "cell_id",
    drug_col: str = "drug_id",
    label_col: str = "label",
    seed: int = 1441,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build N-1 PDXO drug and tumor functional aux targets for one fold.

    Uses only the training PDXO dataset for the current LOO fold. This avoids
    using the held-out tumor's full response profile.
    """
    response_mat = D_t_pdxo.obs.pivot_table(
        index=cell_col,
        columns=drug_col,
        values=label_col,
        aggfunc="mean",
    )

    drug_centered = response_mat.sub(response_mat.mean(axis=0), axis=1)
    drug_sim = drug_centered.corr()

    tumor_centered = response_mat.sub(response_mat.mean(axis=1), axis=0)
    tumor_sim = tumor_centered.T.corr()

    drug_targets = _similarity_matrix_to_targets(
        drug_sim,
        n_components=drug_n_components,
        prefix="drug_function",
        seed=seed,
    )
    cell_targets = _similarity_matrix_to_targets(
        tumor_sim,
        n_components=cell_n_components,
        prefix="cell_function",
        seed=seed,
    )

    return drug_targets, cell_targets


@dataclass
class RunContext:
    """Mutable per-fold state that can be cleared after each fold."""

    R_base: pd.DataFrame | None = None
    R_tune: pd.DataFrame | None = None
    R_screen: pd.DataFrame | None = None

    M_aux: keras.Model | None = None
    M_tune: keras.Model | None = None
    M_screen: keras.Model | None = None

    W_tune: t.Any = None
    W_screen: t.Any = None

    D_t_pdxo_full: Dataset | None = None
    D_e_pdxo_full: Dataset | None = None
    D_s_pdxo: Dataset | None = None

    drug_selector: t.Any = None

    def clear(self) -> None:
        """Drop per-fold references."""
        model_utils.clear_compiled_model(self.M_aux)
        model_utils.clear_compiled_model(self.M_tune)
        model_utils.clear_compiled_model(self.M_screen)

        self.R_base = None
        self.R_tune = None
        self.R_screen = None

        self.M_aux = None
        self.M_tune = None
        self.M_screen = None

        self.W_tune = None
        self.W_screen = None

        self.D_t_pdxo_full = None
        self.D_e_pdxo_full = None
        self.D_s_pdxo = None

        self.drug_selector = None

        cleanup_objects()
        model_utils.trim_malloc()


@dataclass
class ExperimentData:
    """Prepared datasets and base model state for a run."""

    dataset: Dataset | None
    D_t_cell: Dataset | None
    D_v_cell: Dataset | None
    D_pdxo: Dataset | None
    all_drug_ids: list[str] | None
    preprocessing_artifacts: PreprocessingArtifacts | None = None
    M_base: keras.Model | None = None
    W_base: t.Any = None
    M_base_trainable_state: model_utils.TrainableState | None = None

    def clear(self) -> None:
        """Drop run-level references."""
        model_utils.clear_compiled_model(self.M_base)

        self.dataset = None
        self.D_t_cell = None
        self.D_v_cell = None
        self.D_pdxo = None
        self.all_drug_ids = None
        self.preprocessing_artifacts = None
        self.M_base = None
        self.W_base = None
        self.M_base_trainable_state = None

        cleanup_objects()
        model_utils.trim_malloc()

    def require_ready(self) -> None:
        """Validate that run-level objects are available."""
        if self.dataset is None:
            raise RuntimeError("Experiment dataset has been cleared.")
        if self.D_pdxo is None:
            raise RuntimeError("PDXO dataset has been cleared.")
        if self.all_drug_ids is None:
            raise RuntimeError("Drug IDs have been cleared.")
        if self.M_base is None:
            raise RuntimeError("Base model has been cleared.")
        if self.W_base is None:
            raise RuntimeError("Base weights have been cleared.")
        if self.M_base_trainable_state is None:
            raise RuntimeError("Base trainable state has been cleared.")


def prepare_experiment(cfg: DictConfig) -> ExperimentData:
    """Load data, preprocess it, pretrain base model, and save base state."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info("Loading %s...", dataset_name)
    dataset = data_loader(cfg)
    all_drug_ids = list(dataset.drug_encoders["mol"].keys())

    log.info("Splitting %s...", dataset_name)
    D_t_cell, D_v_cell, D_pdxo = data_splitter(cfg, dataset)

    log.info("Preprocessing %s...", dataset_name)
    D_t_cell, D_v_cell, D_pdxo, preprocessing_artifacts = data_preprocessor(
        cfg,
        D_t_cell,
        D_v_cell,
        D_pdxo,
    )

    log.info("Building %s...", model_name)
    M_base = base_model_builder(cfg, D_t_cell)

    log.info("Pretraining %s...", model_name)
    M_base = base_model_trainer(cfg, M_base, D_t_cell, D_v_cell)
    model_utils.save_keras_model(M_base, Path("models") / "base.keras")

    return ExperimentData(
        dataset=dataset,
        D_t_cell=D_t_cell,
        D_v_cell=D_v_cell,
        D_pdxo=D_pdxo,
        all_drug_ids=all_drug_ids,
        preprocessing_artifacts=preprocessing_artifacts,
        M_base=M_base,
        W_base=M_base.get_weights(),
        M_base_trainable_state=model_utils.get_trainable_state(M_base),
    )


def prepare_fold_datasets(
    ctx: RunContext,
    D_t_pdxo: Dataset,
    D_e_pdxo: Dataset,
    all_drug_ids: list[str],
) -> None:
    """Create expanded background and target datasets for one fold."""
    ctx.D_t_pdxo_full = data_utils.expand_dataset(
        D_t_pdxo,
        cell_ids=list(set(D_t_pdxo.cell_ids)),
        drug_ids=all_drug_ids,
    )

    ctx.D_e_pdxo_full = data_utils.expand_dataset(
        D_e_pdxo,
        cell_ids=[D_e_pdxo.cell_ids[0]],
        drug_ids=all_drug_ids,
    )


def _annotate_fold(
    df: pd.DataFrame,
    *,
    fold_i: int,
    heldout_cell_id: str,
) -> pd.DataFrame:
    """Attach fold identity to a prediction chunk."""
    df = df.copy()
    df["fold_i"] = fold_i
    df["heldout_cell_id"] = heldout_cell_id
    return df


def run_base_predictions(
    ctx: RunContext,
    M_base: keras.Model,
    output_path: Path,
    *,
    fold_i: int,
    heldout_cell_id: str,
) -> None:
    """Run base model predictions for one fold."""
    if ctx.D_e_pdxo_full is None or ctx.D_t_pdxo_full is None:
        raise RuntimeError("Fold datasets have not been prepared.")

    ctx.R_base = eval_utils.get_predictions_vs_background(
        M=M_base,
        D_t=ctx.D_e_pdxo_full,
        D_b=ctx.D_t_pdxo_full,
        W_t=None,
        W_b=None,
        model="base",
        was_screened=False,
    )
    ctx.R_base = _annotate_fold(
        ctx.R_base,
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
    )
    write_predictions_chunk(ctx.R_base, output_path)


def _fit_response_only_transfer_model(
    cfg: DictConfig,
    ctx: RunContext,
    M_base: keras.Model,
    W_base: t.Any,
    D_t_pdxo: Dataset,
) -> None:
    """Original response-only transfer path."""
    hp_tune = cfg.xfer.hyper

    ctx.M_tune = model_utils.configure_transfer_model(
        M_base,
        initial_weights=W_base,
        frozen_layer_prefixes=safe_lconfig_as_tuple(hp_tune.frozen_layer_prefixes),
        frozen_layer_names=safe_lconfig_as_tuple(hp_tune.frozen_layer_names),
    )

    if ctx.M_tune is M_base:
        log.warning("Transfer model is the same object as base model.")

    ctx.M_tune = model_utils.fit_transfer_model(
        model=ctx.M_tune,
        dataset=D_t_pdxo,
        batch_size=hp_tune.batch_size,
        epochs=hp_tune.epochs,
        learning_rate=hp_tune.learning_rate,
        weight_decay=hp_tune.weight_decay,
        loss="mean_squared_error",
    )
    ctx.W_tune = ctx.M_tune.get_weights()


def _fit_function_aux_transfer_model(
    cfg: DictConfig,
    ctx: RunContext,
    M_base: keras.Model,
    W_base: t.Any,
    D_t_pdxo: Dataset,
) -> None:
    """Fold-specific response + drug/tumor function-aux transfer path."""
    hp_tune = cfg.xfer.hyper
    aux_cfg = cfg.xfer.hyper.aux

    drug_dim = int(_cfg_get(aux_cfg, "drug_n_components", 16))
    cell_dim = int(_cfg_get(aux_cfg, "cell_n_components", 16))
    seed = int(_cfg_get(aux_cfg, "seed", 1441))

    drug_targets, cell_targets = make_fold_pdxo_function_aux_targets(
        D_t_pdxo,
        drug_n_components=drug_dim,
        cell_n_components=cell_dim,
        seed=seed,
    )

    log.debug(
        "Fold aux targets | drugs=%s | tumors=%s",
        drug_targets.shape,
        cell_targets.shape,
    )

    M_aux_base = None
    gen = None
    seq = None

    try:
        # Critical: never attach aux heads to the long-lived base model.
        M_aux_base = model_utils.clone_model_from_weights(M_base, W_base)
        M_aux_base.trainable = True

        ctx.M_aux = screendl.add_function_auxiliary_heads(
            M_aux_base,
            drug_aux_dim=drug_dim,
            cell_aux_dim=cell_dim,
            drug_hidden_dims=_cfg_get(aux_cfg, "drug_hidden_dims", None),
            cell_hidden_dims=_cfg_get(aux_cfg, "cell_hidden_dims", None),
            activation=_cfg_get(
                aux_cfg,
                "activation",
                hp_tune.get("activation", "relu"),
            ),
            use_l2=bool(_cfg_get(aux_cfg, "use_l2", False)),
            l2_factor=float(_cfg_get(aux_cfg, "l2_factor", 0.01)),
        )

        model_utils.freeze_layers(
            ctx.M_aux,
            names=safe_lconfig_as_tuple(hp_tune.frozen_layer_names),
            prefixes=safe_lconfig_as_tuple(hp_tune.frozen_layer_prefixes),
        )

        ctx.M_aux.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp_tune.learning_rate,
                weight_decay=hp_tune.weight_decay,
            ),
            loss={
                "response": "mean_squared_error",
                "drug_function": "mean_squared_error",
                "cell_function": "mean_squared_error",
            },
            loss_weights={
                "response": float(_cfg_get(aux_cfg, "response_loss_weight", 1.0)),
                "drug_function": float(_cfg_get(aux_cfg, "drug_loss_weight", 0.001)),
                "cell_function": float(_cfg_get(aux_cfg, "cell_loss_weight", 0.001)),
            },
            jit_compile=False,
        )

        gen = FunctionAuxResponseGenerator(D_t_pdxo, hp_tune.batch_size)
        seq = gen.flow_from_dataset(
            D_t_pdxo,
            drug_function_targets=drug_targets,
            cell_function_targets=cell_targets,
            shuffle=True,
            seed=seed,
        )

        ctx.M_aux.fit(
            seq,
            epochs=hp_tune.epochs,
            verbose=0,
        )

        ctx.M_tune = screendl.get_response_model(ctx.M_aux)
        ctx.W_tune = ctx.M_tune.get_weights()

    finally:
        del seq
        del gen
        del M_aux_base
        gc.collect()


def run_transfer_model(
    cfg: DictConfig,
    ctx: RunContext,
    M_base: keras.Model,
    W_base: t.Any,
    D_t_pdxo: Dataset,
    output_path: Path,
    model_dir: Path,
    *,
    fold_i: int,
    heldout_cell_id: str,
) -> None:
    """Fine-tune transfer model and write transfer predictions."""
    if ctx.D_e_pdxo_full is None or ctx.D_t_pdxo_full is None:
        raise RuntimeError("Fold datasets have not been prepared.")

    aux_cfg = _cfg_get(cfg.xfer.hyper, "aux", None)
    if _is_enabled(aux_cfg):
        _fit_function_aux_transfer_model(cfg, ctx, M_base, W_base, D_t_pdxo)
    else:
        _fit_response_only_transfer_model(cfg, ctx, M_base, W_base, D_t_pdxo)

    if ctx.M_tune is None or ctx.W_tune is None:
        raise RuntimeError("Transfer model did not produce tuned model/weights.")

    model_utils.save_keras_model(ctx.M_tune, model_dir / "xfer.keras")

    ctx.R_tune = eval_utils.get_predictions_vs_background(
        M=ctx.M_tune,
        D_t=ctx.D_e_pdxo_full,
        D_b=ctx.D_t_pdxo_full,
        W_t=None,
        W_b=None,
        model="xfer",
        was_screened=False,
    )
    ctx.R_tune = _annotate_fold(
        ctx.R_tune,
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
    )
    write_predictions_chunk(ctx.R_tune, output_path)


def run_screenahead_model(
    cfg: DictConfig,
    ctx: RunContext,
    dataset: Dataset,
    D_pdxo: Dataset,
    D_t_pdxo: Dataset,
    D_e_pdxo: Dataset,
    output_path: Path,
    model_dir: Path,
    *,
    fold_i: int,
    heldout_cell_id: str,
) -> None:
    """Fine-tune ScreenAhead model and write screen predictions."""
    if ctx.M_tune is None or ctx.W_tune is None:
        raise RuntimeError("Transfer model has not been fit.")
    if ctx.D_e_pdxo_full is None or ctx.D_t_pdxo_full is None:
        raise RuntimeError("Fold datasets have not been prepared.")

    hp_screen = cfg.screenahead.hyper
    opt_screen = cfg.screenahead.opt

    ctx.drug_selector = SELECTORS[opt_screen.selector](
        dataset.select_cells(D_t_pdxo.cell_ids).select_drugs(D_pdxo.drug_ids),
        na_threshold=opt_screen.na_thresh,
        seed=opt_screen.seed,
    )

    ctx.D_s_pdxo, _ = get_screenahead_split(
        D_e_pdxo,
        drug_selector=ctx.drug_selector,
        num_drugs=opt_screen.n_drugs,
        exclude_drugs=opt_screen.exclude_drugs,
    )
    save_fold_manifest(
        model_dir / "manifest.json",
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
        screened_drug_ids=ctx.D_s_pdxo.drug_ids,
    )

    ctx.M_screen = model_utils.configure_screenahead_model(
        ctx.M_tune,
        initial_weights=ctx.W_tune,
        frozen_layer_prefixes=safe_lconfig_as_tuple(hp_screen.frozen_layer_prefixes),
        frozen_layer_names=safe_lconfig_as_tuple(hp_screen.frozen_layer_names),
        training=False,
    )

    if ctx.M_screen is ctx.M_tune:
        log.warning("ScreenAhead model is the same object as transfer model.")

    ctx.M_screen = model_utils.fit_screenahead_model(
        model=ctx.M_screen,
        dataset=ctx.D_s_pdxo,
        batch_size=hp_screen.batch_size,
        epochs=hp_screen.epochs,
        learning_rate=hp_screen.learning_rate,
        weight_decay=getattr(hp_screen, "weight_decay", None),
        loss="mean_squared_error",
    )

    ctx.W_screen = ctx.M_screen.get_weights()
    model_utils.save_keras_model(ctx.M_screen, model_dir / "screenahead.keras")

    ctx.R_screen = eval_utils.get_predictions_vs_background(
        M=ctx.M_screen,
        D_t=ctx.D_e_pdxo_full,
        D_b=ctx.D_t_pdxo_full,
        W_t=ctx.W_screen if cfg.experiment.background_correction else None,
        W_b=ctx.W_tune if cfg.experiment.background_correction else None,
        model="screen",
    )
    ctx.R_screen["was_screened"] = ctx.R_screen["drug_id"].isin(ctx.D_s_pdxo.drug_ids)
    ctx.R_screen = _annotate_fold(
        ctx.R_screen,
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
    )
    write_predictions_chunk(ctx.R_screen, output_path)


def run_fold(
    cfg: DictConfig,
    exp: ExperimentData,
    D_t_pdxo: Dataset,
    D_e_pdxo: Dataset,
    output_path: Path,
    *,
    fold_i: int,
) -> None:
    """Run one leave-one-out fold and clean all per-fold objects."""
    ctx = RunContext()
    exp.require_ready()

    assert exp.dataset is not None
    assert exp.D_pdxo is not None
    assert exp.all_drug_ids is not None
    assert exp.M_base is not None
    assert exp.W_base is not None

    heldout_cell_id = str(D_e_pdxo.cell_ids[0])
    model_dir = _fold_model_dir(fold_i=fold_i, heldout_cell_id=heldout_cell_id)
    model_utils.save_keras_model(exp.M_base, model_dir / "base.keras")

    try:
        prepare_fold_datasets(
            ctx=ctx,
            D_t_pdxo=D_t_pdxo,
            D_e_pdxo=D_e_pdxo,
            all_drug_ids=exp.all_drug_ids,
        )

        run_base_predictions(
            ctx=ctx,
            M_base=exp.M_base,
            output_path=output_path,
            fold_i=fold_i,
            heldout_cell_id=heldout_cell_id,
        )

        run_transfer_model(
            cfg=cfg,
            ctx=ctx,
            M_base=exp.M_base,
            W_base=exp.W_base,
            D_t_pdxo=D_t_pdxo,
            output_path=output_path,
            model_dir=model_dir,
            fold_i=fold_i,
            heldout_cell_id=heldout_cell_id,
        )

        run_screenahead_model(
            cfg=cfg,
            ctx=ctx,
            dataset=exp.dataset,
            D_pdxo=exp.D_pdxo,
            D_t_pdxo=D_t_pdxo,
            D_e_pdxo=D_e_pdxo,
            output_path=output_path,
            model_dir=model_dir,
            fold_i=fold_i,
            heldout_cell_id=heldout_cell_id,
        )

    finally:
        ctx.clear()


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="pdxo_validation",
)
def run(cfg: DictConfig) -> None:
    """Runs leave-one-out cross validation for the HCI PDMC dataset."""
    configure_runtime()
    log_memory("After TensorFlow runtime setup")

    exp = prepare_experiment(cfg)

    try:
        exp.require_ready()
        D_pdxo = exp.D_pdxo
        assert D_pdxo is not None

        output_path = Path("predictions.csv")
        if output_path.exists():
            output_path.unlink()

        n_folds = sum(
            1 for cell_id in set(D_pdxo.cell_ids) if ":AUG" not in str(cell_id)
        )
        split_gen = loo_split_generator(D_pdxo)

        log.info("Running experiment...")
        for fold_i, (D_t_pdxo, D_e_pdxo) in enumerate(tqdm(split_gen, total=n_folds)):
            log_memory(f"Fold {fold_i} start")

            run_fold(
                cfg=cfg,
                exp=exp,
                D_t_pdxo=D_t_pdxo,
                D_e_pdxo=D_e_pdxo,
                output_path=output_path,
                fold_i=fold_i,
            )

            model_utils.trim_malloc()
            log_memory(f"Fold {fold_i} end")

    finally:
        exp.clear()
        reset_keras_runtime()
        model_utils.trim_malloc()


if __name__ == "__main__":
    run()
