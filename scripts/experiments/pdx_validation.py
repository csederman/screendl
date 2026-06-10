#!/usr/bin/env python
"""Runs transfer learning experiments on the HCI PDXO/PDX dataset."""

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
from screendl.model import utils as model_utils
from screendl.data import preprocess_pdmc_screendl_datasets, PreprocessingArtifacts
from screendl.pipelines.basic.screendl import (
    data_loader,
    data_splitter,
    model_builder as base_model_builder,
    model_trainer as base_model_trainer,
)
from screendl.pipelines.pdmc.utils import (
    get_screenahead_dataset,
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


def _pdx_clinical_path_is_set(cfg: "DictConfig") -> bool:
    """Return False to skip mRECIST / PDX-animal evaluation (PDXO-only run)."""
    if not cfg.experiment.get("use_pdx_clinical", True):
        return False
    p = cfg.experiment.get("pdx_obs_path")
    if p is None:
        return False
    s = str(p).strip()
    return bool(s) and s.lower() not in ("none", "null")


def get_id_prefix(pdmc_id: str) -> str:
    """Gets matching prefixes to avoid data leak."""
    if pdmc_id.startswith("HCI"):
        return pdmc_id[:6]
    elif pdmc_id.startswith("BCM"):
        return pdmc_id[:7]
    elif pdmc_id.startswith("TOW"):
        return pdmc_id[:5]
    else:
        return pdmc_id


def _is_enabled(obj: t.Any) -> bool:
    """Return True if a config node exists and has enabled=true."""
    return bool(_cfg_get(obj, "enabled", False))


def _fold_model_dir(
    *,
    fold_i: int,
    heldout_cell_id: str,
    root: str | Path = "models",
) -> Path:
    """Return the model output directory for one held-out PDXO/PDX fold."""
    return (
        Path(root)
        / f"fold_{fold_i:03d}__{model_utils.safe_path_token(heldout_cell_id)}"
    )

    Yields
    ------
    Iterator[t.Generator[t.Tuple[Dataset, Dataset], None, None]]
        Iterator of (train, test) dataset tuples
    """
    pdmc_ids = set(D.cell_ids)
    for test_id in pdmc_ids:
        id_prefix = get_id_prefix(test_id)
        train_ids = [x for x in pdmc_ids if not str(x).startswith(id_prefix)]
        train_ds = D.select_cells(train_ids, name="train")
        test_ds = D.select_cells([test_id], name="test")
        yield train_ds, test_ds


def get_screenahead_split(
    dataset: Dataset,
    drug_selector: DrugSelectorType,
    num_drugs: int,
    exclude_drugs: t.List[str] | None = None,
) -> t.Tuple[Dataset, Dataset]:
    """"""
    drug_choices = set(dataset.drug_ids)
    if exclude_drugs is not None:
        drug_choices = set(x for x in drug_choices if x not in exclude_drugs)
    screen_drugs = drug_selector.select(num_drugs, choices=drug_choices)
    holdout_drugs = drug_choices.difference(screen_drugs)

    screen_ds = dataset.select_drugs(screen_drugs, name="this_screen")
    holdout_ds = dataset.select_drugs(holdout_drugs, name="this_holdout")

    return screen_ds, holdout_ds


def get_screenahead_dataset(
    dataset: Dataset,
    mode: t.Literal["screen-all", "screen-selected", "screen-non-pdx"],
    drug_selector: DrugSelectorType,
    num_drugs: int,
    exclude_drugs: t.List[str] | None = None,
) -> Dataset:
    """Build the ScreenAhead fine-tuning drug set for one held-out tumor."""
    if mode == "screen-all":
        screen_drugs = set(dataset.drug_ids)
        if exclude_drugs is not None:
            screen_drugs -= set(exclude_drugs)
        screen_ds = dataset.select_drugs(screen_drugs, name="this_screen")
    elif mode in {"screen-selected", "screen-non-pdx"}:
        screen_ds, _ = get_screenahead_split(
            dataset,
            drug_selector=drug_selector,
            num_drugs=num_drugs,
            exclude_drugs=exclude_drugs,
        )
    else:
        raise ValueError(f"Invalid screenahead mode (got {mode})")

def save_fold_manifest(
    path: str | Path,
    *,
    fold_i: int,
    heldout_cell_id: str,
    screened_drug_ids: t.Iterable[t.Any] | None = None,
    pdx_drug_ids: t.Iterable[t.Any] | None = None,
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
        "pdx_drug_ids": (
            None if pdx_drug_ids is None else [str(x) for x in pdx_drug_ids]
        ),
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_pdx_data(cfg: DictConfig, pdmc_ds: Dataset) -> Dataset:
    """Loads PDX *in vivo* clinical outcomes (mRECIST). Optional; use empty dataset to skip."""
    if not _pdx_clinical_path_is_set(cfg):
        # cdrpy Dataset.obs must include id, cell_id, drug_id, label (see cdrpy.datasets.base)
        pdx_obs = pd.DataFrame(columns=["id", "cell_id", "drug_id", "label"])
        return Dataset(
            pdx_obs,
            cell_encoders=pdmc_ds.cell_encoders,
            drug_encoders=pdmc_ds.drug_encoders,
            name="pdx_ds",
        )

    pdx_obs = pd.read_csv(cfg.experiment.pdx_obs_path)
    pdx_obs = pdx_obs[pdx_obs["cell_id"].isin(pdmc_ds.cell_ids)]
    pdx_obs = pdx_obs[pdx_obs["drug_id"].isin(pdmc_ds.drug_ids)]
    pdx_obs["label"] = pdx_obs["mRECIST"].isin(["CR", "PR", "SD"]).astype(int)

    pdx_dataset = Dataset(
        pdx_obs,
        cell_encoders=pdmc_ds.cell_encoders,
        drug_encoders=pdmc_ds.drug_encoders,
        name="pdx_ds",
    )

    if cfg.experiment.pdx_ids is not None:
        pdx_dataset = pdx_dataset.select_cells(cfg.experiment.pdx_ids)

    return pdx_dataset


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


def get_exclude_drugs(opts: DictConfig, drug_ids: list[str]) -> list[str] | None:
    """Parse drugs to exclude from ScreenAhead options config."""
    return drug_ids if opts.mode == "screen-non-pdx" else opts.exclude_drugs


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
    """Build N-1 PDXO drug and cell functional aux targets for one fold.

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

    cell_centered = response_mat.sub(response_mat.mean(axis=1), axis=0)
    cell_sim = cell_centered.T.corr()

    drug_targets = _similarity_matrix_to_targets(
        drug_sim,
        n_components=drug_n_components,
        prefix="drug_function",
        seed=seed,
    )
    cell_targets = _similarity_matrix_to_targets(
        cell_sim,
        n_components=cell_n_components,
        prefix="cell_function",
        seed=seed,
    )

    return drug_targets, cell_targets


@dataclass
class RunContext:
    """Mutable per-fold state that can be cleared after each fold."""

    R_pdxo_base: pd.DataFrame | None = None
    R_pdxo_tune: pd.DataFrame | None = None
    R_pdxo_screen: pd.DataFrame | None = None
    R_pdx_base: pd.DataFrame | None = None
    R_pdx_tune: pd.DataFrame | None = None
    R_pdx_screen: pd.DataFrame | None = None

    M_aux: keras.Model | None = None
    M_tune: keras.Model | None = None
    M_screen: keras.Model | None = None
    W_tune: t.Any = None
    W_screen: t.Any = None

    D_e_pdx: Dataset | None = None
    D_t_pdxo_full: Dataset | None = None
    D_e_pdxo_full: Dataset | None = None
    D_t_pdx_full: Dataset | None = None
    D_e_pdx_full: Dataset | None = None
    D_s_pdxo: Dataset | None = None

    drug_selector: t.Any = None

    def clear(self) -> None:
        """Drop per-fold references and run garbage collection."""
        model_utils.clear_compiled_model(self.M_aux)
        model_utils.clear_compiled_model(self.M_tune)
        model_utils.clear_compiled_model(self.M_screen)

        self.R_pdxo_base = None
        self.R_pdxo_tune = None
        self.R_pdxo_screen = None
        self.R_pdx_base = None
        self.R_pdx_tune = None
        self.R_pdx_screen = None

        self.M_aux = None
        self.M_tune = None
        self.M_screen = None
        self.W_tune = None
        self.W_screen = None

        self.D_e_pdx = None
        self.D_t_pdxo_full = None
        self.D_e_pdxo_full = None
        self.D_t_pdx_full = None
        self.D_e_pdx_full = None
        self.D_s_pdxo = None

        self.drug_selector = None

        cleanup_objects()
        model_utils.trim_malloc()


@dataclass
class ExperimentData:
    """Prepared datasets and base-model state for a run."""

    dataset: Dataset | None
    D_t_cell: Dataset | None
    D_v_cell: Dataset | None
    D_pdxo: Dataset | None
    D_pdx: Dataset | None
    all_drug_ids: list[str] | None
    pdx_ids: list[str] | None
    pdx_drug_ids: list[str] | None
    preprocessing_artifacts: PreprocessingArtifacts | None = None
    M_base: keras.Model | None = None
    W_base: t.Any = None
    M_base_trainable_state: model_utils.TrainableState | None = None

    def clear(self) -> None:
        """Drop run-level references and run garbage collection."""
        model_utils.clear_compiled_model(self.M_base)

        self.dataset = None
        self.D_t_cell = None
        self.D_v_cell = None
        self.D_pdxo = None
        self.D_pdx = None
        self.all_drug_ids = None
        self.pdx_ids = None
        self.pdx_drug_ids = None
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
        if self.D_pdx is None:
            raise RuntimeError("PDX dataset has been cleared.")
        if self.all_drug_ids is None:
            raise RuntimeError("Drug IDs have been cleared.")
        if self.pdx_ids is None:
            raise RuntimeError("PDX IDs have been cleared.")
        if self.pdx_drug_ids is None:
            raise RuntimeError("PDX drug IDs have been cleared.")
        if self.M_base is None:
            raise RuntimeError("Base model has been cleared.")
        if self.W_base is None:
            raise RuntimeError("Base weights have been cleared.")
        if self.M_base_trainable_state is None:
            raise RuntimeError("Base trainable state has been cleared.")


def prepare_experiment(cfg: DictConfig) -> ExperimentData:
    """Load, preprocess, align PDX data, pretrain base model, and save base state."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info("Loading %s...", dataset_name)
    dataset = data_loader(cfg)
    all_drug_ids = list(dataset.drug_encoders["mol"].data.index)

    log.info("Splitting %s...", dataset_name)
    D_t_cell, D_v_cell, D_pdxo = data_splitter(cfg, dataset)

    log.info("Preprocessing %s...", dataset_name)
    D_t_cell, D_v_cell, D_pdxo, preprocessing_artifacts = data_preprocessor(
        cfg,
        D_t_cell,
        D_v_cell,
        D_pdxo,
    )

    log.info("Loading PDX data...")
    D_pdx = load_pdx_data(cfg, D_pdxo)
    pdx_ids = sorted(set(D_pdx.cell_ids))
    pdx_drug_ids = sorted(set(D_pdx.drug_ids))

    # With PDX clinical labels: keep drugs seen in cell lines or PDX animals (exclude PDXO-only).
    # Without PDX: keep all PDXO drugs that overlap the cell-line drug universe for organoid eval.
    if D_pdx.size > 0:
        target_drugs = set(D_pdx.drug_ids).union(D_t_cell.drug_ids).union(D_v_cell.drug_ids)
    else:
        target_drugs = set(D_pdxo.drug_ids).union(D_t_cell.drug_ids).union(D_v_cell.drug_ids)
    D_pdxo = D_pdxo.select_drugs(target_drugs, name="pdmc_ds")

    log.info("Building %s...", model_name)
    M_base = base_model_builder(cfg, D_t_cell)

    pdxo_train_ids = [
        cell_id for cell_id in set(D_pdxo.cell_ids) if cell_id not in pdx_ids
    ]
    _D_t_pdxo = D_pdxo.select_cells(pdxo_train_ids, name="pdmc_train_ds")
    del _D_t_pdxo

    log.info("Pretraining %s...", model_name)
    M_base = base_model_trainer(cfg, M_base, D_t_cell, D_v_cell)
    model_utils.save_keras_model(M_base, Path("models") / "base.keras")

    return ExperimentData(
        dataset=dataset,
        D_t_cell=D_t_cell,
        D_v_cell=D_v_cell,
        D_pdxo=D_pdxo,
        D_pdx=D_pdx,
        all_drug_ids=all_drug_ids,
        pdx_ids=pdx_ids,
        pdx_drug_ids=pdx_drug_ids,
        preprocessing_artifacts=preprocessing_artifacts,
        M_base=M_base,
        W_base=M_base.get_weights(),
        M_base_trainable_state=model_utils.get_trainable_state(M_base),
    )


def prepare_fold_datasets(
    ctx: RunContext,
    exp: ExperimentData,
    D_t_pdxo: Dataset,
    D_e_pdxo: Dataset,
) -> bool:
    """Create expanded PDXO/PDX target and background datasets for one fold.

    Returns False when the held-out PDMC has no matching PDX data.
    """
    exp.require_ready()
    assert exp.D_pdx is not None
    assert exp.all_drug_ids is not None

    ctx.D_e_pdx = exp.D_pdx.select_cells(set(D_e_pdxo.cell_ids))
    if ctx.D_e_pdx.size == 0:
        return False

    D_bg_pdxo = D_t_pdxo.select_drugs(set(D_e_pdxo.drug_ids))
    D_bg_pdx = D_t_pdxo.select_drugs(set(ctx.D_e_pdx.drug_ids))
    bg_tumor_ids = list(set(D_bg_pdxo.cell_ids))

    ctx.D_e_pdxo_full = data_utils.expand_dataset(
        D_e_pdxo,
        cell_ids=[D_e_pdxo.cell_ids[0]],
        drug_ids=exp.all_drug_ids,
    )
    ctx.D_t_pdxo_full = data_utils.expand_dataset(
        D_bg_pdxo,
        cell_ids=bg_tumor_ids,
        drug_ids=exp.all_drug_ids,
    )
    ctx.D_e_pdx_full = data_utils.expand_dataset(
        ctx.D_e_pdx,
        cell_ids=[ctx.D_e_pdx.cell_ids[0]],
        drug_ids=exp.all_drug_ids,
    )
    ctx.D_t_pdx_full = data_utils.expand_dataset(
        D_bg_pdx,
        cell_ids=bg_tumor_ids,
        drug_ids=exp.all_drug_ids,
    )

    return True


def initialize_drug_selector(
    cfg: DictConfig,
    ctx: RunContext,
    exp: ExperimentData,
    D_t_pdxo: Dataset,
) -> None:
    """Initialize the configured ScreenAhead drug selector for a fold."""
    exp.require_ready()
    assert exp.dataset is not None
    assert exp.D_pdxo is not None

    opt_screen = cfg.screenahead.opt

    W_base = M_base.get_weights()
    split_gen = loo_split_generator(D_pdxo)

    pdx_results = []
    pdxo_results = []
    for D_t_pdxo, D_e_pdxo in tqdm(split_gen, total=D_pdxo.n_cells):
        D_e_pdx = D_pdx.select_cells(set(D_e_pdxo.cell_ids))

        # create background datasets against which we will normalize the predictions
        D_bg_pdxo = D_t_pdxo.select_drugs(set(D_e_pdxo.drug_ids))

        bg_tumor_ids = list(set(D_bg_pdxo.cell_ids))
        # PDX-animal grid + background (only when clinical PDX rows exist for this organoid)
        if D_e_pdx.size > 0:
            D_bg_pdx = D_t_pdxo.select_drugs(set(D_e_pdx.drug_ids))
            D_e_pdx_full = data_utils.expand_dataset(
                D_e_pdx, [D_e_pdx.cell_ids[0]], all_drug_ids
            )
            D_bg_pdx_full = data_utils.expand_dataset(D_bg_pdx, bg_tumor_ids, all_drug_ids)
        else:
            D_e_pdx_full = None
            D_bg_pdx_full = None
        D_e_pdxo_full = data_utils.expand_dataset(
            D_e_pdxo, [D_e_pdxo.cell_ids[0]], all_drug_ids
        )
        D_bg_pdxo_full = data_utils.expand_dataset(D_bg_pdxo, bg_tumor_ids, all_drug_ids)

    log.debug(
        "Fold aux targets | drugs=%s | cells=%s",
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
                _cfg_get(hp_tune, "activation", "relu"),
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
    exp: ExperimentData,
    D_t_pdxo: Dataset,
    output_pdxo_path: Path,
    output_pdx_path: Path,
    model_dir: Path,
    *,
    fold_i: int,
    heldout_cell_id: str,
) -> None:
    """Fine-tune transfer model and write transfer predictions."""
    exp.require_ready()
    assert exp.M_base is not None
    assert exp.W_base is not None
    assert ctx.D_e_pdxo_full is not None
    assert ctx.D_t_pdxo_full is not None
    assert ctx.D_e_pdx_full is not None
    assert ctx.D_t_pdx_full is not None

    aux_cfg = _cfg_get(cfg.xfer.hyper, "aux", None)
    if _is_enabled(aux_cfg):
        _fit_function_aux_transfer_model(cfg, ctx, exp.M_base, exp.W_base, D_t_pdxo)
    else:
        _fit_response_only_transfer_model(cfg, ctx, exp.M_base, exp.W_base, D_t_pdxo)

    if ctx.M_tune is None or ctx.W_tune is None:
        raise RuntimeError("Transfer model did not produce tuned model/weights.")

    model_utils.save_keras_model(ctx.M_tune, model_dir / "xfer.keras")

    ctx.R_pdxo_tune = eval_utils.get_predictions_vs_background(
        M=ctx.M_tune,
        D_t=ctx.D_e_pdxo_full,
        D_b=ctx.D_t_pdxo_full,
        W_t=None,
        W_b=None,
        model="xfer",
        was_screened=False,
    )
    ctx.R_pdxo_tune = _annotate_fold(
        ctx.R_pdxo_tune,
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
    )
    write_predictions_chunk(ctx.R_pdxo_tune, output_pdxo_path)

    ctx.R_pdx_tune = eval_utils.get_predictions_vs_background(
        M=ctx.M_tune,
        D_t=ctx.D_e_pdx_full,
        D_b=ctx.D_t_pdx_full,
        W_t=None,
        W_b=None,
        model="xfer",
        was_screened=False,
    )
    ctx.R_pdx_tune = _annotate_fold(
        ctx.R_pdx_tune,
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
    )
    write_predictions_chunk(ctx.R_pdx_tune, output_pdx_path)


def run_screenahead_model(
    cfg: DictConfig,
    ctx: RunContext,
    exp: ExperimentData,
    D_e_pdxo: Dataset,
    output_pdxo_path: Path,
    output_pdx_path: Path,
    model_dir: Path,
    *,
    fold_i: int,
    heldout_cell_id: str,
) -> None:
    """Fine-tune ScreenAhead model and write PDXO/PDX screen predictions."""
    exp.require_ready()
    assert exp.pdx_drug_ids is not None
    assert ctx.M_tune is not None
    assert ctx.W_tune is not None
    assert ctx.drug_selector is not None
    assert ctx.D_e_pdxo_full is not None
    assert ctx.D_t_pdxo_full is not None
    assert ctx.D_e_pdx_full is not None
    assert ctx.D_t_pdx_full is not None

    hp_screen = cfg.screenahead.hyper
    opt_screen = cfg.screenahead.opt

    ctx.D_s_pdxo = get_screenahead_dataset(
        D_e_pdxo,
        mode=opt_screen.mode,
        drug_selector=ctx.drug_selector,
        num_drugs=opt_screen.n_drugs,
        exclude_drugs=get_exclude_drugs(opt_screen, exp.pdx_drug_ids),
    )

    save_fold_manifest(
        model_dir / "manifest.json",
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
        screened_drug_ids=ctx.D_s_pdxo.drug_ids,
        pdx_drug_ids=exp.pdx_drug_ids,
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

    ctx.R_pdxo_screen = eval_utils.get_predictions_vs_background(
        M=ctx.M_screen,
        D_t=ctx.D_e_pdxo_full,
        D_b=ctx.D_t_pdxo_full,
        W_t=ctx.W_screen if cfg.experiment.background_correction else None,
        W_b=ctx.W_tune if cfg.experiment.background_correction else None,
        model="screen",
    )
    ctx.R_pdxo_screen["was_screened"] = ctx.R_pdxo_screen["drug_id"].isin(
        ctx.D_s_pdxo.drug_ids
    )
    ctx.R_pdxo_screen = _annotate_fold(
        ctx.R_pdxo_screen,
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
    )
    write_predictions_chunk(ctx.R_pdxo_screen, output_pdxo_path)

    ctx.R_pdx_screen = eval_utils.get_predictions_vs_background(
        M=ctx.M_screen,
        D_t=ctx.D_e_pdx_full,
        D_b=ctx.D_t_pdx_full,
        W_t=ctx.W_screen if cfg.experiment.background_correction else None,
        W_b=ctx.W_tune if cfg.experiment.background_correction else None,
        model="screen",
    )
    ctx.R_pdx_screen["was_screened"] = ctx.R_pdx_screen["drug_id"].isin(
        ctx.D_s_pdxo.drug_ids
    )
    ctx.R_pdx_screen = _annotate_fold(
        ctx.R_pdx_screen,
        fold_i=fold_i,
        heldout_cell_id=heldout_cell_id,
    )
    write_predictions_chunk(ctx.R_pdx_screen, output_pdx_path)


def run_fold(
    cfg: DictConfig,
    exp: ExperimentData,
    D_t_pdxo: Dataset,
    D_e_pdxo: Dataset,
    output_pdxo_path: Path,
    output_pdx_path: Path,
    *,
    fold_i: int,
) -> bool:
    """Run one leave-one-PDMC-out fold.

    Returns True if a matching PDX target existed and predictions were written.
    """
    ctx = RunContext()
    exp.require_ready()

    heldout_cell_id = str(D_e_pdxo.cell_ids[0])
    model_dir = _fold_model_dir(fold_i=fold_i, heldout_cell_id=heldout_cell_id)

    try:
        has_pdx_target = prepare_fold_datasets(
            ctx=ctx,
            exp=exp,
            D_t_pdxo=D_t_pdxo,
            D_e_pdxo=D_e_pdxo,
        )
        if not has_pdx_target:
            return False

        assert exp.M_base is not None
        model_utils.save_keras_model(exp.M_base, model_dir / "base.keras")

        initialize_drug_selector(cfg, ctx, exp, D_t_pdxo)

        run_base_predictions(
            ctx=ctx,
            exp=exp,
            output_pdxo_path=output_pdxo_path,
            output_pdx_path=output_pdx_path,
            fold_i=fold_i,
            heldout_cell_id=heldout_cell_id,
        )

        run_transfer_model(
            cfg=cfg,
            ctx=ctx,
            exp=exp,
            D_t_pdxo=D_t_pdxo,
            output_pdxo_path=output_pdxo_path,
            output_pdx_path=output_pdx_path,
            model_dir=model_dir,
            fold_i=fold_i,
            heldout_cell_id=heldout_cell_id,
        )

        run_screenahead_model(
            cfg=cfg,
            ctx=ctx,
            exp=exp,
            D_e_pdxo=D_e_pdxo,
            output_pdxo_path=output_pdxo_path,
            output_pdx_path=output_pdx_path,
            model_dir=model_dir,
            fold_i=fold_i,
            heldout_cell_id=heldout_cell_id,
        )

        return True

    finally:
        ctx.clear()


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="pdx_validation",
)
def run(cfg: DictConfig) -> None:
    """Run leave-one-PDMC-out validation against matched PDX outcomes."""
    configure_runtime()
    log_memory("After TensorFlow runtime setup")

    exp = prepare_experiment(cfg)

    try:
        exp.require_ready()
        assert exp.D_pdxo is not None

        output_pdxo_path = Path("predictions_pdxo.csv")
        output_pdx_path = Path("predictions_pdx.csv")

        for path in (output_pdxo_path, output_pdx_path):
            if path.exists():
                path.unlink()

        n_folds = sum(
            1 for cell_id in set(exp.D_pdxo.cell_ids) if ":AUG" not in str(cell_id)
        )
        split_gen = loo_split_generator(exp.D_pdxo)

        log.info("Running experiment...")
        n_completed = 0
        n_skipped = 0

        for fold_i, (D_t_pdxo, D_e_pdxo) in enumerate(tqdm(split_gen, total=n_folds)):
            log_memory(f"Fold {fold_i} start")

            completed = run_fold(
                cfg=cfg,
                exp=exp,
                D_t_pdxo=D_t_pdxo,
                D_e_pdxo=D_e_pdxo,
                output_pdxo_path=output_pdxo_path,
                output_pdx_path=output_pdx_path,
                fold_i=fold_i,
            )

            if completed:
                n_completed += 1
            else:
                n_skipped += 1

            model_utils.trim_malloc()
            log_memory(f"Fold {fold_i} end")

        log.info(
            "Finished experiment | completed_folds=%d | skipped_no_pdx=%d",
            n_completed,
            n_skipped,
        )

    if pdx_results:
        pdx_results = pd.concat(pdx_results)
        pdx_results.to_csv("predictions_pdx.csv", index=False)
    else:
        log.info("No PDX clinical evaluation (empty pdx_ds); skipping predictions_pdx.csv")


if __name__ == "__main__":
    run()
