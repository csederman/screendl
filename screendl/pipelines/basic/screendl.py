"""
Run utilities for ScreenDL.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from cdrpy.datasets import Dataset
from cdrpy.mapper import BatchedResponseGenerator

from screendl import model as screendl
from screendl.utils.evaluation import ScoreDict, get_eval_metrics, make_pred_df
from screendl.utils.serialization import to_jsonable
from screendl.data.preprocess import (
    PreprocessingArtifacts,
    preprocess_screendl_datasets,
)

log = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """Outputs from a pipeline run."""

    model: keras.Model
    scores: dict[str, ScoreDict]
    datasets: dict[str, Dataset]


def cfg_get(cfg: t.Any, key: str, default: t.Any = None) -> t.Any:
    """Safely read nested OmegaConf keys using dot notation."""
    try:
        value = OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default
    return default if value is None else value


def data_loader(cfg: DictConfig) -> Dataset:
    """Load the input dataset."""
    paths = cfg.dataset.sources

    mol_path = paths.screendl.mol
    exp_path = paths.screendl.exp
    mut_path = paths.screendl.mut if cfg.model.feat.use_mut else None
    cnv_path = paths.screendl.cnv if cfg.model.feat.use_cnv else None
    ont_path = paths.screendl.ont if cfg.model.feat.use_ont else None

    drug_encoders = screendl.load_drug_features(mol_path)
    cell_encoders = screendl.load_cell_features(
        exp_path=exp_path,
        mut_path=mut_path,
        cnv_path=cnv_path,
        ont_path=ont_path,
    )

    cell_meta = None
    if hasattr(paths, "cell_meta"):
        cell_meta = pd.read_csv(paths.cell_meta, index_col=0)

    drug_meta = None
    if hasattr(paths, "drug_meta"):
        drug_meta = pd.read_csv(paths.drug_meta, index_col=0)

    return Dataset.from_csv(
        paths.labels,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
        cell_meta=cell_meta,
        drug_meta=drug_meta,
        name=cfg.dataset.name,
    )


def data_splitter(
    cfg: DictConfig, dataset: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
    """Split the dataset into train/validation/test sets."""
    split_id = cfg.dataset.split.id
    split_dir = cfg.dataset.split.dir
    split_name = cfg.dataset.split.name
    split_path = os.path.join(split_dir, split_name, f"fold_{split_id}.pkl")

    with open(split_path, "rb") as fh:
        split = pickle.load(fh)

    return (
        dataset.select(split["train"], name="train"),
        dataset.select(split["val"], name="val"),
        dataset.select(split["test"], name="test"),
    )


def data_preprocessor(
    cfg: DictConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> tuple[Dataset, Dataset | None, Dataset | None, PreprocessingArtifacts]:
    """Preprocess expression, CNV, and response values."""
    return preprocess_screendl_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        exp_norm_method=cfg.dataset.preprocess.norm_exp,
        response_norm_method=cfg.dataset.preprocess.norm,
        normalize_cnv=True,
        artifact_path="preprocessing_artifacts.pkl",
    )


def _similarity_matrix_to_targets(
    sim_df: pd.DataFrame,
    *,
    n_components: int,
    prefix: str,
    fill_value: float = 0.0,
    random_state: int = 1441,
) -> pd.DataFrame:
    """Convert a square similarity matrix to fixed-width PCA targets."""
    common = sim_df.index.intersection(sim_df.columns)
    sim_df = sim_df.loc[common, common]

    x = sim_df.fillna(fill_value).to_numpy(dtype="float32")
    if x.shape[0] == 0:
        return pd.DataFrame(
            columns=[f"{prefix}_{i}" for i in range(n_components)],
            dtype="float32",
        )

    k = min(n_components, x.shape[0], x.shape[1])
    z = (
        PCA(n_components=k, random_state=random_state)
        .fit_transform(x)
        .astype("float32")
    )

    if k < n_components:
        padded = np.zeros((z.shape[0], n_components), dtype="float32")
        padded[:, :k] = z
        z = padded

    return pd.DataFrame(
        z,
        index=sim_df.index,
        columns=[f"{prefix}_{i}" for i in range(n_components)],
    )


def make_function_aux_targets_from_train_obs(
    train_ds: Dataset,
    *,
    drug_n_components: int = 16,
    cell_n_components: int = 16,
    cell_col: str = "cell_id",
    drug_col: str = "drug_id",
    label_col: str = "label",
    random_state: int = 1441,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build drug/tumor functional auxiliary targets from train_ds.obs only.

    Response matrix is train tumors x train drugs. Similarities use the same
    centered-response approach as the notebook experiments:

    - drug-drug similarity: correlate drug response profiles after centering
      each drug across tumors.
    - tumor-tumor similarity: correlate tumor response profiles after centering
      each tumor across drugs.

    PCA coordinates from these similarity matrices become per-drug and per-tumor
    auxiliary targets.
    """
    rmat = train_ds.obs.pivot_table(
        index=cell_col,
        columns=drug_col,
        values=label_col,
        aggfunc="mean",
    )

    drug_centered = rmat.sub(rmat.mean(axis=0), axis=1)
    drug_sim = drug_centered.corr()

    cell_centered = rmat.sub(rmat.mean(axis=1), axis=0)
    cell_sim = cell_centered.T.corr()

    drug_targets = _similarity_matrix_to_targets(
        drug_sim,
        n_components=drug_n_components,
        prefix="drug_function",
        random_state=random_state,
    )

    cell_targets = _similarity_matrix_to_targets(
        cell_sim,
        n_components=cell_n_components,
        prefix="cell_function",
        random_state=random_state,
    )

    return drug_targets, cell_targets


def prepare_function_aux_targets(
    cfg: DictConfig,
    train_ds: Dataset,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Create functional auxiliary target tables from the training dataset."""
    if not bool(cfg_get(cfg, "model.hyper.aux.enabled", False)):
        return None, None

    drug_targets, cell_targets = make_function_aux_targets_from_train_obs(
        train_ds,
        drug_n_components=int(cfg_get(cfg, "model.hyper.aux.drug_n_components", 16)),
        cell_n_components=int(cfg_get(cfg, "model.hyper.aux.cell_n_components", 16)),
        cell_col=str(cfg_get(cfg, "model.hyper.aux.cell_col", "cell_id")),
        drug_col=str(cfg_get(cfg, "model.hyper.aux.drug_col", "drug_id")),
        label_col=str(cfg_get(cfg, "model.hyper.aux.label_col", "label")),
        random_state=int(cfg_get(cfg, "model.hyper.aux.random_state", 1441)),
    )

    log.info(
        "Built train-derived functional aux targets | drugs=%d x %d | tumors=%d x %d",
        drug_targets.shape[0],
        drug_targets.shape[1],
        cell_targets.shape[0],
        cell_targets.shape[1],
    )

    return drug_targets, cell_targets


def model_builder(cfg: DictConfig, train_dataset: Dataset) -> keras.Model:
    """Build the ScreenDL model."""
    params = cfg.model

    exp_dim = train_dataset.cell_encoders["exp"].shape[-1]

    mut_dim = None
    if "mut" in train_dataset.cell_encoders:
        mut_dim = train_dataset.cell_encoders["mut"].shape[-1]

    cnv_dim = None
    if "cnv" in train_dataset.cell_encoders:
        cnv_dim = train_dataset.cell_encoders["cnv"].shape[-1]

    ont_dim = None
    if "ont" in train_dataset.cell_encoders:
        ont_dim = train_dataset.cell_encoders["ont"].shape[-1]

    mol_dim = train_dataset.drug_encoders["mol"].shape[-1]

    return screendl.create_model(
        exp_dim,
        mol_dim,
        mut_dim,
        cnv_dim,
        ont_dim,
        exp_hidden_dims=params.hyper.hidden_dims.exp,
        mut_hidden_dims=params.hyper.hidden_dims.mut,
        cnv_hidden_dims=params.hyper.hidden_dims.cnv,
        ont_hidden_dims=params.hyper.hidden_dims.ont,
        mol_hidden_dims=params.hyper.hidden_dims.mol,
        shared_hidden_dims=params.hyper.hidden_dims.shared,
        use_noise=params.hyper.use_noise,
        use_normalization=params.hyper.use_normalization,
        use_dropout=params.hyper.use_dropout,
        use_l2=params.hyper.use_l2,
        noise_stddev=params.hyper.noise_stddev,
        l2_factor=params.hyper.l2_factor,
        dropout_rate=params.hyper.dropout_rate,
        activation=params.hyper.activation,
        norm_type=params.hyper.norm_type,
        interaction_mode=str(cfg_get(cfg, "model.hyper.interaction_mode", "concat")),
        bilinear_dim=int(cfg_get(cfg, "model.hyper.bilinear_dim", 64)),
        include_bilinear_product=bool(
            cfg_get(cfg, "model.hyper.include_bilinear_product", True)
        ),
        include_bilinear_score=bool(
            cfg_get(cfg, "model.hyper.include_bilinear_score", True)
        ),
    )


def maybe_add_auxiliary_heads(
    cfg: DictConfig,
    model: keras.Model,
    drug_targets: pd.DataFrame | None,
    cell_targets: pd.DataFrame | None,
) -> keras.Model:
    """Attach functional auxiliary heads if target tables are present."""
    if drug_targets is None and cell_targets is None:
        return model

    return screendl.add_function_auxiliary_heads(
        model,
        drug_aux_dim=None if drug_targets is None else drug_targets.shape[1],
        cell_aux_dim=None if cell_targets is None else cell_targets.shape[1],
        drug_hidden_dims=cfg_get(cfg, "model.hyper.aux.hidden_dims.drug", []),
        cell_hidden_dims=cfg_get(cfg, "model.hyper.aux.hidden_dims.cell", []),
        activation=cfg.model.hyper.activation,
        use_l2=cfg.model.hyper.use_l2,
        l2_factor=cfg.model.hyper.l2_factor,
    )


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> keras.Model:
    """Train response-only or auxiliary ScreenDL model."""
    params = cfg.model
    hyper = params.hyper

    drug_targets, cell_targets = prepare_function_aux_targets(cfg, train_dataset)
    model = maybe_add_auxiliary_heads(cfg, model, drug_targets, cell_targets)

    opt = keras.optimizers.Adam(
        learning_rate=hyper.learning_rate,
        weight_decay=hyper.weight_decay,
    )

    save_dir = "." if params.io.save is True else None
    log_dir = "./logs" if params.io.tensorboard is True else None

    model = screendl.train_model(
        model,
        opt,
        train_dataset,
        val_dataset,
        batch_size=hyper.batch_size,
        epochs=hyper.epochs,
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping=hyper.early_stopping,
        tensorboard=params.io.tensorboard,
        drug_function_targets=drug_targets,
        cell_function_targets=cell_targets,
        drug_function_loss_weight=float(
            cfg_get(cfg, "model.hyper.aux.drug_loss_weight", 0.01)
        ),
        cell_function_loss_weight=float(
            cfg_get(cfg, "model.hyper.aux.cell_loss_weight", 0.01)
        ),
        early_stopping_monitor=hyper.early_stopping_monitor,
        early_stopping_mode=hyper.early_stopping_mode,
        early_stopping_patience=int(hyper.early_stopping_patience),
        early_stopping_start_from_epoch=int(hyper.early_stopping_start_from_epoch),
    )

    if drug_targets is not None or cell_targets is not None:
        if bool(cfg_get(cfg, "model.hyper.aux.extract_response_model", True)):
            model = screendl.get_response_model(model)

    return model


def model_evaluator(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[Dataset],
) -> dict[str, ScoreDict]:
    """Evaluate model and write predictions/scores."""
    param_dict = {
        "model": cfg.model.name,
        "split_id": cfg.dataset.split.id,
        "split_type": cfg.dataset.split.name,
        "norm_method": cfg.dataset.preprocess.norm,
    }

    pred_dfs = []
    scores = {}
    for ds in datasets:
        gen = BatchedResponseGenerator(ds, cfg.model.hyper.batch_size)
        preds: np.ndarray | dict[str, np.ndarray] = model.predict(
            gen.flow(ds.cell_ids, ds.drug_ids)
        )
        if isinstance(preds, dict):
            preds = preds["response"]
        pred_df = make_pred_df(ds, preds, split_group=ds.name, **param_dict)
        pred_dfs.append(pred_df)
        scores[ds.name] = get_eval_metrics(pred_df)

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions.csv", index=False)

    with open("scores.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(scores), fh, ensure_ascii=False, indent=4)

    if cfg.dataset.output.save:
        ds_dir = Path("./datasets")
        ds_dir.mkdir(exist_ok=True)
        for ds in datasets:
            ds.save(ds_dir / f"{ds.name}.h5")

    return scores


class ScreenDLPipeline:
    """Small pipeline object around the existing function API."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dataset: Dataset | None = None
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
        self.model: keras.Model | None = None
        self.scores: dict[str, ScoreDict] | None = None
        self.preprocessing_artifacts: PreprocessingArtifacts | None = None

    def __enter__(self) -> "ScreenDLPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def load(self) -> Dataset:
        self.dataset = data_loader(self.cfg)
        return self.dataset

    def split(self) -> tuple[Dataset, Dataset, Dataset]:
        if self.dataset is None:
            raise RuntimeError("Call load() before split().")
        self.train_ds, self.val_ds, self.test_ds = data_splitter(self.cfg, self.dataset)
        return self.train_ds, self.val_ds, self.test_ds

    def preprocess(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        if self.train_ds is None:
            raise RuntimeError("Call split() before preprocess().")

        (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.preprocessing_artifacts,
        ) = data_preprocessor(
            self.cfg,
            self.train_ds,
            self.val_ds,
            self.test_ds,
        )

        return self.train_ds, self.val_ds, self.test_ds

    def build(self) -> keras.Model:
        if self.train_ds is None:
            raise RuntimeError("Call preprocess() before build().")
        self.model = model_builder(self.cfg, self.train_ds)
        return self.model

    def train(self) -> keras.Model:
        if self.model is None or self.train_ds is None or self.val_ds is None:
            raise RuntimeError("Call build() before train().")
        self.model = model_trainer(self.cfg, self.model, self.train_ds, self.val_ds)
        return self.model

    def evaluate(self) -> dict[str, ScoreDict]:
        if self.model is None or self.train_ds is None or self.val_ds is None:
            raise RuntimeError("Call train() before evaluate().")
        datasets: list[Dataset] = [self.train_ds, self.val_ds]
        if self.test_ds is not None:
            datasets.append(self.test_ds)
        self.scores = model_evaluator(self.cfg, self.model, datasets)
        return self.scores

    def run(self) -> PipelineArtifacts:
        dataset_name = self.cfg.dataset.name
        model_name = self.cfg.model.name

        log.info("Loading %s...", dataset_name)
        self.load()

        log.info("Splitting %s...", dataset_name)
        self.split()

        log.info("Preprocessing %s...", dataset_name)
        self.preprocess()

        log.info("Building %s...", model_name)
        self.build()

        log.info("Training %s...", model_name)
        self.train()

        log.info("Evaluating %s...", model_name)
        self.evaluate()

        assert self.model is not None
        assert self.scores is not None
        assert self.dataset is not None
        assert self.train_ds is not None
        assert self.val_ds is not None
        assert self.test_ds is not None

        return PipelineArtifacts(
            model=self.model,
            scores=self.scores,
            datasets={
                "full": self.dataset,
                "train": self.train_ds,
                "val": self.val_ds,
                "test": self.test_ds,
            },
        )


def run_pipeline(
    cfg: DictConfig,
) -> tuple[keras.Model, dict[str, ScoreDict], dict[str, Dataset]]:
    """Run the ScreenDL training pipeline."""
    artifacts = ScreenDLPipeline(cfg).run()
    return artifacts.model, artifacts.scores, artifacts.datasets


def run_hp_pipeline(cfg: DictConfig) -> float:
    """Run cross-validation optimization pipeline."""
    cfg.model.io.save = False
    cfg.dataset.output.save = False

    pipeline = ScreenDLPipeline(cfg)
    pipeline.load()
    pipeline.split()
    pipeline.preprocess()
    pipeline.build()
    pipeline.train()

    assert pipeline.model is not None
    assert pipeline.val_ds is not None

    batch_size = cfg.model.hyper.batch_size
    gen = BatchedResponseGenerator(pipeline.val_ds, batch_size)
    seq = gen.flow(
        pipeline.val_ds.cell_ids,
        pipeline.val_ds.drug_ids,
        targets=pipeline.val_ds.labels,
    )
    loss, *_ = pipeline.model.evaluate(seq)
    return loss


def run_sa_pipeline(
    cfg: DictConfig,
) -> tuple[keras.Model, Dataset, Dataset, Dataset, Dataset]:
    """Run ScreenDL pipeline and return datasets for ScreenAhead."""
    pipeline = ScreenDLPipeline(cfg)
    pipeline.load()
    pipeline.split()
    pipeline.preprocess()
    pipeline.build()
    pipeline.train()
    pipeline.evaluate()

    assert pipeline.model is not None
    assert pipeline.dataset is not None
    assert pipeline.train_ds is not None
    assert pipeline.val_ds is not None
    assert pipeline.test_ds is not None

    return (
        pipeline.model,
        pipeline.dataset,
        pipeline.train_ds,
        pipeline.val_ds,
        pipeline.test_ds,
    )
