"""
Pipeline for running the original (legacy) DualGCN code.

>>> DUALGCN_ROOT="pkg/DualGCN/code" python scripts/runners/run.py --multirun \
        model=DualGCN-legacy \
        dataset.preprocess.norm=global
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tensorflow import keras

from cdrpy.data.preprocess import normalize_responses
from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import DictEncoder, RepeatEncoder
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.metrics import tf_metrics
from cdrpy.util.io import read_pickled_dict
from cdrpy.util.validation import check_same_columns, check_same_indexes

from screendl.utils.evaluation import ScoreDict, get_eval_metrics, make_pred_df
from screendl.utils.serialization import to_jsonable

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


def import_dualgcn_namespace() -> SimpleNamespace:
    """Import DualGCN model and graph utilities from DUALGCN_ROOT."""
    dualgcn_root = os.environ.get("DUALGCN_ROOT")
    if not dualgcn_root:
        raise RuntimeError("DUALGCN_ROOT must point to the DualGCN/code directory.")

    sys.path.insert(1, dualgcn_root)
    try:
        from model import KerasMultiSourceDualGCNModel
        from DualGCN import CalculateGraphFeat, CelllineGraphAdjNorm  # type: ignore[import]
    finally:
        del sys.path[1]

    return SimpleNamespace(
        model=KerasMultiSourceDualGCNModel,
        calc_graph_feat=CalculateGraphFeat,
        get_ppi_adj=CelllineGraphAdjNorm,
    )


dualgcn = import_dualgcn_namespace()


def data_loader(cfg: DictConfig) -> Dataset:
    """Load DualGCN omics, PPI, molecular graph, and response data."""
    paths = cfg.dataset.sources

    mol_path = paths.dualgcn.mol
    exp_path = paths.dualgcn.exp
    cnv_path = paths.dualgcn.cnv
    ppi_path = paths.dualgcn.ppi

    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    cnv_mat = pd.read_csv(cnv_path, index_col=0).astype("float32")

    check_same_columns(exp_mat, cnv_mat)
    check_same_indexes(exp_mat, cnv_mat)
    common_genes = list(exp_mat.columns)

    omics_dict = {}
    for cell_id in exp_mat.index:
        cell_exp = exp_mat.loc[cell_id].values.reshape(-1, 1)
        cell_cnv = cnv_mat.loc[cell_id].values.reshape(-1, 1)
        omics_dict[cell_id] = np.hstack((cell_exp, cell_cnv))

    omics_enc = DictEncoder(omics_dict, name="omics_encoder")

    idx_dict = {gene: idx for idx, gene in enumerate(common_genes)}
    ppi_edges = pd.read_csv(ppi_path)
    ppi_adj_info = [[] for _ in common_genes]

    for gene_1, gene_2 in zip(ppi_edges["gene_1"], ppi_edges["gene_2"]):
        if gene_1 not in idx_dict or gene_2 not in idx_dict:
            continue
        if idx_dict[gene_1] <= idx_dict[gene_2]:
            ppi_adj_info[idx_dict[gene_1]].append(idx_dict[gene_2])
            ppi_adj_info[idx_dict[gene_2]].append(idx_dict[gene_1])

    with tempfile.TemporaryDirectory() as tmpdir:
        # `DualGCN.CelllineGraphAdjNorm` requires genes saved as a .txt file.
        gene_list_file = Path(tmpdir) / "gene_list.txt"
        with open(gene_list_file, "w", encoding="utf-8") as fh:
            for gene in common_genes:
                fh.write(f"{gene}\n")

        ppi_adj_norm = dualgcn.get_ppi_adj(ppi_adj_info, gene_list_file)
        ppi_adj_norm = ppi_adj_norm.astype(np.float32)

    ppi_adj_enc = RepeatEncoder(ppi_adj_norm, name="ppi_adj_encoder")

    drug_dict = read_pickled_dict(mol_path)
    drug_feat = {}
    drug_adj = {}
    for drug_id, (feat, _, adj) in drug_dict.items():
        drug_feat[drug_id], drug_adj[drug_id] = dualgcn.calc_graph_feat(feat, adj)

    cell_encoders = {
        "omics": omics_enc,
        "ppi": ppi_adj_enc,
    }
    drug_encoders = {
        "feat": DictEncoder(drug_feat, name="drug_feature_encoder"),
        "adj": DictEncoder(drug_adj, name="drug_adj_encoder"),
    }

    cell_meta = None
    if hasattr(paths, "cell_meta"):
        cell_meta = pd.read_csv(paths.cell_meta, index_col=0)

    drug_meta = None
    if hasattr(paths, "drug_meta"):
        drug_meta = pd.read_csv(paths.drug_meta, index_col=0)

    return Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
        cell_meta=cell_meta,
        drug_meta=drug_meta,
        encode_drugs_first=True,
    )


def data_splitter(
    cfg: DictConfig,
    dataset: Dataset,
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
) -> tuple[Dataset, Dataset, Dataset]:
    """Normalize response values and DualGCN omics tensors."""
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    omics_enc: DictEncoder = train_dataset.cell_encoders["omics"]
    x_omics = np.array(omics_enc.encode(train_dataset.cell_ids))
    x_mean = np.mean(x_omics, axis=0)
    x_std = np.std(x_omics, axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)

    omics_enc.data = {
        cell_id: ((values - x_mean) / x_std).astype("float32")
        for cell_id, values in omics_enc.data.items()
    }

    if val_dataset is None or test_dataset is None:
        raise RuntimeError(
            "DualGCN preprocessing expects train, val, and test datasets."
        )

    val_dataset.cell_encoders = train_dataset.cell_encoders
    test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, train_dataset: Dataset) -> keras.Model:
    """Build the DualGCN model."""
    cell_feat_dim = train_dataset.cell_encoders["omics"].shape[-1]
    drug_feat_dim = train_dataset.drug_encoders["feat"].shape[-1]

    return dualgcn.model().createMaster(
        drug_dim=drug_feat_dim,
        cell_line_dim=cell_feat_dim,
        units_list=cfg.model.hyper.units_list,
    )


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> keras.Model:
    """Train the DualGCN model."""
    params = cfg.model
    hyper = params.hyper

    adam_kwargs: dict[str, t.Any] = {"learning_rate": hyper.learning_rate}
    weight_decay = cfg_get(cfg, "model.hyper.weight_decay", None)
    if weight_decay is not None:
        adam_kwargs["weight_decay"] = weight_decay
    opt = keras.optimizers.Adam(**adam_kwargs)

    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=[tf_metrics.pearson],
    )

    callbacks: list[keras.callbacks.Callback] = []
    if bool(hyper.early_stopping):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=str(
                    cfg_get(cfg, "model.hyper.early_stopping_monitor", "val_loss")
                ),
                mode=str(cfg_get(cfg, "model.hyper.early_stopping_mode", "min")),
                patience=int(cfg_get(cfg, "model.hyper.early_stopping_patience", 15)),
                restore_best_weights=True,
                start_from_epoch=int(
                    cfg_get(cfg, "model.hyper.early_stopping_start_from_epoch", 3)
                ),
                verbose=1,
            )
        )

    if bool(cfg_get(cfg, "model.io.checkpoints", False)):
        ckpt_path = Path("./checkpoint")
        ckpt_path.mkdir(exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(ckpt_path),
                monitor=str(cfg_get(cfg, "model.hyper.checkpoint_monitor", "val_loss")),
                mode=str(cfg_get(cfg, "model.hyper.checkpoint_mode", "min")),
                save_best_only=True,
            )
        )

    log_dir = "./logs" if bool(cfg_get(cfg, "model.io.tensorboard", False)) else None
    if log_dir is not None:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

    batch_size = hyper.batch_size
    train_gen = BatchedResponseGenerator(train_dataset, batch_size)
    val_gen = BatchedResponseGenerator(val_dataset, batch_size)

    train_seq = train_gen.flow_from_dataset(
        train_dataset,
        drugs_first=True,
        shuffle=True,
        seed=int(cfg_get(cfg, "model.hyper.shuffle_seed", 4114)),
    )
    val_seq = val_gen.flow_from_dataset(
        val_dataset,
        drugs_first=True,
        shuffle=False,
    )

    model.fit(
        train_seq,
        epochs=hyper.epochs,
        validation_data=val_seq,
        callbacks=callbacks,
    )

    if bool(params.io.save):
        model.save("model")
        model.save_weights("weights")

    return model


def _predict_dataset(
    model: keras.Model,
    dataset: Dataset,
    *,
    batch_size: int,
) -> np.ndarray:
    """Predict a DualGCN Dataset using the DualGCN drug-first input order."""
    gen = BatchedResponseGenerator(dataset, batch_size)
    preds = model.predict(
        gen.flow_from_dataset(dataset, drugs_first=True, shuffle=False)
    )
    if isinstance(preds, dict):
        preds = preds["response"]
    return t.cast(np.ndarray, preds)


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
        preds = _predict_dataset(
            model,
            ds,
            batch_size=int(cfg.model.hyper.batch_size),
        )
        pred_df = make_pred_df(ds, preds, split_group=ds.name, **param_dict)
        pred_dfs.append(pred_df)
        scores[ds.name] = get_eval_metrics(pred_df)

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions.csv", index=False)

    with open("scores.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(scores), fh, ensure_ascii=False, indent=4)

    if bool(cfg_get(cfg, "dataset.output.save", False)):
        ds_dir = Path("./datasets")
        ds_dir.mkdir(exist_ok=True)
        for ds in datasets:
            ds.save(ds_dir / f"{ds.name}.h5")

    return scores


class DualGCNPipeline:
    """Small pipeline object around the existing function API."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dataset: Dataset | None = None
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
        self.model: keras.Model | None = None
        self.scores: dict[str, ScoreDict] | None = None

    def __enter__(self) -> DualGCNPipeline:
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

    def preprocess(self) -> tuple[Dataset, Dataset, Dataset]:
        if self.train_ds is None:
            raise RuntimeError("Call split() before preprocess().")
        self.train_ds, self.val_ds, self.test_ds = data_preprocessor(
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
    """Run the DualGCN training pipeline."""
    artifacts = DualGCNPipeline(cfg).run()
    return artifacts.model, artifacts.scores, artifacts.datasets
