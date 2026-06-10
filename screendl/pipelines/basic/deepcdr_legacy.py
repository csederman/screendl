"""
DeepCDR training and evaluation pipeline.

>>> DEEPCDR_ROOT="pkg/DeepCDR/prog" python scripts/runners/run.py --multirun \
        model=DeepCDR-legacy \
        dataset.preprocess.norm=global
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import typing as t
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tensorflow import keras

from cdrpy.data.preprocess import normalize_responses
from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import DictEncoder, PandasEncoder
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.metrics import tf_metrics
from cdrpy.util.io import read_pickled_dict

from screendl.utils import data_utils
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


def import_deepcdr_namespace() -> SimpleNamespace:
    """Import the DeepCDR model and graph featurizer from DEEPCDR_ROOT."""
    deepcdr_root = os.environ.get("DEEPCDR_ROOT")
    if not deepcdr_root:
        raise RuntimeError("DEEPCDR_ROOT must point to the DeepCDR/prog directory.")

    sys.path.insert(1, deepcdr_root)
    try:
        from model import KerasMultiSourceGCNModel
        from run_DeepCDR import CalculateGraphFeat  # type: ignore[import]
    finally:
        del sys.path[1]

    return SimpleNamespace(
        model=KerasMultiSourceGCNModel,
        calc_graph_feat=CalculateGraphFeat,
    )


deepcdr = import_deepcdr_namespace()


def data_loader(cfg: DictConfig) -> Dataset:
    """Load DeepCDR molecular graph, expression, mutation, and response data."""
    paths = cfg.dataset.sources

    mol_path = paths.deepcdr.mol
    exp_path = paths.deepcdr.exp
    mut_path = paths.deepcdr.mut

    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    mut_mat = pd.read_csv(mut_path, index_col=0).astype("int32")

    mut_dict = {
        cell_id: mut_mat.loc[cell_id].values.reshape(1, -1, 1)
        for cell_id in mut_mat.index
    }
    cell_encoders = {
        "mut": DictEncoder(mut_dict, name="mut_encoder"),
        "exp": PandasEncoder(exp_mat, name="exp_encoder"),
    }

    drug_dict = read_pickled_dict(mol_path)
    drug_feat = {}
    drug_adj = {}
    for drug_id, (feat, _, adj) in drug_dict.items():
        drug_feat[drug_id], drug_adj[drug_id] = deepcdr.calc_graph_feat(feat, adj)

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
    """Normalize response values and DeepCDR expression features."""
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    exp_encoder: PandasEncoder = train_dataset.cell_encoders["exp"]
    x_exp = np.array(exp_encoder.encode(list(set(train_dataset.cell_ids))))

    if bool(cfg_get(cfg, "model.preprocess.use_quantile_norm", False)):
        exp_transformer = QuantileTransformer(
            output_distribution="normal",
            random_state=int(cfg_get(cfg, "model.preprocess.random_state", 1771)),
        ).fit(x_exp)
    else:
        exp_transformer = StandardScaler().fit(x_exp)

    exp_encoder.data.loc[:, :] = exp_transformer.transform(exp_encoder.data.values)

    num_genes = exp_encoder.data.shape[-1]
    exp_encoder.data = exp_encoder.data.dropna(axis=1)
    num_missing = num_genes - exp_encoder.shape[-1]
    if num_missing > 0:
        log.warning("Dropped %d genes with NaN values.", num_missing)

    if val_dataset is None or test_dataset is None:
        raise RuntimeError(
            "DeepCDR preprocessing expects train, val, and test datasets."
        )

    val_dataset.cell_encoders = train_dataset.cell_encoders
    test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, train_dataset: Dataset) -> keras.Model:
    """Build the DeepCDR model."""
    mut_dim = train_dataset.cell_encoders["mut"].shape[1]
    exp_dim = train_dataset.cell_encoders["exp"].shape[-1]
    drug_dim = train_dataset.drug_encoders["feat"].shape[-1]

    return deepcdr.model(use_mut=True, use_gexp=True, use_methy=False).createMaster(
        drug_dim=drug_dim,
        mutation_dim=mut_dim,
        gexpr_dim=exp_dim,
        units_list=cfg.model.hyper.units_list,
        use_relu=True,
        use_bn=True,
        use_GMP=True,
    )


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> keras.Model:
    """Train the DeepCDR model."""
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
    """Predict a DeepCDR Dataset using the DeepCDR drug-first input order."""
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


class DeepCDRPipeline:
    """Small pipeline object around the existing function API."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dataset: Dataset | None = None
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
        self.model: keras.Model | None = None
        self.scores: dict[str, ScoreDict] | None = None

    def __enter__(self) -> DeepCDRPipeline:
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
    """Run the DeepCDR training pipeline."""
    artifacts = DeepCDRPipeline(cfg).run()
    return artifacts.model, artifacts.scores, artifacts.datasets


def run_hp_pipeline(cfg: DictConfig) -> float:
    """Run cross-validation optimization pipeline."""
    if cfg_get(cfg, "model.io.save", None) is not None:
        cfg.model.io.save = False
    if cfg_get(cfg, "dataset.output.save", None) is not None:
        cfg.dataset.output.save = False

    pipeline = DeepCDRPipeline(cfg)
    pipeline.load()
    pipeline.split()
    pipeline.preprocess()
    pipeline.build()
    pipeline.train()

    assert pipeline.model is not None
    assert pipeline.val_ds is not None

    batch_size = int(cfg.model.hyper.batch_size)
    val_gen = BatchedResponseGenerator(pipeline.val_ds, batch_size)
    val_seq = val_gen.flow_from_dataset(
        pipeline.val_ds,
        drugs_first=True,
        shuffle=False,
    )
    loss, *_ = pipeline.model.evaluate(val_seq)
    return loss


def run_pdx_pipeline(
    cfg: DictConfig,
) -> tuple[keras.Model, dict[str, ScoreDict], dict[str, Dataset]]:
    """Run DeepCDR and predict expanded PDX tumor-drug combinations.

    This is the expanded-combination behavior from the old run_pdx_pipeline_v2.
    """
    artifacts = DeepCDRPipeline(cfg).run()
    model = artifacts.model
    scores = artifacts.scores
    ds_dict = artifacts.datasets

    pdmc_ds = ds_dict["test"]
    all_drug_ids = list(dict.fromkeys(ds_dict["full"].drug_ids))
    all_pdmc_ids = list(dict.fromkeys(pdmc_ds.cell_ids))

    pdx_obs = pd.read_csv(cfg.pdx_path)
    pdx_obs = pdx_obs[pdx_obs["cell_id"].isin(pdmc_ds.cell_ids)]
    pdx_obs = pdx_obs[pdx_obs["drug_id"].isin(pdmc_ds.drug_ids)]
    pdx_obs["label"] = pdx_obs["mRECIST"].isin(["CR", "PR", "SD"]).astype(int)

    pdx_ds = Dataset(
        pdx_obs,
        cell_encoders=pdmc_ds.cell_encoders,
        drug_encoders=pdmc_ds.drug_encoders,
        name="pdx_ds",
    )

    pdx_ds_full = data_utils.expand_dataset(pdx_ds, all_pdmc_ids, all_drug_ids)
    pdx_preds = _predict_dataset(
        model,
        pdx_ds_full,
        batch_size=int(cfg_get(cfg, "pdx.batch_size", cfg.model.hyper.batch_size)),
    )

    param_dict = {
        "model": cfg.model.name,
        "split_id": cfg.dataset.split.id,
        "split_type": cfg.dataset.split.name,
        "norm_method": cfg.dataset.preprocess.norm,
    }
    pdx_pred_df = make_pred_df(pdx_ds_full, pdx_preds, **param_dict)
    pdx_pred_df.to_csv("predictions_pdx.csv", index=False)

    return model, scores, ds_dict
