"""
HiDRA training and evaluation pipeline.

>>> HIDRA_ROOT="pkg/HiDRA" python scripts/runners/run.py --multirun \
        model=HiDRA-legacy \
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
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from cdrpy.data.preprocess import normalize_responses
from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import PandasEncoder
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


def import_hidra_namespace() -> SimpleNamespace:
    """Import HiDRA model-building code from HIDRA_ROOT."""
    try:
        hidra_root = os.environ["HIDRA_ROOT"]
    except KeyError as e:
        raise RuntimeError("HIDRA_ROOT must point to the HiDRA package root.") from e

    import importlib.util

    path = os.path.join(hidra_root, "Training/HiDRA_training.py")
    spec = importlib.util.spec_from_file_location("hidra", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HiDRA module from {path!r}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["hidra"] = module
    spec.loader.exec_module(module)

    return SimpleNamespace(create_model=module.Making_Model)


hidra = import_hidra_namespace()


def data_loader(cfg: DictConfig) -> tuple[Dataset, dict[str, list[str]]]:
    """Load the input dataset and HiDRA gene-set definitions."""
    paths = cfg.dataset.sources

    mol_path = paths.hidra.mol
    exp_path = paths.hidra.exp
    gene_path = paths.hidra.gene

    geneset_dict: dict[str, list[str]] = read_pickled_dict(gene_path)
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    mol_mat = pd.read_csv(mol_path, index_col=0).astype("int32")

    cell_encoders: dict[str, PandasEncoder] = {}
    for geneset_name, geneset_genes in geneset_dict.items():
        genes = list(geneset_genes)
        geneset_exp_mat = exp_mat[genes]
        geneset_enc_data = pd.DataFrame(
            geneset_exp_mat.values,
            index=geneset_exp_mat.index,
            columns=genes,
        )
        cell_encoders[geneset_name] = PandasEncoder(
            geneset_enc_data,
            name=geneset_name,
        )

    drug_encoders = {"mol": PandasEncoder(mol_mat, name="mol_encoder")}

    cell_meta = None
    if hasattr(paths, "cell_meta"):
        cell_meta = pd.read_csv(paths.cell_meta, index_col=0)

    drug_meta = None
    if hasattr(paths, "drug_meta"):
        drug_meta = pd.read_csv(paths.drug_meta, index_col=0)

    dataset = Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
        cell_meta=cell_meta,
        drug_meta=drug_meta,
    )

    return dataset, geneset_dict


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
    geneset_dict: dict[str, list[str]],
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Normalize responses and HiDRA gene-set expression encoders."""
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    for geneset_name, enc in train_dataset.cell_encoders.items():
        scaler = StandardScaler()
        x_train = np.array(enc.encode(list(set(train_dataset.cell_ids))))
        scaler.fit(x_train)
        enc.data.loc[:, :] = scaler.transform(enc.data.values)

        if enc.data.isnull().any().any():
            n_genes = enc.data.shape[-1]
            enc.data = enc.data.dropna(axis=1)
            n_dropped = n_genes - enc.data.shape[-1]
            log.warning(
                "Dropped %d genes with NaN values from %s.",
                n_dropped,
                enc.name,
            )
            geneset_dict[geneset_name] = list(enc.data.columns)

    if val_dataset is None or test_dataset is None:
        raise ValueError("HiDRA preprocessing expects train, val, and test datasets.")

    val_dataset.cell_encoders = train_dataset.cell_encoders
    test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(
    cfg: DictConfig,
    geneset_dict: dict[str, list[str]],
) -> keras.Model:
    """Build the HiDRA model."""
    del cfg
    return hidra.create_model(geneset_dict)


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> keras.Model:
    """Train the HiDRA model."""
    params = cfg.model
    hyper = params.hyper

    opt_kwargs: dict[str, t.Any] = {"learning_rate": hyper.learning_rate}
    weight_decay = cfg_get(cfg, "model.hyper.weight_decay")
    if weight_decay is not None:
        opt_kwargs["weight_decay"] = weight_decay
    opt = keras.optimizers.Adam(**opt_kwargs)

    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=["mse", tf_metrics.pearson],
    )

    callbacks: list[keras.callbacks.Callback] = []
    if hyper.early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=cfg_get(cfg, "model.hyper.early_stopping_monitor", "val_loss"),
                mode=cfg_get(cfg, "model.hyper.early_stopping_mode", "auto"),
                patience=int(cfg_get(cfg, "model.hyper.early_stopping_patience", 15)),
                restore_best_weights=True,
                start_from_epoch=int(
                    cfg_get(cfg, "model.hyper.early_stopping_start_from_epoch", 3)
                ),
                verbose=1,
            )
        )

    batch_size = hyper.batch_size
    train_gen = BatchedResponseGenerator(train_dataset, batch_size)
    val_gen = BatchedResponseGenerator(val_dataset, batch_size)

    train_seq = train_gen.flow_from_dataset(
        train_dataset,
        shuffle=True,
        seed=int(cfg_get(cfg, "model.hyper.shuffle_seed", 4114)),
    )
    val_seq = val_gen.flow_from_dataset(val_dataset, shuffle=False)

    model.fit(
        train_seq,
        epochs=hyper.epochs,
        validation_data=val_seq,
        callbacks=callbacks,
    )

    if params.io.save:
        save_dir = Path(".")
        model.save(save_dir / "model")
        model.save_weights(save_dir / "weights")

    return model


def model_evaluator(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[Dataset],
) -> dict[str, ScoreDict]:
    """Evaluate the HiDRA model and write predictions/scores."""
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
        preds: np.ndarray = model.predict(gen.flow_from_dataset(ds, shuffle=False))
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


class HiDRAPipeline:
    """Small pipeline object around the existing HiDRA function API."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dataset: Dataset | None = None
        self.geneset_dict: dict[str, list[str]] | None = None
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
        self.model: keras.Model | None = None
        self.scores: dict[str, ScoreDict] | None = None

    def __enter__(self) -> "HiDRAPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def load(self) -> Dataset:
        self.dataset, self.geneset_dict = data_loader(self.cfg)
        return self.dataset

    def split(self) -> tuple[Dataset, Dataset, Dataset]:
        if self.dataset is None:
            raise RuntimeError("Call load() before split().")
        self.train_ds, self.val_ds, self.test_ds = data_splitter(self.cfg, self.dataset)
        return self.train_ds, self.val_ds, self.test_ds

    def preprocess(self) -> tuple[Dataset, Dataset, Dataset]:
        if self.geneset_dict is None or self.train_ds is None:
            raise RuntimeError("Call split() before preprocess().")
        self.train_ds, self.val_ds, self.test_ds = data_preprocessor(
            self.cfg,
            self.geneset_dict,
            self.train_ds,
            self.val_ds,
            self.test_ds,
        )
        return self.train_ds, self.val_ds, self.test_ds

    def build(self) -> keras.Model:
        if self.geneset_dict is None:
            raise RuntimeError("Call preprocess() before build().")
        self.model = model_builder(self.cfg, self.geneset_dict)
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
    """Run the HiDRA training pipeline."""
    artifacts = HiDRAPipeline(cfg).run()
    return artifacts.model, artifacts.scores, artifacts.datasets


def run_hp_pipeline(cfg: DictConfig) -> float:
    """Run validation-loss optimization pipeline."""
    cfg.model.io.save = False
    if cfg_get(cfg, "dataset.output.save") is not None:
        cfg.dataset.output.save = False

    pipeline = HiDRAPipeline(cfg)
    pipeline.load()
    pipeline.split()
    pipeline.preprocess()
    pipeline.build()
    pipeline.train()

    assert pipeline.model is not None
    assert pipeline.val_ds is not None

    batch_size = cfg.model.hyper.batch_size
    gen = BatchedResponseGenerator(pipeline.val_ds, batch_size)
    seq = gen.flow_from_dataset(pipeline.val_ds, shuffle=False)
    loss, *_ = pipeline.model.evaluate(seq)
    return loss


def run_pdx_pipeline(
    cfg: DictConfig,
) -> tuple[keras.Model, dict[str, ScoreDict], dict[str, Dataset]]:
    """Run HiDRA, then predict all held-out PDXO x full-drug PDX combinations."""
    artifacts = HiDRAPipeline(cfg).run()

    model = artifacts.model
    scores = artifacts.scores
    ds_dict = artifacts.datasets

    pdmc_ds = ds_dict["test"]
    all_drug_ids = list(set(ds_dict["full"].drug_ids))
    all_pdmc_ids = list(set(pdmc_ds.cell_ids))

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

    pdx_batch_size = int(cfg_get(cfg, "pdx_batch_size", 256))
    pdx_gen = BatchedResponseGenerator(pdx_ds_full, pdx_batch_size)
    pdx_preds: np.ndarray = model.predict(
        pdx_gen.flow_from_dataset(pdx_ds_full, shuffle=False)
    )

    param_dict = {"model": cfg.model.name}
    pdx_pred_df = make_pred_df(pdx_ds_full, pdx_preds, **param_dict)
    pdx_pred_df.to_csv("predictions_pdx.csv", index=False)

    return model, scores, ds_dict
