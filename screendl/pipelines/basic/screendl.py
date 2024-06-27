"""
Run utilities for ScreenDL.
"""

from __future__ import annotations

import os
import json
import logging
import pickle

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from cdrpy.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.feat.encoders import PandasEncoder
from cdrpy.mapper import BatchedResponseGenerator

from screendl import model as screendl
from screendl.utils.evaluation import make_pred_df, get_eval_metrics, ScoreDict


log = logging.getLogger(__name__)


def data_loader(cfg: DictConfig) -> Dataset:
    """Loads the input dataset.

    Parameters
    ----------
        cfg:

    Returns
    -------
    """
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
) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Splits the dataset into train/validation/test sets.

    Parameters
    ----------
        cfg:
        dataset:

    Returns
    -------
    """
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
) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Preprocessing pipeline.

    Parameters
    ----------
        cfg:
        train_dataset:
        val_dataset:
        test_dataset:

    Returns
    -------
        A (train, validation, test) tuple of processed datasets.
    """

    # normalize the gene expression
    exp_enc: PandasEncoder = train_dataset.cell_encoders["exp"]
    X_exp = np.array(exp_enc.encode(list(set(train_dataset.cell_ids))))
    exp_scaler = StandardScaler().fit(X_exp)
    exp_enc.data[:] = exp_scaler.transform(exp_enc.data.values)

    # normalize copy number data if present
    if "cnv" in train_dataset.cell_encoders:
        cnv_enc: PandasEncoder = train_dataset.cell_encoders["cnv"]
        X_cnv = np.array(cnv_enc.encode(list(set(train_dataset.cell_ids))))
        cnv_scaler = StandardScaler().fit(X_cnv)
        cnv_enc.data[:] = cnv_scaler.transform(cnv_enc.data.values)

    # normalize the drug responses
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    if cfg.model.feat.use_mr:
        train_mr = (
            train_dataset.obs.groupby("cell_id")["label"].mean().to_frame(name="value")
        )
        train_mr[:] = StandardScaler().fit_transform(train_mr)

        val_cell_ids = list(set(val_dataset.cell_ids))
        test_cell_ids = list(set(test_dataset.cell_ids))
        val_mr = pd.DataFrame({"value": 0}, index=val_cell_ids)
        test_mr = pd.DataFrame({"value": 0}, index=test_cell_ids)

        mr_data = pd.concat([train_mr, val_mr, test_mr])
        train_dataset.cell_encoders["mr"] = PandasEncoder(mr_data, name="mr")

    # val_dataset.cell_encoders = train_dataset.cell_encoders
    # test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, train_dataset: Dataset) -> keras.Model:
    """Builds the ScreenDL model."""
    params = cfg.model

    # extract exp shape from cell encoders
    exp_enc: PandasEncoder = train_dataset.cell_encoders["exp"]
    exp_dim = exp_enc.shape[-1]

    # extract mut shape from cell encoders
    mut_dim = None
    if "mut" in train_dataset.cell_encoders:
        mut_enc: PandasEncoder = train_dataset.cell_encoders["mut"]
        mut_dim = mut_enc.shape[-1]

    # extract cnv shape from cell encoders
    cnv_dim = None
    if "cnv" in train_dataset.cell_encoders:
        cnv_enc: PandasEncoder = train_dataset.cell_encoders["cnv"]
        cnv_dim = cnv_enc.shape[-1]

    # extract tissue ontology shape from cell encoders
    ont_dim = None
    if "ont" in train_dataset.cell_encoders:
        ont_enc: PandasEncoder = train_dataset.cell_encoders["ont"]
        ont_dim = ont_enc.shape[-1]

    # extract mol shape from drug encoders
    mol_enc: PandasEncoder = train_dataset.drug_encoders["mol"]
    mol_dim = mol_enc.shape[-1]

    model = screendl.create_model(
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
        use_mr=params.feat.use_mr,
        use_noise=params.hyper.use_noise,
        use_batch_norm=params.hyper.use_batch_norm,
        use_dropout=params.hyper.use_dropout,
        use_l2=params.hyper.use_l2,
        noise_stddev=params.hyper.noise_stddev,
        l2_factor=params.hyper.l2_factor,
        dropout_rate=params.hyper.dropout_rate,
        activation=params.hyper.activation,
    )

    return model


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> keras.Model:
    """Trains the ScreenDL model.

    Parameters
    ----------
        cfg:
        model:
        train_dataset:
        val_dataset:

    Returns
    -------
        The trained `keras.Model` instance.
    """
    params = cfg.model

    opt = keras.optimizers.Adam(learning_rate=params.hyper.learning_rate)

    save_dir = "." if params.io.save is True else None
    log_dir = "./logs" if params.io.tensorboard is True else None

    model = screendl.train_model(
        model,
        opt,
        train_dataset,
        val_dataset,
        batch_size=params.hyper.batch_size,
        epochs=params.hyper.epochs,
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping=params.hyper.early_stopping,
        tensorboard=params.io.tensorboard,
    )

    return model


def model_evaluator(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[Dataset],
) -> t.Dict[str, ScoreDict]:
    """Evaluates the ScreenDL Model and returns validation metrics.

    Parameters
    ----------
        cfg:
        model:
        datasets:
    """
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
        preds: np.ndarray = model.predict(gen.flow(ds.cell_ids, ds.drug_ids))
        pred_df = make_pred_df(ds, preds, split_group=ds.name, **param_dict)
        pred_dfs.append(pred_df)
        scores[ds.name] = get_eval_metrics(pred_df)

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions.csv", index=False)

    with open("scores.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh, ensure_ascii=False, indent=4)

    if cfg.dataset.output.save:
        ds_dir = Path("./datasets")
        ds_dir.mkdir()
        for ds in datasets:
            ds.save(ds_dir / f"{ds.name}.h5")

    return scores


def run_pipeline(
    cfg: DictConfig,
) -> t.Tuple[keras.Model, t.Dict[str, ScoreDict], t.Dict[str, Dataset]]:
    """Runs the ScreenDL training pipeline."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    ds = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    train_ds, val_ds, test_ds = data_splitter(cfg, ds)

    log.info(f"Preprocessing {dataset_name}...")
    train_ds, val_ds, test_ds = data_preprocessor(cfg, train_ds, val_ds, test_ds)

    log.info(f"Building {model_name}...")
    model = model_builder(cfg, train_ds)

    log.info(f"Training {model_name}...")
    model = model_trainer(cfg, model, train_ds, val_ds)

    log.info(f"Evaluating {model_name}...")
    scores = model_evaluator(cfg, model, [train_ds, val_ds, test_ds])

    ds_dict = {"full": ds, "train": train_ds, "val": val_ds, "test": test_ds}

    return model, scores, ds_dict


def run_hp_pipeline(cfg: DictConfig) -> float:
    """Runs the ScreenDL cross validation optimization pipline."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    # don't save outputs during hyperparameter optimization
    cfg.model.io.save = False
    cfg.dataset.output.save = False

    log.info(f"Loading {dataset_name}...")
    dataset = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    train_dataset, val_dataset, test_dataset = data_splitter(cfg, dataset)

    log.info(f"Preprocessing {dataset_name}...")
    train_dataset, val_dataset, test_dataset = data_preprocessor(
        cfg, train_dataset, val_dataset, test_dataset
    )

    log.info(f"Building {model_name}...")
    model = model_builder(cfg, train_dataset)

    log.info(f"Training {model_name}...")
    model = model_trainer(cfg, model, train_dataset, val_dataset)

    log.info(f"Evaluating {model_name}...")
    batch_size = cfg.model.hyper.batch_size

    cell_ids = val_dataset.cell_ids
    drug_ids = val_dataset.drug_ids
    targets = val_dataset.labels

    gen = BatchedResponseGenerator(val_dataset, batch_size)
    seq = gen.flow(cell_ids, drug_ids, targets=targets)

    loss, *_ = model.evaluate(seq)

    return loss


def run_sa_pipeline(
    cfg: DictConfig,
) -> t.Tuple[keras.Model, Dataset, Dataset, Dataset, Dataset]:
    """Runs the ScreenDL pipeline with ScreenAhead."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    ds = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    train_ds, val_ds, test_ds = data_splitter(cfg, ds)

    log.info(f"Preprocessing {dataset_name}...")
    train_ds, val_ds, test_ds = data_preprocessor(cfg, train_ds, val_ds, test_ds)

    log.info(f"Building {model_name}...")
    model = model_builder(cfg, train_ds)

    log.info(f"Training {model_name}...")
    model = model_trainer(cfg, model, train_ds, val_ds)

    log.info(f"Evaluating {model_name}...")
    model_evaluator(cfg, model, [train_ds, val_ds, test_ds])

    return model, ds, train_ds, val_ds, test_ds
