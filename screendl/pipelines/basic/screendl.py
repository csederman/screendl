"""
Run utilities for ScreenDL.
"""

from __future__ import annotations

import os
import logging
import pickle

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.mapper import BatchedResponseGenerator

from screendl import model as screendl


if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder


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

    return Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
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
        datasets:

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
        exp_norm_layer=None,
        cnv_norm_layer=None,
        exp_hidden_dims=params.hyper.hidden_dims.exp,
        mut_hidden_dims=params.hyper.hidden_dims.mut,
        cnv_hidden_dims=params.hyper.hidden_dims.cnv,
        ont_hidden_dims=params.hyper.hidden_dims.ont,
        mol_hidden_dims=params.hyper.hidden_dims.mol,
        shared_hidden_dims=params.hyper.hidden_dims.shared,
        use_batch_norm=params.hyper.use_batch_norm,
        use_dropout=params.hyper.use_dropout,
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
) -> None:
    """Evaluates the ScreenDL Model.

    Parameters
    ----------
        cfg:
        model:
        datasets:
    """
    pred_dfs = []
    for ds in datasets:
        gen = BatchedResponseGenerator(ds, cfg.model.hyper.batch_size)
        preds: np.ndarray = model.predict(gen.flow(ds.cell_ids, ds.drug_ids))
        pred_dfs.append(
            pd.DataFrame(
                {
                    "cell_id": ds.cell_ids,
                    "drug_id": ds.drug_ids,
                    "y_true": ds.labels,
                    "y_pred": preds.reshape(-1),
                    "split": ds.name,
                }
            )
        )

    pred_df = pd.concat(pred_dfs)
    pred_df["fold"] = cfg.dataset.split.id
    pred_df["model"] = "ScreenDL"

    pred_df.to_csv("predictions.csv", index=False)

    if cfg.dataset.output.save:
        root_dir = Path("./datasets")
        root_dir.mkdir()
        for ds in datasets:
            file_path = root_dir / f"{ds.name}.h5"
            ds.save(file_path)


def run_pipeline(cfg: DictConfig) -> None:
    """Runs the ScreenDL training pipeline."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

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
    model_evaluator(cfg, model, [train_dataset, val_dataset, test_dataset])


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
) -> t.Tuple[keras.Model, Dataset, Dataset, Dataset]:
    """Runs the ScreenDL pipeline with ScreenAhead."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

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
    model_evaluator(cfg, model, [train_dataset, val_dataset, test_dataset])

    return model, dataset, train_dataset, val_dataset, test_dataset
