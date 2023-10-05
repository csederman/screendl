"""
HiDRA training and evaluation pipeline.

TODO: create a custom HiDRAEncoder class that encoders using the genelist
    for the specific type of output of hydra instead of multiple encoders for
    each pathway

>>> HIDRA_ROOT="/scratch/ucgd/lustre-work/marth/u0871891/projects/screendl/pkg/DualGCN/code" \
        python scripts/runners/run.py model=DualGCN-legacy
"""

from __future__ import annotations

import os
import logging
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from tensorflow import keras
from types import SimpleNamespace

from cdrpy.feat.encoders import PandasEncoder
from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.metrics import tf_metrics
from cdrpy.util.io import read_pickled_dict
from cdrpy.mapper import BatchedResponseGenerator


log = logging.getLogger(__name__)


def import_hidra_namespace() -> SimpleNamespace:
    """Imports the necessary function/classe definitions from DualGCN."""
    try:
        hidra_root = os.environ["HIDRA_ROOT"]
    except KeyError as e:
        raise e

    import importlib

    path = os.path.join(hidra_root, "Training/HiDRA_training.py")
    spec = importlib.util.spec_from_file_location("hidra", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hidra"] = module
    spec.loader.exec_module(module)

    return SimpleNamespace(create_model=module.Making_Model)


hidra = import_hidra_namespace()


def data_loader(cfg: DictConfig) -> tuple[Dataset, dict[str, list[str]]]:
    """Loads the input dataset.

    Parameters
    ----------
        cfg:

    Returns
    -------
    """
    paths = cfg.dataset.sources

    mol_path = paths.hidra.mol
    exp_path = paths.hidra.exp
    gene_path = paths.hidra.gene

    gs_dict = read_pickled_dict(gene_path)
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    mol_mat = pd.read_csv(mol_path, index_col=0).astype("int32")

    cell_encoders = []
    for gs_name, gs_genes in gs_dict.items():
        gs_exp = exp_mat[gs_genes]
        _temp = pd.DataFrame(
            gs_exp.values, index=gs_exp.index, columns=gs_genes
        )
        cell_encoders.append(PandasEncoder(_temp, name=gs_name))

    drug_encoders = [PandasEncoder(mol_mat, name="mol_encoder")]

    dataset = Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
    )

    return dataset, gs_dict


def data_splitter(
    cfg: DictConfig, dataset: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
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
    geneset_dict: dict[str, list[str]],
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Preprocessing pipeline.

    Parameters
    ----------
        cfg:
        geneset_dict:
        train_dataset:
        val_dataset:
        test_dataset:

    Returns
    -------
        A (train, validation, test) tuple of processed datasets.
    """

    # STEP 1: Normalize the drug responses
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    # STEP 2: Normalize the gene expression data.
    for enc in train_dataset.cell_encoders:
        x_enc = np.array(enc.encode(list(set(train_dataset.cell_ids))))
        x_mean = x_enc.mean(axis=0)
        x_std = x_enc.std(axis=0)
        enc.data: pd.DataFrame = (enc.data - x_mean) / x_std

        if enc.data.isnull().any().any():
            # make sure that the normalization hasn't created any NaN values
            n_genes = enc.data.shape[-1]
            enc.data = enc.data.dropna(axis=1)

            n_dropped = n_genes - enc.data.shape[-1]
            warnings.warn(
                f"Dropped {n_dropped} genes with NaN values from {enc.name}"
            )

            geneset_dict[enc.name] = list(enc.data.columns)

    val_dataset.cell_encoders = train_dataset.cell_encoders
    test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(
    cfg: DictConfig, geneset_dict: dict[str, list[str]]
) -> keras.Model:
    """Builds the HiDRA model.

    Parameters
    ----------
        cfg:
        geneset_dict:

    Returns
    -------
    """
    return hidra.create_model(geneset_dict)


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_ds: Dataset,
    val_ds: Dataset,
) -> keras.Model:
    """Trains the HiDRA model.

    Parameters
    ----------
        cfg:
        model:
        train_ds:
        val_ds:

    Returns
    -------
        The trained `keras.Model` instance.
    """
    params = cfg.model
    opt = keras.optimizers.Adam(learning_rate=params.hyper.learning_rate)

    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=["mse", tf_metrics.pearson],
    )

    callbacks = []

    if params.hyper.early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                "val_loss",
                patience=15,
                restore_best_weights=True,
                start_from_epoch=3,
                verbose=1,
            )
        )

    batch_size = params.hyper.batch_size
    train_gen = BatchedResponseGenerator(train_ds, batch_size)
    val_gen = BatchedResponseGenerator(val_ds, batch_size)

    train_sequence = train_gen.flow(
        train_ds.cell_ids,
        train_ds.drug_ids,
        targets=train_ds.labels,
        shuffle=True,
        seed=4114,
    )

    val_sequence = val_gen.flow(
        val_ds.cell_ids,
        val_ds.drug_ids,
        targets=val_ds.labels,
        shuffle=False,  # don't shuffle for validation
    )

    _ = model.fit(
        train_sequence,
        epochs=params.hyper.epochs,
        validation_data=val_sequence,
        callbacks=callbacks,
    )

    if params.io.save:
        save_dir = "."
        model.save(os.path.join(save_dir, "model"))
        model.save_weights(os.path.join(save_dir, "weights"))

    return model


def model_evaluator(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[Dataset],
) -> None:
    """Evaluates the HiDRA Model.

    Parameters
    ----------
        cfg:
        model:
        datasets:
    """

    # FIXME: convert this to use the sequence method

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

    pred_df.to_csv("predictions.csv", index=False)


def run_pipeline(cfg: DictConfig) -> None:
    """"""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    dataset, geneset_dict = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    train_dataset, val_dataset, test_dataset = data_splitter(cfg, dataset)

    log.info(f"Preprocessing {dataset_name}...")
    train_dataset, val_dataset, test_dataset = data_preprocessor(
        cfg, geneset_dict, train_dataset, val_dataset, test_dataset
    )

    log.info(f"Building {model_name}...")
    model = model_builder(cfg, geneset_dict)

    log.info(f"Training {model_name}...")
    model = model_trainer(cfg, model, train_dataset, val_dataset)

    log.info(f"Evaluating {model_name}...")
    model_evaluator(cfg, model, [train_dataset, val_dataset, test_dataset])
