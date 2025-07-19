"""
HiDRA training and evaluation pipeline.

>>> HIDRA_ROOT="pkg/HiDRA" python scripts/runners/run.py --multirun \
        model=HiDRA-legacy \
        dataset.preprocess.norm=global
"""

from __future__ import annotations

import os
import json
import logging
import sys
import pickle

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from types import SimpleNamespace

from cdrpy.feat.encoders import PandasEncoder
from cdrpy.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.metrics import tf_metrics
from cdrpy.util.io import read_pickled_dict
from cdrpy.mapper import BatchedResponseGenerator

from screendl.utils.evaluation import make_pred_df, get_eval_metrics, ScoreDict
from screendl.utils import data_utils


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


def data_loader(cfg: DictConfig) -> t.Tuple[Dataset, t.Dict[str, t.List[str]]]:
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

    cell_encoders = {}
    for gs_name, gs_genes in gs_dict.items():
        gs_exp_mat = exp_mat[gs_genes]
        gs_enc_data = pd.DataFrame(
            gs_exp_mat.values, index=gs_exp_mat.index, columns=gs_genes
        )
        cell_encoders[gs_name] = PandasEncoder(gs_enc_data, name=gs_name)

    drug_encoders = {"mol": PandasEncoder(mol_mat, name="mol_encoder")}

    dataset = Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
    )

    return dataset, gs_dict


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
    geneset_dict: t.Dict[str, t.List[str]],
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> t.Tuple[Dataset, Dataset, Dataset]:
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
    for name, enc in train_dataset.cell_encoders.items():
        ss = StandardScaler()
        X = np.array(enc.encode(list(set(train_dataset.cell_ids))))
        _ = ss.fit(X)
        enc.data[:] = ss.transform(enc.data.values)

        if enc.data.isnull().any().any():
            # make sure that the normalization hasn't created any NaN values
            n_genes = enc.data.shape[-1]
            enc.data = enc.data.dropna(axis=1)
            n_dropped = n_genes - enc.data.shape[-1]
            log.warning(f"Dropped {n_dropped} genes with NaN values from {enc.name}")
            geneset_dict[name] = list(enc.data.columns)

    val_dataset.cell_encoders = train_dataset.cell_encoders
    test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, geneset_dict: t.Dict[str, t.List[str]]) -> keras.Model:
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

    train_seq = train_gen.flow_from_dataset(train_ds, shuffle=True, seed=4114)
    val_seq = val_gen.flow_from_dataset(val_ds, shuffle=False)

    _ = model.fit(
        train_seq,
        epochs=params.hyper.epochs,
        validation_data=val_seq,
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
) -> t.Dict[str, ScoreDict]:
    """Evaluates the HiDRA Model.

    Parameters
    ----------
        cfg:
        model:
        datasets:
    """

    param_dict = {
        "model": "HiDRA",
        "split_id": cfg.dataset.split.id,
        "split_type": cfg.dataset.split.name,
        "norm_method": cfg.dataset.preprocess.norm,
    }

    pred_dfs = []
    scores = {}
    for ds in datasets:
        gen = BatchedResponseGenerator(ds, cfg.model.hyper.batch_size)
        preds: np.ndarray = model.predict(gen.flow_from_dataset(ds))
        pred_df = make_pred_df(ds, preds, split_group=ds.name, **param_dict)
        pred_dfs.append(pred_df)
        scores[ds.name] = get_eval_metrics(pred_df)

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv("predictions.csv", index=False)

    with open("scores.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh, ensure_ascii=False, indent=4)

    return scores


def run_pipeline(
    cfg: DictConfig,
) -> t.Tuple[keras.Model, t.Dict[str, ScoreDict], t.Dict[str, Dataset]]:
    """"""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    ds, gs_dict = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    train_ds, val_ds, test_ds = data_splitter(cfg, ds)

    log.info(f"Preprocessing {dataset_name}...")
    train_ds, val_ds, test_ds = data_preprocessor(cfg, gs_dict, train_ds, val_ds, test_ds)

    log.info(f"Building {model_name}...")
    model = model_builder(cfg, gs_dict)

    log.info(f"Training {model_name}...")
    model = model_trainer(cfg, model, train_ds, val_ds)

    log.info(f"Evaluating {model_name}...")
    scores = model_evaluator(cfg, model, [train_ds, val_ds, test_ds])

    ds_dict = {"full": ds, "train": train_ds, "val": val_ds, "test": test_ds}

    return model, scores, ds_dict


def run_pdx_pipeline(
    cfg: DictConfig,
) -> t.Tuple[keras.Model, t.Dict[str, ScoreDict], t.Dict[str, Dataset]]:
    """"""
    model, scores, ds_dict = run_pipeline(cfg)

    pdmc_ds = ds_dict["test"]

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

    pdx_gen = BatchedResponseGenerator(pdx_ds, 256)
    pdx_seq = pdx_gen.flow_from_dataset(pdx_ds)
    pdx_preds: np.ndarray = model.predict(pdx_seq)

    param_dict = {"model": "HiDRA"}
    pdx_pred_df = make_pred_df(pdx_ds, pdx_preds, **param_dict)
    pdx_pred_df.to_csv("predictions_pdx.csv", index=False)

    return model, scores, ds_dict


def run_pdx_pipeline_v2(
    cfg: DictConfig,
) -> t.Tuple[keras.Model, t.Dict[str, ScoreDict], t.Dict[str, Dataset]]:
    """"""
    model, scores, ds_dict = run_pipeline(cfg)

    all_drug_ids = list(set(ds_dict["full"].drug_ids))
    all_pdmc_ids = list(set(ds_dict["test"].cell_ids))

    pdmc_ds = ds_dict["test"]

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

    # Expand to all tumor-drug combinations
    pdx_ds_full = data_utils.expand_dataset(pdx_ds, all_pdmc_ids, all_drug_ids)

    pdx_gen = BatchedResponseGenerator(pdx_ds_full, 256)
    pdx_seq = pdx_gen.flow_from_dataset(pdx_ds_full)
    pdx_preds: np.ndarray = model.predict(pdx_seq)

    param_dict = {"model": "HiDRA"}
    pdx_pred_df = make_pred_df(pdx_ds_full, pdx_preds, **param_dict)
    pdx_pred_df.to_csv("predictions_pdx.csv", index=False)

    return model, scores, ds_dict
