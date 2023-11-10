"""
Pipeline for running the original (legacy) DualGCN code.

>>> DUALGCN_ROOT="pkg/DualGCN/code" python scripts/runners/run.py --multirun \
        model=DualGCN-legacy \
        dataset.preprocess.norm=global
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import sys
import tempfile
import pickle

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from pathlib import Path
from tensorflow import keras
from types import SimpleNamespace

from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.feat.encoders import DictEncoder, RepeatEncoder
from cdrpy.metrics import tf_metrics
from cdrpy.util.io import read_pickled_dict
from cdrpy.util.validation import check_same_columns, check_same_indexes
from cdrpy.mapper import BatchedResponseGenerator


log = logging.getLogger(__name__)


def import_dualgcn_namespace() -> SimpleNamespace:
    """Imports the necessary function/classe definitions from DualGCN."""
    try:
        path = os.environ["DUALGCN_ROOT"]
    except KeyError as e:
        raise e

    sys.path.insert(1, path)

    from model import KerasMultiSourceDualGCNModel
    from DualGCN import CalculateGraphFeat, CelllineGraphAdjNorm

    del sys.path[1]

    return SimpleNamespace(
        model=KerasMultiSourceDualGCNModel,
        calc_graph_feat=CalculateGraphFeat,
        get_ppi_adj=CelllineGraphAdjNorm,
    )


dualgcn = import_dualgcn_namespace()


def data_loader(cfg: DictConfig) -> Dataset:
    """Refactored DualGCN data loading and preprocessing pipeline.

    Parameters
    ----------
        cfg:

    Returns
    -------
    """
    paths = cfg.dataset.sources

    mol_path = paths.dualgcn.mol
    exp_path = paths.dualgcn.exp
    cnv_path = paths.dualgcn.cnv
    ppi_path = paths.dualgcn.ppi

    # STEP 1. Load the cell line omics data
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

    # STEP 2. Load the protein-protein interaction network
    idx_dict = {}
    for index, item in enumerate(common_genes):
        idx_dict[item] = index

    ppi_edges = pd.read_csv(ppi_path)
    ppi_adj_info = [[] for _ in common_genes]

    for gene_1, gene_2 in zip(ppi_edges["gene_1"], ppi_edges["gene_2"]):
        if idx_dict[gene_1] <= idx_dict[gene_2]:
            ppi_adj_info[idx_dict[gene_1]].append(idx_dict[gene_2])
            ppi_adj_info[idx_dict[gene_2]].append(idx_dict[gene_1])

    with tempfile.TemporaryDirectory() as tmpdir:
        # `DualGCN.CelllineGraphAdjNorm` requires genes saved as a .txt file.
        gene_list_file = Path(tmpdir) / "gene_list.txt"

        with open(gene_list_file, "w") as fh:
            for gene in common_genes:
                fh.write(f"{gene}\n")

        ppi_adj_norm = dualgcn.get_ppi_adj(ppi_adj_info, gene_list_file)
        ppi_adj_norm = ppi_adj_norm.astype(np.float32)

    ppi_adj_enc = RepeatEncoder(ppi_adj_norm, name="ppi_adj_encoder")

    # STEP 3. Load and preprocess drug molecular features.
    drug_dict = read_pickled_dict(mol_path)

    drug_feat = {}
    drug_adj = {}
    for k, (feat, _, adj) in drug_dict.items():
        drug_feat[k], drug_adj[k] = dualgcn.calc_graph_feat(feat, adj)

    drug_feat_enc = DictEncoder(drug_feat, name="drug_feature_encoder")
    drug_adj_enc = DictEncoder(drug_adj, name="drug_adj_encoder")

    cell_encoders = {"omics": omics_enc, "ppi": ppi_adj_enc}
    drug_encoders = {"feat": drug_feat_enc, "adj": drug_adj_enc}

    # STEP 4. Create the dataset
    dataset = Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
        encode_drugs_first=True,
    )

    return dataset


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
) -> t.Tuple[Dataset, Dataset | None, Dataset | None]:
    """Preprocessing pipeline.

    Assumes the first dataset provided is the training set.

    Parameters
    ----------
        cfg:
        datasets:

    Returns
    -------
        A (train, validation, test) tuple of processed datasets.
    """

    # 1. normalize the drug responses
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    # 2. normalize omics data
    omics_enc = train_dataset.cell_encoders["omics"]
    X_omics = np.array(omics_enc.encode(train_dataset.cell_ids))
    X_mean = np.mean(X_omics, axis=0)
    X_std = np.std(X_omics, axis=0)

    omics_enc.data = {k: (v - X_mean) / X_std for k, v in omics_enc.data.items()}

    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, train_ds: Dataset) -> keras.Model:
    """Builds the DualGCN model.

    Parameters
    ----------
        cfg:
        train_ds:

    Returns
    -------
    """

    # extract shapes from encoders
    cell_feat_dim = train_ds.cell_encoders["omics"].shape[-1]
    drug_feat_dim = train_ds.drug_encoders["feat"].shape[-1]

    model = dualgcn.model().createMaster(
        drug_dim=drug_feat_dim,
        cell_line_dim=cell_feat_dim,
        units_list=cfg.model.hyper.units_list,
    )

    return model


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_ds: Dataset,
    val_ds: Dataset,
) -> keras.Model:
    """Trains the DualGCN model.

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
        metrics=[tf_metrics.pearson],
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

    if params.io.checkpoints:
        ckpt_path = "./checkpoint"
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    ckpt_path, monitor="val_loss", save_best_only=True
                )
            )

    batch_size = params.hyper.batch_size
    train_gen = BatchedResponseGenerator(train_ds, batch_size)
    val_gen = BatchedResponseGenerator(val_ds, batch_size)

    train_seq = train_gen.flow_from_dataset(
        train_ds, drugs_first=True, shuffle=True, seed=4114
    )
    val_seq = val_gen.flow_from_dataset(val_ds, drugs_first=True, shuffle=False)

    _ = model.fit(
        train_seq,
        epochs=params.hyper.epochs,
        validation_data=val_seq,
        callbacks=callbacks,
    )

    return model


def model_evaluator(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[Dataset],
) -> None:
    """Evaluates the DualGCN Model.

    Parameters
    ----------
        cfg:
        model:
        datasets:
    """

    pred_df = []
    for ds in datasets:
        gen = BatchedResponseGenerator(ds, cfg.model.hyper.batch_size)
        seq = gen.flow_from_dataset(ds, drugs_first=True, shuffle=False)
        preds: np.ndarray = model.predict(seq)
        preds = pd.DataFrame(
            {
                "cell_id": ds.cell_ids,
                "drug_id": ds.drug_ids,
                "y_true": ds.labels,
                "y_pred": preds.reshape(-1),
                "split": ds.name,
            }
        )
        pred_df.append(preds)

    pred_df = pd.concat(pred_df)
    pred_df["fold"] = cfg.dataset.split.id
    pred_df["model"] = "DualGCN"

    pred_df.to_csv("./predictions.csv", index=False)


def run_pipeline(cfg: DictConfig) -> None:
    """"""
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
    model_evaluator(
        cfg,
        model,
        [train_dataset, val_dataset, test_dataset],
    )
