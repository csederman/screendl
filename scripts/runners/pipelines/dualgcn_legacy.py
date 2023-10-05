"""
Pipeline for running the original (legacy) DualGCN code.

>>> DUALGCN_ROOT="/scratch/ucgd/lustre-work/marth/u0871891/projects/screendl/pkg/DualGCN/code" \
        python scripts/runners/run.py model=DualGCN-legacy
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
import tensorflow as tf

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

    # STEP 4. Create the dataset
    dataset = Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=[omics_enc, ppi_adj_enc],
        drug_encoders=[drug_feat_enc, drug_adj_enc],
        encode_drugs_first=True,
    )

    return dataset


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
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset | None, tf.data.Dataset | None]:
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
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    X_train, y_train = train_dataset.encode()
    X_train = list(X_train)

    # STEP 1: Normalize the response data.
    X_train_omics = np.array(X_train[2])
    X_train_omics_mean = np.mean(X_train_omics, axis=0)
    X_train_omics_std = np.std(X_train_omics, axis=0)
    X_train_omics = (X_train_omics - X_train_omics_mean) / X_train_omics_std
    X_train[2] = X_train_omics

    # STEP 2: Create the data generator.
    def make_generator(X, Y) -> t.Callable[[], t.Generator[t.Any, None, None]]:
        def generator():
            for x, y in zip(zip(*X), Y):
                yield x, y

        return generator

    output_signature = train_dataset._infer_tf_output_signature()

    train_tfds = tf.data.Dataset.from_generator(
        make_generator(X_train, y_train), output_signature=output_signature
    )

    # STEP 3: Apply the transformations to validation/test sets.
    val_tfds = None
    if val_dataset is not None:
        X_val, y_val = val_dataset.encode()
        X_val = list(X_val)
        X_val_omics = np.array(X_val[2])
        X_val_omics = (X_val_omics - X_train_omics_mean) / X_train_omics_std
        X_val[2] = X_val_omics

        val_tfds = tf.data.Dataset.from_generator(
            make_generator(X_val, y_val), output_signature=output_signature
        )

    test_tfds = None
    if test_dataset is not None:
        X_test, y_test = test_dataset.encode()
        X_test = list(X_test)
        X_test_omics = np.array(X_test[2])
        X_test_omics = (X_test_omics - X_train_omics_mean) / X_train_omics_std
        X_test[2] = X_test_omics

        test_tfds = tf.data.Dataset.from_generator(
            make_generator(X_test, y_test), output_signature=output_signature
        )

    return train_tfds, val_tfds, test_tfds


def model_builder(
    cfg: DictConfig, cell_feat_dim: int, drug_feat_dim: int
) -> keras.Model:
    """Builds the DualGCN model.

    Parameters
    ----------
        cfg:
        cell_feat_dim:
        drug_feat_dim:

    Returns
    -------
    """
    model = dualgcn.model().createMaster(
        drug_dim=drug_feat_dim,
        cell_line_dim=cell_feat_dim,
        units_list=cfg.model.hyper.units_list,
    )

    return model


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
) -> keras.Model:
    """Trains the DualGCN model.

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

    _ = model.fit(
        train_dataset.shuffle(10000).batch(params.hyper.batch_size),
        epochs=params.hyper.epochs,
        validation_data=val_dataset.shuffle(10000).batch(
            params.hyper.batch_size
        ),
        callbacks=callbacks,
    )

    return model


def model_evaluator(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[tuple[Dataset, tf.data.Dataset]],
) -> None:
    """Evaluates the DualGCN Model.

    Parameters
    ----------
        cfg:
        model:
        datasets:
    """

    pred_df = []
    for ds, tfds in datasets:
        preds = model.predict(tfds.batch(32)).reshape(-1)
        preds = pd.DataFrame(
            {
                "cell_id": ds.cell_ids,
                "drug_id": ds.drug_ids,
                "y_true": ds.labels,
                "y_pred": preds,
                "split": ds.name,
            }
        )
        pred_df.append(preds)

    pred_df = pd.concat(pred_df)
    pred_df["fold"] = cfg.dataset.split.id
    pred_df.to_csv("./predictions.csv", index=False)

    if cfg.dataset.output.save:
        root_dir = Path("./datasets")
        root_dir.mkdir()
        for ds in datasets:
            subdir = root_dir / str(ds.name)
            subdir.mkdir()
            ds.save(subdir)


def run_pipeline(cfg: DictConfig) -> None:
    """"""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    dataset = data_loader(cfg)

    cell_feat_dim = dataset.cell_encoders[0].shape[-1]
    drug_feat_dim = dataset.drug_encoders[0].shape[-1]

    log.info(f"Splitting {dataset_name}...")
    train_dataset, val_dataset, test_dataset = data_splitter(cfg, dataset)

    log.info(f"Preprocessing {dataset_name}...")
    train_tfds, val_tfds, test_tfds = data_preprocessor(
        cfg, train_dataset, val_dataset, test_dataset
    )

    log.info(f"Building {model_name}...")
    model = model_builder(cfg, cell_feat_dim, drug_feat_dim)

    log.info(f"Training {model_name}...")
    model = model_trainer(cfg, model, train_tfds, val_tfds)

    log.info(f"Evaluating {model_name}...")
    model_evaluator(
        cfg,
        model,
        datasets=[
            (train_dataset, train_tfds),
            (val_dataset, val_tfds),
            (test_dataset, test_tfds),
        ],
    )
