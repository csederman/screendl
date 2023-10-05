"""
DeepCDR training and evaluation pipeline.

>>> DEEPCDR_ROOT="/scratch/ucgd/lustre-work/marth/u0871891/projects/screendl/pkg/DeepCDR/prog" \
        python scripts/runners/run.py model=DeepCDR-legacy dataset.preprocess.norm=global
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
from sklearn.preprocessing import QuantileTransformer
from tensorflow import keras
from types import SimpleNamespace

from cdrpy.feat.encoders import PandasEncoder, DictEncoder
from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.metrics import tf_metrics
from cdrpy.util.io import read_pickled_dict
from cdrpy.mapper import BatchedResponseGenerator


log = logging.getLogger(__name__)


def import_deepcdr_namespace() -> SimpleNamespace:
    """Imports the necessary function/classe definitions from DualGCN."""
    try:
        deepcdr_root = os.environ["DEEPCDR_ROOT"]
    except KeyError as e:
        raise e

    sys.path.insert(1, deepcdr_root)

    from model import KerasMultiSourceGCNModel
    from run_DeepCDR import CalculateGraphFeat

    del sys.path[1]

    return SimpleNamespace(
        model=KerasMultiSourceGCNModel, calc_graph_feat=CalculateGraphFeat
    )


deepcdr = import_deepcdr_namespace()


def data_loader(cfg: DictConfig) -> Dataset:
    """Refactored DualGCN data loading and preprocessing pipeline.

    Parameters
    ----------
        cfg:

    Returns
    -------
    """
    paths = cfg.dataset.sources

    mol_path = paths.deepcdr.mol
    exp_path = paths.deepcdr.exp
    mut_path = paths.deepcdr.mut

    # STEP 1. Load the cell line omics data
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    mut_mat = pd.read_csv(mut_path, index_col=0).astype("int32")

    mut_dict = {}
    for cell_id in mut_mat.index:
        mut_dict[cell_id] = mut_mat.loc[cell_id].values.reshape(1, -1, 1)

    exp_enc = PandasEncoder(exp_mat, name="exp_encoder")
    mut_enc = DictEncoder(mut_dict, name="mut_encoder")

    # STEP 3. Load and preprocess drug molecular features.
    drug_dict = read_pickled_dict(mol_path)

    drug_feat = {}
    drug_adj = {}
    for k, (feat, _, adj) in drug_dict.items():
        drug_feat[k], drug_adj[k] = deepcdr.calc_graph_feat(feat, adj)

    drug_feat_enc = DictEncoder(drug_feat, name="drug_feature_encoder")
    drug_adj_enc = DictEncoder(drug_adj, name="drug_adj_encoder")

    # STEP 4. Create the dataset
    dataset = Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=[mut_enc, exp_enc],
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
) -> tuple[Dataset, Dataset, Dataset]:
    """Preprocessing pipeline.

    Parameters
    ----------
        cfg:
        datasets:

    Returns
    -------
        A (train, validation, test) tuple of processed datasets.
    """
    # STEP 1: Normalize the drug response data
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    # STEP 2: Normalize the gene expression data
    exp_encoder: PandasEncoder = train_dataset.cell_encoders[1]
    X = np.array(exp_encoder.encode(list(set(train_dataset.cell_ids))))

    if cfg.model.preprocess.use_quantile_norm:
        # apply quantile normalization
        qt = QuantileTransformer(
            output_distribution="normal", random_state=1771
        ).fit(X)
        exp_encoder.data[:] = qt.transform(exp_encoder.data.values)
    else:
        # apply zscore normalization
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        exp_encoder.data: pd.DataFrame = (exp_encoder.data - x_mean) / x_std

    num_genes = exp_encoder.data.shape[-1]
    exp_encoder.data = exp_encoder.data.dropna(axis=1)
    num_missing = num_genes - exp_encoder.shape[-1]
    if num_missing > 0:
        warnings.warn(f"Dropped {num_missing} genes with NaN values.")

    val_dataset.cell_encoders = train_dataset.cell_encoders
    test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, train_dataset: Dataset) -> keras.Model:
    """Builds the DeepCDR model.

    Parameters
    ----------
        cfg:
        cell_feat_dim:
        drug_feat_dim:

    Returns
    -------
    """

    # extract mut shape from cell encoders
    mut_enc: DictEncoder = train_dataset.cell_encoders[0]
    mut_dim = mut_enc.shape[1]

    # extract exp shape from cell encoders
    exp_enc: PandasEncoder = train_dataset.cell_encoders[1]
    exp_dim = exp_enc.shape[-1]

    drug_dim = train_dataset.drug_encoders[0].shape[-1]

    model = deepcdr.model(
        use_mut=True, use_gexp=True, use_methy=False
    ).createMaster(
        drug_dim=drug_dim,
        mutation_dim=mut_dim,
        gexpr_dim=exp_dim,
        units_list=cfg.model.hyper.units_list,
        use_relu=True,
        use_bn=True,
        use_GMP=True,
    )

    return model


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_ds: Dataset,
    val_ds: Dataset,
) -> keras.Model:
    """Trains the DeepCDR model.

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
        optimizer=opt, loss="mean_squared_error", metrics=[tf_metrics.pearson]
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
        drugs_first=True,
        shuffle=True,
        seed=4114,
    )

    val_sequence = val_gen.flow(
        val_ds.cell_ids,
        val_ds.drug_ids,
        targets=val_ds.labels,
        drugs_first=True,
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
        preds: np.ndarray = model.predict(
            gen.flow(ds.cell_ids, ds.drug_ids, drugs_first=True, shuffle=False)
        )
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