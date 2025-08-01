"""
DeepCDR training and evaluation pipeline.

>>> DEEPCDR_ROOT="pkg/DeepCDR/prog" python scripts/runners/run.py --multirun \
        model=DeepCDR-legacy \
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
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tensorflow import keras
from types import SimpleNamespace

from cdrpy.feat.encoders import PandasEncoder, DictEncoder
from cdrpy.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.metrics import tf_metrics
from cdrpy.util.io import read_pickled_dict
from cdrpy.mapper import BatchedResponseGenerator

from screendl.utils.evaluation import make_pred_df, get_eval_metrics, ScoreDict
from screendl.utils import data_utils


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

    cell_encoders = {
        "mut": DictEncoder(mut_dict, name="mut_encoder"),
        "exp": PandasEncoder(exp_mat, name="exp_encoder"),
    }

    # STEP 2. Load and preprocess drug molecular features.
    drug_dict = read_pickled_dict(mol_path)

    drug_feat = {}
    drug_adj = {}
    for k, (feat, _, adj) in drug_dict.items():
        drug_feat[k], drug_adj[k] = deepcdr.calc_graph_feat(feat, adj)

    drug_encoders = {
        "feat": DictEncoder(drug_feat, name="drug_feature_encoder"),
        "adj": DictEncoder(drug_adj, name="drug_adj_encoder"),
    }

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
    # STEP 1: Normalize the drug response data
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )

    # STEP 2: Normalize the gene expression data
    exp_encoder: PandasEncoder = train_dataset.cell_encoders["exp"]
    X = np.array(exp_encoder.encode(list(set(train_dataset.cell_ids))))

    if cfg.model.preprocess.use_quantile_norm:
        # apply quantile normalization
        qt = QuantileTransformer(output_distribution="normal", random_state=1771)
        _ = qt.fit(X)
        exp_encoder.data[:] = qt.transform(exp_encoder.data.values)
    else:
        # apply zscore normalization
        ss = StandardScaler().fit(X)
        exp_encoder.data[:] = ss.transform(exp_encoder.data.values)

    num_genes = exp_encoder.data.shape[-1]
    exp_encoder.data = exp_encoder.data.dropna(axis=1)
    num_missing = num_genes - exp_encoder.shape[-1]
    if num_missing > 0:
        log.warning(f"Dropped {num_missing} genes with NaN values.")

    # FIXME: add if not None
    val_dataset.cell_encoders = train_dataset.cell_encoders
    test_dataset.cell_encoders = train_dataset.cell_encoders

    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, train_dataset: Dataset) -> keras.Model:
    """Builds the DeepCDR model.

    Parameters
    ----------
        cfg:
        train_dataset:

    Returns
    -------
    """

    # extract shapes from encoders
    mut_dim = train_dataset.cell_encoders["mut"].shape[1]
    exp_dim = train_dataset.cell_encoders["exp"].shape[-1]
    drug_dim = train_dataset.drug_encoders["feat"].shape[-1]

    model = deepcdr.model(use_mut=True, use_gexp=True, use_methy=False).createMaster(
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

    model.compile(optimizer=opt, loss="mean_squared_error", metrics=[tf_metrics.pearson])

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

    # FIXME: convert this to use the sequence method

    param_dict = {
        "model": "DeepCDR",
        "split_id": cfg.dataset.split.id,
        "split_type": cfg.dataset.split.name,
        "norm_method": cfg.dataset.preprocess.norm,
    }

    pred_dfs = []
    scores = {}
    for ds in datasets:
        gen = BatchedResponseGenerator(ds, cfg.model.hyper.batch_size)
        seq = gen.flow_from_dataset(ds, drugs_first=True, shuffle=False)
        preds: np.ndarray = model.predict(seq)
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
    pdx_seq = pdx_gen.flow_from_dataset(pdx_ds, drugs_first=True)
    pdx_preds: np.ndarray = model.predict(pdx_seq)

    param_dict = {"model": "DeepCDR"}
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
    pdx_seq = pdx_gen.flow_from_dataset(pdx_ds_full, drugs_first=True)
    pdx_preds: np.ndarray = model.predict(pdx_seq)

    param_dict = {"model": "DeepCDR"}
    pdx_pred_df = make_pred_df(pdx_ds_full, pdx_preds, **param_dict)
    pdx_pred_df.to_csv("predictions_pdx.csv", index=False)

    return model, scores, ds_dict
