#!/usr/bin/env python
"""Trains and evaluates baseline models in cell lines."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import logging

import numpy as np
import pandas as pd
import typing as t

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

from cdrpy.datasets import Dataset
from cdrpy.data.preprocess import GroupStandardScaler

from screendl.pipelines.basic.screendl import (
    data_loader,
    data_splitter,
    data_preprocessor,
)
from screendl.utils import data_utils

if t.TYPE_CHECKING:
    from omegaconf import DictConfig


log = logging.getLogger(__name__)


BASELINES = {
    "ridge": Ridge,
    "forest": RandomForestRegressor,
}

BASELINE_PARAM_GRIDS = {
    "ridge": {
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    },
    "forest": {
        "max_depth": [20, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 3, 4],
        "n_estimators": [200, 400],
    },
}


def _prepare_data_for_sklearn_model(D: Dataset) -> t.Tuple[pd.DataFrame, pd.Series]:
    """Reshape data for sklearn estimators."""

    y = D.obs.copy()

    x_cell = (
        D.cell_encoders["exp"]
        .data.loc[y["cell_id"]]
        .reset_index(drop=True)
        .astype(np.float32)
    )
    x_drug = (
        D.drug_encoders["mol"]
        .data.loc[y["drug_id"]]
        .reset_index(drop=True)
        .astype(np.float32)
    )

    return pd.concat([x_cell, x_drug], axis=1), y["label"]


def _preds_vs_background(
    estimator: GridSearchCV,
    D_t: Dataset,
    D_e: Dataset,
    X_t: pd.DataFrame,
    X_e: pd.DataFrame,
) -> pd.DataFrame:
    """Predict z-scores using the training set as a background distribution."""
    preds_t = D_t.obs.rename(columns={"label": "y_true"}).drop(columns="id").copy()
    preds_t["y_pred"] = estimator.predict(X_t)

    preds_e = D_e.obs.rename(columns={"label": "y_true"}).drop(columns="id").copy()
    preds_e["y_pred"] = estimator.predict(X_e)

    gss = GroupStandardScaler().fit(preds_t[["y_pred"]], groups=preds_t["drug_id"])
    preds_e["y_pred"] = gss.transform(preds_e[["y_pred"]], groups=preds_e["drug_id"])

    return preds_e


def _decomp_tumor_features(
    X: pd.DataFrame, n_features: int | None = None
) -> pd.DataFrame:
    """Reduce tumor feature dimensionality using PCA."""
    pca = PCA(n_components=n_features, random_state=42)
    X_decomp = pca.fit_transform(X)  # features are already normalized
    X_decomp = X_decomp / np.sqrt(pca.explained_variance_)  # whiten the PCs
    return pd.DataFrame(
        X_decomp,
        index=X.index,
        columns=[f"PC_T{i}" for i in range(X_decomp.shape[1])],
    )


def _decomp_drug_features(
    X: pd.DataFrame, n_features: int | None = None
) -> pd.DataFrame:
    """Reduce drug feature dimensionality using PCA."""
    pca = PCA(n_components=n_features, random_state=42)
    X_decomp = pca.fit_transform(X.transform(stats.zscore).dropna(axis=1))
    X_decomp = X_decomp / np.sqrt(pca.explained_variance_)  # whiten the PCs
    return pd.DataFrame(
        X_decomp,
        index=X.index,
        columns=[f"PC_D{i}" for i in range(X_decomp.shape[1])],
    )


def _get_drug_ids(*datasets: Dataset) -> t.List[str]:
    """Get all drug IDs from the provided datasets."""
    all_drug_ids = set()
    for D in datasets:
        if D.drug_ids is not None:
            all_drug_ids.update(D.drug_ids)
    return list(all_drug_ids)


def train_baseline(
    cfg: DictConfig, D_t: Dataset, D_e: Dataset, all_drug_ids: t.List[str] | None = None
) -> pd.DataFrame:
    """Trains and evaluates baseline."""
    if cfg.baseline.estimator not in BASELINES:
        raise KeyError(f"Invalid baseline (got '{cfg.baseline.estimator}')")

    if all_drug_ids is None:
        all_drug_ids = _get_drug_ids(D_t, D_e)

    estimator = BASELINES[cfg.baseline.estimator]()
    param_grid = BASELINE_PARAM_GRIDS[cfg.baseline.estimator]

    reg = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=GroupKFold(n_splits=5),
        n_jobs=cfg.baseline.n_jobs,
        verbose=3,
        refit=True,
    )

    X_t, y_t = _prepare_data_for_sklearn_model(D_t)
    # X_e, _ = _prepare_data_for_sklearn_model(D_e)

    D_t.shuffle(seed=cfg.baseline.seed)
    _ = reg.fit(X_t, y_t, groups=D_t.cell_ids)
    print(reg.best_params_)

    D_t_full = data_utils.expand_dataset(
        D_t,
        cell_ids=list(set(D_t.cell_ids)),
        drug_ids=list(set(D_t.drug_ids) | set(all_drug_ids)),
    )
    D_e_full = data_utils.expand_dataset(
        D_e,
        cell_ids=list(set(D_e.cell_ids)),
        drug_ids=list(set(D_t.drug_ids) | set(all_drug_ids)),
    )

    X_t_full, _ = _prepare_data_for_sklearn_model(D_t_full)
    X_e_full, _ = _prepare_data_for_sklearn_model(D_e_full)

    preds = _preds_vs_background(reg, D_t_full, D_e_full, X_t_full, X_e_full)
    preds["model"] = cfg.baseline.estimator
    preds["split_id"] = cfg.dataset.split.id

    return preds


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="baselines",
)
def run(cfg: DictConfig) -> None:
    """Runs baseline validation.

    Parameters
    ----------
    cfg : DictConfig
        Experiment configuration.
    """
    dataset_name = cfg.dataset.name
    assert cfg.dataset.preprocess.norm == "grouped"

    log.info(f"Loading {dataset_name}...")
    D = data_loader(cfg)
    all_drug_ids = list(D.drug_encoders["mol"].keys())

    log.info(f"Splitting {dataset_name}...")
    D_t, _, D_e = data_splitter(cfg, D)

    log.info(f"Preprocessing {dataset_name}...")
    D_t, _, D_e = data_preprocessor(cfg, D_t, None, D_e)

    if cfg.experiment.reduce_tumor_features:
        # use PCA to reduce the dimensionality of the tumor features
        D.cell_encoders["exp"].data = _decomp_tumor_features(
            D.cell_encoders["exp"].data, n_features=cfg.experiment.num_tumor_features
        )

    if cfg.experiment.reduce_drug_features:
        # use PCA to reduce the dimensionality of the drug features
        D.drug_encoders["mol"].data = _decomp_drug_features(
            D.drug_encoders["mol"].data, n_features=cfg.experiment.num_drug_features
        )

    log.info(f"Fitting baselines...")
    results_df = train_baseline(cfg, D_t, D_e, all_drug_ids=all_drug_ids)
    results_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    run()
