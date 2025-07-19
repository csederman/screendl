#!/usr/bin/env python
"""Trains and evaluates baseline models in breast cancer PDXs."""

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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.decomposition import PCA
from omegaconf.listconfig import ListConfig


from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import PandasEncoder
from cdrpy.data.preprocess import normalize_responses, GroupStandardScaler

from screendl.pipelines.basic.screendl import data_loader
from screendl.utils import data_utils

if t.TYPE_CHECKING:
    from omegaconf import DictConfig
    from cdrpy.feat.encoders import PandasEncoder


log = logging.getLogger(__name__)


BASELINES = {
    "ridge": Ridge,
    "forest": RandomForestRegressor,
}

BASELINE_PARAM_GRIDS = {
    "ridge": {
        "alpha": [0.001, 0.01, 0.1, 10, 100, 1000, 10000],
    },
    "forest": {
        "max_depth": [20, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 3, 4],
        "n_estimators": [200, 400],
    },
}


def get_id_prefix(pdmc_id: str) -> str:
    """Gets matching prefixes to avoid data leak."""
    if pdmc_id.startswith("HCI"):
        return pdmc_id[:6]
    elif pdmc_id.startswith("BCM"):
        return pdmc_id[:7]
    elif pdmc_id.startswith("TOW"):
        return pdmc_id[:5]
    else:
        return pdmc_id


def loo_split_generator(
    D: Dataset,
) -> t.Generator[t.Tuple[Dataset, Dataset], None, None]:
    """Generates leave-one-pdmc-out splits for model evaluation.

    Parameters
    ----------
    D : Dataset
        The PDMC dataset to split.

    Returns
    -------
    t.Generator[t.Tuple[Dataset, Dataset], None, None]
        Generator of (train, test) dataset tuples

    Yields
    ------
    Iterator[t.Generator[t.Tuple[Dataset, Dataset], None, None]]
        Iterator of (train, test) dataset tuples
    """
    pdmc_ids = set(D.cell_ids)
    for test_id in pdmc_ids:
        id_prefix = get_id_prefix(test_id)
        train_ids = [x for x in pdmc_ids if not str(x).startswith(id_prefix)]
        train_ds = D.select_cells(train_ids, name="train")
        test_ds = D.select_cells([test_id], name="test")
        yield train_ds, test_ds


def apply_pdmc_drug_filters(
    cfg: DictConfig, pdmc_ds: Dataset, cell_ds: Dataset | t.Iterable[Dataset]
) -> Dataset:
    """Filters drugs based on configured parameters."""
    if isinstance(cell_ds, Dataset):
        keep_drugs = set(cell_ds.drug_ids)
    elif isinstance(cell_ds, list):
        keep_drugs = set()
        for D in cell_ds:
            keep_drugs = keep_drugs.union(D.drug_ids)

    if cfg.experiment.keep_pdmc_only_drugs is not None:
        if cfg.experiment.keep_pdmc_only_drugs == "all":
            keep_drugs = keep_drugs.union(pdmc_ds.drug_ids)
        elif isinstance(cfg.experiment.keep_pdmc_only_drugs, ListConfig):
            keep_drugs = keep_drugs.union(cfg.experiment.keep_pdmc_only_drugs)
        else:
            raise TypeError(
                "Unsupported parameter type (experiment.keep_pdmc_only_drugs)"
            )

    if cfg.experiment.min_pdmcs_per_drug is not None:
        pdmcs_per_drug = pdmc_ds.obs["drug_id"].value_counts()
        drugs_with_min_pdmcs = pdmcs_per_drug[
            pdmcs_per_drug >= cfg.experiment.min_pdmcs_per_drug
        ]
        keep_drugs = keep_drugs.intersection(drugs_with_min_pdmcs.index)

    return pdmc_ds.select_drugs(keep_drugs, name=pdmc_ds.name)


def copy_dataset(D: Dataset) -> Dataset:
    """Creates a deep copy of the Dataset."""
    obs = D.obs.copy(deep=True)

    cell_encoders = None
    if D.cell_encoders is not None:
        cell_encoders = {k: v.copy() for k, v in D.cell_encoders.items()}

    drug_encoders = None
    if D.drug_encoders is not None:
        drug_encoders = {k: v.copy() for k, v in D.drug_encoders.items()}

    cell_meta = None
    if D.cell_meta is not None:
        cell_meta = D.cell_meta.copy(deep=True)

    drug_meta = None
    if D.drug_meta is not None:
        drug_meta = D.drug_meta.copy(deep=True)

    return Dataset(
        obs,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
        cell_meta=cell_meta,
        drug_meta=drug_meta,
        transforms=None,
        encode_drugs_first=D.encode_drugs_first,
        name=D.name,
        desc=D.desc,
    )


def data_preprocessor(
    cfg: DictConfig, D_c: Dataset, D_p: Dataset
) -> t.Tuple[Dataset, Dataset]:
    """Preprocesses and split the raw data.

    Parameters
    ----------
    cfg : DictConfig
        Hidra experiment config (ignored)
    D_c : Dataset
        The cell line (source) training dataset
    D_p : Dataset
        The PDMC dataset

    Returns
    -------
    t.Tuple[Dataset, Dataset]
        A tuple containing the processed dataset objects.
    """

    cell_ids = list(set(D_c.cell_ids))
    pdmc_ids = list(set(D_p.cell_ids))

    # 1. normalize the gene expression
    enc: PandasEncoder = D_c.cell_encoders["exp"]

    x_cell = enc.data.loc[cell_ids]
    x_pdmc = enc.data.loc[pdmc_ids]

    ss = StandardScaler().fit(x_cell)
    x_cell[:] = ss.transform(x_cell)
    x_pdmc[:] = ss.transform(x_pdmc)

    enc.data = pd.concat([x_cell, x_pdmc])

    if cfg.experiment.reduce_tumor_features:
        # use PCA to reduce the dimensionality of the tumor features
        enc.data = _decomp_tumor_features(
            enc.data, n_features=cfg.experiment.num_tumor_features
        )

    if cfg.experiment.reduce_drug_features:
        # use PCA to reduce the dimensionality of the drug features
        D_p.drug_encoders["mol"].data = _decomp_drug_features(
            D_p.drug_encoders["mol"].data, n_features=cfg.experiment.num_drug_features
        )

    # FIXME: this could be done external to this function
    D_p = apply_pdmc_drug_filters(cfg, D_p, D_c)

    # normalize the drug responses
    D_c, *_ = normalize_responses(D_c, norm_method="grouped")
    D_p, *_ = normalize_responses(D_p, norm_method="grouped")

    return D_c, D_p


def data_preprocessor_0(cfg: DictConfig, D_c: Dataset, D_p: Dataset) -> Dataset:
    """Preprocesses and split the raw data.

    Parameters
    ----------
    cfg : DictConfig
        Hidra experiment config (ignored)
    train_cell_ds : Dataset
        The cell line (source) training dataset
    val_cell_ds : Dataset
        The cell line validation dataset (used for early stopping)
    pdmc_ds : Dataset
        The PDMC dataset

    Returns
    -------
    t.Tuple[Dataset, Dataset, Dataset]
        A tuple containing the processed dataset objects.
    """

    pdmc_ids = list(set(D_p.cell_ids))

    # 1. normalize the gene expression
    enc: PandasEncoder = D_p.cell_encoders["exp"]
    X = enc.data.loc[pdmc_ids]

    ss = StandardScaler().fit(X)
    X[:] = ss.transform(X)
    enc.data = X

    if cfg.experiment.reduce_tumor_features:
        # use PCA to reduce the dimensionality of the tumor features
        enc.data = _decomp_tumor_features(
            enc.data, n_features=cfg.experiment.num_tumor_features
        )

    if cfg.experiment.reduce_drug_features:
        # use PCA to reduce the dimensionality of the drug features
        D_p.drug_encoders["mol"].data = _decomp_drug_features(
            D_p.drug_encoders["mol"].data, n_features=cfg.experiment.num_drug_features
        )

    # FIXME: this could be done external to this function
    D_p = apply_pdmc_drug_filters(cfg, D_p, D_c)

    # normalize the drug responses
    D_p, *_ = normalize_responses(D_p, norm_method="grouped")

    return D_p


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
    X_decomp = pca.fit_transform(X)
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
    cfg: DictConfig,
    D_cell: Dataset,
    D_pdxo: Dataset,
    D_pdx: Dataset,
    all_drug_ids: t.List[str] | None = None,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluates a baseline."""
    if cfg.baseline.estimator not in BASELINES:
        raise KeyError(f"Invalid baseline (got '{cfg.baseline.estimator}')")

    if all_drug_ids is None:
        all_drug_ids = _get_drug_ids(D_pdxo, D_pdx, D_cell)

    # estimator = BASELINES[cfg.baseline.estimator](random_state=cfg.baseline.seed)
    estimator = BASELINES[cfg.baseline.estimator]()
    param_grid = BASELINE_PARAM_GRIDS[cfg.baseline.estimator]

    # 1. fit the baseline model
    reg = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=GroupKFold(n_splits=5),
        n_jobs=cfg.baseline.n_jobs,
        refit=True,
        verbose=3,
    )

    D_cell.shuffle(seed=cfg.baseline.seed)
    X_cell, y_cell = _prepare_data_for_sklearn_model(D_cell)

    _ = reg.fit(X_cell, y_cell, groups=D_cell.cell_ids)
    print(reg.best_params_)

    # generate predictions for all tumor/drug combinations
    D_pdxo_full = data_utils.expand_dataset(
        D_pdxo,
        cell_ids=list(set(D_pdxo.cell_ids)),
        drug_ids=all_drug_ids,
    )

    # generate predictions for all tumor/drug combinations
    D_pdx_full = data_utils.expand_dataset(
        D_pdx,
        cell_ids=list(set(D_pdx.cell_ids)),
        drug_ids=all_drug_ids,
    )

    X_pdxo_full, _ = _prepare_data_for_sklearn_model(D_pdxo_full)
    X_pdx_full, _ = _prepare_data_for_sklearn_model(D_pdx_full)

    preds_pdxo = _preds_vs_background(
        reg, D_t=D_pdxo_full, D_e=D_pdxo_full, X_t=X_pdxo_full, X_e=X_pdxo_full
    )
    preds_pdxo["model"] = cfg.baseline.estimator

    preds_pdx = _preds_vs_background(
        reg, D_t=D_pdxo_full, D_e=D_pdx_full, X_t=X_pdxo_full, X_e=X_pdx_full
    )
    preds_pdx["model"] = cfg.baseline.estimator

    return preds_pdxo, preds_pdx


def train_baseline_0(
    cfg: DictConfig,
    D_cell: Dataset,
    D_pdxo: Dataset,
    D_pdx: Dataset,
    all_drug_ids: t.List[str] | None = None,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Trains and evaluates baseline from scratch using LOO cross validation."""
    if cfg.baseline.estimator not in BASELINES:
        raise KeyError(f"Invalid baseline (got '{cfg.baseline.estimator}')")

    if all_drug_ids is None:
        all_drug_ids = _get_drug_ids(D_pdxo, D_pdx, D_cell)

    results_pdxo = []
    results_pdx = []
    for i, (D_t_pdxo, D_e_pdxo) in enumerate(loo_split_generator(D_pdxo)):
        log.info(f"Running for {D_e_pdxo.cell_ids[0]} ({i+1:3d}/{D_pdxo.n_cells:3d})")
        estimator = BASELINES[cfg.baseline.estimator](random_state=cfg.baseline.seed)
        param_grid = BASELINE_PARAM_GRIDS[cfg.baseline.estimator]

        D_e_pdx = D_pdx.select_cells(set(D_e_pdxo.cell_ids))
        D_t_pdxo.shuffle(seed=cfg.baseline.seed)

        # generate predictions for all tumor/drug combinations
        D_e_pdxo_full = data_utils.expand_dataset(
            D_e_pdxo,
            cell_ids=list(set(D_e_pdxo.cell_ids)),
            drug_ids=all_drug_ids,
        )
        D_t_pdxo_full = data_utils.expand_dataset(
            # full dataset for background distribution
            D_t_pdxo,
            cell_ids=list(set(D_t_pdxo.cell_ids)),
            drug_ids=all_drug_ids,
        )

        X_t_pdxo, y_t_pdxo = _prepare_data_for_sklearn_model(D_t_pdxo)
        X_e_pdxo_full, _ = _prepare_data_for_sklearn_model(D_e_pdxo_full)

        # we use k-fold grouped by tumor ID so we are doing HPO for tumor-blind
        reg = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=GroupKFold(n_splits=5),
            n_jobs=cfg.baseline.n_jobs,
            refit=True,
        )
        _ = reg.fit(X_t_pdxo, y_t_pdxo, groups=D_t_pdxo.cell_ids)
        print(reg.best_params_)

        X_t_pdxo_full, _ = _prepare_data_for_sklearn_model(D_t_pdxo_full)
        preds_e_pdxo = _preds_vs_background(
            reg, D_t_pdxo_full, D_e_pdxo_full, X_t_pdxo_full, X_e_pdxo_full
        )
        results_pdxo.append(preds_e_pdxo)

        if D_e_pdx.n_cells > 0:
            # generate predictions for all tumor/drug combinations
            D_e_pdx_full = data_utils.expand_dataset(
                D_e_pdx,
                cell_ids=list(set(D_e_pdx.cell_ids)),
                drug_ids=all_drug_ids,
            )

            X_e_pdx_full, _ = _prepare_data_for_sklearn_model(D_e_pdx_full)
            preds_e_pdx = _preds_vs_background(
                reg, D_t_pdxo_full, D_e_pdx_full, X_t_pdxo_full, X_e_pdx_full
            )
            results_pdx.append(preds_e_pdx)

    results_pdxo = (
        pd.concat(results_pdxo)
        .reset_index(drop=True)
        .assign(model=f"{cfg.baseline.estimator}-0")
    )
    results_pdx = (
        pd.concat(results_pdx)
        .reset_index(drop=True)
        .assign(model=f"{cfg.baseline.estimator}-0")
    )

    return results_pdxo, results_pdx


def load_pdx_data(cfg: DictConfig, D_p: Dataset) -> Dataset:
    """Loads the raw PDX screening data."""
    obs = pd.read_csv(cfg.experiment.pdx_obs_path)
    obs = obs[obs["cell_id"].isin(D_p.cell_ids)]
    obs = obs[obs["drug_id"].isin(D_p.drug_ids)]
    obs["label"] = obs["mRECIST"].isin(["CR", "PR", "SD"]).astype(int)

    D_pdx = Dataset(
        obs,
        cell_encoders=D_p.cell_encoders,
        drug_encoders=D_p.drug_encoders,
        name="pdx_ds",
    )

    if cfg.experiment.pdx_ids is not None:
        D_pdx = D_pdx.select_cells(cfg.experiment.pdx_ids)

    return D_pdx


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="pdx_baselines",
)
def run(cfg: DictConfig) -> None:
    """Runs leave-one-out cross validation for the HCI PDMC dataset.

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
    cell_ids = D.cell_meta[D.cell_meta["domain"] == "CELL"].index
    pdxo_ids = D.cell_meta[D.cell_meta["domain"] == "PDMC"].index

    D_cell = D.select_cells(cell_ids, name="cell_ds")
    D_pdxo = D.select_cells(pdxo_ids, name="pdmc_ds")
    D_pdxo_0 = copy_dataset(D_pdxo)

    log.info(f"Preprocessing {dataset_name}...")
    D_cell, D_pdxo = data_preprocessor(cfg, D_cell, D_pdxo)
    D_pdxo_0 = data_preprocessor_0(cfg, D_cell, D_pdxo_0)

    D_pdx = load_pdx_data(cfg, D_pdxo)
    D_pdx_0 = load_pdx_data(cfg, D_pdxo_0)

    log.info(f"Fitting baseline...")
    preds_pdxo, preds_pdx = train_baseline(
        cfg, D_cell=D_cell, D_pdxo=D_pdxo, D_pdx=D_pdx, all_drug_ids=all_drug_ids
    )

    log.info(f"Fitting baseline (0)...")
    preds_pdxo_0, preds_pdx_0 = train_baseline_0(
        cfg, D_cell=D_cell, D_pdxo=D_pdxo_0, D_pdx=D_pdx_0, all_drug_ids=all_drug_ids
    )

    pdxo_results_df = pd.concat([preds_pdxo, preds_pdxo_0])
    pdxo_results_df.to_csv("predictions.pdxo.csv", index=False)

    pdx_results_df = pd.concat([preds_pdx, preds_pdx_0])
    pdx_results_df.to_csv("predictions.pdx.csv", index=False)


if __name__ == "__main__":
    run()
