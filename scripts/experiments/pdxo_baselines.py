#!/usr/bin/env python
"""Trains and evaluates baseline models for breast cancer PDXOs."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import gc
import hydra
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# from joblib import parallel_backend
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.decomposition import PCA

from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import PandasEncoder
from cdrpy.data.preprocess import normalize_responses, GroupStandardScaler

from screendl.data import filter_drugs, normalize_feat_dfs
from screendl.pipelines.basic.screendl import data_loader
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
        "alpha": [10, 100, 1000, 10000],
    },
    "forest": {
        "max_depth": [20],
        "max_features": ["sqrt"],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2],
        "n_estimators": [200],
    },
}


@dataclass
class BaselineSearchResult:
    """Fitted estimator plus CV search metadata.

    This intentionally mimics the small subset of GridSearchCV used downstream:
    `.predict(...)` and `.best_params_`.
    """

    estimator: Ridge | RandomForestRegressor
    best_params_: dict[str, t.Any]
    best_score_: float
    cv_results_: pd.DataFrame

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the refit best estimator."""
        return self.estimator.predict(X)


@dataclass
class ExperimentData:
    """Prepared datasets for one baseline run."""

    dataset: Dataset | None
    D_cell: Dataset | None
    D_pdxo: Dataset | None
    D_pdxo_0: Dataset | None
    all_drug_ids: list[str] | None

    def require_ready(self) -> None:
        """Validate that prepared run-level datasets are available."""
        if self.dataset is None:
            raise RuntimeError("Full dataset has been cleared.")
        if self.D_cell is None:
            raise RuntimeError("Cell-line dataset has been cleared.")
        if self.D_pdxo is None:
            raise RuntimeError("PDXO dataset has been cleared.")
        if self.D_pdxo_0 is None:
            raise RuntimeError("PDXO-0 dataset has been cleared.")
        if self.all_drug_ids is None:
            raise RuntimeError("Drug IDs have been cleared.")

    def clear(self) -> None:
        """Drop run-level references."""
        self.dataset = None
        self.D_cell = None
        self.D_pdxo = None
        self.D_pdxo_0 = None
        self.all_drug_ids = None
        gc.collect()


@dataclass
class FoldContext:
    """Mutable per-fold state for LOO baseline-0 evaluation."""

    D_t_pdxo: Dataset | None = None
    D_e_pdxo: Dataset | None = None

    D_t_pdxo_full: Dataset | None = None
    D_e_pdxo_full: Dataset | None = None

    X_t_pdxo: np.ndarray | None = None
    y_t_pdxo: np.ndarray | None = None
    X_t_pdxo_full: np.ndarray | None = None
    X_e_pdxo_full: np.ndarray | None = None

    preds_pdxo: pd.DataFrame | None = None

    def clear(self) -> None:
        """Drop per-fold references."""
        self.D_t_pdxo = None
        self.D_e_pdxo = None

        self.D_t_pdxo_full = None
        self.D_e_pdxo_full = None

        self.X_t_pdxo = None
        self.y_t_pdxo = None
        self.X_t_pdxo_full = None
        self.X_e_pdxo_full = None

        self.preds_pdxo = None

        gc.collect()


def get_id_prefix(pdmc_id: str) -> str:
    """Get matching prefixes to avoid data leakage across related models."""
    if pdmc_id.startswith("HCI"):
        return pdmc_id[:6]
    if pdmc_id.startswith("BCM"):
        return pdmc_id[:7]
    if pdmc_id.startswith("TOW"):
        return pdmc_id[:5]
    return pdmc_id


def loo_split_generator(
    D: Dataset,
) -> t.Generator[tuple[Dataset, Dataset], None, None]:
    """Generate leave-one-PDMC-out splits."""
    pdmc_ids = set(D.cell_ids)

    for test_id in sorted(pdmc_ids):
        id_prefix = get_id_prefix(str(test_id))
        train_ids = [x for x in pdmc_ids if not str(x).startswith(id_prefix)]

        train_ds = D.select_cells(train_ids, name="train")
        test_ds = D.select_cells([test_id], name="test")

        yield train_ds, test_ds


def copy_dataset(D: Dataset) -> Dataset:
    """Create a deep copy of a Dataset."""
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


def _decomp_tumor_features(
    X: pd.DataFrame,
    n_features: int | None = None,
) -> pd.DataFrame:
    """Reduce tumor feature dimensionality using whitened PCA."""
    pca = PCA(n_components=n_features, random_state=42)
    X_decomp = pca.fit_transform(X)
    X_decomp = X_decomp / np.sqrt(pca.explained_variance_)

    return pd.DataFrame(
        X_decomp,
        index=X.index,
        columns=[f"PC_T{i}" for i in range(X_decomp.shape[1])],
    )


def _decomp_drug_features(
    X: pd.DataFrame,
    n_features: int | None = None,
) -> pd.DataFrame:
    """Reduce drug feature dimensionality using whitened PCA."""
    pca = PCA(n_components=n_features, random_state=42)
    X_decomp = pca.fit_transform(X.transform(stats.zscore).dropna(axis=1))
    X_decomp = X_decomp / np.sqrt(pca.explained_variance_)

    return pd.DataFrame(
        X_decomp,
        index=X.index,
        columns=[f"PC_D{i}" for i in range(X_decomp.shape[1])],
    )


def data_preprocessor(
    cfg: "DictConfig",
    D_c: Dataset,
    D_p: Dataset,
) -> tuple[Dataset, Dataset]:
    """Preprocess cell-line and PDXO datasets."""
    cell_ids = sorted(set(D_c.cell_ids))
    pdxo_ids = sorted(set(D_p.cell_ids))

    exp_enc: PandasEncoder = D_c.cell_encoders["exp"]

    x_cell = exp_enc.data.loc[cell_ids].copy()
    x_pdxo = exp_enc.data.loc[pdxo_ids].copy()

    x_cell, x_pdxo, _, _ = normalize_feat_dfs(
        x_cell,
        x_pdxo,
        norm_method=cfg.dataset.preprocess.norm_exp,
    )

    assert x_pdxo is not None
    exp_enc.data = pd.concat([x_cell, x_pdxo], axis=0).copy()

    if cfg.experiment.reduce_tumor_features:
        exp_enc.data = _decomp_tumor_features(
            exp_enc.data,
            n_features=cfg.experiment.num_tumor_features,
        )

    if cfg.experiment.reduce_drug_features:
        D_p.drug_encoders["mol"].data = _decomp_drug_features(
            D_p.drug_encoders["mol"].data,
            n_features=cfg.experiment.num_drug_features,
        )

    D_p = filter_drugs(
        D_p,
        reference_datasets=[D_c],
        include_dataset_drugs=cfg.experiment.keep_pdmc_only_drugs,
        min_cells_per_drug=cfg.experiment.min_pdmcs_per_drug,
        drug_col="drug_id",
        name=D_p.name,
    )

    D_c, *_ = normalize_responses(D_c, norm_method="grouped")
    D_p, *_ = normalize_responses(D_p, norm_method="grouped")

    return D_c, D_p


def data_preprocessor_0(
    cfg: "DictConfig",
    D_c: Dataset,
    D_p: Dataset,
) -> Dataset:
    """Preprocess the PDXO-only baseline-0 dataset."""
    pdxo_ids = sorted(set(D_p.cell_ids))

    exp_enc: PandasEncoder = D_p.cell_encoders["exp"]
    x_pdxo = exp_enc.data.loc[pdxo_ids].copy()

    x_pdxo, _, _, _ = normalize_feat_dfs(
        x_pdxo,
        norm_method=cfg.dataset.preprocess.norm_exp,
    )

    exp_enc.data = x_pdxo.copy()

    if cfg.experiment.reduce_tumor_features:
        exp_enc.data = _decomp_tumor_features(
            exp_enc.data,
            n_features=cfg.experiment.num_tumor_features,
        )

    if cfg.experiment.reduce_drug_features:
        D_p.drug_encoders["mol"].data = _decomp_drug_features(
            D_p.drug_encoders["mol"].data,
            n_features=cfg.experiment.num_drug_features,
        )

    D_p = filter_drugs(
        D_p,
        reference_datasets=[D_c],
        include_dataset_drugs=cfg.experiment.keep_pdmc_only_drugs,
        min_cells_per_drug=cfg.experiment.min_pdmcs_per_drug,
        drug_col="drug_id",
        name=D_p.name,
    )

    D_p, *_ = normalize_responses(D_p, norm_method="grouped")

    return D_p


def _prepare_data_for_sklearn_model(D: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a Dataset into sklearn feature and target arrays."""
    obs = D.obs

    x_cell = D.cell_encoders["exp"].data.loc[obs["cell_id"]]
    x_drug = D.drug_encoders["mol"].data.loc[obs["drug_id"]]

    x_cell_arr = np.asarray(x_cell, dtype=np.float32)
    x_drug_arr = np.asarray(x_drug, dtype=np.float32)

    X = np.empty(
        (obs.shape[0], x_cell_arr.shape[1] + x_drug_arr.shape[1]),
        dtype=np.float32,
    )
    X[:, : x_cell_arr.shape[1]] = x_cell_arr
    X[:, x_cell_arr.shape[1] :] = x_drug_arr

    y = obs["label"].to_numpy(dtype=np.float32, copy=True)

    return X, y


def _preds_vs_background(
    estimator: BaselineSearchResult,
    D_t: Dataset,
    D_e: Dataset,
    X_t: np.ndarray,
    X_e: np.ndarray,
) -> pd.DataFrame:
    """Predict z-scores using the training set as a background distribution."""
    preds_t = (
        D_t.obs.rename(columns={"label": "y_true"})
        .drop(columns="id", errors="ignore")
        .copy()
    )
    preds_t["y_pred"] = estimator.predict(X_t)

    preds_e = (
        D_e.obs.rename(columns={"label": "y_true"})
        .drop(columns="id", errors="ignore")
        .copy()
    )
    preds_e["y_pred"] = estimator.predict(X_e)

    gss = GroupStandardScaler().fit(
        preds_t[["y_pred"]],
        groups=preds_t["drug_id"],
    )
    preds_e["y_pred"] = gss.transform(
        preds_e[["y_pred"]],
        groups=preds_e["drug_id"],
    )

    return preds_e


def _get_drug_ids(*datasets: Dataset) -> list[str]:
    """Get all drug IDs from the provided datasets."""
    all_drug_ids = set()

    for D in datasets:
        if D.drug_ids is not None:
            all_drug_ids.update(D.drug_ids)

    return sorted(all_drug_ids)


def _make_estimator(cfg: "DictConfig") -> Ridge | RandomForestRegressor:
    """Instantiate the configured sklearn estimator."""
    if cfg.baseline.estimator not in BASELINES:
        raise KeyError(f"Invalid baseline: {cfg.baseline.estimator}")

    estimator_cls = BASELINES[cfg.baseline.estimator]

    if cfg.baseline.estimator == "forest":
        return estimator_cls(
            random_state=cfg.baseline.seed,
            n_jobs=int(cfg.baseline.n_jobs),
        )

    return estimator_cls(random_state=cfg.baseline.seed)


def _make_cv(groups: t.Sequence[t.Any], max_splits: int = 5) -> GroupKFold:
    """Create a GroupKFold that does not exceed the available group count."""
    n_groups = len(set(groups))
    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups for GroupKFold; got {n_groups}.")

    return GroupKFold(n_splits=min(max_splits, n_groups))


def _fit_grid_search(
    cfg: "DictConfig",
    estimator: Ridge | RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
    groups: t.Sequence[t.Any],
) -> BaselineSearchResult:
    """Fit baseline hyperparameter search with tumor-blind CV.

    Manual implementation of the GridSearchCV behavior used here:
    - same ParameterGrid expansion
    - same GroupKFold splits
    - same default regressor scoring behavior, R^2
    - same refit=True behavior on the full training set
    - no joblib/loky/GridSearchCV execution path
    """
    param_grid = BASELINE_PARAM_GRIDS[cfg.baseline.estimator]
    cv = _make_cv(groups, max_splits=5)
    groups_arr = np.asarray(groups)

    log.info(
        "Starting manual CV search | estimator=%s | X=%s %s | y=%s %s | "
        "groups=%d | estimator_n_jobs=%s",
        cfg.baseline.estimator,
        X.shape,
        X.dtype,
        y.shape,
        y.dtype,
        len(set(groups)),
        getattr(estimator, "n_jobs", None),
    )

    best_params: dict[str, t.Any] | None = None
    best_score = -np.inf
    rows: list[dict[str, t.Any]] = []

    candidates = list(ParameterGrid(param_grid))
    n_splits = cv.get_n_splits(X, y, groups_arr)
    n_fits = len(candidates) * n_splits

    log.info(
        "Manual CV search will fit %d candidates x %d folds = %d fits",
        len(candidates),
        n_splits,
        n_fits,
    )

    fit_i = 0
    for cand_i, params in enumerate(candidates, start=1):
        fold_scores: list[float] = []

        for fold_i, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups=groups_arr),
            start=1,
        ):
            fit_i += 1

            model = clone(estimator)
            model.set_params(**params)

            log.info(
                "Fitting CV model %d/%d | candidate=%d/%d | fold=%d/%d | params=%s",
                fit_i,
                n_fits,
                cand_i,
                len(candidates),
                fold_i,
                n_splits,
                params,
            )

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = float(r2_score(y_val, y_pred))
            finally:
                del X_train
                del y_train
                del X_val
                del y_val
                gc.collect()

            fold_scores.append(score)

            log.info(
                "Finished CV model %d/%d | candidate=%d/%d | fold=%d/%d | "
                "score=%.6f",
                fit_i,
                n_fits,
                cand_i,
                len(candidates),
                fold_i,
                n_splits,
                score,
            )

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        row = {
            **params,
            "mean_test_score": mean_score,
            "std_test_score": std_score,
        }
        for i, score in enumerate(fold_scores):
            row[f"split{i}_test_score"] = score
        rows.append(row)

        log.info(
            "Finished candidate %d/%d | mean_score=%.6f | std_score=%.6f | params=%s",
            cand_i,
            len(candidates),
            mean_score,
            std_score,
            params,
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("Manual CV search did not evaluate any candidates.")

    log.info(
        "Refitting best baseline on full training data | best_score=%.6f | params=%s",
        best_score,
        best_params,
    )

    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    best_estimator.fit(X, y)

    log.info("Finished refitting best baseline")

    return BaselineSearchResult(
        estimator=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
        cv_results_=pd.DataFrame(rows),
    )


def train_baseline(
    cfg: "DictConfig",
    D_cell: Dataset,
    D_pdxo: Dataset,
    all_drug_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Train the cell-line baseline and evaluate PDXO predictions."""
    if all_drug_ids is None:
        all_drug_ids = _get_drug_ids(D_pdxo, D_cell)

    estimator = _make_estimator(cfg)

    D_cell.shuffle(seed=cfg.baseline.seed)
    X_cell, y_cell = _prepare_data_for_sklearn_model(D_cell)

    reg = _fit_grid_search(
        cfg=cfg,
        estimator=estimator,
        X=X_cell,
        y=y_cell,
        groups=D_cell.cell_ids,
    )

    D_pdxo_full = data_utils.expand_dataset(
        D_pdxo,
        cell_ids=list(set(D_pdxo.cell_ids)),
        drug_ids=all_drug_ids,
    )

    X_pdxo_full, _ = _prepare_data_for_sklearn_model(D_pdxo_full)

    preds = _preds_vs_background(
        reg,
        D_t=D_pdxo_full,
        D_e=D_pdxo_full,
        X_t=X_pdxo_full,
        X_e=X_pdxo_full,
    )
    preds["model"] = cfg.baseline.estimator

    return preds


def run_baseline_0_fold(
    cfg: "DictConfig",
    D_t_pdxo: Dataset,
    D_e_pdxo: Dataset,
    all_drug_ids: list[str],
) -> pd.DataFrame:
    """Train and evaluate one leave-one-PDMC-out baseline-0 fold."""
    ctx = FoldContext()

    try:
        ctx.D_t_pdxo = D_t_pdxo
        ctx.D_e_pdxo = D_e_pdxo

        estimator = _make_estimator(cfg)

        ctx.D_t_pdxo.shuffle(seed=cfg.baseline.seed)

        ctx.X_t_pdxo, ctx.y_t_pdxo = _prepare_data_for_sklearn_model(ctx.D_t_pdxo)

        reg = _fit_grid_search(
            cfg=cfg,
            estimator=estimator,
            X=ctx.X_t_pdxo,
            y=ctx.y_t_pdxo,
            groups=ctx.D_t_pdxo.cell_ids,
        )

        ctx.D_e_pdxo_full = data_utils.expand_dataset(
            ctx.D_e_pdxo,
            cell_ids=list(set(ctx.D_e_pdxo.cell_ids)),
            drug_ids=all_drug_ids,
        )
        ctx.D_t_pdxo_full = data_utils.expand_dataset(
            ctx.D_t_pdxo,
            cell_ids=list(set(ctx.D_t_pdxo.cell_ids)),
            drug_ids=all_drug_ids,
        )

        ctx.X_e_pdxo_full, _ = _prepare_data_for_sklearn_model(ctx.D_e_pdxo_full)
        ctx.X_t_pdxo_full, _ = _prepare_data_for_sklearn_model(ctx.D_t_pdxo_full)

        ctx.preds_pdxo = _preds_vs_background(
            reg,
            D_t=ctx.D_t_pdxo_full,
            D_e=ctx.D_e_pdxo_full,
            X_t=ctx.X_t_pdxo_full,
            X_e=ctx.X_e_pdxo_full,
        )

        return ctx.preds_pdxo

    finally:
        ctx.clear()


def train_baseline_0(
    cfg: "DictConfig",
    D_cell: Dataset,
    D_pdxo: Dataset,
    all_drug_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Train and evaluate baseline-0 using LOO cross validation."""
    if cfg.baseline.estimator not in BASELINES:
        raise KeyError(f"Invalid baseline: {cfg.baseline.estimator}")

    if all_drug_ids is None:
        all_drug_ids = _get_drug_ids(D_pdxo, D_cell)

    results = []

    n_folds = len(set(D_pdxo.cell_ids))
    split_gen = loo_split_generator(D_pdxo)

    for i, (D_t_pdxo, D_e_pdxo) in enumerate(split_gen):
        heldout_id = D_e_pdxo.cell_ids[0]
        log.info(
            "Running baseline-0 for %s (%3d/%3d)",
            heldout_id,
            i + 1,
            n_folds,
        )

        try:
            preds_e = run_baseline_0_fold(
                cfg=cfg,
                D_t_pdxo=D_t_pdxo,
                D_e_pdxo=D_e_pdxo,
                all_drug_ids=all_drug_ids,
            )

        except Exception:
            log.exception("Failed baseline-0 fold for heldout_id=%s", heldout_id)
            continue

        results.append(preds_e)

    if not results:
        raise RuntimeError("No PDXO baseline-0 predictions were generated.")

    return (
        pd.concat(results)
        .reset_index(drop=True)
        .assign(model=f"{cfg.baseline.estimator}-0")
    )


def prepare_experiment(cfg: "DictConfig") -> ExperimentData:
    """Load, split, and preprocess the PDXO baseline datasets."""
    dataset_name = cfg.dataset.name

    log.info("Loading %s...", dataset_name)
    D = data_loader(cfg)
    all_drug_ids = list(D.drug_encoders["mol"].keys())

    log.info("Splitting %s...", dataset_name)
    cell_ids = D.cell_meta[D.cell_meta["domain"] == "CELL"].index
    pdxo_ids = D.cell_meta[D.cell_meta["domain"] == "PDMC"].index

    D_cell = D.select_cells(cell_ids, name="cell_ds")
    D_pdxo = D.select_cells(pdxo_ids, name="pdmc_ds")
    D_pdxo_0 = copy_dataset(D_pdxo)

    log.info("Preprocessing %s...", dataset_name)
    D_cell, D_pdxo = data_preprocessor(cfg, D_cell, D_pdxo)
    D_pdxo_0 = data_preprocessor_0(cfg, D_cell, D_pdxo_0)

    return ExperimentData(
        dataset=D,
        D_cell=D_cell,
        D_pdxo=D_pdxo,
        D_pdxo_0=D_pdxo_0,
        all_drug_ids=all_drug_ids,
    )


def run_cell_line_baseline(
    cfg: "DictConfig",
    exp: ExperimentData,
    output_path: Path,
) -> None:
    """Train the cell-line baseline and write PDXO predictions."""
    exp.require_ready()

    assert exp.D_cell is not None
    assert exp.D_pdxo is not None
    assert exp.all_drug_ids is not None

    log.info("Fitting baseline...")
    preds = train_baseline(
        cfg,
        D_cell=exp.D_cell,
        D_pdxo=exp.D_pdxo,
        all_drug_ids=exp.all_drug_ids,
    )

    preds.to_csv(
        output_path,
        index=False,
        mode="a",
        header=not output_path.exists(),
    )


def run_pdxo_only_baseline(
    cfg: "DictConfig",
    exp: ExperimentData,
    output_path: Path,
) -> None:
    """Train the PDXO-only LOO baseline and write PDXO predictions."""
    exp.require_ready()

    assert exp.D_cell is not None
    assert exp.D_pdxo_0 is not None
    assert exp.all_drug_ids is not None

    log.info("Fitting baseline (0)...")
    preds_0 = train_baseline_0(
        cfg,
        D_cell=exp.D_cell,
        D_pdxo=exp.D_pdxo_0,
        all_drug_ids=exp.all_drug_ids,
    )

    preds_0.to_csv(
        output_path,
        index=False,
        mode="a",
        header=not output_path.exists(),
    )


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="pdxo_baselines",
)
def run(cfg: "DictConfig") -> None:
    """Run PDXO baseline analyses."""
    assert cfg.dataset.preprocess.norm == "grouped"

    output_path = Path("predictions.csv")
    if output_path.exists():
        output_path.unlink()

    exp = prepare_experiment(cfg)

    try:
        run_cell_line_baseline(
            cfg=cfg,
            exp=exp,
            output_path=output_path,
        )

        run_pdxo_only_baseline(
            cfg=cfg,
            exp=exp,
            output_path=output_path,
        )

    finally:
        exp.clear()


if __name__ == "__main__":
    run()
