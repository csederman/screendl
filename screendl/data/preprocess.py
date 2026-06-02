"""Data preprocessing utils."""

from __future__ import annotations

import pickle
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import ListConfig
from scipy import stats
from sklearn.preprocessing import StandardScaler

from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import PandasEncoder
from cdrpy.data.preprocess import normalize_responses as _normalize_responses

if t.TYPE_CHECKING:
    from cdrpy.datasets import Dataset


NormMethod = t.Literal["between", "within"]


def _maybe_transform_df(
    df: pd.DataFrame | None,
    transform: t.Callable[[pd.DataFrame], np.ndarray],
) -> pd.DataFrame | None:
    """Apply a transform to a DataFrame while preserving index/columns."""
    if df is None:
        return None

    out = df.copy()
    out.loc[:, :] = transform(out)
    return out


def normalize_feat_dfs(
    t_df: pd.DataFrame,
    v_df: pd.DataFrame | None = None,
    e_df: pd.DataFrame | None = None,
    norm_method: NormMethod = "between",
) -> tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, StandardScaler | None
]:
    """Normalize feature DataFrames.

    Parameters
    ----------
    t_df
        Training feature matrix. Used to fit the scaler for between-sample normalization.
    v_df
        Optional validation feature matrix.
    e_df
        Optional external/test feature matrix.
    norm_method
        "between": fit StandardScaler on training rows and apply to all sets.
        "within": z-score each row independently.

    Returns
    -------
    tuple
        Normalized train, validation, test DataFrames and fitted normalizer
    """
    t_df = t_df.copy()
    v_df = None if v_df is None else v_df.copy()
    e_df = None if e_df is None else e_df.copy()
    scaler = None

    if norm_method == "between":
        scaler = StandardScaler().fit(t_df.to_numpy())

        def transform_between(df: pd.DataFrame) -> np.ndarray:
            return scaler.transform(df.to_numpy())

        t_df.loc[:, :] = transform_between(t_df)
        v_df = _maybe_transform_df(v_df, transform_between)
        e_df = _maybe_transform_df(e_df, transform_between)

    elif norm_method == "within":

        def transform_within(df: pd.DataFrame) -> np.ndarray:
            return stats.zscore(df.to_numpy(), axis=1, ddof=0)

        t_df.loc[:, :] = transform_within(t_df)
        v_df = _maybe_transform_df(v_df, transform_within)
        e_df = _maybe_transform_df(e_df, transform_within)

    else:
        raise ValueError(
            f"Invalid feature normalization method: {norm_method!r}. "
            "Expected one of {'between', 'within'}."
        )

    return t_df, v_df, e_df, scaler


DrugId = t.Any
DrugSetLike = t.Iterable[DrugId]


def _as_dataset_iterable(
    datasets: Dataset | t.Iterable[Dataset] | None,
) -> t.Iterable["Dataset"]:
    """Normalize optional single/multiple datasets to an iterable."""
    if datasets is None:
        return ()

    # Avoid importing Dataset at runtime if this lives in a utility module.
    if hasattr(datasets, "drug_ids") and hasattr(datasets, "obs"):
        return (datasets,)

    return datasets


def _collect_drugs_from_datasets(
    datasets: Dataset | t.Iterable[Dataset] | None,
) -> set[DrugId]:
    """Collect union of drug IDs from one or more datasets."""
    keep_drugs: set[DrugId] = set()

    for ds in _as_dataset_iterable(datasets):
        keep_drugs.update(ds.drug_ids)

    return keep_drugs


def _normalize_extra_drugs(
    extra_drugs: str | DrugSetLike | None,
    *,
    dataset_drug_ids: DrugSetLike,
    param_name: str = "extra_drugs",
) -> set[DrugId]:
    """Normalize extra drug inclusion parameter.

    Supported:
    - None: add no extra drugs
    - "all": add all drugs from the target dataset
    - iterable/list/ListConfig/set/tuple: add those explicit drug IDs
    """
    if extra_drugs is None:
        return set()

    if extra_drugs == "all":
        return set(dataset_drug_ids)

    if isinstance(extra_drugs, (list, tuple, set, ListConfig)):
        return set(extra_drugs)

    raise TypeError(
        f"Unsupported parameter type for {param_name}: {type(extra_drugs)!r}. "
        "Expected None, 'all', or an iterable of drug IDs."
    )


def filter_drugs(
    dataset: "Dataset",
    *,
    reference_datasets: Dataset | t.Iterable[Dataset] | None = None,
    include_dataset_drugs: str | DrugSetLike | None = None,
    min_cells_per_drug: int | None = None,
    drug_col: str = "drug_id",
    name: str | None = None,
) -> "Dataset":
    """Filter a dataset to a configurable drug set.

    Parameters
    ----------
    dataset
        Dataset to filter.
    reference_datasets
        Optional dataset or datasets whose drugs define the initial keep set.
        For the old PDXO use case, this is the train/val cell-line dataset(s).
    include_dataset_drugs
        Additional drugs to include from `dataset`.
        - None: no extra drugs
        - "all": include all drugs present in `dataset`
        - iterable: include those explicit drug IDs
    min_cells_per_drug
        If provided, keep only drugs with at least this many rows in
        `dataset.obs[drug_col]`.
    drug_col
        Drug ID column in `dataset.obs`.
    name
        Optional name for the returned selected dataset. Defaults to dataset.name.

    Returns
    -------
    Dataset
        Filtered dataset.
    """
    keep_drugs = _collect_drugs_from_datasets(reference_datasets)

    keep_drugs.update(
        _normalize_extra_drugs(
            include_dataset_drugs,
            dataset_drug_ids=dataset.drug_ids,
            param_name="include_dataset_drugs",
        )
    )

    # If no reference/extra drugs were provided, start from all dataset drugs.
    if not keep_drugs:
        keep_drugs = set(dataset.drug_ids)

    if min_cells_per_drug is not None:
        if drug_col not in dataset.obs.columns:
            raise ValueError(f"dataset.obs is missing drug column {drug_col!r}.")

        drugs_with_min_obs = (
            dataset.obs[drug_col]
            .value_counts()
            .loc[lambda s: s >= min_cells_per_drug]
            .index
        )
        keep_drugs = keep_drugs.intersection(drugs_with_min_obs)

    return dataset.select_drugs(keep_drugs, name=dataset.name if name is None else name)


SklearnObject = t.Any


@dataclass
class PreprocessingArtifacts:
    """Fitted preprocessing objects."""

    feature_scalers: dict[str, SklearnObject | None]
    response_scalers: dict[str, SklearnObject | None]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as fh:
            pickle.dump(self, fh)


def normalize_response_datasets(
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    test_ds: Dataset | None = None,
    *,
    norm_method: str = "global",
) -> tuple[Dataset, Dataset | None, Dataset | None, SklearnObject | None]:
    """Normalize response labels and return the fitted normalizer."""
    if norm_method == "global":
        scaler = StandardScaler()
        train_ds.obs["label"] = scaler.fit_transform(train_ds.obs[["label"]])

        if val_ds is not None:
            val_ds.obs["label"] = scaler.transform(val_ds.obs[["label"]])

        if test_ds is not None:
            test_ds.obs["label"] = scaler.transform(test_ds.obs[["label"]])

    elif norm_method == "grouped":
        from cdrpy.feat.transformers import GroupStandardScaler

        scaler = GroupStandardScaler()
        train_ds.obs["label"] = scaler.fit_transform(
            train_ds.obs[["label"]],
            groups=train_ds.obs["drug_id"],
        )

        if val_ds is not None:
            val_ds.obs["label"] = scaler.transform(
                val_ds.obs[["label"]],
                groups=val_ds.obs["drug_id"],
            )

        if test_ds is not None:
            test_ds.obs["label"] = scaler.transform(
                test_ds.obs[["label"]],
                groups=test_ds.obs["drug_id"],
            )

    else:
        raise ValueError("norm_method must be one of {'global', 'grouped'}.")

    return train_ds, val_ds, test_ds, scaler


def _unique_ids(ds: Dataset | None) -> list[str]:
    """Return sorted unique cell IDs from a dataset."""
    if ds is None:
        return []
    return sorted(set(ds.cell_ids))


def normalize_cell_feature(
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    test_ds: Dataset | None = None,
    *,
    feature_name: str,
    norm_method: str,
) -> SklearnObject | None:
    """Normalize one cell feature encoder using train cells as the fit set."""
    if feature_name not in train_ds.cell_encoders:
        return None

    enc: PandasEncoder = train_ds.cell_encoders[feature_name]

    train_ids = _unique_ids(train_ds)
    val_ids = _unique_ids(val_ds)
    test_ids = _unique_ids(test_ds)

    train_df = enc.data.loc[train_ids].copy()
    val_df = enc.data.loc[val_ids].copy() if val_ids else None
    test_df = enc.data.loc[test_ids].copy() if test_ids else None

    train_df, val_df, test_df, scaler = normalize_feat_dfs(
        train_df,
        val_df,
        test_df,
        norm_method=norm_method,
    )

    parts = [train_df]
    if val_df is not None:
        parts.append(val_df)
    if test_df is not None:
        parts.append(test_df)

    enc.data = pd.concat(parts, axis=0).copy()

    if val_ds is not None and feature_name in val_ds.cell_encoders:
        val_ds.cell_encoders[feature_name].data = enc.data

    if test_ds is not None and feature_name in test_ds.cell_encoders:
        test_ds.cell_encoders[feature_name].data = enc.data

    return scaler


def preprocess_screendl_datasets(
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    test_ds: Dataset | None = None,
    *,
    exp_norm_method: str = "between",
    response_norm_method: str = "global",
    normalize_cnv: bool = True,
    artifact_path: str | Path | None = "preprocessing_artifacts.pkl",
) -> tuple[Dataset, Dataset | None, Dataset | None, PreprocessingArtifacts]:
    """Normalize same-domain ScreenDL datasets and save fitted preprocessing objects."""
    feature_scalers: dict[str, SklearnObject | None] = {}

    feature_scalers["exp"] = normalize_cell_feature(
        train_ds,
        val_ds,
        test_ds,
        feature_name="exp",
        norm_method=exp_norm_method,
    )

    if normalize_cnv and "cnv" in train_ds.cell_encoders:
        feature_scalers["cnv"] = normalize_cell_feature(
            train_ds,
            val_ds,
            test_ds,
            feature_name="cnv",
            norm_method="between",
        )

    train_ds, val_ds, test_ds, response_scaler = normalize_response_datasets(
        train_ds,
        val_ds,
        test_ds,
        norm_method=response_norm_method,
    )

    artifacts = PreprocessingArtifacts(
        feature_scalers=feature_scalers,
        response_scalers={"response": response_scaler},
    )

    if artifact_path is not None:
        artifacts.save(artifact_path)

    return train_ds, val_ds, test_ds, artifacts


def preprocess_pdmc_screendl_datasets(
    train_cell_ds: Dataset,
    val_cell_ds: Dataset,
    pdmc_ds: Dataset,
    *,
    exp_norm_method: str = "between",
    keep_pdmc_only_drugs: t.Any = None,
    min_pdmcs_per_drug: int | None = None,
    normalize_cnv: bool = True,
    artifact_path: str | Path | None = "preprocessing_artifacts.pkl",
) -> tuple[Dataset, Dataset, Dataset, PreprocessingArtifacts]:
    """Normalize cell-line + PDMC/PDXO datasets.

    Preserves the original PDMC/PDXO behavior:
    - feature scalers fit on train_cell_ds cells
    - feature scalers applied to val_cell_ds and pdmc_ds cells
    - PDMC/PDXO drugs filtered after feature normalization
    - train/val cell-line responses normalized together
    - PDMC/PDXO responses normalized separately
    """
    feature_scalers: dict[str, SklearnObject | None] = {}

    feature_scalers["exp"] = normalize_cell_feature(
        train_cell_ds,
        val_cell_ds,
        pdmc_ds,
        feature_name="exp",
        norm_method=exp_norm_method,
    )

    if normalize_cnv and "cnv" in train_cell_ds.cell_encoders:
        feature_scalers["cnv"] = normalize_cell_feature(
            train_cell_ds,
            val_cell_ds,
            pdmc_ds,
            feature_name="cnv",
            norm_method="between",
        )

    pdmc_ds = filter_drugs(
        pdmc_ds,
        reference_datasets=[train_cell_ds, val_cell_ds],
        include_dataset_drugs=keep_pdmc_only_drugs,
        min_cells_per_drug=min_pdmcs_per_drug,
        drug_col="drug_id",
        name=pdmc_ds.name,
    )

    train_cell_ds, val_cell_ds, _, cell_response_scaler = normalize_response_datasets(
        train_cell_ds,
        val_cell_ds,
        None,
        norm_method="grouped",
    )

    pdmc_ds, _, _, pdmc_response_scaler = normalize_response_datasets(
        pdmc_ds,
        None,
        None,
        norm_method="grouped",
    )

    artifacts = PreprocessingArtifacts(
        feature_scalers=feature_scalers,
        response_scalers={
            "cell": cell_response_scaler,
            "pdmc": pdmc_response_scaler,
        },
    )

    if artifact_path is not None:
        artifacts.save(artifact_path)

    return train_cell_ds, val_cell_ds, pdmc_ds, artifacts
