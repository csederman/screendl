#!/usr/bin/env python
"""IMPROVE-compatible training script for ScreenDL."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle

import numpy as np
import typing as t
import benchmark as bmk
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace

from cdrpy.data import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.feat.encoders import PandasEncoder

from screendl import model as screendl
from screendl.utils import evaluation as eval_utils


GParams = t.Dict[str, t.Any]


file_path = os.path.dirname(os.path.realpath(__file__))
init_params = bmk.make_param_initializer(file_path)


paths = SimpleNamespace(
    dataset="CellModelPassportsGDSCv2.h5",
    split="splits/tumor_blind/fold_1.pkl",
)


def split_data(g_params: GParams, D: Dataset) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Splits the dataset into train/val/test."""
    data_dir = Path(g_params["data_dir"])

    split_type = g_params["split_type"]
    split_id = g_params["split_id"]
    split_path = data_dir / split_type / f"fold_{split_id}.pkl"

    with open(split_path, "rb") as fh:
        split_dict = pickle.load(fh)

    return (
        D.select(split_dict["train"]),
        D.select(split_dict["val"]),
        D.select(split_dict["test"]),
    )


def preprocess_data(
    g_params: GParams, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset
) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Runs preprocessing."""

    # FIXME: add g_params passing of norm_method
    # FIXME: add saving of the sklearn transformer
    train_ds, val_ds, test_ds = normalize_responses(
        train_ds, val_ds, test_ds, norm_method="global"
    )

    # normalize gene expression features
    # FIXME: add saving of the standard scaler
    exp_enc: PandasEncoder = train_ds.cell_encoders["exp"]
    X_exp = np.array(exp_enc.encode(list(set(train_ds.cell_ids))))
    exp_scaler = StandardScaler().fit(X_exp)
    exp_enc.data[:] = exp_scaler.transform(exp_enc.data.values)

    return train_ds, val_ds, test_ds


def run(g_params: GParams) -> t.Dict[str, float]:
    """Trains and evaluates ScreenDL for the specified parameters."""
    print(g_params)

    data_dir = Path(g_params["data_dir"])

    D = Dataset.load(data_dir / paths.dataset)

    train_ds, val_ds, test_ds = split_data(g_params, D)
    train_ds, val_ds, test_ds = preprocess_data(g_params, train_ds, val_ds, test_ds)

    print(train_ds, val_ds, test_ds)


def main() -> None:
    g_params = init_params()
    scores = run(g_params)


if __name__ == "__main__":
    main()

    try:
        K.clear_session()
    except AttributeError:
        pass
