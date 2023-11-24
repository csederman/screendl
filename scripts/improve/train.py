#!/usr/bin/env python
"""IMPROVE-compatible training script for ScreenDL."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle

import typing as t
import benchmark as bmk
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

from types import SimpleNamespace
from pathlib import Path

from cdrpy.data import Dataset
from screendl import model as screendl
from screendl.utils import evaluation as eval_utils

file_path = os.path.dirname(os.path.realpath(__file__))
initialize_params = bmk.make_initialize_params(file_path)


GParams = t.Dict[str, t.Any]

# NOTE: split should be in additional parameters

input_paths = SimpleNamespace(
    labels="LabelsLogIC50.csv",
    split="splits/tumor_blind/fold_1.pkl",
    mol="ScreenDL/FeatureMorganFingerprints.csv",
    exp="ScreenDL/FeatureGeneExpression.csv",
)


def split_dataset(g_params: GParams, D: Dataset) -> t.Tuple[Dataset, Dataset, Dataset]:
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


def run(g_params: GParams) -> t.Dict[str, float]:
    """Trains and evaluates ScreenDL for the specified parameters."""
    print(g_params)

    data_dir = Path(g_params["data_dir"])

    cell_encoders = screendl.load_cell_features(data_dir / input_paths.exp)
    drug_encoders = screendl.load_drug_features(data_dir / input_paths.mol)

    D = Dataset.from_csv(
        data_dir / input_paths.labels,
        name="cmp-gdsc2",
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
    )

    train_ds, val_ds, test_ds = split_dataset(g_params, D)

    print(train_ds, val_ds, test_ds)


def main() -> None:
    g_params = initialize_params()
    scores = run(g_params)


if __name__ == "__main__":
    main()

    try:
        K.clear_session()
    except AttributeError:
        pass
