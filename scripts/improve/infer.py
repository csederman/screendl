"""IMPROVE-compatable inference script."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pickle

import numpy as np
import typing as t
import benchmark as bmk
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

from pathlib import Path

from cdrpy.data import Dataset


GParams = t.Dict[str, t.Any]


file_path = os.path.dirname(os.path.realpath(__file__))
init_params = bmk.make_param_initializer(file_path)


bmk.configure_session()


def load_test_data(g_params: GParams) -> Dataset:
    """Loads the test dataset."""
    data_dir = Path(g_params["data_dir"])

    D = Dataset.load(data_dir / "CellModelPassportsGDSCv2.h5")

    split_type = g_params["split_type"]
    split_id = g_params["split_id"]
    split_path = data_dir / "splits" / split_type / f"fold_{split_id}.pkl"

    with open(split_path, "rb") as fh:
        split_dict = pickle.load(fh)

    return D.select(split_dict["train"], name="test")


def preprocess_data(g_params: GParams, test_ds: Dataset) -> Dataset:
    """Apply preprocessing pipeline."""
    output_dir = Path(g_params["output_dir"])
    print(os.listdir(g_params["output_dir"]))


def infer(g_params: GParams) -> None:
    """Runs inference with ScreenDL."""
    data_dir = Path(g_params["data_dir"])
    output_dir = Path(g_params["output_dir"])

    test_ds = load_test_data(g_params)


def main() -> None:
    g_params = init_params()
    infer(g_params)


if __name__ == "__main__":
    main()
    try:
        K.clear_session()

    except AttributeError:
        pass
