#!/usr/bin/env python
"""IMPROVE-compatible training script for ScreenDL."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import typing as t

from types import SimpleNamespace
from pathlib import Path

from cdrpy.data import Dataset
from screendl import model as screendl
from screendl.utils import evaluation as eval_utils

from . import benchmark as bmk

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


def split_data(g_params: GParams, D: Dataset) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Splits the dataset into train/val/test."""


def run(g_params: GParams) -> t.Dict[str, float]:
    """Trains and evaluates ScreenDL for the specified parameters."""

    data_dir = Path(g_params["data_dir"])

    cell_encoders = screendl.load_cell_features(data_dir / input_paths.exp)
    drug_encoders = screendl.load_drug_features(data_dir / input_paths.mol)

    D = Dataset.from_csv(data_dir / input_paths.labels)

    # with open(split_)


def main() -> None:
    g_parameters = initialize_params()
    scores = run(g_parameters)
