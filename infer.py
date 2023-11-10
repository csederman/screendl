#!/usr/bin/env python
"""IMPROVE-compatible training script for ScreenDL."""

from __future__ import annotations

import os
import pickle
import candle

import numpy as np
import typing as t
import tensorflow as tf

from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import backend as K

from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.mapper import BatchedResponseGenerator

from screendl import model as screendl
from screendl.utils.evaluation import make_pred_df

if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder

data_dir = os.environ["CANDLE_DATA_DIR"].rstrip("/")
file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = []
required = ["seed", "epochs", "batch_size", "learning_rate", "output_dir"]


class ScreenDL(candle.Benchmark):
    def set_locals(self):
        if required is not None:
            self.required = set(required)

        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


if K.backend() == "tensorflow" and "NUM_INTRA_THREADS" in os.environ:
    sess = tf.Session(
        config=tf.ConfigProto(
            inter_op_parallelism_threads=int(os.environ["NUM_INTER_THREADS"]),
            intra_op_parallelism_threads=int(os.environ["NUM_INTRA_THREADS"]),
        )
    )
    K.set_session(sess)


def initialize_parameters() -> t.Dict[str, t.Any]:
    """Initialize parameters for the run."""
    screendl_common = ScreenDL(
        file_path,
        "screendl_default_model.txt",
        "keras",
        prog="ScreenDL_candle",
        desc="ScreenDL run",
    )

    # Initialize parameters
    g_parameters = candle.finalize_parameters(screendl_common)

    return g_parameters


def run(g_parameters: t.Dict[str, t.Any]) -> None:
    """"""
    batch_size = g_parameters["batch_size"]
    output_dir = Path(g_parameters["output_dir"])
    output_dir.mkdir(exist_ok=True)

    # 1. load the data

    # NOTE: could move this to preprocess.py
    data_dir = Path(data_dir)
    mol_path = data_dir / "ScreenDL/FeatureMorganFingerprints.csv"
    exp_path = data_dir / "ScreenDL/FeatureGeneExpression.csv"

    cell_encoders = screendl.load_cell_features(exp_path)
    drug_encoders = screendl.load_drug_features(mol_path)

    ds = Dataset.from_csv(
        data_dir / "LabelsLogIC50.csv",
        name="CMP_GDSCv2",
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
    )

    # 2. preprocessing

    # preprocess the labels
    ds.obs["label"] = ds.obs.groupby("drug_id")["label"].transform(stats.zscore)

    # preprocess the gene expression
    exp_enc: PandasEncoder = ds.cell_encoders["exp"]
    exp_enc.data[:] = StandardScaler().fit_transform(exp_enc.data)

    # 3. load the model
    model = keras.models.load_model(output_dir / "model")

    # 4. get test set
    split_id = 1
    split_name = "tumor_blind"
    split_path = data_dir / "splits" / split_name / split_id

    with open(split_path, "rb") as fh:
        split = pickle.load(fh)

    test_ds = ds.select(split["test"], name="test")
    test_gen = BatchedResponseGenerator(test_ds, batch_size=batch_size)
    test_seq = test_gen.flow_from_dataset(test_ds)

    # 5. generate predictions
    preds = model.predict(test_seq)
    result = make_pred_df(test_ds, preds)
    result.to_csv(output_dir / "test_results.csv")


def main() -> None:
    g_parameters = initialize_parameters()
    run(g_parameters)


if __name__ == "__main__":
    main()
    try:
        K.clear_session()

    except AttributeError:
        pass
