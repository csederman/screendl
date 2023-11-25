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
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from cdrpy.data import Dataset
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.feat.transformers import GroupStandardScaler

from screendl.utils import evaluation as eval_utils

if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder


GParams = t.Dict[str, t.Any]
LabelScaler = t.Union[StandardScaler, GroupStandardScaler]


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


def preprocess_data(g_params: GParams, D: Dataset) -> Dataset:
    """Apply preprocessing pipeline."""
    output_dir = Path(g_params["output_dir"])
    norm_method = g_params["label_norm_method"]

    with open(output_dir / "label_scaler.pkl", "rb") as fh:
        label_scaler: LabelScaler = pickle.load(fh)

    Y = D.obs[["label"]]
    if norm_method == "grouped":
        D.obs["label"] = label_scaler.transform(Y, groups=D.obs["drug_id"])
    elif norm_method == "global":
        D.obs["label"] = label_scaler.transform(Y)
    else:
        raise ValueError(f"Unsupported label normalization method ({norm_method})")

    # normalize gene expression features
    with open(output_dir / "exp_scaler.pkl", "rb") as fh:
        exp_scaler: StandardScaler = pickle.load(fh)

    exp_enc: PandasEncoder = D.cell_encoders["exp"]
    exp_enc.data[:] = exp_scaler.transform(exp_enc.data.values)

    return D


def infer(g_params: GParams) -> t.Dict[str, float]:
    """Runs inference with ScreenDL."""
    output_dir = Path(g_params["output_dir"])

    D = load_test_data(g_params)
    D = preprocess_data(g_params, D)

    model = keras.models.load_model(output_dir / "model")

    gen = BatchedResponseGenerator(D, g_params["batch_size"])
    preds = model.predict(gen.flow_from_dataset(D))
    result = eval_utils.make_pred_df(D, preds)
    result.to_csv(output_dir / "test_predictions.csv", index=False)

    scores = eval_utils.get_eval_metrics(result)
    with open(output_dir / "test_scores.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh, ensure_ascii=False, indent=4)

    return scores


def main() -> None:
    g_params = init_params()
    infer(g_params)


if __name__ == "__main__":
    main()
    try:
        K.clear_session()

    except AttributeError:
        pass
