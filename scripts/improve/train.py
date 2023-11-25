#!/usr/bin/env python
"""IMPROVE-compatible training script for ScreenDL."""

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
from cdrpy.feat.encoders import PandasEncoder
from cdrpy.feat.transformers import GroupStandardScaler
from cdrpy.mapper import BatchedResponseGenerator

from screendl import model as screendl
from screendl.utils import evaluation as eval_utils


GParams = t.Dict[str, t.Any]


file_path = os.path.dirname(os.path.realpath(__file__))
init_params = bmk.make_param_initializer(file_path)
bmk.configure_session()


def split_data(g_params: GParams, D: Dataset) -> t.Tuple[Dataset, Dataset]:
    """Splits the dataset into train/val/test."""
    data_dir = Path(g_params["data_dir"])

    split_type = g_params["split_type"]
    split_id = g_params["split_id"]
    split_path = data_dir / "splits" / split_type / f"fold_{split_id}.pkl"

    with open(split_path, "rb") as fh:
        split_dict = pickle.load(fh)

    train_ds = D.select(split_dict["train"], name="train")
    val_ds = D.select(split_dict["val"], name="val")

    return train_ds, val_ds


def preprocess_data(
    g_params: GParams, train_ds: Dataset, val_ds: Dataset
) -> t.Tuple[Dataset, Dataset]:
    """Normalizers drug response labels and expression data."""
    output_dir = Path(g_params["output_dir"])
    norm_method = g_params["label_norm_method"]

    # normalize the drug response labels
    y_train = train_ds.obs[["label"]]
    y_val = val_ds.obs[["label"]]

    if norm_method == "grouped":
        groups_train = train_ds.obs["drug_id"]
        groups_val = val_ds.obs["drug_id"]

        scaler = GroupStandardScaler()
        train_ds.obs["label"] = scaler.fit_transform(y_train, groups=groups_train)
        val_ds.obs["label"] = scaler.transform(y_val, groups=groups_val)

    elif norm_method == "global":
        scaler = StandardScaler()
        train_ds.obs["label"] = scaler.fit_transform(y_train)
        val_ds.obs["label"] = scaler.transform(y_val)

    else:
        raise ValueError(f"Unsupported label normalization method ({norm_method})")

    with open(output_dir / "label_scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    # normalize gene expression features
    exp_enc: PandasEncoder = train_ds.cell_encoders["exp"]
    x_train = np.array(exp_enc.encode(list(set(train_ds.cell_ids))))
    exp_scaler = StandardScaler().fit(x_train)
    exp_enc.data[:] = exp_scaler.transform(exp_enc.data.values)

    with open(output_dir / "exp_scaler.pkl", "wb") as fh:
        pickle.dump(exp_scaler, fh)

    return train_ds, val_ds


def train(g_params: GParams) -> t.Dict[str, float]:
    """Trains and evaluates ScreenDL for the specified parameters."""
    data_dir = Path(g_params["data_dir"])
    output_dir = Path(g_params["output_dir"])

    D = Dataset.load(data_dir / "CellModelPassportsGDSCv2.h5")
    train_ds, val_ds = split_data(g_params, D)
    train_ds, val_ds = preprocess_data(g_params, train_ds, val_ds)

    exp_dim = train_ds.cell_encoders["exp"].shape[-1]
    mol_dim = train_ds.drug_encoders["mol"].shape[-1]

    model = screendl.create_model(
        exp_dim=exp_dim,
        mol_dim=mol_dim,
        exp_hidden_dims=g_params["exp_hidden_dims"],
        mol_hidden_dims=g_params["mol_hidden_dims"],
        shared_hidden_dims=g_params["shared_hidden_dims"],
        use_batch_norm=g_params["use_batch_norm"],
        use_dropout=g_params["use_dropout"],
        dropout_rate=g_params["dropout_rate"],
        activation=g_params["activation"],
    )

    model.compile(
        optimizer=keras.optimizers.Adam(g_params["learning_rate"]),
        loss="mean_squared_error",
    )

    train_gen = BatchedResponseGenerator(train_ds, g_params["batch_size"])
    val_gen = BatchedResponseGenerator(val_ds, g_params["batch_size"])
    train_seq = train_gen.flow_from_dataset(train_ds, shuffle=True, seed=4114)
    val_seq = val_gen.flow_from_dataset(val_ds)

    early_stopping = keras.callbacks.EarlyStopping(
        "val_loss",
        patience=10,
        restore_best_weights=True,
        start_from_epoch=3,
        verbose=1,
    )

    hx = model.fit(
        train_seq,
        epochs=g_params["epochs"],
        validation_data=val_seq,
        callbacks=[early_stopping],
    )

    model.save(output_dir / "model")

    val_preds = model.predict(val_seq)
    val_result = eval_utils.make_pred_df(val_ds, val_preds)
    val_result.to_csv(output_dir / "val_predictions.csv")

    val_scores = eval_utils.get_eval_metrics(val_result)
    with open(output_dir / "val_scores.json", "w", encoding="utf-8") as fh:
        json.dump(val_scores, fh, ensure_ascii=False, indent=4)

    print("IMPROVE_RESULT val_loss:\t{}".format(val_scores["loss"]))

    return val_scores


def main() -> None:
    g_params = init_params()
    scores = train(g_params)


if __name__ == "__main__":
    main()

    try:
        K.clear_session()
    except AttributeError:
        pass
