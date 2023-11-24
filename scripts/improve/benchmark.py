"""ScreenDL benchmark configuration utils."""

from __future__ import annotations

import os
import candle  # pyright: ignore[reportMissingImports]

import tensorflow as tf
import typing as t

from tensorflow.keras import backend as K  # pyright: ignore[reportMissingImports]


additional_definitions = [
    # hyperparameters
    {
        "name": "activation",
        "type": str,
        "default": "leaky_relu",
    },
    {
        "name": "use_dropout",
        "type": bool,
        "default": False,
    },
    {
        "name": "dropout_rate",
        "type": float,
        "default": 0.1,
    },
    {
        "name": "use_batch_norm",
        "type": bool,
        "default": False,
    },
    # architecture
    {
        "name": "shared_hidden_dims",
        "type": t.List[str],
        "default": [64, 32, 16, 8],
    },
    {
        "name": "exp_hidden_dims",
        "type": t.List[str],
        "default": [512, 256, 128, 64],
    },
    {
        "name": "mol_hidden_dims",
        "type": t.List[str],
        "default": [256, 128, 64],
    },
    # train/val/test split
    {
        "name": "split_id",
        "type": int,
        "default": 1,
    },
    {
        "name": "split_type",
        "type": str,
        "default": "tumor_blind",
    },
    # preprocessing
    {
        "name": "label_norm_method",
        "type": str,
        "default": "grouped",
    },
]
required_definitions = ["epochs", "batch_size", "learning_rate"]


class ScreenDL(candle.Benchmark):
    def set_locals(self):
        if required_definitions is not None:
            self.required = set(required_definitions)

        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def make_param_initializer(file_path: str) -> t.Callabe[[], t.Dict[str, t.Any]]:
    """Creates parameter initializer."""

    def initialize_params() -> t.Dict[str, t.Any]:
        screendl_bmk = ScreenDL(
            file_path,
            "screendl_default_model.txt",
            "keras",
            prog="ScreenDL_baseline",
            desc="ScreenDL Banchmark",
        )

        g_parameters = candle.finalize_parameters(screendl_bmk)

        return g_parameters

    return initialize_params


def configure_session() -> None:
    if K.backend() == "tensorflow" and "NUM_INTRA_THREADS" in os.environ:
        sess = tf.Session(
            config=tf.ConfigProto(
                inter_op_parallelism_threads=int(os.environ["NUM_INTER_THREADS"]),
                intra_op_parallelism_threads=int(os.environ["NUM_INTRA_THREADS"]),
            )
        )
        K.set_session(sess)
