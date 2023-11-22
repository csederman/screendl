"""ScreenDL benchmark configuration utils."""

from __future__ import annotations

import os
import candle  # pyright: ignore[reportMissingImports]

import tensorflow as tf
import typing as t

from tensorflow.keras import backend as K  # pyright: ignore[reportMissingImports]


additional_definitions = [
    {
        "name": "activation",
        "type": str,
        "default": "leaky_relu",
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
]
required_definitions = ["epochs", "batch_size", "learning_rate"]


class ScreenDL(candle.Benchmark):
    def set_locals(self):
        if required_definitions is not None:
            self.required = set(required_definitions)

        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def make_initialize_params(file_path: str) -> t.Callabe[[], t.Dict[str, t.Any]]:
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
