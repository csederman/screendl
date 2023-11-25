"""ScreenDL benchmark configuration utils."""

from __future__ import annotations

import os
import candle  # pyright: ignore[reportMissingImports]

import tensorflow as tf
import typing as t

from tensorflow.keras import backend as K  # pyright: ignore[reportMissingImports]

from constants import IMPROVE_ADDITIONAL_DEFINITIONS as additional_definitions
from constants import IMPROVE_REQUIRED_DEFINITIONS as required_definitions


class ScreenDLBenchmark(candle.Benchmark):
    def set_locals(self):
        if required_definitions is not None:
            self.required = set(required_definitions)

        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def make_param_initializer(file_path: str) -> t.Callabe[[], t.Dict[str, t.Any]]:
    """Creates parameter initializer."""

    def initialize_params() -> t.Dict[str, t.Any]:
        screendl_bmk = ScreenDLBenchmark(
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
