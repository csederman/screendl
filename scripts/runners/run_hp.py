#!/usr/bin/env python
"""Hyper-parameter optimization for ScreenDL with Optuna."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import importlib
import logging
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from omegaconf import DictConfig


log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}


def configure_session() -> None:
    """Enable memory growth to avoid mem leak between hydra jobs."""
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="hp_config",
)
def run_hp(cfg: DictConfig) -> float:
    """"""
    # What I should do here is just use importlib
    if not cfg.model.name in PIPELINES:
        raise ValueError("Unsupported model.")

    module_file = PIPELINES[cfg.model.name]
    module_name = f"screendl.pipelines.basic.{module_file}"
    module = importlib.import_module(module_name)

    _, scores, _ = module.run_pipeline(cfg)

    # return scores["val"]["value"]
    return scores["val"]["mean_drug_pcc"]


if __name__ == "__main__":
    configure_session()
    run_hp()
    try:
        K.clear_session()
    except AttributeError:
        pass
