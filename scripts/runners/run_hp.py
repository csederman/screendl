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

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from omegaconf import DictConfig


log = logging.getLogger(__name__)


PIPELINES = {"ScreenDL": "screendl"}


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

    return scores["val"]["value"]


if __name__ == "__main__":
    run_hp()
