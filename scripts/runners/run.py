#!/usr/bin/env python
""""""

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


PIPELINES = {
    "DeepCDR-legacy": "deepcdr_legacy",
    "DualGCN-legacy": "dualgcn_legacy",
    "HiDRA-legacy": "hidra_legacy",
    "ScreenDL": "screendl",
}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """"""
    # What I should do here is just use importlib
    if not cfg.model.name in PIPELINES:
        raise ValueError("Unsupported model.")

    module_file = PIPELINES[cfg.model.name]
    module_path = f"pipelines.{module_file}"
    module = importlib.import_module(module_path)

    module.run_pipeline(cfg)


if __name__ == "__main__":
    run()
