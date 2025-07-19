#!/usr/bin/env python
"""Pretrains ScreenDL in cancer cell lines.

Examples
========

Pretrain ScreenDL on the full cell line dataset:

>>> python scripts/core/pretrain.py \
        --config-name=pretrain \
        pretrain.independent_norm=false \
        pretrain.full_dataset_mode=true \
        dataset=CellModelPassports-GDSCv1v2 \
        model.io.save=true \
        model.hyper.early_stopping=false \
        model.hyper.epochs=20

Pretrain ScreenDL ensemble members for application in PDxOs:

>>> python scripts/core/pretrain.py -m \
        --config-name=pretrain \
        pretrain.independent_norm=true \
        pretrain.full_dataset_mode=false \
        dataset=CellModelPassports-GDSCv1v2-HCI \
        model.io.save=true \
        model.hyper.early_stopping=false \
        model.hyper.epochs=15
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import json
import hydra
import logging
import random

import typing as t
import numpy as np
import tensorflow as tf
import pprint as pp

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from screendl.pipelines.core.screendl import (
    build_model_from_config,
    evaluate_model,
    load_dataset,
    preprocess_dataset,
    pretrain_model_from_config,
    split_dataset,
)

if t.TYPE_CHECKING:
    from omegaconf import DictConfig


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/core",
    config_name="pretrain",
)
def pretrain(cfg: DictConfig) -> None:
    """"""
    if cfg.model.name != "ScreenDL":
        raise ValueError("Unsupported model.")

    if cfg.pretrain.full_dataset_mode:
        # full dataset training is only supported for regular runs
        assert HydraConfig().get().mode == RunMode.RUN

    log.info(f"Loading {cfg.dataset.name}...")
    D = load_dataset(cfg)

    log.info(f"Splitting {cfg.dataset.name}...")
    Dt, Dv, De = split_dataset(cfg, D)

    log.info(f"Preprocessing {cfg.dataset.name}...")
    Dt, Dv, De = preprocess_dataset(cfg, Dt, Dv, De)

    log.info(f"Building {cfg.model.name}...")
    model = build_model_from_config(
        cfg,
        exp_dim=D.cell_encoders["exp"].shape[-1],
        mol_dim=D.drug_encoders["mol"].shape[-1],
    )

    log.info(f"Training {cfg.model.name}...")
    model = pretrain_model_from_config(cfg, model, Dt, Dv)

    log.info(f"Evaluating {cfg.model.name}...")
    preds, scores = evaluate_model(cfg, model, list(filter(None, [Dt, Dv, De])))
    preds.to_csv("predictions.pt.csv", index=False)

    with open("scores.pt.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh, ensure_ascii=False, indent=4)

    pp.pprint(scores)

    return None


if __name__ == "__main__":
    pretrain()
