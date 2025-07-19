#!/usr/bin/env python
"""Fine-tune ScreenDL on breast cancer PDxOs.

FIXME: this should really not require the dataset format and should not use hydra
    -> it should instead simply take paths to input files and such to make it more
       flexible

Examples
========

Fine-tune ScreenDL ensemble members for application in PDxOs:

>>> python scripts/core/finetune.py \
        --config=finetune \
        --pretrain-dir="path/to/pretrained/model/dir"

Fine-tune using local directory configuration:

>>> python scripts/core/finetune.py \
        --config=finetune.local \
        --pretrain-dir="path/to/pretrained/model/dir"
"""

from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import logging
import random

import numpy as np
import pprint as pp
import tensorflow as tf
import typing as t
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from omegaconf import OmegaConf, ListConfig

from screendl.pipelines.core.screendl import (
    apply_preprocessing_pipeline,
    evaluate_model,
    load_dataset,
    load_pretrained_model,
    load_pretraining_configs,
    split_dataset,
)
from screendl.utils import model_utils


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser(description="Finetune ScreenDL on PDxO data.")
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to pretrained model."
    )
    parser.add_argument("--config", type=str, required=True, help="Config path.")
    args = parser.parse_args()
    return args


def safe_list_config_as_tuple(item: t.Any) -> t.Any:
    """Converts ListConfig instances to tuples or does nothing."""
    return tuple(item) if isinstance(item, ListConfig) else item


def finetune(args: argparse.Namespace) -> None:
    """"""
    pt_cfg, _ = load_pretraining_configs(args.dir)
    ft_cfg = OmegaConf.load(args.config)

    # load the dataset and apply the preprocessing pipeline
    log.info(f"Loading dataset...")
    D = load_dataset(pt_cfg)
    Dt, Dv, De = split_dataset(pt_cfg, D)
    Dt, Dv, De = apply_preprocessing_pipeline(args.dir, Dt, Dv, De)

    # load and the pretrained model
    pt_model = load_pretrained_model(args.dir)

    log.info(f"Tuning ScreenDL...")
    ft_model = model_utils.fit_transfer_model(
        pt_model,
        De.select_cells(set(De.cell_ids).difference(ft_cfg.exclude_tumors)),
        batch_size=ft_cfg.batch_size,
        epochs=ft_cfg.epochs,
        learning_rate=ft_cfg.learning_rate,
        weight_decay=ft_cfg.weight_decay,
        frozen_layer_prefixes=safe_list_config_as_tuple(ft_cfg.frozen_layer_prefixes),
        frozen_layer_names=safe_list_config_as_tuple(ft_cfg.frozen_layer_names),
    )

    log.info(f"Saving ScreenDL-FT...")
    ft_model.save(os.path.join(args.dir, "ScreenDL-FT.model"))
    ft_model.save_weights(os.path.join(args.dir, "ScreenDL-FT.weights"))

    log.info(f"Evaluating ScreenDL-FT...")
    preds, scores = evaluate_model(pt_cfg, ft_model, list(filter(None, [Dt, Dv, De])))
    preds_file = os.path.join(args.dir, "predictions.ft.csv")
    preds.to_csv(preds_file, index=False)

    scores_file = os.path.join(args.dir, "scores.ft.json")
    with open(scores_file, "w", encoding="utf-8") as fh:
        json.dump(scores, fh, ensure_ascii=False, indent=4)

    pp.pprint(scores)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S,%03d",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    finetune(args)

    try:
        K.clear_session()
    except AttributeError:
        pass
