#!/usr/bin/env python
"""Run ScreenAhead on breast cancer PDxOs.

FIXME: this should really not require the dataset format and should not use hydra
    -> it should instead simply take paths to input files and such to make it more
       flexible

Examples
========

Run ScreenAhead for each ScreenDL-FT ensemble member for application in PDxOs:

>>> python scripts/core/screenahead.py \
        --config=screenahead \
        --dir="/path/to/pretrained/model/dir"
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
from pathlib import Path

from screendl.pipelines.core.screendl import (
    apply_preprocessing_pipeline,
    evaluate_model,
    load_dataset,
    load_pretrained_model,
    load_finetuned_model,
    load_pretraining_configs,
    split_dataset,
)
from screendl.utils import model_utils
from screendl.utils.drug_selectors import SELECTORS
from cdrpy.datasets.base import merge

if t.TYPE_CHECKING:
    from keras import Model
    from cdrpy.datasets import Dataset


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser(description="Run ScreenAhead for a PDxO.")
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to pretrained model."
    )
    parser.add_argument("--config", type=str, required=True, help="Config path.")
    args = parser.parse_args()
    return args


def safe_list_config_as_tuple(item: t.Any) -> t.Any:
    """Converts ListConfig instances to tuples or does nothing."""
    return tuple(item) if isinstance(item, ListConfig) else item


def load_base_model(
    dir_: str | Path, type_: t.Literal["ScreenDL-PT", "ScreenDL-FT"]
) -> Model:
    """"""
    if type_ == "ScreenDL-PT":
        return load_pretrained_model(dir_)
    elif type_ == "ScreenDL-FT":
        return load_finetuned_model(dir_)
    else:
        raise ValueError(f"Invalid base model type (got {type_})")


def screenahead(args: argparse.Namespace) -> None:
    """"""
    pt_cfg, _ = load_pretraining_configs(args.dir)
    sa_cfg = OmegaConf.load(args.config)

    # load the dataset and apply the preprocessing pipeline
    log.info(f"Loading datasets...")
    D = load_dataset(pt_cfg)
    D_t, D_v, D_e = split_dataset(pt_cfg, D)
    D_t, D_v, D_e = apply_preprocessing_pipeline(args.dir, D_t, D_v, D_e)

    # load and the pretrained or fine-tuned model model
    base_model = load_base_model(args.dir, sa_cfg.base_model)

    # dataset for ScreenAhead drug selection
    D_selection = D_t if D_v is None else merge(D_t, D_v)

    # dataset for evaluation
    D_tumor = D_e.select_cells([sa_cfg.tumor_id])

    # initialize the drug selector
    drug_selector = SELECTORS[sa_cfg.selector](
        # TODO: try to do the selection with the pdmc dataset (no na threshold)
        D_selection,
        seed=sa_cfg.seed,
        na_threshold=sa_cfg.na_thresh,
    )

    drug_choices = set(D_tumor.drug_ids)
    if sa_cfg.exclude_drugs is not None:
        drug_choices = set(x for x in drug_choices if x not in sa_cfg.exclude_drugs)
    prescreen_drugs = drug_selector.select(sa_cfg.n_drugs, choices=drug_choices)

    log.info(f"Tuning ScreenDL...")
    tuned_model = model_utils.fit_screenahead_model(
        base_model,
        D_tumor.select_drugs(prescreen_drugs),
        batch_size=sa_cfg.batch_size,
        epochs=sa_cfg.epochs,
        learning_rate=sa_cfg.learning_rate,
        weight_decay=sa_cfg.weight_decay,
        frozen_layer_prefixes=safe_list_config_as_tuple(sa_cfg.frozen_layer_prefixes),
        frozen_layer_names=safe_list_config_as_tuple(sa_cfg.frozen_layer_names),
    )

    log.info(f"Saving ScreenDL-SA...")
    prefix = f"ScreenDL-SA.{sa_cfg.tumor_id}"
    tuned_model.save(os.path.join(args.dir, prefix + ".model"))
    tuned_model.save_weights(os.path.join(args.dir, prefix + ".weights"))

    log.info(f"Evaluating ScreenDL-SA...")
    eval_datasets = list(filter(None, [D_t, D_v, D_e]))
    preds, scores = evaluate_model(pt_cfg, tuned_model, eval_datasets)
    preds_file = os.path.join(args.dir, "predictions.sa.csv")
    preds.to_csv(preds_file, index=False)

    scores_file = os.path.join(args.dir, "scores.sa.json")
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
    screenahead(args)

    try:
        K.clear_session()
    except AttributeError:
        pass
