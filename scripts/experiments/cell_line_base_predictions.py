#!/usr/bin/env python
"""Train the base ScreenDL model on cell lines and save predictions (no transfer, no ScreenAhead).

Uses the same data loading, CV split, preprocessing, and base train loop as
``pdx_validation.py`` (expression scaling, grouped response normalization, PDMC drug
filters applied to the PDMC split so normalization matches the PDXO pipeline).

Output CSV columns align with ``predictions_pdxo.csv`` for the base model::

    cell_id, drug_id, y_true, y_pred, model, was_screened

Unlike ``get_predictions_vs_background`` in the PDXO script, ``y_pred`` here is the
**raw** model output on **observed** (cell_line, drug) pairs (not drug-wise z-scored
against a background panel).

Example::

    python scripts/experiments/cell_line_base_predictions.py \\
      dataset.dir=/path/to/CellModelPassports-... \\
      dataset.split.id=1 \\
      experiment.min_pdmcs_per_drug=1 \\
      _datastore_=/path/to/writable_runs
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import pandas as pd
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

from cdrpy.mapper import BatchedResponseGenerator
from omegaconf import DictConfig

from screendl.pipelines.basic.screendl import (
    data_loader,
    data_splitter,
    model_builder as base_model_builder,
    model_trainer as base_model_trainer,
)
from screendl.utils import evaluation as eval_utils

log = logging.getLogger(__name__)


def _load_pdx_validation_module():
    """Load sibling module without running ``main``."""
    path = Path(__file__).resolve().parent / "pdx_validation.py"
    spec = importlib.util.spec_from_file_location("pdx_validation_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["pdx_validation_helpers"] = mod
    spec.loader.exec_module(mod)
    return mod


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="pdx_validation",
)
def run(cfg: DictConfig) -> None:
    """Train base model on cell lines; write ``predictions_cell_lines_base.csv``."""
    pv = _load_pdx_validation_module()

    log.info("Loading %s...", cfg.dataset.name)
    dataset = data_loader(cfg)

    log.info("Splitting %s...", cfg.dataset.name)
    D_t_cell, D_v_cell, D_pdxo = data_splitter(cfg, dataset)

    log.info("Preprocessing (same as pdx_validation)...")
    D_t_cell, D_v_cell, D_pdxo = pv.data_preprocessor(cfg, D_t_cell, D_v_cell, D_pdxo)

    log.info("Building and training base model...")
    M = base_model_builder(cfg, D_t_cell)
    M = base_model_trainer(cfg, M, D_t_cell, D_v_cell)

    batch_size = int(cfg.model.hyper.batch_size)

    def _preds(D, split_name: str) -> pd.DataFrame:
        gen = BatchedResponseGenerator(D, batch_size)
        yhat = M.predict(gen.flow_from_dataset(D), verbose=0)
        return eval_utils.make_pred_df(
            D,
            yhat,
            model="base",
            was_screened=False,
            split=split_name,
        )

    log.info("Predicting on cell-line train and val (observed pairs only)...")
    out = pd.concat(
        [_preds(D_t_cell, "train"), _preds(D_v_cell, "val")],
        ignore_index=True,
    )
    out_path = Path("predictions_cell_lines_base.csv")
    out.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path.resolve(), len(out))


if __name__ == "__main__":
    run()
    K.clear_session()
