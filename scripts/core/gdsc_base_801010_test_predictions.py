#!/usr/bin/env python
"""GDSC cell-line baseline: 80/10/10 split by cell_id, train ScreenDL, test predictions only.

Uses the core pipeline ([`load_dataset`], [`preprocess_dataset`], [`build_model_from_config`],
[`pretrain_model_from_config`]).

**Data layout:** the raw ``.../datasets/CellModelPassports-GDSCv1v2`` tree has Omics*.csv only
(no ``ScreenDL/FeatureMorganFingerprints.csv``). Use the **preprocessed inputs** bundle instead,
e.g. ``.../inputs/CellModelPassports-GDSCv1v2-HCI-v1.0.0`` with
``dataset=CellModelPassports-GDSCv1v2-HCI-inputs`` (``MetaSampleAnnotations.csv`` + ``ScreenDL/``).
By default, rows are restricted to **cell lines** (``domain == CELL`` in sample meta) so PDMC
lines are excluded from the 80/10/10 split; set ``gdsc_split.cell_lines_domain_only=false`` to keep all.

**Split:** 80% / 10% / 10% of unique ``cell_id``. First ``train_test_split`` holds out 20%
of lines; second splits that 20% evenly into validation and test (10% each of all lines).
If ``gdsc_split.stratify`` is true, stratifies on ``cancer_type`` from ``MetaCellAnnotations``
when each stratum has at least two lines; otherwise logs a warning and splits without stratify.

**Outputs** (Hydra run directory):

- ``predictions_base_gdsc_test_80_10_10_seed{seed}.csv`` with columns
  ``cell_id``, ``drug_id``, ``true_label`` (post group-normalization, same as training target),
  ``predicted_label`` (raw model output).
- ``split_cells_80_10_10.json`` listing train/val/test ``cell_id`` and metadata.

**Not** PDXO background z-scored predictions; comparable to core ``pretrain.py`` evaluation.

Example::

    python scripts/core/gdsc_base_801010_test_predictions.py \\
      dataset=CellModelPassports-GDSCv1v2-HCI-inputs \\
      dataset.dir=/path/to/inputs/CellModelPassports-GDSCv1v2-HCI-v1.0.0 \\
      gdsc_split.seed=42
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

from cdrpy.datasets import Dataset
from cdrpy.mapper import BatchedResponseGenerator

from screendl.pipelines.core.screendl import (
    build_model_from_config,
    load_dataset,
    preprocess_dataset,
    pretrain_model_from_config,
)

log = logging.getLogger(__name__)


def _cancer_type_per_cell(D: Dataset) -> pd.Series | None:
    """Map cell_id -> cancer_type if available."""
    meta = D.cell_meta
    if meta is None or meta.empty:
        return None
    if "cancer_type" not in meta.columns:
        return None

    cells = set(D.obs["cell_id"].astype(str).unique())

    if "cell_id" in meta.columns:
        m = meta.drop_duplicates(subset=["cell_id"])
        s = m.set_index(m["cell_id"].astype(str))["cancer_type"].astype(str)
    else:
        idx = meta.index.astype(str)
        s = pd.Series(meta["cancer_type"].astype(str).values, index=idx)

    s = s[~s.index.duplicated(keep="first")]
    hit = sum(1 for c in cells if c in s.index)
    if hit < max(1, int(0.5 * len(cells))):
        log.warning(
            "cell_meta index/cell_id alignment weak (%d/%d cells matched); stratify may be wrong",
            hit,
            len(cells),
        )
    out = pd.Series({c: s.get(c, "unknown") for c in cells})
    out = out.fillna("unknown").astype(str)
    return out


def _stratify_ok(labels: np.ndarray) -> bool:
    if labels.size == 0:
        return False
    _, counts = np.unique(labels, return_counts=True)
    return bool(np.all(counts >= 2) and len(counts) >= 2)


def split_cell_ids_801010(
    cells: np.ndarray,
    cancer: pd.Series | None,
    *,
    seed: int,
    stratify: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train, val, test) cell_id lists."""
    cells = np.array(sorted(set(str(c) for c in cells)))

    if stratify and cancer is not None:
        strata = np.array([cancer.get(c, "unknown") for c in cells])
        use_s1 = _stratify_ok(strata)
        if not use_s1:
            log.warning("Stratify disabled for first split (need >=2 lines per cancer_type).")
    else:
        strata = None
        use_s1 = False

    if use_s1:
        c_train, c_temp, _, st_temp = train_test_split(
            cells,
            strata,
            test_size=0.2,
            random_state=seed,
            stratify=strata,
        )
    else:
        c_train, c_temp = train_test_split(
            cells,
            test_size=0.2,
            random_state=seed,
        )
        st_temp = np.array([cancer.get(c, "unknown") for c in c_temp]) if cancer is not None else None

    if st_temp is not None and _stratify_ok(st_temp):
        c_val, c_test = train_test_split(
            c_temp,
            test_size=0.5,
            random_state=seed + 1,
            stratify=st_temp,
        )
    else:
        if st_temp is not None:
            log.warning("Stratify disabled for val/test split (small held-out set).")
        c_val, c_test = train_test_split(
            c_temp,
            test_size=0.5,
            random_state=seed + 1,
        )

    return list(c_train), list(c_val), list(c_test)


@hydra.main(version_base=None, config_path="../../conf/core", config_name="pretrain")
def run(cfg: DictConfig) -> None:
    if cfg.pretrain.full_dataset_mode:
        raise ValueError("Set pretrain.full_dataset_mode=false for this script.")

    seed = int(cfg.gdsc_split.seed)
    stratify = bool(cfg.gdsc_split.stratify)

    log.info("Loading %s...", cfg.dataset.name)
    D = load_dataset(cfg)

    if bool(cfg.gdsc_split.get("cell_lines_domain_only", True)):
        meta = D.cell_meta
        if meta is not None and "domain" in meta.columns:
            n_cells_before = len(set(D.obs["cell_id"].astype(str)))
            keep_cells = meta.index[meta["domain"].astype(str) == "CELL"].astype(str).tolist()
            if not keep_cells:
                raise RuntimeError("cell_lines_domain_only=true but no CELL rows in cell_meta.")
            D = D.select_cells(keep_cells, name=D.name)
            log.info(
                "Filtered to domain=CELL: %d -> %d unique cell_id in obs",
                n_cells_before,
                len(set(D.obs["cell_id"].astype(str))),
            )
        else:
            log.info(
                "cell_lines_domain_only=true but no `domain` column in cell_meta; using all cells."
            )

    cancer = _cancer_type_per_cell(D) if stratify else None
    cells = D.obs["cell_id"].unique()
    train_cells, val_cells, test_cells = split_cell_ids_801010(
        cells, cancer, seed=seed, stratify=stratify and cancer is not None
    )

    split_path = Path("split_cells_80_10_10.json")
    split_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "stratify_requested": stratify,
                "n_cells_train": len(train_cells),
                "n_cells_val": len(val_cells),
                "n_cells_test": len(test_cells),
                "train": sorted(train_cells),
                "val": sorted(val_cells),
                "test": sorted(test_cells),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Wrote %s", split_path.resolve())

    Dt = D.select_cells(train_cells, name="train")
    Dv = D.select_cells(val_cells, name="val")
    Dte = D.select_cells(test_cells, name="test")

    log.info("Preprocessing...")
    Dt, Dv, Dte = preprocess_dataset(cfg, Dt, Dv, Dte)

    log.info("Building model...")
    model = build_model_from_config(
        cfg,
        exp_dim=Dt.cell_encoders["exp"].shape[-1],
        mol_dim=Dt.drug_encoders["mol"].shape[-1],
    )

    log.info("Training...")
    model = pretrain_model_from_config(cfg, model, Dt, Dv)

    log.info("Predicting on test (observed pairs only)...")
    bs = int(cfg.model.hyper.batch_size)
    gen = BatchedResponseGenerator(Dte, bs)
    yhat = model.predict(gen.flow_from_dataset(Dte), verbose=0).flatten()

    out = pd.DataFrame(
        {
            "cell_id": Dte.cell_ids,
            "drug_id": Dte.drug_ids,
            "true_label": Dte.labels.astype(np.float64),
            "predicted_label": yhat.astype(np.float64),
        }
    )
    out_name = f"predictions_base_gdsc_test_80_10_10_seed{seed}.csv"
    out.to_csv(out_name, index=False)
    log.info("Wrote %s (%d rows)", Path(out_name).resolve(), len(out))


if __name__ == "__main__":
    run()
    K.clear_session()
