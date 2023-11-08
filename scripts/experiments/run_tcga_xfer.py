#!/usr/bin/env python
"""Runs transfer learning experiments on the TCGA dataset."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import typing as t

from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tqdm import tqdm

from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.metrics import tf_metrics

from screendl import model as screendl
from screendl.utils.evaluation import make_pred_df

if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder


log = logging.getLogger(__name__)


def data_loader(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    """Loads the input datasets."""

    pt_paths = cfg.dataset.sources.patient
    cell_paths = cfg.dataset.sources.cell

    pt_drug_encoders = screendl.load_drug_features(pt_paths.screendl.mol)
    pt_sample_encoders = screendl.load_cell_features(pt_paths.screendl.exp)

    cell_drug_encoders = screendl.load_drug_features(cell_paths.screendl.mol)
    cell_sample_encoders = screendl.load_cell_features(cell_paths.screendl.exp)

    pt_ds = Dataset.from_csv(
        pt_paths.labels,
        name=cfg.dataset.name,
        cell_encoders=pt_sample_encoders,
        drug_encoders=pt_drug_encoders,
    )

    cell_ds = Dataset.from_csv(
        cell_paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_sample_encoders,
        drug_encoders=cell_drug_encoders,
    )

    return cell_ds, pt_ds
