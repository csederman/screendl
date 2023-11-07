"""Evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

if t.TYPE_CHECKING:
    from cdrpy.data import Dataset


def make_pred_df(ds: Dataset, preds: np.ndarray, **kwargs) -> pd.DataFrame:
    """"""
    return pd.DataFrame(
        dict(
            cell_id=ds.cell_ids,
            drug_id=ds.drug_ids,
            y_true=ds.labels,
            y_pred=preds.flatten(),
            **kwargs,
        )
    )
