"""Evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

from scipy import stats

if t.TYPE_CHECKING:
    from cdrpy.data import Dataset


ScoreDict = t.Dict[str, float]
ScoreKey: t.Literal[
    "loss",
    "pcc",
    "scc",
    "rmse",
    "mean_drug_loss",
    "mean_drug_pcc",
    "mean_drug_scc",
    "mean_drug_rmse",
]


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


def get_eval_metrics(pred_df: pd.DataFrame, key: ScoreKey = "loss") -> ScoreDict:
    """Computes performance metrics."""

    def _get_metrics(df: pd.DataFrame):
        y_true = df["y_true"]
        y_pred = df["y_pred"]

        mse = ((y_true - y_pred) ** 2).mean().astype(np.float64)
        pcc = scc = np.nan
        if df.shape[0] >= 5:
            pcc = stats.pearsonr(y_true, y_pred)[0]
            scc = stats.spearmanr(y_true, y_pred)[0]

        metrics = {"loss": mse, "pcc": pcc, "scc": scc, "rmse": mse**0.5}

        return metrics

    drug_metrics = pred_df.groupby("drug_id").apply(_get_metrics)
    drug_metrics = dict(pd.DataFrame(list(drug_metrics)).mean(skipna=True))

    metrics = _get_metrics(pred_df)
    metrics.update({f"mean_drug_{k}": v for k, v in drug_metrics.items()})

    return {"key": key, "value": metrics[key], **metrics}
