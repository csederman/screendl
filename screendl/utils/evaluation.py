"""Evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import typing as t

from scipy import stats
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from cdrpy.mapper import BatchedResponseGenerator
from cdrpy.data.preprocess import GroupStandardScaler

if t.TYPE_CHECKING:
    from cdrpy.data import Dataset
    from cdrpy.mapper.sequences import ResponseSequence


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


def pcorr(
    df: pd.DataFrame, c1: str = "y_true", c2: str = "y_pred", min_obs: int = 5
) -> float:
    if df.shape[0] < min_obs:
        return np.nan
    return stats.pearsonr(df[c1], df[c2])[0]


def get_predictions(model: keras.Model, batch_seq: ResponseSequence) -> np.ndarray:
    """Generate predictions without triggering tf.function retracing."""
    predictions = []
    for batch_x, *_ in batch_seq:
        batch_preds: tf.Tensor = model(batch_x, training=False)
        predictions.append(batch_preds.numpy().flatten())
    return np.concatenate(predictions)


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


# def get_preds_vs_background(
#     M: keras.Model,
#     target_ds: Dataset,
#     background_ds: Dataset,
#     batch_size: int = 256,
#     grouped: bool = True,
#     **kwargs,
# ) -> pd.DataFrame:
#     """Computes z-score predictions against a background distribution."""
#     t_gen = BatchedResponseGenerator(target_ds, batch_size)
#     t_preds = M.predict(t_gen.flow_from_dataset(target_ds), verbose=0)
#     t_pred_df = make_pred_df(target_ds, t_preds, **dict(kwargs, _bg=False))

#     b_gen = BatchedResponseGenerator(background_ds, batch_size)
#     b_preds = M.predict(b_gen.flow_from_dataset(background_ds), verbose=0)
#     b_pred_df = make_pred_df(background_ds, b_preds, **dict(kwargs, _bg=True))

#     pred_df = pd.concat([t_pred_df, b_pred_df])

#     if grouped:
#         pred_df["y_pred"] = pred_df.groupby("drug_id")["y_pred"].transform(stats.zscore)
#     else:
#         pred_df["y_pred"] = stats.zscore(pred_df["y_pred"])

#     pred_df = pred_df[pred_df["_bg"] == False].drop(columns="_bg")

#     return pred_df


def get_preds_vs_background(
    M: keras.Model,
    D_target: Dataset,
    D_background: Dataset,
    batch_size: int = 256,
    grouped: bool = True,
    return_all: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Computes z-score predictions against a background distribution."""
    tgt_gen = BatchedResponseGenerator(D_target, batch_size)
    tgt_seq = tgt_gen.flow_from_dataset(D_target)
    tgt_preds = M.predict(tgt_seq, verbose=0)
    tgt_preds = make_pred_df(D_target, tgt_preds, **kwargs)

    bg_gen = BatchedResponseGenerator(D_background, batch_size)
    bg_seq = bg_gen.flow_from_dataset(D_background)
    bg_preds = M.predict(bg_seq, verbose=0)
    bg_preds = make_pred_df(D_background, bg_preds, **kwargs)

    if grouped:
        scaler = GroupStandardScaler().fit(
            bg_preds[["y_pred"]], groups=bg_preds["drug_id"]
        )
        tgt_preds["y_pred"] = scaler.transform(
            tgt_preds[["y_pred"]], groups=tgt_preds["drug_id"]
        )
        if return_all:
            bg_preds["y_pred"] = scaler.transform(
                bg_preds[["y_pred"]], groups=bg_preds["drug_id"]
            )
    else:
        scaler = StandardScaler().fit(bg_preds[["y_pred"]])
        tgt_preds["y_pred"] = scaler.transform(tgt_preds[["y_pred"]])
        if return_all:
            bg_preds["y_pred"] = scaler.transform(bg_preds[["y_pred"]])

    if return_all:
        bg_preds = bg_preds.assign(partition="background")
        tgt_preds = tgt_preds.assign(partition="target")
        return pd.concat([tgt_preds, bg_preds]).reset_index(drop=True)

    return tgt_preds
