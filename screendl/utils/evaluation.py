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
ScoreKey = t.Literal[
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
    mean_drug_metrics = dict(pd.DataFrame(list(drug_metrics)).mean(skipna=True))
    median_drug_metrics = dict(pd.DataFrame(list(drug_metrics)).median(skipna=True))

    metrics = _get_metrics(pred_df)
    metrics.update({f"mean_drug_{k}": v for k, v in mean_drug_metrics.items()})
    metrics.update({f"median_drug_{k}": v for k, v in median_drug_metrics.items()})

    return {"key": key, "value": metrics[key], **metrics}


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
    # NOTE: deprecated, use `get_predictions_vs_background` instead

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


def get_preds(
    M: keras.Model, D: Dataset, batch_size: int = 256, **kwargs
) -> pd.DataFrame:
    """Computes z-score predictions against a background distribution."""
    gen = BatchedResponseGenerator(D, batch_size)
    seq = gen.flow_from_dataset(D)
    preds = M.predict(seq, verbose=0)

    return make_pred_df(D, preds, **kwargs)


def _predict_internal(M: keras.Model, D: Dataset, batch_size: bool = 256) -> np.ndarray:
    """"""
    gen = BatchedResponseGenerator(D, batch_size)
    return M.predict(gen.flow_from_dataset(D), verbose=0)


def _normalize_to_background_global(
    t_preds: pd.DataFrame,
    b_preds: pd.DataFrame,
    y_var: str = "y_pred",
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """"""
    scaler = StandardScaler().fit(b_preds[[y_var]])
    t_preds[y_var] = scaler.transform(t_preds[[y_var]])
    b_preds[y_var] = scaler.transform(b_preds[[y_var]])

    return t_preds, b_preds


def _normalize_to_background_grouped(
    t_preds: pd.DataFrame,
    b_preds: pd.DataFrame,
    y_var: str = "y_pred",
    group_var: str = "drug_id",
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """"""
    scaler = GroupStandardScaler().fit(b_preds[[y_var]], groups=b_preds[group_var])
    t_preds[y_var] = scaler.transform(t_preds[[y_var]], groups=t_preds[group_var])
    b_preds[y_var] = scaler.transform(b_preds[[y_var]], groups=b_preds[group_var])
    return t_preds, b_preds


def get_predictions_vs_background(
    M: keras.Model,
    D_t: Dataset,
    D_b: Dataset,
    W_t: t.Any = None,
    W_b: t.Any = None,
    batch_size: int = 256,
    grouped: bool = True,
    return_all: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Computes z-score predictions against a background distribution."""
    W_0 = M.get_weights()

    W_t = W_0 if W_t is None else W_t
    W_b = W_0 if W_b is None else W_b

    M.set_weights(W_t)
    t_preds = _predict_internal(M, D_t, batch_size)
    t_preds = make_pred_df(D_t, t_preds, **kwargs)

    M.set_weights(W_b)
    b_preds = _predict_internal(M, D_b, batch_size)
    b_preds = make_pred_df(D_b, b_preds, **kwargs)

    # restore initial weights
    M.set_weights(W_0)

    norm_func = (
        _normalize_to_background_grouped if grouped else _normalize_to_background_global
    )
    t_preds, b_preds = norm_func(t_preds, b_preds)

    if return_all:
        b_preds = b_preds.assign(partition="bkg")
        t_preds = t_preds.assign(partition="tgt")
        return pd.concat([t_preds, b_preds]).reset_index(drop=True)

    return t_preds
