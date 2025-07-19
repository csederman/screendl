"""Helpers for model evaluation."""

from __future__ import annotations

import pandas as pd
import numpy as np
import sklearn.metrics as skm


def auroc(
    df: pd.DataFrame,
    col1: str = "y_true",
    col2: str = "y_pred",
    min_obs: int = 10,
) -> float:
    """Calculate the Area Under the Receiver Operating Characteristic Curve (auROC)."""
    if df[col1].nunique() <= 1 or len(df) < min_obs:
        return np.nan
    return skm.roc_auc_score(df[col1], -1 * df[col2])


def _check_series_is_binary(series: pd.Series) -> bool:
    """Check if the values are binary."""
    return series.isin([0, 1]).all()


class ResponseRateEvaluator:
    """Utility class for assessing response rate.

    Parameters
    ----------
    y_pred_var : str
        Name of the predictions variable in the DataFrame.
    y_true_var : str
        Name of the true labels variable in the DataFrame.
    n_iter : int
        Number of iterations for sampling to estimate response rate.
    epsilon : float
        Tolerance for determining the best therapy based on predictions.
    """

    def __init__(
        self,
        y_pred_var: str = "y_pred",
        y_true_var: str = "y_true",
        n_iter: int = 1000,
        epsilon: float = 1e-7,
    ) -> None:
        self.y_pred_var = y_pred_var
        self.y_true_var = y_true_var
        self.n_iter = n_iter
        self.epsilon = epsilon

    def eval(self, df: pd.DataFrame) -> int:
        """"""
        if not _check_series_is_binary(df[self.y_true_var]):
            raise ValueError(f"Expected binary values in {self.y_true_var}.")

        grouped = df.groupby("cell_id")

        rrs = []
        for _, g in grouped:
            # Select the best therapy based on predictions
            y_pred_min = g[self.y_pred_var].min()
            df_sel = g[(g[self.y_pred_var] - y_pred_min).abs() < self.epsilon]
            y_vals = df_sel[self.y_true_var].values

            if len(df_sel) == 1:
                # no ties - just take the true value for the selected drug
                rrs.append(y_vals[0])
            else:
                # ties found - run n_iters of random sampling and take the mean
                rrs.append(
                    np.mean([np.random.choice(y_vals) for _ in range(self.n_iter)])
                )

        return np.mean(rrs)


def select_best_therapy(
    df: pd.DataFrame, y_pred_var: str = "y_pred", epsilon: float = 1e-7
) -> pd.DataFrame:
    """Selects the best therapy for each tumor based on predicted response."""
    y_pred_min = df[y_pred_var].min()
    return df[(df[y_pred_var] - y_pred_min).abs() < epsilon].sample(n=1)
