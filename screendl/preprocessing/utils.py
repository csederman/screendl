"""Preprocessing utilities for ScreenDL."""

from __future__ import annotations


import pandas as pd
import typing as t


def intersect_columns(
    df1: pd.DataFrame, df2: pd.DataFrame, col1: t.Any, col2: t.Any = None
) -> t.Set[t.Any]:
    """Helper function to get common values between columns of two pd.DataFrames."""
    if col2 is None:
        col2 = col1
    df1_values = set(df1[col1])
    df2_values = set(df2[col2])
    return set.intersection(df1_values, df2_values)


def filter_by_value_counts(df: pd.DataFrame, col: t.Any, n: int) -> pd.DataFrame:
    """Filter a pd.DataFrame by value_counts in a given column."""
    counts = df[col].value_counts()
    keep_values = counts[counts >= n].index
    return df[df[col].isin(keep_values)]
