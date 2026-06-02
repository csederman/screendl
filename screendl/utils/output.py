"""Output utils for screendl."""

from __future__ import annotations

import pandas as pd

from pathlib import Path


def write_predictions_chunk(df: pd.DataFrame, path: str | Path) -> None:
    """Append prediction rows to CSV without keeping all folds in memory."""
    path = Path(path)
    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )