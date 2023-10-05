"""DepMap preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from pathlib import Path


def load_oncotree_annotations(file_path: str | Path) -> dict[str, str]:
    """Reads in the oncotree annotations from DepMap."""
    df = (
        pd.read_csv(
            file_path,
            usecols=["SangerModelID", "OncotreeCode"],
        )
        .dropna(subset="SangerModelID")
        .drop_duplicates(subset="SangerModelID")
    )

    # manual curation
    cell_to_oncotree = dict(zip(df["SangerModelID"], df["OncotreeCode"]))
    cell_to_oncotree.update({"SIDM00427": "ES", "SIDM01117": "CSCLC"})

    return cell_to_oncotree
