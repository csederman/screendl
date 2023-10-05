"""
Data loading utilities for ScreenDL.
"""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from cdrpy.feat.encoders import PandasEncoder


def load_cell_features(
    exp_path: str | Path,
    mut_path: str | Path | None = None,
    cnv_path: str | Path | None = None,
    ont_path: str | Path | None = None,
) -> tuple[
    PandasEncoder,
    PandasEncoder | None,
    PandasEncoder | None,
    PandasEncoder | None,
]:
    """Load cell features for ScreenDL."""
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    exp_enc = PandasEncoder(exp_mat, name="cell_encoder")

    mut_enc = None
    if mut_path is not None:
        mut_mat = pd.read_csv(mut_path, index_col=0).astype("int32")
        mut_enc = PandasEncoder(mut_mat, name="mut_encoder")

    cnv_enc = None
    if cnv_path is not None:
        cnv_mat = pd.read_csv(cnv_path, index_col=0).astype("float32")
        cnv_enc = PandasEncoder(cnv_mat, name="cnv_encoder")

    ont_enc = None
    if ont_path is not None:
        ont_mat = pd.read_csv(ont_path, index_col=0).astype("float32")
        ont_enc = PandasEncoder(ont_mat, name="ont_encoder")

    return exp_enc, mut_enc, cnv_enc, ont_enc


def load_drug_features(mol_path: str | Path) -> PandasEncoder:
    """Load drug features for ScreenDL."""
    mol_mat = pd.read_csv(mol_path, index_col=0).astype("int32")
    return PandasEncoder(mol_mat, name="drug_encoder")
