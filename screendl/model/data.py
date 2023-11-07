"""
Data loading utilities for ScreenDL.
"""

from __future__ import annotations

import pandas as pd
import typing as t

from pathlib import Path

from cdrpy.feat.encoders import PandasEncoder


if t.TYPE_CHECKING:
    from cdrpy.data.datasets import EncoderDict


def load_cell_features(
    exp_path: str | Path,
    mut_path: str | Path | None = None,
    cnv_path: str | Path | None = None,
    ont_path: str | Path | None = None,
) -> EncoderDict:
    """Loads cell features for ScreenDL."""
    enc_dict = {}

    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    enc_dict["exp"] = PandasEncoder(exp_mat, name="cell_encoder")

    if mut_path is not None:
        mut_mat = pd.read_csv(mut_path, index_col=0).astype("int32")
        enc_dict["mut"] = PandasEncoder(mut_mat, name="mut_encoder")

    if cnv_path is not None:
        cnv_mat = pd.read_csv(cnv_path, index_col=0).astype("float32")
        enc_dict["cnv"] = PandasEncoder(cnv_mat, name="cnv_encoder")

    if ont_path is not None:
        ont_mat = pd.read_csv(ont_path, index_col=0).astype("float32")
        enc_dict["ont"] = PandasEncoder(ont_mat, name="ont_encoder")

    return enc_dict


def load_drug_features(mol_path: str | Path) -> EncoderDict:
    """Loads drug features for ScreenDL."""
    enc_dict = {}
    mol_mat = pd.read_csv(mol_path, index_col=0).astype("int32")
    enc_dict["mol"] = PandasEncoder(mol_mat, name="drug_encoder")
    return enc_dict


def load_cell_metadata(file_path: str | Path) -> pd.DataFrame:
    """Loads the cell line metadata annotations."""
    return pd.read_csv(file_path, index_col=0)


def load_drug_metadata(file_path: str | Path) -> pd.DataFrame:
    """Loads the drug metadata annotations."""
    return pd.read_csv(file_path, index_col=0)
