"""Configuration and constant variables."""

from __future__ import annotations

import typing as t

IMPROVE_ADDITIONAL_DEFINITIONS = [
    # hyperparameters
    {
        "name": "activation",
        "type": str,
        "default": "leaky_relu",
    },
    {
        "name": "use_dropout",
        "type": bool,
        "default": False,
    },
    {
        "name": "dropout_rate",
        "type": float,
        "default": 0.1,
    },
    {
        "name": "use_batch_norm",
        "type": bool,
        "default": False,
    },
    # architecture
    {
        "name": "shared_hidden_dims",
        "type": t.List[str],
        "default": [64, 32, 16, 8],
    },
    {
        "name": "exp_hidden_dims",
        "type": t.List[str],
        "default": [512, 256, 128, 64],
    },
    {
        "name": "mol_hidden_dims",
        "type": t.List[str],
        "default": [256, 128, 64],
    },
    # train/val/test split
    {
        "name": "split_id",
        "type": int,
        "default": 1,
    },
    {
        "name": "split_type",
        "type": str,
        "default": "tumor_blind",
    },
    # preprocessing
    {
        "name": "label_norm_method",
        "type": str,
        "default": "grouped",
    },
]

IMPROVE_REQUIRED_DEFINITIONS = ["epochs", "batch_size", "learning_rate"]


GENELIST_CHOICES = {
    "cgc": "CancerGeneCensus736Genes.pkl",
    "hmark": "HallmarkPathways4384Genes.pkl",
    "lincs": "LINCS978Genes.pkl",
    "mcg": "MiniCancerGenome1815Genes.pkl",
    "okb": "OncoKB1102Genes.pkl",
}
