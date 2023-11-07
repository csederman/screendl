"""Gene list parsers."""

from __future__ import annotations

import pickle

import pandas as pd
import typing as t

from pathlib import Path
from types import SimpleNamespace

from cdrpy.util import io


def pickle_genelist(file_path: Path, genelist: list[str]) -> None:
    """Pickles the genelist for later."""
    with open(file_path, "wb") as fh:
        pickle.dump(genelist, fh)


def parse_cancer_gene_census(file_path: Path) -> list[str]:
    """Parses the raw CancerGeneCensus gene list."""
    df = pd.read_csv(file_path)
    return sorted(list(df["Gene Symbol"].dropna().unique()))


def parse_mini_cancer_genome(file_path: Path) -> list[str]:
    """Parses the raw MiniCancerGenome gene list."""
    df = pd.read_csv(file_path)
    return sorted(list(df["HGNC_SYMBOL"].dropna().unique()))


def parse_lincs_genes(file_path: Path) -> list[str]:
    """Parses the raw LINCS1000 gene list."""
    df = pd.read_csv(file_path, sep="\t")
    df_landmark = df[df["feature_space"] == "landmark"]
    return sorted(list(df_landmark["gene_symbol"].unique()))


def parse_hallmark_genes(file_path: Path) -> list[str]:
    """Parses the raw MSigDB hallmark pathway GMT file."""
    gsets = io.read_gmt(file_path)
    genes = set([g for gs in gsets.values() for g in gs])
    return sorted(list(genes))


def parse_oncokb_genes(file_path: Path) -> list[str]:
    """Parses the raw OncoKB gene list."""
    df = pd.read_csv(file_path, sep="\t")
    return sorted(list(df["Hugo Symbol"].dropna().unique()))
