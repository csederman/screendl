"""Preprocessing utilities for DualGCN."""

from __future__ import annotations

import logging
import pickle

import pandas as pd
import numpy as np
import typing as t

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem as AllChem
from deepchem.feat.graph_features import ConvMolFeaturizer

from .common import compute_copy_number_ratios


log = logging.getLogger(__name__)


ConvMolFeat = t.Dict[str, t.Tuple[np.ndarray, t.List[int], t.List[t.List[int]]]]


def read_dualgcn_ppi(file_path: str) -> pd.DataFrame:
    """Reads the raw DualGCN PPI network file."""
    ppi_df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["gene_1", "gene_2"],
    )

    # map proteins to genes for non-overlapping proteins
    unique_prots = list(ppi_df["gene_1"].unique())

    prot_to_gene = dict(zip(unique_prots, unique_prots))
    prot_to_gene.update(
        {
            "CARS": "CARS1",
            "FGFR1OP": "CEP43",
            "SEPT5": "SEPTIN5",
            "SEPT6": "SEPTIN6",
            "SEPT9": "SEPTIN9",
            "H3F3A": "H3-3A",
            "H3F3B": "H3-3B",
        }
    )

    ppi_df["gene_1"] = ppi_df["gene_1"].map(prot_to_gene)
    ppi_df["gene_2"] = ppi_df["gene_2"].map(prot_to_gene)

    return ppi_df


def generate_dualgcn_inputs(
    exp_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    cell_info_df: pd.DataFrame,
    drug_info_df: pd.DataFrame,
) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ConvMolFeat]:
    """"""

    # 1. get PPI genes with expression and CNV features
    ppi_genes = set(ppi_df["gene_1"]).intersection(ppi_df["gene_2"])
    common_genes = ppi_genes.intersection(exp_df.columns, cnv_df.columns)
    common_genes = sorted(list(common_genes))

    num_missing_ppi_genes = len(ppi_genes.difference(common_genes))
    if num_missing_ppi_genes > 0:
        log.warning(f"Found {num_missing_ppi_genes} missing PPI genes.")

    # 2. extract gene expression features
    # exp_feat: pd.DataFrame = np.log2(exp_df[common_genes] + 1)
    exp_feat = exp_df[common_genes]

    # 3. extract copy number features
    cnv_feat = compute_copy_number_ratios(
        cnv_df=cnv_df[common_genes], cell_info_df=cell_info_df
    )
    cnv_feat: pd.DataFrame = np.log2(cnv_feat + 1)

    # 4. extract the PPI features
    ppi_feat = ppi_df[
        ppi_df["gene_1"].isin(common_genes) & ppi_df["gene_2"].isin(common_genes)
    ]

    # 5. generate the drug features
    drug_to_smiles = drug_info_df[["drug_name", "smiles"]]
    drug_to_conv_mol_feat = {}
    for drug_name, smiles in drug_to_smiles.itertuples(index=False):
        mol = Chem.MolFromSmiles(smiles)
        featurizer = ConvMolFeaturizer()
        mol_object = featurizer.featurize([mol])
        drug_to_conv_mol_feat[drug_name] = (
            mol_object[0].atom_features,
            mol_object[0].deg_list,
            mol_object[0].canon_adj_list,
        )

    return exp_feat, cnv_feat, ppi_feat, drug_to_conv_mol_feat


@dataclass
class _PathConfig:
    """Path config for DualGCN input features."""

    ppi: str | Path


@dataclass
class _ParamConfig:
    """Param config for DualGCN input features.."""


@dataclass
class DualGCNFeatureConfig:
    """DualGCn input feature config."""

    paths: _PathConfig
    params: _ParamConfig


def generate_and_save_dualgcn_inputs(
    cfg: DualGCNFeatureConfig,
    out_dir: Path,
    exp_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    cell_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
) -> None:
    """Generates and saves DualGCN input features."""

    out_dir.mkdir(exist_ok=True)

    paths = cfg.paths

    ppi_df = read_dualgcn_ppi(paths.ppi)
    exp_feat, cnv_feat, ppi_feat, mol_feat = generate_dualgcn_inputs(
        exp_df=exp_df,
        cnv_df=cnv_df,
        ppi_df=ppi_df,
        cell_info_df=cell_meta,
        drug_info_df=drug_meta,
    )

    exp_feat.to_csv(out_dir / "FeatureGeneExpression.csv")
    cnv_feat.to_csv(out_dir / "FeatureCopyNumberRatio.csv")
    ppi_feat.to_csv(out_dir / "MetaPPIEdges.csv", index=False)

    with open(out_dir / "FeatureConvMol.pkl", "wb") as fh:
        pickle.dump(mol_feat, fh)
