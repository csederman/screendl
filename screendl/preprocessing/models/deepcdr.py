"""Preprocessing utilities for DeepCDR."""

from __future__ import annotations

import logging
import pickle

import pandas as pd
import numpy as np
import typing as t

from dataclasses import dataclass
from pathlib import Path

from cdrpy.util import io
from rdkit import Chem
from rdkit.Chem import AllChem as AllChem
from deepchem.feat.graph_features import ConvMolFeaturizer


log = logging.getLogger(__name__)


ConvMolFeat = t.Dict[str, t.Tuple[np.ndarray, t.List[int], t.List[t.List[int]]]]


def generate_deepcdr_inputs(
    exp_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    drug_info_df: pd.DataFrame,
    gene_list: t.Iterable[str],
) -> t.Tuple[pd.DataFrame, pd.DataFrame, ConvMolFeat]:
    """Prepares input features for DeepCDR."""

    # 1. extract gene expression features
    exp_common_genes = exp_df.columns.intersection(gene_list)

    # log transform raw TPM values
    # exp_feat: pd.DataFrame = np.log2(exp_df[exp_common_genes] + 1)
    exp_feat = exp_df[exp_common_genes].sort_index()

    num_missing_exp = len(set(gene_list).difference(exp_common_genes))
    if num_missing_exp > 0:
        log.warning(f"Found {num_missing_exp} missing expression features.")

    # 2. extract the mutation features
    mut_df = mut_df[mut_df["gene_symbol"].isin(gene_list)]
    mut_df = mut_df[mut_df["protein_mutation"] != "-"]

    # convert chromosomes to intergers for sorting
    ordered_chroms = [f"chr{x}" for x in range(1, 23)] + ["chrX", "chrY"]
    chrom_to_order = {x: int(i) for i, x in enumerate(ordered_chroms, 1)}
    mut_df["chrom_int"] = mut_df["chrom"].map(chrom_to_order)

    feat_cols = ["chrom", "pos", "gene_symbol"]
    mut_df["chrom_int"] = mut_df["chrom"].map(chrom_to_order)
    mut_df["feature_id"] = mut_df[feat_cols].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    mut_feat = (
        mut_df.sort_values(["chrom_int", "pos"])
        .groupby(["model_id", "feature_id"], sort=False)
        .size()
        .unstack()
        .fillna(0)
        .clip(upper=1)
        .astype(np.float32)
    )

    # 3. generate the drug features
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

    return exp_feat, mut_feat, drug_to_conv_mol_feat


@dataclass
class _PathConfig:
    """Path config for DeepCDR input features."""

    gene_list: str | Path


@dataclass
class _ParamConfig:
    """Param config for DeepCDR input features.."""


@dataclass
class DeepCDRFeatureConfig:
    """DeepCDR input feature config."""

    paths: _PathConfig
    params: _ParamConfig


def generate_and_save_deepcdr_inputs(
    cfg: DeepCDRFeatureConfig,
    out_dir: Path,
    exp_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    drug_meta: pd.DataFrame,
) -> None:
    """Generates and saves DeepCDR input features."""

    out_dir.mkdir(exist_ok=True)

    paths = cfg.paths

    gene_list = io.read_pickled_list(paths.gene_list)
    exp_feat, mut_feat, mol_feat = generate_deepcdr_inputs(
        exp_df=exp_df,
        mut_df=mut_df,
        drug_info_df=drug_meta,
        gene_list=gene_list,
    )

    exp_feat.to_csv(out_dir / "FeatureGeneExpression.csv")
    mut_feat.to_csv(out_dir / "FeatureSomaticMutations.csv")

    with open(out_dir / "FeatureConvMol.pkl", "wb") as fh:
        pickle.dump(mol_feat, fh)
