"""Preprocessing utilities for DeepCDR."""

from __future__ import annotations

import click
import pickle
import warnings

import pandas as pd
import numpy as np
import typing as t

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem as AllChem
from deepchem.feat.graph_features import ConvMolFeaturizer
from cdrpy.util import io

if t.TYPE_CHECKING:
    from make_inputs import LazyDataStore


def prepare_deepcdr_inputs(
    exp_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    drug_info_df: pd.DataFrame,
    gene_list: t.Iterable[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[np.ndarray, list[int], list[list[int]]]],
]:
    """Prepares input features for DeepCDR."""

    # 1. extract gene expression features
    exp_common_genes = exp_df.columns.intersection(gene_list)

    # log transform raw TPM values
    exp_feat: pd.DataFrame = np.log2(exp_df[exp_common_genes] + 1)
    exp_feat = exp_feat.sort_index()

    num_missing_exp = len(set(gene_list).difference(exp_common_genes))
    if num_missing_exp > 0:
        warnings.warn(f"Found {num_missing_exp} missing expression features.")

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
    drug_to_smiles = drug_info_df[["Name", "CanonicalSMILES"]]
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


@click.command()
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory where outputs should be saved.",
)
@click.option(
    "--gene-list-path",
    type=str,
    required=True,
    help="Path to Cancer Gene Census gene list pickle file.",
)
@click.pass_obj
def command(
    store: LazyDataStore,
    output_dir: str,
    gene_list_path: str,
) -> None:
    """Creates the DeepCDR input files."""
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    click.echo("Generating DeepCDR inputs...")
    exp_feat, mut_feat, mol_feat = prepare_deepcdr_inputs(
        exp_df=store.cell_exp,
        mut_df=store.cell_mut,
        drug_info_df=store.drug_info,
        gene_list=io.read_pickled_list(gene_list_path),
    )

    click.echo("Saving DeepCDR inputs...")
    exp_feat.to_csv(output_dir / "FeatureGeneExpressionTPMLogp1.csv")
    mut_feat.to_csv(output_dir / "FeatureSomaticMutationsPositionEncoded.csv")

    with open(output_dir / "FeatureConvMol.pkl", "wb") as fh:
        pickle.dump(mol_feat, fh)
