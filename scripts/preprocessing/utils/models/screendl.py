"""Preprocessing utilities for ScreenDL."""

from __future__ import annotations

import click
import warnings

import pandas as pd
import numpy as np
import typing as t

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem as AllChem
from sklearn.preprocessing import OneHotEncoder
from cdrpy.util import io

if t.TYPE_CHECKING:
    from make_inputs import LazyDataStore

from .common import compute_copy_number_ratios


def prepare_screendl_inputs(
    exp_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    cell_info_df: pd.DataFrame,
    drug_info_df: pd.DataFrame,
    exp_gene_list: t.Iterable[str],
    cnv_gene_list: t.Iterable[str],
    mut_gene_list: t.Iterable[str],
    min_mut_cells_per_gene: int = 10,
    min_cells_per_ct: int = 5,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Prepares feature inputs for ScreenDL.

    Parameters
    ----------
        exp_df:
        cnv_df:
        mut_df:
        cell_info_df:
        drug_info_df:
        exp_gene_list:
        cnv_gene_list:
        mut_gene_list:
        min_mut_cells_per_gene:

    Returns
    -------
    """

    # 1. extract gene expression features
    exp_common_genes = exp_df.columns.intersection(exp_gene_list)

    # log transform raw TPM values
    exp_feat: pd.DataFrame = np.log2(exp_df[exp_common_genes] + 1)
    exp_feat = exp_feat.sort_index(axis=1)

    num_missing_exp = len(set(exp_gene_list).difference(exp_common_genes))
    if num_missing_exp > 0:
        warnings.warn(f"Found {num_missing_exp} missing expression features.")

    # 2. extract copy number features
    cnv_common_genes = cnv_df.columns.intersection(cnv_gene_list)
    cnv_feat = compute_copy_number_ratios(
        cnv_df=cnv_df[cnv_common_genes], cell_info_df=cell_info_df
    )

    # log transform copy number ratios
    cnv_feat = np.log2(cnv_feat + 1).sort_index(axis=1)

    num_missing_cnv = len(set(cnv_gene_list).difference(cnv_common_genes))
    if num_missing_cnv > 0:
        warnings.warn(f"Found {num_missing_cnv} missing copy number features.")

    # 3. extract somatic mutation features
    mut_df_common = mut_df[mut_df["gene_symbol"].isin(mut_gene_list)]
    mut_feat = (
        mut_df_common[mut_df_common["protein_mutation"] != "-"]
        .groupby(["model_id", "gene_symbol"])
        .size()
        .unstack()
        .fillna(0)
        .clip(upper=1)
        .astype(np.float32)
    )

    # filter genes mutated in less than `min_mut_cells_per_gene` cells
    mut_counts = mut_feat.sum()
    keep_mut_genes = mut_counts[mut_counts >= min_mut_cells_per_gene].index
    mut_feat = mut_feat[keep_mut_genes].sort_index(axis=1)

    num_missing_mut = len(set(mut_gene_list).difference(mut_feat))
    if num_missing_mut > 0:
        warnings.warn(f"Found {num_missing_mut} missing mutation features.")

    # 4. generate cancer ontology features
    enc = OneHotEncoder(sparse_output=False)
    ct_feat = enc.fit_transform(cell_info_df[["cancer_type"]])
    ct_feat = pd.DataFrame(
        ct_feat, index=list(cell_info_df["model_id"]), columns=enc.categories_
    )
    ct_feat = ct_feat.sort_index()

    # filter out cancer types with fewer than `min_cells_per_ct` cell lines
    ct_counts = ct_feat.sum()
    keep_cts = ct_counts[ct_counts >= min_cells_per_ct].index
    ct_feat = ct_feat[keep_cts]

    # 5. generate Morgan fingerprints (drug features)
    drug_to_smiles = drug_info_df[["Name", "CanonicalSMILES"]]
    drug_to_morgan = {}
    for drug_name, smiles in drug_to_smiles.itertuples(index=False):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        drug_to_morgan[drug_name] = list(fp)

    mol_feat = pd.DataFrame.from_dict(drug_to_morgan, orient="index")

    return exp_feat, cnv_feat, mut_feat, ct_feat, mol_feat


@click.command()
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory where outputs should be saved.",
)
@click.option(
    "--exp-gene-path",
    type=str,
    required=True,
    help="Path to expression gene list.",
)
@click.option(
    "--cnv-gene-path",
    type=str,
    required=True,
    help="Path to copy number gene list.",
)
@click.option(
    "--mut-gene-path",
    type=str,
    required=True,
    help="Path to copy number gene list.",
)
@click.pass_obj
def command(
    store: LazyDataStore,
    output_dir: str,
    exp_gene_path: str,
    cnv_gene_path: str,
    mut_gene_path: str,
) -> None:
    """Creates the ScreenDL input feature files"""
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    click.echo("Generating ScreenDL inputs...")
    exp_feat, cnv_feat, mut_feat, ct_feat, mol_feat = prepare_screendl_inputs(
        store.cell_exp,
        store.cell_cnv,
        store.cell_mut,
        store.cell_info,
        store.drug_info,
        exp_gene_list=io.read_pickled_list(exp_gene_path),
        cnv_gene_list=io.read_pickled_list(cnv_gene_path),
        mut_gene_list=io.read_pickled_list(mut_gene_path),
        min_mut_cells_per_gene=10,
    )

    click.echo("Saving ScreenDL inputs...")
    exp_feat.to_csv(output_dir / "FeatureGeneExpressionTPMLogp1.csv")
    cnv_feat.to_csv(output_dir / "FeatureCopyNumberRatioLogp1.csv")
    mut_feat.to_csv(output_dir / "FeatureSomaticMutations.csv")
    ct_feat.to_csv(output_dir / "FeatureCancerTypeOntologyOHE.csv")
    mol_feat.to_csv(output_dir / "FeatureMorganFingerprints1024Bit.csv")
