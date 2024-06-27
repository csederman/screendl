"""Preprocessing utilities for ScreenDL."""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np
import typing as t

from dataclasses import dataclass
from pathlib import Path

from cdrpy.util import io
from rdkit import Chem
from rdkit.Chem import AllChem as AllChem
from sklearn.preprocessing import OneHotEncoder

from .common import compute_copy_number_ratios


log = logging.getLogger(__name__)


def _generate_exp_features(
    exp_df: pd.DataFrame,
    exp_gene_list: t.Iterable[str],
    min_var_threshold: float | None = None,
    log_transform: bool = False,
) -> pd.DataFrame:
    """Generates gene expression inputs for ScreenDL.

    Parameters
    ----------
        exp_df: A `pd.DataFrame` instance with shape (n_samples, n_genes).
        exp_gene_list: An iterable of gene features to include.
        min_var_threshold: The minimum variance across samples for a gene to be
            included in the feature set.
        log_transform: Whether or not to log transform gene expression values.

    Returns
    -------
        A `pd.DataFrame` instance of gene expression features.
    """
    common_genes = exp_df.columns.intersection(exp_gene_list)
    exp_feat = exp_df[common_genes].sort_index(axis=1)

    num_missing = len(set(exp_gene_list).difference(common_genes))
    if num_missing > 0:
        msg = "Found {} missing exp genes in the specified gene list."
        log.warning(msg.format(num_missing))

    # if log_transform:
    #     exp_feat: pd.DataFrame = np.log2(exp_feat + 1)

    if min_var_threshold is not None:
        gene_vars = exp_feat.var()
        low_var_genes = gene_vars[gene_vars < min_var_threshold].index

        num_low_var_genes = len(low_var_genes)
        if num_low_var_genes > 0:
            exp_feat = exp_feat[exp_feat.columns.difference(low_var_genes)]
            msg = "Removed {} low variance genes from expression features."
            log.info(msg.format(num_low_var_genes))

    return exp_feat


def _generate_cnv_features(
    cnv_df: pd.DataFrame,
    cell_info_df: pd.DataFrame,
    cnv_gene_list: t.Iterable[str],
    min_var_threshold: float | None = None,
    log_transform: bool = True,
) -> pd.DataFrame:
    """Generates copy number features for ScreenDL.

    Parameters
    ----------
        cnv_df: A `pd.DataFrame` instance with shape (n_samples, n_genes).
        cell_info_df: A `pd.DataFrame` instance containing ploidy annotations.
        cnv_gene_list: An iterable of gene features to include.
        min_var_threshold: The minimum variance across samples for a gene to be
            included in the feature set.
        log_transform: Whether or not to log transform gene expression values.

    Returns
    -------
    """
    common_genes = cnv_df.columns.intersection(cnv_gene_list)
    cnv_feat = compute_copy_number_ratios(
        cnv_df=cnv_df[common_genes], cell_info_df=cell_info_df
    )
    cnv_feat = cnv_feat.sort_index(axis=1)

    num_missing = len(set(cnv_gene_list).difference(common_genes))
    if num_missing > 0:
        msg = "Found {} missing cnv genes in the specified gene list."
        log.warning(msg.format(num_missing))

    if log_transform:
        cnv_feat: pd.DataFrame = np.log2(cnv_feat + 1)

    if min_var_threshold is not None:
        gene_vars = cnv_feat.var()
        low_var_genes = gene_vars[gene_vars < min_var_threshold].index

        num_low_var_genes = len(low_var_genes)
        if num_low_var_genes > 0:
            cnv_feat = cnv_feat[cnv_feat.columns.difference(low_var_genes)]
            msg = "Removed {} low variance genes from cnv features."
            log.info(msg.format(num_low_var_genes))

    return cnv_feat


def _generate_mut_features(
    mut_df: pd.DataFrame,
    mut_gene_list: t.Iterable[str],
    min_samples_per_gene: int = 10,
) -> pd.DataFrame:
    """Generates mutation features for ScreenDL.

    Parameters
    ----------
        mut_df:
        mut_gene_list:
        min_samples_per_gene:

    Returns
    -------
    """

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

    mut_counts = mut_feat.sum()
    keep_genes = mut_counts[mut_counts >= min_samples_per_gene].index
    mut_feat = mut_feat[keep_genes].sort_index(axis=1)

    num_missing = len(set(mut_gene_list).difference(mut_feat))
    if num_missing > 0:
        msg = "Found {} missing mut genes in the specified gene list."
        log.warning(msg.format(num_missing))

    return mut_feat


def _generate_ont_features(
    cell_info_df: pd.DataFrame,
    min_samples_per_ct: int = 5,
    ct_blacklist: t.Iterable[str] | None = None,
) -> pd.DataFrame:
    """Generate cancer type ontology features for ScreenDL.

    Parameters
    ----------
        cell_info_df:
        min_samples_per_ct:
        ct_blacklist:

    Returns
    -------
    """
    enc = OneHotEncoder(sparse_output=False)
    ct_feat = enc.fit_transform(cell_info_df[["cancer_type"]])

    ct_feat = pd.DataFrame(
        ct_feat,
        columns=enc.categories_[0],
        index=list(cell_info_df["model_id"]),
    )

    # filter out cancer types with fewer than `min_samples_per_ct` cell lines
    counts = ct_feat.sum()
    keep_cts = counts[counts >= min_samples_per_ct].index
    ct_feat = ct_feat[keep_cts].sort_index(axis=1)

    # remove blacklist cancer types from the ontology encoding
    if ct_blacklist is not None:
        ct_feat = ct_feat[ct_feat.columns.difference(ct_blacklist)]

    return ct_feat


def _generate_mol_features(
    drug_info_df: pd.DataFrame, n_bits: int = 1024, radius: int = 2
) -> pd.DataFrame:
    """Generates Morgan fingerprint features for ScreenDL.

    Parameters
    ----------
        drug_info_df:
        n_bits:
        radius:

    Returns
    -------
    """
    drug_to_smiles = drug_info_df[["drug_name", "smiles"]]
    drug_to_morgan = {}
    for drug_name, smiles in drug_to_smiles.itertuples(index=False):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        drug_to_morgan[drug_name] = list(fp)

    mol_feat = pd.DataFrame.from_dict(drug_to_morgan, orient="index")
    mol_feat.columns = mol_feat.columns.astype(str)

    return mol_feat


def generate_screendl_inputs(
    cell_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
    exp_df: pd.DataFrame,
    exp_gene_list: t.Iterable[str],
    cnv_df: pd.DataFrame | None = None,
    cnv_gene_list: t.Iterable[str] | None = None,
    mut_df: pd.DataFrame | None = None,
    mut_gene_list: t.Iterable[str] | None = None,
    exp_min_var_threshold: float | None = None,
    exp_log_transform: bool = False,
    cnv_min_var_threshold: float | None = None,
    cnv_log_transform: bool = False,
    mut_min_samples_per_gene: int = 10,
    ont_min_samples_per_ct: int = 5,
    ont_ct_blacklist: t.Iterable[str] | None = None,
    mol_n_bits: int = 1024,
    mol_radius: int = 2,
) -> t.Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Prepares feature inputs for ScreenDL.

    Parameters
    ----------
        exp_df: A `pd.DataFrame` instance with shape (n_samples, n_features).
        cnv_df: A `pd.DataFrame` instance with shape (n_samples, n_features).
        mut_df: An MAF-like `pd.DataFrame` instance containing mutation data.
        cell_meta:
        drug_meta:
        exp_gene_list: An iterable of gene expression genes to include.
        cnv_gene_list:
        mut_gene_list:
        exp_min_var_threshold:
        exp_log_transform:
        cnv_min_var_threshold:
        cnv_log_transform:
        mut_min_samples_per_gene:
        ont_min_samples_per_ct:
        ont_ct_blacklist:
        mol_n_bits:
        mol_radius:

    Returns
    -------
    """

    exp_feat = _generate_exp_features(
        exp_df,
        exp_gene_list,
        min_var_threshold=exp_min_var_threshold,
        log_transform=exp_log_transform,
    )

    cnv_feat = None
    if cnv_df is not None:
        if cnv_gene_list is None:
            raise ValueError("No CNV gene list specified.")
        cnv_feat = _generate_cnv_features(
            cnv_df,
            cell_meta,
            cnv_gene_list,
            min_var_threshold=cnv_min_var_threshold,
            log_transform=cnv_log_transform,
        )

    mut_feat = None
    if mut_df is not None:
        if mut_gene_list is None:
            raise ValueError("No mut gene list specified.")
        mut_feat = _generate_mut_features(
            mut_df,
            mut_gene_list,
            min_samples_per_gene=mut_min_samples_per_gene,
        )

    ont_feat = _generate_ont_features(
        cell_meta,
        min_samples_per_ct=ont_min_samples_per_ct,
        ct_blacklist=ont_ct_blacklist,
    )

    mol_feat = _generate_mol_features(drug_meta, n_bits=mol_n_bits, radius=mol_radius)

    return exp_feat, cnv_feat, mut_feat, ont_feat, mol_feat


@dataclass
class _PathConfig:
    """Path config for ScreenDL input features."""

    exp_gene_list: str | Path
    cnv_gene_list: str | Path | None = None
    mut_gene_list: str | Path | None = None


@dataclass
class _ParamConfig:
    """Param config for ScreenDL input features.."""

    exp_min_var_threshold: float = 0.5
    exp_log_transform: bool = True
    cnv_min_var_threshold: float = 0.1
    cnv_log_transform: bool = True
    mut_min_samples_per_gene: int = 10
    ont_min_samples_per_ct: int = 5
    ont_ct_blacklist: t.List[str] | None = None
    mol_n_bits: int = 1024
    mol_radius: int = 2


@dataclass
class ScreenDLFeatureConfig:
    """ScreenDL input feature config."""

    paths: _PathConfig
    params: _ParamConfig


def generate_and_save_screendl_inputs(
    cfg: ScreenDLFeatureConfig,
    out_dir: Path,
    cell_meta: pd.DataFrame,
    drug_meta: pd.DataFrame,
    exp_df: pd.DataFrame,
    cnv_df: pd.DataFrame | None = None,
    mut_df: pd.DataFrame | None = None,
) -> None:
    """Generates and saves ScreenDL input features."""
    out_dir.mkdir(exist_ok=True)

    paths = cfg.paths
    params = cfg.params

    exp_gene_list = io.read_pickled_list(paths.exp_gene_list)

    cnv_gene_list = None
    cnv_gene_list_path = paths.get("cnv_gene_list")
    if cnv_gene_list_path is not None:
        cnv_gene_list = io.read_pickled_list(cnv_gene_list_path)

    mut_gene_list = None
    mut_gene_list_path = paths.get("mut_gene_list")
    if mut_gene_list_path is not None:
        mut_gene_list = io.read_pickled_list(mut_gene_list_path)

    exp_feat, cnv_feat, mut_feat, ct_feat, mol_feat = generate_screendl_inputs(
        cell_meta,
        drug_meta,
        exp_df=exp_df,
        exp_gene_list=exp_gene_list,
        cnv_df=cnv_df,
        cnv_gene_list=cnv_gene_list,
        mut_df=mut_df,
        mut_gene_list=mut_gene_list,
        exp_min_var_threshold=params.exp_min_var_threshold,
        exp_log_transform=params.exp_log_transform,
        cnv_min_var_threshold=params.cnv_min_var_threshold,
        cnv_log_transform=params.cnv_log_transform,
        mut_min_samples_per_gene=params.mut_min_samples_per_gene,
        ont_min_samples_per_ct=params.ont_min_samples_per_ct,
        ont_ct_blacklist=params.ont_ct_blacklist,
        mol_n_bits=params.mol_n_bits,
        mol_radius=params.mol_radius,
    )

    exp_feat.to_csv(out_dir / "FeatureGeneExpression.csv")
    ct_feat.to_csv(out_dir / "FeatureCancerTypeOntology.csv")
    mol_feat.to_csv(out_dir / "FeatureMorganFingerprints.csv")

    if cnv_feat is not None:
        cnv_feat.to_csv(out_dir / "FeatureCopyNumberRatio.csv")

    if mut_feat is not None:
        mut_feat.to_csv(out_dir / "FeatureSomaticMutations.csv")
