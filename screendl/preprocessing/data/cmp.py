"""Preprocessing functionality for Cell Model Passports data."""

from __future__ import annotations

import pandas as pd

from pathlib import Path


def load_cmp_mutations(vcf_dir: str | Path) -> pd.DataFrame:
    """Parses a single Cell Model Passports WES VCF file.

    Parameters
    ----------
        vcf_dir: directory containing the WES VCF files.

    Returns
    -------
        A `pd.DataFrame` instance containing the extracted mutations.
    """
    if isinstance(vcf_dir, str):
        vcf_dir = Path(vcf_dir)

    def parse_vcf(file_path: Path) -> pd.DataFrame:
        with open(file_path, "rt") as fh:
            cell_id, source, *_ = file_path.stem.split("_")
            muts = []
            for line in fh:
                if line.startswith("#"):  # skip VCF header
                    continue
                chrom, pos, _, ref, alt, _, _, info, *_ = line.split("\t")
                info = info.split(";")

                # extract VAGrENT default classification
                vd = list(filter(lambda x: str(x).startswith("VD="), info))[0]
                vd = vd.lstrip("VD=").split("|")
                gene, _, rna_mut, cdna_mut, protein_mut, *_ = vd

                # extract VAGrENT variant consequence
                vc = list(filter(lambda x: str(x).startswith("VC="), info))[0]
                effect = vc.lstrip("VC=")

                # extract driver and predisposition status
                drv = "DRV" in info
                cpv = "CPV" in info

                muts.append(
                    [
                        cell_id,
                        chrom,
                        pos,
                        ref,
                        alt,
                        gene,
                        rna_mut,
                        cdna_mut,
                        protein_mut,
                        drv,
                        cpv,
                        effect,
                        source,
                    ]
                )

        return pd.DataFrame(
            muts,
            columns=[
                "model_id",
                "chrom",
                "pos",
                "ref",
                "alt",
                "gene_symbol",
                "rna_mutation",
                "cdna_mutation",
                "protein_mutation",
                "cancer_driver",
                "cancer_predisposition_variant",
                "effect",
                "source",
            ],
        )

    mut_dfs = []
    for file_path in vcf_dir.glob("*.vcf"):
        mut_dfs.append(parse_vcf(file_path))

    return pd.concat(mut_dfs).drop_duplicates()


def load_cmp_expression(file_path: str | Path) -> pd.DataFrame:
    """Loads the raw Cell Model Passports gene expression file."""
    df = (
        pd.read_csv(file_path, skiprows=[1, 2, 3, 4], low_memory=False)
        .drop(columns="model_id")
        .set_index("Unnamed: 1")
        .rename_axis(columns="model_id", index="gene_symbol")
        .sort_index()
        .transpose()
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .dropna(axis=1)
    )

    return df.loc[:, ~df.columns.duplicated()]


def load_cmp_copy_number(file_path: str | Path) -> pd.DataFrame:
    """Loads the raw Cell Model Passports copy number file."""
    df = (
        pd.read_csv(file_path, skiprows=[0, 2, 3], index_col=0)
        .rename_axis(columns="model_id", index="gene_symbol")
        .sort_index()
        .transpose()
        .dropna(axis=1)
    )

    return df.loc[:, ~df.columns.duplicated()]


def load_cmp_data(
    exp_path: str | Path,
    meta_path: str | Path,
    cnv_path: str | Path | None = None,
    vcf_dir: str | Path | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None
]:
    """Loads the raw Cell Model Passports Data.

    Parameters
    ----------
        exp_path: Path to the raw expression data.
        cnv_path: Path to the raw copy number data.
        vcf_dir: Directory containing the WES VCF files.
        meta_path: Path to the raw cell line metadata.

    Returns
    -------
        A tuple (exp_df, meta_df, cnv_df, mut_df) of `pd.DataFrame`
            instances.
    """
    if isinstance(vcf_dir, str):
        vcf_dir = Path(vcf_dir)

    exp_df = load_cmp_expression(exp_path)
    meta_df = pd.read_csv(meta_path)
    cnv_df = None if cnv_path is None else load_cmp_copy_number(cnv_path)
    mut_df = None if vcf_dir is None else load_cmp_mutations(vcf_dir)

    return exp_df, meta_df, cnv_df, mut_df


def harmonize_cmp_data(
    exp_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cnv_df: pd.DataFrame | None = None,
    mut_df: pd.DataFrame | None = None,
    min_cells_per_cancer_type: int = 20,
    required_info_columns: list[str] | None = None,
    cancer_type_blacklist: list[str] | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None
]:
    """Harmonizes Cell Model Passports data.

    Parameters
    ----------
        exp_df: The raw expression data.
        meta_df: The raw cell line metadata.
        cnv_df: The raw copy number data.
        mut_df: The raw somatic mutation data.
        min_cells_per_cancer_type: Min number of cell lines per cancer type.
        required_info_columns: Columns in `meta_df` which must not
            contain NaN values.
        cancer_type_blacklist: List of cancer types to exclued.

    Returns
    -------
        A tuple of (exp_df, cnv_df, mut_df, mut_df_pos, meta_df)
            `pd.DataFrame` instances.

    """
    if required_info_columns is not None:
        meta_df = meta_df.dropna(subset=required_info_columns)

    if cancer_type_blacklist is not None:
        meta_df = meta_df[~meta_df["cancer_type"].isin(cancer_type_blacklist)]

    meta_df = meta_df.drop_duplicates(subset="model_id")

    # retain cell lines with all feature types
    common_cells = set(meta_df["model_id"]).intersection(exp_df.index)
    if cnv_df is not None:
        common_cells = common_cells.intersection(cnv_df.index)
    if mut_df is not None:
        common_cells = common_cells.intersection(mut_df["model_id"])

    meta_df = meta_df[meta_df["model_id"].isin(common_cells)]

    # filter cancer types with fewer than `min_cells_per_cancer_type` cells
    ct_counts = meta_df["cancer_type"].value_counts()
    keep_cts = ct_counts[ct_counts >= min_cells_per_cancer_type].index
    meta_df = meta_df[meta_df["cancer_type"].isin(keep_cts)]

    # harmonize features and metadata
    final_cells = sorted(list(meta_df["model_id"]))
    exp_df = exp_df.loc[final_cells]

    if cnv_df is not None:
        cnv_df = cnv_df.loc[final_cells]

    if mut_df is not None:
        mut_df = mut_df[mut_df["model_id"].isin(final_cells)]
        mut_df = mut_df.sort_values("model_id")

    return exp_df, meta_df, cnv_df, mut_df
