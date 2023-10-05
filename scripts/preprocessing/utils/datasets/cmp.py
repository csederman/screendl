"""Preprocessing utilities for Cell Model Passports data."""

from __future__ import annotations

import pandas as pd

from pathlib import Path


def parse_cmp_wes_vcfs(vcf_dir: str | Path) -> pd.DataFrame:
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
                    {
                        "model_id": cell_id,
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                        "gene_symbol": gene,
                        "rna_mutation": rna_mut,
                        "cdna_mutation": cdna_mut,
                        "protein_mutation": protein_mut,
                        "cancer_driver": drv,
                        "cancer_predisposition_variant": cpv,
                        "effect": effect,
                        "source": source,
                    }
                )

        return pd.DataFrame(muts)

    mut_dfs = []
    for file_path in vcf_dir.glob("*.vcf"):
        mut_dfs.append(parse_vcf(file_path))

    return pd.concat(mut_dfs).drop_duplicates()


def load_cmp_data(
    exp_path: str | Path,
    cnv_path: str | Path,
    mut_path: str | Path,
    vcf_dir: str | Path,
    cell_info_path: str | Path,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Loads the raw Cell Model Passports Data.

    Parameters
    ----------
        exp_path: Path to the raw expression data.
        cnv_path: Path to the raw copy number data.
        mut_path: Path to the raw somatic mutation data.
        vcf_dir: Directory containing the WES VCF files.
        cell_info_path: Path to the raw cell line metadata.

    Returns
    -------
        A tuple of (exp_df, cnv_df, mut_df, mut_df_pos, cell_info_df)
            `pd.DataFrame` instances.

    """
    if isinstance(vcf_dir, str):
        vcf_dir = Path(vcf_dir)

    # load the raw expression data
    exp_df = (
        pd.read_csv(exp_path, skiprows=[1, 2, 3, 4])
        .drop(columns="model_id")
        .rename(columns={"Unnamed: 1": "gene_symbol"})
        .set_index("gene_symbol")
        .sort_index()
        .rename_axis(columns="model_id")
        .transpose()
        .dropna(axis=1)
    )

    # load the raw somatic mutation data
    mut_df = pd.read_csv(mut_path).drop_duplicates()
    mut_df_pos = parse_cmp_wes_vcfs(vcf_dir)
    mut_df_pos = mut_df_pos.drop_duplicates()

    # load the raw copy number data
    cnv_df = (
        pd.read_csv(cnv_path, skiprows=[0, 2, 3], index_col=0)
        .rename_axis(columns="model_id", index="gene_symbol")
        .sort_index()
        .transpose()
        .dropna(axis=1)
    )

    # load cell info
    cell_info_df = pd.read_csv(cell_info_path)

    return exp_df, cnv_df, mut_df, mut_df_pos, cell_info_df


def harmonize_cmp_data(
    exp_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    mut_df_pos: pd.DataFrame,
    cell_info_df: pd.DataFrame,
    min_cells_per_cancer_type: int = 20,
    required_info_columns: list[str] | None = None,
    cancer_type_blacklist: list[str] | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Harmonizes Cell Model Passports data.

    Parameters
    ----------
        exp_df: The raw expression data.
        cnv_df: The raw copy number data.
        mut_df: The raw somatic mutation data.
        mut_df_pos: The raw somatic mutation data extracted from the VCF files.
        cell_info_df: The raw cell line metadata.
        min_cells_per_cancer_type: Minimum number of cell lines per cancer
            type.
        required_info_columns: Columns which must not contain NaN values.
        cancer_type_blacklist: List of cancer types to exclued.

    Returns
    -------
        A tuple of (exp_df, cnv_df, mut_df, mut_df_pos, cell_info_df)
            `pd.DataFrame` instances.

    """
    if required_info_columns is None:
        required_info_columns = ["ploidy_wes", "cancer_type"]

    if cancer_type_blacklist is None:
        cancer_type_blacklist = ["Non-Cancerous", "Unknown"]

    cell_info_df = cell_info_df.dropna(subset=required_info_columns)
    cell_info_df = cell_info_df.drop_duplicates(subset="model_id")

    # retain cell lines with WES, CNV, and RNA-seq
    common_cells = set(cell_info_df["model_id"]).intersection(
        mut_df["model_id"],
        mut_df_pos["model_id"],
        exp_df.index,
        cnv_df.index,
    )
    cell_info_df = cell_info_df[cell_info_df["model_id"].isin(common_cells)]

    # filter cancer types with fewer than `min_cells_per_cancer_type` cells
    ct_counts = cell_info_df["cancer_type"].value_counts()
    keep_cts = ct_counts[ct_counts >= min_cells_per_cancer_type].index
    cell_info_df = cell_info_df[cell_info_df["cancer_type"].isin(keep_cts)]

    # harmonize the features and metadata
    final_cells = sorted(list(cell_info_df["model_id"]))

    exp_df = exp_df.loc[final_cells]
    cnv_df = cnv_df.loc[final_cells]

    mut_df = mut_df[mut_df["model_id"].isin(final_cells)]
    mut_df = mut_df.sort_values("model_id")

    mut_df_pos = mut_df_pos[mut_df_pos["model_id"].isin(final_cells)]
    mut_df_pos = mut_df_pos.sort_values("model_id")

    return exp_df, cnv_df, mut_df, mut_df_pos, cell_info_df
