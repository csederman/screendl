"""Preprocessing functionality for Cell Model Passports data."""

from __future__ import annotations

import pandas as pd
import pandas._typing as pdt
import typing as t

from dataclasses import dataclass
from pathlib import Path

from ..utils import filter_by_value_counts


StrOrPath = t.Union[str, Path]
FilePathOrBuff = t.Union[pdt.FilePath, pdt.ReadCsvBuffer[bytes], pdt.ReadCsvBuffer[str]]


@dataclass(repr=False)
class CMPData:
    """Container for Cell Model Passports data sources."""

    exp: pd.DataFrame
    meta: pd.DataFrame
    cnv: pd.DataFrame | None = None
    mut: pd.DataFrame | None = None


def load_cmp_mutations(vcf_dir: StrOrPath) -> pd.DataFrame:
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


def load_cmp_expression(file_path: FilePathOrBuff) -> pd.DataFrame:
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


def load_cmp_copy_number(file_path: FilePathOrBuff) -> pd.DataFrame:
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
    exp_path: FilePathOrBuff,
    meta_path: FilePathOrBuff,
    cnv_path: FilePathOrBuff | None = None,
    vcf_dir: StrOrPath | None = None,
) -> CMPData:
    """Loads the raw Cell Model Passports Data.

    Parameters
    ----------
        exp_path: Path to the raw expression data.
        meta_path: Path to the raw cell line metadata.
        cnv_path: Path to the raw copy number data.
        vcf_dir: Directory containing the WES VCF files.

    Returns
    -------
        The raw CMPData object.
    """
    if isinstance(vcf_dir, str):
        vcf_dir = Path(vcf_dir)

    exp_data = load_cmp_expression(exp_path)
    meta_data = pd.read_csv(meta_path)

    cnv_data = None if cnv_path is None else load_cmp_copy_number(cnv_path)
    mut_data = None if vcf_dir is None else load_cmp_mutations(vcf_dir)

    return CMPData(exp_data, meta_data, cnv_data, mut_data)


def clean_cmp_data(
    data: CMPData,
    min_cells_per_cancer_type: int = 20,
    required_info_columns: t.List[str] | None = None,
    cancer_type_blacklist: t.List[str] | None = None,
) -> CMPData:
    """Harmonizes Cell Model Passports data.

    Parameters
    ----------
        data: The raw CMPData object.
        min_cells_per_cancer_type: Min number of cell lines per cancer type.
        required_info_columns: Required sample metadata columns.
        cancer_type_blacklist: List of cancer types to exclued.

    Returns
    -------
        The harmonized CMPData object.

    """
    if required_info_columns is not None:
        data.meta = data.meta.dropna(subset=required_info_columns)

    if cancer_type_blacklist is not None:
        data.meta = data.meta[~data.meta["cancer_type"].isin(cancer_type_blacklist)]

    data.meta = data.meta.drop_duplicates(subset="model_id")

    # retain cell lines with all feature types
    common_samples = set(data.meta["model_id"]).intersection(data.exp.index)
    if data.cnv is not None:
        common_samples = common_samples.intersection(data.cnv.index)
    if data.mut is not None:
        common_samples = common_samples.intersection(data.mut["model_id"])
    data.meta = data.meta[data.meta["model_id"].isin(common_samples)]

    # filter cancer types with fewer than `min_cells_per_cancer_type` cells
    data.meta = filter_by_value_counts(
        data.meta, "cancer_type", min_cells_per_cancer_type
    )

    # harmonize features and metadata
    final_samples = sorted(list(data.meta["model_id"]))
    data.exp = data.exp.loc[final_samples]
    if data.cnv is not None:
        data.cnv = data.cnv.loc[final_samples]
    if data.mut is not None:
        data.mut = data.mut[data.mut["model_id"].isin(final_samples)]
        data.mut = data.mut.sort_values("model_id")

    return data


def load_and_clean_cmp_data(
    exp_path: FilePathOrBuff,
    meta_path: FilePathOrBuff,
    cnv_path: FilePathOrBuff | None = None,
    vcf_dir: StrOrPath | None = None,
    min_cells_per_cancer_type: int = 20,
    required_info_columns: t.List[str] | None = None,
    cancer_type_blacklist: t.List[str] | None = None,
) -> CMPData:
    """Loads and cleans the raw CMP data.

    Parameters
    ----------
        exp_path: Path to the raw expression data.
        meta_path: Path to the raw cell line metadata.
        cnv_path: Path to the raw copy number data.
        vcf_dir: Directory containing the WES VCF files.
        min_cells_per_cancer_type: Min number of cell lines per cancer type.
        required_info_columns: Required sample metadata columns.
        cancer_type_blacklist: List of cancer types to exclued.

    Returns
    -------
        The cleaned CMPData object.
    """
    cmp_data = load_cmp_data(exp_path, meta_path, cnv_path, vcf_dir)
    cmp_data = clean_cmp_data(
        cmp_data,
        min_cells_per_cancer_type=min_cells_per_cancer_type,
        required_info_columns=required_info_columns,
        cancer_type_blacklist=cancer_type_blacklist,
    )
    return cmp_data
