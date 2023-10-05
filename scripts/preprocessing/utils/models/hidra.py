"""HiDRA preprocessing utilities."""

from __future__ import annotations

import click
import pickle
import warnings

import numpy as np
import pandas as pd
import typing as t

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem as AllChem
from cdrpy.util import io

if t.TYPE_CHECKING:
    from make_inputs import LazyDataStore


def read_hidra_gmt(
    gmt_file_path: str | Path, gene_symbol_map: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """Reads the HiDRA genesets from the GMT file."""
    gene_sets = io.read_gmt(gmt_file_path)

    if gene_symbol_map is not None:
        for name, gene_set in gene_sets.items():
            gene_sets[name] = [gene_symbol_map.get(g, g) for g in gene_set]

    return gene_sets


def prepare_hidra_inputs(
    exp_df: pd.DataFrame,
    exp_gene_sets: dict[str, list[str]],
    drug_info_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """Prepares inputs features for HiDRA.

    Parameters
    ----------
        exp_df:
        exp_gene_sets:
        drug_info_df:

    Returns
    -------
    """

    # 1. harmonize the gene sets and the expression
    gene_list = set([g for gs in exp_gene_sets.values() for g in gs])
    common_genes = sorted(list(gene_list.intersection(exp_df.columns)))

    # log transform TPM values
    exp_feat: pd.DataFrame = np.log2(exp_df[common_genes] + 1)

    num_missing_exp = len(set(gene_list).difference(common_genes))
    if num_missing_exp > 0:
        warnings.warn(f"Found {num_missing_exp} missing expression features.")

    gene_sets = {}
    for pathway, genes in exp_gene_sets.items():
        gene_sets[pathway] = sorted([x for x in genes if x in common_genes])

    # 2. generate Morgan fingerpritns
    drug_to_smiles = drug_info_df[["Name", "CanonicalSMILES"]]
    drug_to_morgan = {}
    for drug_name, smiles in drug_to_smiles.itertuples(index=False):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        drug_to_morgan[drug_name] = list(fp)

    mol_feat = pd.DataFrame.from_dict(drug_to_morgan, orient="index")

    return exp_feat, gene_sets, mol_feat


@click.command()
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory where outputs should be saved.",
)
@click.option(
    "--gmt-path",
    type=str,
    required=True,
    help="Path to the HiDRA geneset .gmt file.",
)
@click.pass_obj
def command(
    store: LazyDataStore,
    output_dir: str,
    gmt_path: str,
) -> None:
    """Creates the HiDRA input files."""
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    geneset_dict = read_hidra_gmt(
        gmt_path, gene_symbol_map=HIDRA_GENE_SYMBOL_MAP
    )

    click.echo("Generating HiDRA inputs...")
    exp_feat, gene_set_meta, mol_feat = prepare_hidra_inputs(
        exp_df=store.cell_exp,
        exp_gene_sets=geneset_dict,
        drug_info_df=store.drug_info,
    )

    click.echo("Saving HiDRA inputs...")
    exp_feat.to_csv(output_dir / "FeatureGeneExpressionTPMLogp1.csv")
    mol_feat.to_csv(output_dir / "FeatureMorganFingerprints512Bit.csv")

    with open(output_dir / "MetaGenesetDict.pkl", "wb") as fh:
        pickle.dump(gene_set_meta, fh)


HIDRA_GENE_SYMBOL_MAP = {
    # manual curation of missing gene symbols from Cell Model Passports
    "AARS": "AARS1",
    "ABP1": "AOC1",
    "ACCN1": "ASIC2",
    "ACPP": "ACP3",
    "ACPT": "ACP4",
    "ADC": "AZIN2",
    "ADRBK1": "GRK2",
    "ADRBK2": "GRK3",
    "ADSS": "ADSS2",
    "ADSSL1": "ADSS1",
    "AGPAT6": "GPAT4",
    "AGPAT9": "GPAT3",
    "ALPPL2": "ALPG",
    "ASAH2C": "ASAH2B",
    "ATP5A1": "ATP5F1A",
    "ATP5B": "ATP5F1B",
    "ATP5C1": "ATP5F1C",
    "ATP5D": "ATP5F1D",
    "ATP5E": "ATP5F1E",
    "ATP5F1": "ATP5PB",
    "ATP5G1": "ATP5MC1",
    "ATP5G1P5": "ATP5MC1P5",
    "ATP5G2": "ATP5MC2",
    "ATP5G3": "ATP5MC3",
    "ATP5H": "ATP5PD",
    "ATP5I": "ATP5ME",
    "ATP5J": "ATP5PF",
    "ATP5J2": "ATP5MF",
    "ATP5L": "ATP5MG",
    "ATP5O": "ATP5PO",
    "ATP6": "MT-ATP6",
    "ATP8": "MT-ATP8",
    "B3GNT1": "B4GAT1",
    "BAI1": "ADGRB1",
    "BCMO1": "BCO1",
    "BECN1P1": "BECN2",
    "C17orf48": "ADPRM",
    "C9orf95": "NMRK1",
    "CARS": "CARS1",
    "CHP": "CHP1",
    "COX1": "MT-CO1",
    "COX2": "MT-CO2",
    "COX3": "MT-CO3",
    "CSDA": "YBX3",
    "CTPS": "CTPS1",
    "CTSL1": "CTSL",
    "CTSL2": "CTSV",
    "CYTB": "MT-CYB",
    "DAK": "TKFC",
    "DARS": "DARS1",
    "EPRS": "EPRS1",
    "ERBB2IP": "ERBIN",
    "ERO1L": "ERO1A",
    "FAM125A": "MVB12A",
    "FAM125B": "MVB12B",
    "FIGF": "VEGFD",
    "FUK": "FCSK",
    "G6PC": "G6PC1",
    "GALNTL1": "GALNT16",
    "GALNTL2": "GALNT15",
    "GALNTL4": "GALNT18",
    "GARS": "GARS1",
    "GSTT1": "GSTT1",
    "GUCY1A3": "GUCY1A1",
    "GUCY1B3": "GUCY1B1",
    "H2AFB1": "H2AB1",
    "H2AFB2": "H2AB2",
    "H2AFB3": "H2AB3",
    "H2AFJ": "H2AJ",
    "H2AFV": "H2AZ2",
    "H2AFX": "H2AX",
    "H2AFY": "MACROH2A1",
    "H2AFY2": "MACROH2A2",
    "H2AFZ": "H2AZ1",
    "H2BFM": "H2BW2",
    "H2BFWT": "H2BW1",
    "H3.X": "H3Y2",
    "H3.Y": "H3Y1",
    "H3F3A": "H3-3A",
    "H3F3B": "H3-3B",
    "H3F3C": "H3-5",
    "HARS": "HARS1",
    "HIST1H2AA": "H2AC1",
    "HIST1H2AB": "H2AC4",
    "HIST1H2AC": "H2AC6",
    "HIST1H2AD": "H2AC7",
    "HIST1H2AE": "H2AC8",
    "HIST1H2AG": "H2AC11",
    "HIST1H2AH": "H2AC12",
    "HIST1H2AI": "H2AC13",
    "HIST1H2AJ": "H2AC14",
    "HIST1H2AK": "H2AC15",
    "HIST1H2AL": "H2AC16",
    "HIST1H2AM": "H2AC17",
    "HIST1H2APS6": "H2ACP1",
    "HIST1H2BA": "H2BC1",
    "HIST1H2BB": "H2BC3",
    "HIST1H2BC": "H2BC4",
    "HIST1H2BD": "H2BC5",
    "HIST1H2BE": "H2BC6",
    "HIST1H2BF": "H2BC7",
    "HIST1H2BG": "H2BC8",
    "HIST1H2BH": "H2BC9",
    "HIST1H2BI": "H2BC10",
    "HIST1H2BJ": "H2BC11",
    "HIST1H2BK": "H2BC12",
    "HIST1H2BL": "H2BC13",
    "HIST1H2BM": "H2BC14",
    "HIST1H2BN": "H2BC15",
    "HIST1H2BO": "H2BC17",
    "HIST1H3A": "H3C1",
    "HIST1H3B": "H3C2",
    "HIST1H3C": "H3C3",
    "HIST1H3D": "H3C4",
    "HIST1H3E": "H3C6",
    "HIST1H3F": "H3C7",
    "HIST1H3G": "H3C8",
    "HIST1H3H": "H3C10",
    "HIST1H3I": "H3C11",
    "HIST1H3J": "H3C12",
    "HIST1H4A": "H4C1",
    "HIST1H4B": "H4C2",
    "HIST1H4C": "H4C3",
    "HIST1H4D": "H4C4",
    "HIST1H4E": "H4C5",
    "HIST1H4F": "H4C6",
    "HIST1H4G": "H4C7",
    "HIST1H4H": "H4C8",
    "HIST1H4I": "H4C9",
    "HIST1H4J": "H4C11",
    "HIST1H4K": "H4C12",
    "HIST1H4L": "H4C13",
    "HIST2H2AA3": "H2AC18",
    "HIST2H2AA4": "H2AC19",
    "HIST2H2AB": "H2AC21",
    "HIST2H2AC": "H2AC20",
    "HIST2H2BE": "H2BC21",
    "HIST2H2BF": "H2BC18",
    "HIST2H3A": "H3C15",
    "HIST2H3C": "H3C14",
    "HIST2H3D": "H3C13",
    "HIST2H4A": "H4C14",
    "HIST2H4B": "H4C15",
    "HIST3H2A": "H2AC25",
    "HIST3H2BB": "H2BC26",
    "HIST3H3": "H3-4",
    "HIST4H4": "H4C16",
    "IARS": "IARS1",
    "IL28A": "IFNL2",
    "IL28B": "IFNL3",
    "IL28RA": "IFNLR1",
    "IL29": "IFNL1",
    "IL8": "CXCL8",
    "INADL": "PATJ",
    "KARS": "KARS1",
    "LARS": "LARS1",
    "MARS": "MARS1",
    "MLLT4": "AFDN",
    "MRE11A": "MRE11",
    "MRVI1": "IRAG1",
    "MUT": "MMUT",
    "NARS": "NARS1",
    "NAT6": "NAA80",
    "ND1": "MT-ND1",
    "ND2": "MT-ND2",
    "ND3": "MT-ND3",
    "ND4": "MT-ND4",
    "ND4L": "MT-ND4L",
    "ND5": "MT-ND5",
    "ND6": "MT-ND6",
    "NGFRAP1": "BEX3",
    "NHP2L1": "SNU13",
    "NT5C3": "NT5C3A",
    "OR5R1": "OR8U3",
    "OR8G2": "OR8G2P",
    "PAK7": "PAK5",
    "PAPD7": "TENT4A",
    "PARK2": "PRKN",
    "PIDD": "PIDD1",
    "PKM2": "PKM",
    "PPAP2A": "PLPP1",
    "PPAP2B": "PLPP3",
    "PPAP2C": "PLPP2",
    "PPBPL1": "PPBPP1",
    "PPYR1": "NPY4R",
    "PRHOXNB": "URAD",
    "PRUNE": "PRUNE1",
    "PTPLA": "HACD1",
    "PTPLB": "HACD2",
    "PVRL1": "NECTIN1",
    "PVRL2": "NECTIN2",
    "PVRL3": "NECTIN3",
    "PVRL4": "NECTIN4",
    "PYCRL": "PYCR3",
    "QARS": "QARS1",
    "RARS": "RARS1",
    "RFWD2": "COP1",
    "RQCD1": "CNOT9",
    "SARS": "SARS1",
    "SC5DL": "SC5D",
    "SEPT5": "SEPTIN5",
    "SETD8": "KMT5A",
    "SF3B14": "SF3B6",
    "SGOL1": "SGO1",
    "SHFM1": "SEM1",
    "SKIV2L2": "MTREX",
    "SRPR": "SRPRA",
    "SUV420H1": "KMT5B",
    "SUV420H2": "KMT5C",
    "TARS": "TARS1",
    "TARSL2": "TARS3",
    "TCEB1": "ELOC",
    "TCEB2": "ELOB",
    "THOC4": "ALYREF",
    "TMEM173": "STING1",
    "TMSL3": "TMSB4XP8",
    "TROVE2": "RO60",
    "TSTA3": "GFUS",
    "WARS": "WARS1",
    "WBSCR17": "GALNT17",
    "WBSCR22": "BUD23",
    "WHSC1": "NSD2",
    "WHSC1L1": "NSD3",
    "YARS": "YARS1",
    "ZAK": "MAP3K20",
    "ZFYVE20": "RBSN",
    "ZNRD1": "POLR1H",
}
