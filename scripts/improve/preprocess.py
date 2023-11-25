#!/usr/bin/env python
"""ScreenDL data preprocessing."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import io
import sys
import argparse
import pickle
import requests
import zipfile

import pandas as pd
import typing as t

from cdrpy.data import Dataset
from cdrpy.feat.encoders import PandasEncoder
from cdrpy.util import io as io_utils

from screendl.preprocessing.data import gdsc, cmp, pubchem
from screendl.preprocessing.data import harmonize_cmp_gdsc_data
from screendl.preprocessing.models import generate_screendl_inputs
from screendl.preprocessing.splits import kfold_split_generator

from constants import GENELIST_CHOICES


CMP_EXP_PATH = "https://cog.sanger.ac.uk/cmp/download/rnaseq_all_20220624.zip"
CMP_META_PATH = "https://cog.sanger.ac.uk/cmp/download/model_list_20230923.csv"
GDSC_SCREEN_PATH = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx"


def get_gdsc_data() -> gdsc.GDSCData:
    """Downloads and preprocesses the GDSC data."""
    gdsc_drug_meta = gdsc.fetch_gdsc_drug_info()
    gdsc_data = gdsc.load_and_clean_gdsc_data(GDSC_SCREEN_PATH, gdsc_drug_meta)

    # get drug properties from PubCHEM
    pchem_ids = list(gdsc_data.meta["pubchem_id"])
    pchem_props = pubchem.fetch_pubchem_properties(pchem_ids)
    pchem_props = pchem_props.rename(columns={"CanonicalSMILES": "smiles"})
    pchem_props["CID"] = pchem_props["CID"].astype(str)

    gdsc_data.meta = gdsc_data.meta.merge(
        pchem_props, left_on="pubchem_id", right_on="CID"
    )

    return gdsc_data


def get_cmp_data() -> cmp.CMPData:
    """Downloads and preprocesses the Cell Model Passports data."""
    response = requests.get(CMP_EXP_PATH)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zfh:
        assert "rnaseq_tpm_20220624.csv" in zfh.namelist()
        exp_data = cmp.load_cmp_expression(zfh.open("rnaseq_tpm_20220624.csv"))

    meta_data = pd.read_csv(CMP_META_PATH)

    cmp_data = cmp.clean_cmp_data(
        cmp.CMPData(exp_data, meta_data),
        min_cells_per_cancer_type=20,
        required_info_columns=["cancer_type", "ploidy_wes"],
    )

    return cmp_data


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and formats the labels."""
    col_map = {"model_id": "cell_id", "drug_name": "drug_id", "ln_ic50": "label"}
    df_valid = df[~df["ln_ic50"].isin(gdsc.INVALID_RESPONSE_VALUES)]
    df_valid["id"] = range(df_valid.shape[0])
    return df_valid[["id", *col_map]].rename(columns=col_map)


def clean_drug_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the drug meta data."""
    cols = ["drug_name", "targets", "target_pathway", "pubchem_id", "smiles"]
    return df[cols].set_index("drug_name").rename_axis(index="drug_id")


def clean_sample_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the sample meta data."""
    cols = ["model_id", "tissue", "cancer_type", "cancer_type_detail"]
    return df[cols].set_index("model_id").rename_axis(index="cell_id")


def generate_splits(
    sample_meta: pd.DataFrame, labels: pd.DataFrame
) -> t.Generator[t.Dict[str, t.List[int]]]:
    """Generates sample blind train/val/test splits."""
    sample_ids = pd.Series(sample_meta.index)
    sample_groups = sample_meta["cancer_type"]

    gen = kfold_split_generator(
        sample_ids, sample_groups, n_splits=10, random_state=1771
    )

    for split in gen:
        train_ids = labels[labels["cell_id"].isin(split["train"])]["id"]
        val_ids = labels[labels["cell_id"].isin(split["val"])]["id"]
        test_ids = labels[labels["cell_id"].isin(split["test"])]["id"]

        yield {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}


def parse_args(args: t.List[str]) -> argparse.Namespace:
    """Parse command line arguements."""

    parser = argparse.ArgumentParser(description="Generate dataset for ScreenDL")

    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        # default="/improve_data_dir/screendl/Data",
        help="Output dir to store the generated files.",
    )
    parser.add_argument(
        "--genelist",
        type=str,
        required=True,
        choices=list(GENELIST_CHOICES),
        default="mcg",
        help="The genelist for selection of gene expression features.",
    )

    args = parser.parse_args(args)
    return args


def preprocess(args: t.List[str]) -> None:
    """"""
    args = parse_args(args)

    out_root = args.outdir
    os.makedirs(out_root, exist_ok=True)

    # fetch and clean the raw data
    gdsc_data = get_gdsc_data()
    cmp_data = get_cmp_data()

    # harmonize the raw data
    cmp_data, gdsc_data = harmonize_cmp_gdsc_data(cmp_data, gdsc_data)

    # generate inputs
    genelist_path = os.path.join("/genelist_dir", GENELIST_CHOICES[args.genelist])
    exp_feat, *_, mol_feat = generate_screendl_inputs(
        cmp_data.meta,
        gdsc_data.meta,
        cmp_data.exp,
        exp_gene_list=io_utils.read_pickled_list(genelist_path),
    )

    # extract labels and metadata
    labels = clean_labels(gdsc_data.resp)
    drug_meta = clean_drug_meta(gdsc_data.meta)
    cell_meta = clean_sample_meta(cmp_data.meta)

    D = Dataset(
        labels,
        cell_encoders={"exp": PandasEncoder(exp_feat)},
        drug_encoders={"mol": PandasEncoder(mol_feat)},
        cell_meta=cell_meta,
        drug_meta=drug_meta,
        name="CellModelPassportsGDSCv2",
    )

    D.save(os.path.join(out_root, f"{D.name}.h5"))

    # generate train/val/test splits
    split_dir = os.path.join(out_root, "splits", "tumor_blind")
    os.makedirs(split_dir, exist_ok=True)

    split_gen = generate_splits(D.cell_meta, D.obs)
    for i, split in enumerate(split_gen, 1):
        split_file = os.path.join(split_dir, f"fold_{i}.pkl")
        with open(split_file, "wb") as fh:
            pickle.dump(split, fh)


if __name__ == "__main__":
    preprocess(sys.argv[1:])
