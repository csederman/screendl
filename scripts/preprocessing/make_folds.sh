#!/usr/bin/env bash

set -u

root_dir="/scratch/ucgd/lustre-work/marth/u0871891/datastore"
output_dir="${root_dir}/inputs/CellModelPassportsGDSCv2/splits"
labels_path="${root_dir}/inputs/CellModelPassportsGDSCv2/LabelsLogIC50.csv"
cell_info_path="${root_dir}/datasets/CellModelPassportsGDSCv2/CellLineAnnotations.csv"

mkdir -p $output_dir

python ./make_folds.py --output-dir=$output_dir --labels-path=$labels_path --cell-info-path=$cell_info_path