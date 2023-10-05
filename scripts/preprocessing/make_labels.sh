#!/usr/bin/env bash

set -u

root_dir="/scratch/ucgd/lustre-work/marth/u0871891/datastore"
output_dir="${root_dir}/inputs/CellModelPassportsGDSCv2"
drug_resp_path="${root_dir}/datasets/CellModelPassportsGDSCv2/ScreenDoseResponse.csv"

mkdir -p $output_dir

python ./make_labels.py --output-dir=$output_dir --drug-resp-path=$drug_resp_path