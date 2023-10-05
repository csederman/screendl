#!/usr/bin/env bash

# NOTE: consider converting this into a MakeFile

set -u

root_dir="/scratch/ucgd/lustre-work/marth/u0871891/datastore"
dataset_dir="${root_dir}/datasets/CellModelPassportsGDSCv2"
genelist_dir="${root_dir}/processed/genelists"
output_dir="${root_dir}/inputs/CellModelPassportsGDSCv2"

mkdir -p $output_dir

python ./make_inputs.py \
    --exp-path="${dataset_dir}/OmicsGeneExpressionTPM.csv" \
    --cnv-path="${dataset_dir}/OmicsTotalCopyNumber.csv" \
    --mut-path="${dataset_dir}/OmicsSomaticMutationsWithPosition.csv" \
    --cell-info-path="${dataset_dir}/CellLineAnnotations.csv" \
    --drug-info-path="${dataset_dir}/DrugAnnotations.csv" \
    --drug-resp-path="${dataset_dir}/ScreenDoseResponse.csv" \
    screendl \
        --output-dir="${output_dir}/ScreenDL" \
        --exp-gene-path="${genelist_dir}/MiniCancerGenome1815Genes.pkl" \
        --cnv-gene-path="${genelist_dir}/MiniCancerGenome1815Genes.pkl" \
        --mut-gene-path="${genelist_dir}/MiniCancerGenome1815Genes.pkl"

# python ./make_inputs.py \
#     --exp-path="${dataset_dir}/OmicsGeneExpressionTPM.csv" \
#     --cnv-path="${dataset_dir}/OmicsTotalCopyNumber.csv" \
#     --mut-path="${dataset_dir}/OmicsSomaticMutationsWithPosition.csv" \
#     --cell-info-path="${dataset_dir}/CellLineAnnotations.csv" \
#     --drug-info-path="${dataset_dir}/DrugAnnotations.csv" \
#     --drug-resp-path="${dataset_dir}/ScreenDoseResponse.csv" \
#         deepcdr \
#             --output-dir="${output_dir}/DeepCDR" \
#             --gene-list-path="${genelist_dir}/CancerGeneCensus736Genes.pkl"

# python ./make_inputs.py \
#     --exp-path="${dataset_dir}/OmicsGeneExpressionTPM.csv" \
#     --cnv-path="${dataset_dir}/OmicsTotalCopyNumber.csv" \
#     --mut-path="${dataset_dir}/OmicsSomaticMutationsWithPosition.csv" \
#     --cell-info-path="${dataset_dir}/CellLineAnnotations.csv" \
#     --drug-info-path="${dataset_dir}/DrugAnnotations.csv" \
#     --drug-resp-path="${dataset_dir}/ScreenDoseResponse.csv" \
#         dualgcn \
#             --output-dir="${output_dir}/DualGCN" \
#             --ppi-path="/scratch/ucgd/lustre-work/marth/u0871891/projects/screendl/pkg/DualGCN/data/PPI/PPI_network.txt"

# python ./make_inputs.py \
#     --exp-path="${dataset_dir}/OmicsGeneExpressionTPM.csv" \
#     --cnv-path="${dataset_dir}/OmicsTotalCopyNumber.csv" \
#     --mut-path="${dataset_dir}/OmicsSomaticMutationsWithPosition.csv" \
#     --cell-info-path="${dataset_dir}/CellLineAnnotations.csv" \
#     --drug-info-path="${dataset_dir}/DrugAnnotations.csv" \
#     --drug-resp-path="${dataset_dir}/ScreenDoseResponse.csv" \
#         hidra \
#             --output-dir="${output_dir}/HiDRA" \
#             --gmt-path="/scratch/ucgd/lustre-work/marth/u0871891/projects/screendl/pkg/HiDRA/Training/geneset.gmt"