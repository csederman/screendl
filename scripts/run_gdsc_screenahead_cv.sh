#!/bin/bash
# Native ScreenDL + ScreenAhead on GDSC (CellModelPassports-GDSCv1v2-HCI, 10-fold CV, 20 drugs screened).
#
# Outputs (per fold under hydra multirun subdir 0..9):
#   predictions.csv      — baseline ScreenDL (train/val/test)
#   predictions_sa.csv   — ScreenAhead with was_screened column
#   scores.json
#
# Submit: sbatch scripts/run_gdsc_screenahead_cv.sh
# Single fold: SPLIT_IDS=3 sbatch scripts/run_gdsc_screenahead_cv.sh

#SBATCH --job-name=screendl_sa_gdsc
#SBATCH --time=80:00:00
#SBATCH --account=marth-rw
#SBATCH --partition=marth-rw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --chdir=/scratch/ucgd/lustre-labs/marth/scratch/u1521317/screendl
#SBATCH -o /scratch/ucgd/lustre-labs/marth/scratch/u1521317/screendl_runs/slurm-%j.out-%N
#SBATCH -e /scratch/ucgd/lustre-labs/marth/scratch/u1521317/screendl_runs/slurm-%j.err-%N

set -euo pipefail

SCREENDL_ROOT="/scratch/ucgd/lustre-labs/marth/scratch/u1521317/screendl"
PYTHON_BIN="/uufs/chpc.utah.edu/common/HIPAA/u1521317/software/pkg/miniforge3/envs/distillmd/bin/python"
SPLIT_IDS="${SPLIT_IDS:-1,2,3,4,5,6,7,8,9,10}"

export PYTHONPATH="${SCREENDL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TF_CPP_MIN_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${SCREENDL_ROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}" /scratch/ucgd/lustre-labs/marth/scratch/u1521317/screendl_runs

cd "${SCREENDL_ROOT}"

echo "=== Native ScreenDL ScreenAhead (GDSC HCI) ==="
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "Python: ${PYTHON_BIN}"
echo "Splits: ${SPLIT_IDS}"
echo "Outputs: screendl_runs/outputs/screenahead/CellModelPassports-GDSCv1v2-HCI/ScreenDL/multiruns/<timestamp>/"

"${PYTHON_BIN}" scripts/runners/run_screenahead.py -m \
  --config-name screenahead_gdsc_hci \
  "dataset.split.id=${SPLIT_IDS}"
