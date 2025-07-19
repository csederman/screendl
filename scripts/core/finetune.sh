#!/usr/bin/env bash
# Runs fine-tuning for all ensemble members

DATASTORE="./data"
RUN_DATE="2025-06-24_09-26-35" # # replace with the date of your run
RUN_DIR="${DATASTORE}/outputs/core/CellModelPassports-GDSCv1v2-HCI/ScreenDL/multiruns/${RUN_DATE}"
RUN_CMD="python scripts/core/finetune.py"
RUN_CFG="./conf/core/finetune.yaml"

find $RUN_DIR -mindepth 1 -maxdepth 1 -type d -name '[0-9]' -print0 \
    | xargs -0 -I {} $RUN_CMD --dir {} --config $RUN_CFG