#!/usr/bin/env bash

### Path to preprocessing script
GENELIST_DIR=/genelist_dir
CANDLE_PREPROCESS=/usr/local/screendl/scripts/improve/preprocess.py

if [ $# -lt 2 ] ; then
        echo "Illegal number of parameters"
        echo "CUDA_VISIBLE_DEVICES and CANDLE_DATA_DIR are required"
        exit 1
fi

if [ $# -eq 2 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift
        CMD="python ${CANDLE_PREPROCESS}"
        echo "CMD = $CMD"

elif [ $# -ge 3 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift

        # if original $3 is a file, set candle_config and passthrough $@
        if [ -f "$CANDLE_DATA_DIR/$1" ] ; then
		    echo "$1 is a file"
            CANDLE_CONFIG=$1 ; shift
            CMD="python ${CANDLE_PREPROCESS} --config_file $CANDLE_CONFIG $*"
            echo "CMD = $CMD $*"

        # else passthrough $@
        else
		echo "$1 is not a file"
                CMD="python ${CANDLE_PREPROCESS} $*"
                echo "CMD = $CMD"

        fi
fi

# Display runtime arguments
python -V

echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"

# Set up environmental variables and execute preprocessing script
EXE_DIR=$(dirname ${CANDLE_PREPROCESS})
cd "$EXE_DIR" || exit

echo "running command ${CMD}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} GENELIST_DIR=${GENELIST_DIR} $CMD

# Check if successful
exit 0