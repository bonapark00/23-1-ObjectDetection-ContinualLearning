#/bin/bash

# CIL CONFIG
NOTE="incremental_default" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
DATASET="clad" # clad, shift
BATCHSIZE=16
NUM_EPOCHS=10
SEED_NUM=3


if [ "$DATASET" == "clad" ]; then
    SCRIPT_NAME="upperbound_clad.py"
else
    SCRIPT_NAME="upperbound_shift.py"
fi

# Check if debug mode is on
if [ "$1" == "debug" ]; then
    DEBUG="--debug"
    BATCHSIZE=4
    NUM_EPOCHS=2
    NOTE="debug"
else
    DEBUG=""
fi

python3 $SCRIPT_NAME --dataset $DATASET --batchsize $BATCHSIZE \
--num_epochs $NUM_EPOCHS --note $NOTE --seed_num $SEED_NUM $DEBUG
