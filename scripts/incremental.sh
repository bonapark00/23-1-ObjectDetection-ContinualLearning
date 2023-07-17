#/bin/bash

# CIL CONFIG
NOTE="incremental_default" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="der"
DATASET="clad" # clad, shift
SEED=$1 # This will now take seed value as a command line argument
CKPT_EPOCH=10

if [ "$DATASET" == "clad" ]; then
    MEM_SIZE=150 ONLINE_ITER=1
    BATCHSIZE=16
    TEMP_BATCHSIZE=8
    EVAL_PERIOD=100

elif [ "$DATASET" == "shift" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    BATCHSIZE=16
    TEMP_BATCHSIZE=8
    EVAL_PERIOD=1000

else
    echo "Undefined setting"
    exit 1
fi

if [ "$DATASET" == "clad" ]; then
    SCRIPT_NAME="main_incremental.py"
else
    SCRIPT_NAME="main_incremental_shift.py"
fi

# Check if debug mode is on
if [ "$2" == "debug" ]; then
    DEBUG="--debug"
    BATCHSIZE=4
    TEMP_BATCHSIZE=2
    EVAL_PERIOD=40
    NOTE="debug"
else
    DEBUG=""
fi

python $SCRIPT_NAME --mode $MODE --dataset $DATASET \
--seed_num $SEED --note $NOTE --batchsize $BATCHSIZE \
--temp_batchsize $TEMP_BATCHSIZE --memory_size $MEM_SIZE \
--online_iter $ONLINE_ITER --eval_period $EVAL_PERIOD --ckpt_epoch $CKPT_EPOCH $DEBUG
