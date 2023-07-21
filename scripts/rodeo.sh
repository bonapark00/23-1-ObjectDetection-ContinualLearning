#/bin/bash

# CIL CONFIG
NOTE="default_13928" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="rodeo"
DATASET="clad" # clad, shift
SEED=$1 # This will now take seed value as a command line argument

if [ "$DATASET" == "clad" ]; then
    MEM_SIZE=13928 ONLINE_ITER=1
    BATCHSIZE=4
    TEMP_BATCHSIZE=0
    EVAL_PERIOD=100
    PRETRAIN_TASK_NUM=2
    CODEBOOK_SIZE=32

# TODO: Change these values for shift dataset
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
    SCRIPT_NAME="main.py"
else
    SCRIPT_NAME="main_shift.py"
fi

# Check if debug mode is on
if [ "$2" == "debug" ]; then
    DEBUG="--debug"
    MEM_SIZE=50
    BATCHSIZE=4
    TEMP_BATCHSIZE=0
    EVAL_PERIOD=40
    NOTE="debug"
else
    DEBUG=""
fi

python $SCRIPT_NAME --mode $MODE --dataset $DATASET \
--seed_num $SEED --note $NOTE --batchsize $BATCHSIZE \
--temp_batchsize $TEMP_BATCHSIZE --memory_size $MEM_SIZE \
--online_iter $ONLINE_ITER --eval_period $EVAL_PERIOD \
--pretrain_task_num $PRETRAIN_TASK_NUM --codebook_size $CODEBOOK_SIZE $DEBUG
