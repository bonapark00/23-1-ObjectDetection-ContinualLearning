#/bin/bash

# CIL CONFIG
NOTE="default" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="finetune"
DATASET="clad" # clad, shift
SEED="1"

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
    SCRIPT_NAME="main.py"
else
    SCRIPT_NAME="main_shift.py"
fi

# Check if debug mode is on
if [ "$1" == "debug" ]; then
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
--online_iter $ONLINE_ITER --eval_period $EVAL_PERIOD $DEBUG

# python $SCRIPT_NAME --mode $MODE --dataset $DATASET \
# --batchsize $BATCHSIZE --temp_batchsize $TEMP_BATCHSIZE \
# --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
# --seed_num $RND_SEED --eval_period $EVAL_PERIOD --note $NOTE

