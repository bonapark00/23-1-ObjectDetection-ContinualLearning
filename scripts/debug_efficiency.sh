#/bin/bash

# CIL CONFIG
NOTE="shift_baseline" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="baseline"
DATASET="shift"
SEEDS="1"
EVAL_PERIOD=1000
MEM_SIZE=150
ONLINE_ITER=1
MODEL_NAME="faster_rcnn"
BATCHSIZE=16
TEMP_BATCHSIZE=8

for RND_SEED in $SEEDS
do
    python main_shift_for_debug.py --mode $MODE \
    --model_name $MODEL_NAME --dataset $DATASET \
    --batchsize $BATCHSIZE --temp_batchsize $TEMP_BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --seed_num $RND_SEED --eval_period $EVAL_PERIOD
done
