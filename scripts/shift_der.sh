#/bin/bash

# CIL CONFIG
NOTE="shift_der" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="shift_der"
DATASET="shift" # cifar10, cifar100, tinyimagenet, imagenet
SEEDS="1"
ALPHA=0.05
BETA=0.5
THETA=1.0

if [ "$DATASET" == "shift" ]; then
    MEM_SIZE=150 ONLINE_ITER=1
    MODEL_NAME="faster_rcnn"
    BATCHSIZE=16
    TEMP_BATCHSIZE=8

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python eval_shift.py --mode $MODE \
    --model_name $MODEL_NAME --dataset $DATASET \
    --batchsize $BATCHSIZE --temp_batchsize $TEMP_BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --seed_num $RND_SEED --alpha $ALPHA --beta $BETA --theta $THETA 
done