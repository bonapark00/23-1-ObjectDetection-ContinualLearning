#/bin/bash

# CIL CONFIG
NOTE="clad_ilod" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="ilod"
DATASET="clad" # cifar10, cifar100, tinyimagenet, imagenet
SEEDS="1"
EVAL_PERIOD=30

if [ "$DATASET" == "clad" ]; then
    MEM_SIZE=150 ONLINE_ITER=1
    MODEL_NAME="faster_rcnn"
    BATCHSIZE=4
    TEMP_BATCHSIZE=2

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --model_name $MODEL_NAME --dataset $DATASET \
    --batchsize $BATCHSIZE --temp_batchsize $TEMP_BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --seed_num $RND_SEED --eval_period $EVAL_PERIOD --debug
done

