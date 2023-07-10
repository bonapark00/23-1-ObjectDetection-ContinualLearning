#/bin/bash

# CIL CONFIG
NOTE="clad_rodeo" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="rodeo"
DATASET="clad" # cifar10, cifar100, tinyimagenet, imagenet
SEEDS="1 2 3"
EVAL_PERIOD=30

if [ "$DATASET" == "clad" ]; then
    MEM_SIZE=4 ONLINE_ITER=1
    MODEL_NAME="fast_rcnn"
    BATCHSIZE=2
    TEMP_BATCHSIZE=0

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
