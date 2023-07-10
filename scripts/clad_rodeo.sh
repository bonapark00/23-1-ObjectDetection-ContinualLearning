#/bin/bash

# CIL CONFIG
NOTE="clad_rodeo" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="rodeo"
DATASET="clad" # cifar10, cifar100, tinyimagenet, imagenet
SEEDS="3"
EVAL_PERIOD=100


if [ "$DATASET" == "clad" ]; then
    MEM_SIZE=150 ONLINE_ITER=1
    MODEL_NAME="fast_rcnn"
    BATCHSIZE=16
    TEMP_BATCHSIZE=0
    PRETRAIN_TASK_NUM=2
    CODEBOOK_SIZE=32

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
    --seed_num $RND_SEED --pretrain_task_num $PRETRAIN_TASK_NUM \
    --codebook_size $CODEBOOK_SIZE --eval_period $EVAL_PERIOD
done