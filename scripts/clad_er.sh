#/bin/bash

# CIL CONFIG
NOTE="clad_er" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="clad_er"
DATASET="clad" # cifar10, cifar100, tinyimagenet, imagenet
SEEDS="1"


if [ "$DATASET" == "clad" ]; then
    MEM_SIZE=150 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=4; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default"

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python train.py --mode $MODE \
    --model_name $MODEL_NAME \
    --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER
done
