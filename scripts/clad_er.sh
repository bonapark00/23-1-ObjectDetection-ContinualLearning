#/bin/bash

# CIL CONFIG
NOTE="clad_er" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="clad_er"
DATASET="clad" # cifar10, cifar100, tinyimagenet, imagenet
SEEDS="1 2 3"


if [ "$DATASET" == "clad" ]; then
    MEM_SIZE=150 ONLINE_ITER=1
    MODEL_NAME="faster_rcnn"
    BATCHSIZE=4

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --model_name $MODEL_NAME --dataset $DATASET \
    --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --seed_num $RND_SEED

     
    python eval.py --mode $MODE \
    --dataset $DATASET --batchsize $BATCHSIZE \
    --seed_num $RND_SEED
done
