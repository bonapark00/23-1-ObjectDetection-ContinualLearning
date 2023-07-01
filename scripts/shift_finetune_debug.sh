#/bin/bash

# CIL CONFIG
NOTE="shift_finetune" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="finetune"
DATASET="shift" # cifar10, cifar100, tinyimagenet, imagenet, clad, shift
SEEDS="2"
EVAL_PERIOD=40


if [ "$DATASET" == "shift" ]; then
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
    python main_shift.py --mode $MODE \
    --model_name $MODEL_NAME --dataset $DATASET \
    --batchsize $BATCHSIZE --temp_batchsize $TEMP_BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --seed_num $RND_SEED --eval_period $EVAL_PERIOD --debug
     
    # python eval.py --mode $MODE \
    # --model_name $MODEL_NAME --dataset $DATASET \
    # --batchsize $BATCHSIZE --temp_batchsize $TEMP_BATCHSIZE \
    # --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    # --seed_num $RND_SEED
    
done