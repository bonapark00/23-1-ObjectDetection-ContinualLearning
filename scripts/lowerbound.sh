#/bin/bash

BATCHSIZE=16
NUM_ITERS=16
SEED=1
EVAL_PERIOD=100

python lowerbound.py --batch_size $BATCHSIZE \
    --num_iters $NUM_ITERS --seed $SEED \
    --eval_period $EVAL_PERIOD
