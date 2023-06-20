#/bin/bash

BATCHSIZE=4
NUM_ITERS=2
SEED=1
EVAL_PERIOD=80

python lowerbound.py --batch_size $BATCHSIZE \
    --num_iters $NUM_ITERS --seed $SEED \
    --eval_period $EVAL_PERIOD --debug
