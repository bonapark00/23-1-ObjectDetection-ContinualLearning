#/bin/bash

BATCHSIZE=16
EPOCHS=16
DATASET="shift"

python upperbound_shift.py --batchsize $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 1000 --upperbound --dataset $DATASET