#/bin/bash

BATCHSIZE=4
EPOCHS=2
DATASET="shift"

python upperbound_shift.py --batchsize $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 20 --upperbound --dataset $DATASET \
--debug
