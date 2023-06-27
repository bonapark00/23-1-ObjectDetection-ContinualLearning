#/bin/bash

BATCHSIZE=4
EPOCHS=2

python upperbound_shift.py --batchsize $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 20 --upperbound --debug