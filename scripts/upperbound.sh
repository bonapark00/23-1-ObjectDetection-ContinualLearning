#/bin/bash

BATCHSIZE=16
EPOCHS=16

python upperbound.py --batch_size $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 100 --upperbound