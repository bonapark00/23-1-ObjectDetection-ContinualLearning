#/bin/bash

BATCHSIZE=16
EPOCHS=8

python upperbound.py --batch_size $BATCHSIZE \
--num_epochs $EPOCHS  --eval_period 100