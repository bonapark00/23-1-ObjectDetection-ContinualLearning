#/bin/bash

BATCHSIZE=4
EPOCHS=3

python upperbound.py --batch_size $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 10 --debug