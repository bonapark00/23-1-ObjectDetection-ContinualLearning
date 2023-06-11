#/bin/bash

BATCHSIZE=4
EPOCHS=2

python upperbound.py --batch_size $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 20 --upperbound --debug