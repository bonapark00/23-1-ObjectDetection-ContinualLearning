#/bin/bash

BATCHSIZE=4
EPOCHS=2
SEED=1

python upperbound.py --batch_size $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 20 --seed_num $SEED --debug