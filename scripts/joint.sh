#/bin/bash

BATCHSIZE=16
EPOCHS=16
SEED=1

python upperbound.py --batch_size $BATCHSIZE \
--num_epochs $EPOCHS --eval_period 100 --seed_num $SEED