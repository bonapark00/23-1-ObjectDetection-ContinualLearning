#/bin/bash

BATCHSIZE=4
EPOCHS=4

python upperbound.py --batch_size $BATCHSIZE \
--num_epochs $EPOCHS