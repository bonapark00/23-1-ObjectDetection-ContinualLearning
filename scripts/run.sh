#!/bin/bash

# Prepare to run experiments
# 1. Modify {METHOD_NAME}.sh file
#   - Set NOTE
#   - Set DATASET
#   - Set MEMORY_SIZE (different from below)
# 2. Modify this file
#   - Set METHOD_NAME
#   - Set MEMORY_SIZE (for just logging purpose - you must set manually for each .sh file)
#   - Set NOTE (if you want to add some note to the experiment)
# 3. Comment out the line belows to modify seed values
# 4. Run this file

METHOD_NAME="der"
MEMORY_SIZE=500
NOTE="default" # You can write NOTE here

# If NOTE is not empty, add underscore before it
NOTE_SUFFIX=${NOTE:+_$NOTE}

# Run 3 seeds in multiple GPUs
CUDA_VISIBLE_DEVICES=0 ./scripts/${METHOD_NAME}.sh 1 > logs/shift_${METHOD_NAME}_${MEMORY_SIZE}_1${NOTE_SUFFIX}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 ./scripts/${METHOD_NAME}.sh 2 > logs/shift_${METHOD_NAME}_${MEMORY_SIZE}_2${NOTE_SUFFIX}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 ./scripts/${METHOD_NAME}.sh 3 > logs/shift_${METHOD_NAME}_${MEMORY_SIZE}_3${NOTE_SUFFIX}.txt 2>&1 &
