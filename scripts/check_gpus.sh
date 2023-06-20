#!/bin/bash

# Get the PID of each GPU process
pids=$(nvidia-smi | awk 'BEGIN{FS=" "}{if (NR>=10) print $4}')
echo $pids
# for pid in $pids; do
#     # Show details of the process
#     echo "Details for PID $pid:"
#     ps -p $pid -o user,etime,cmd= -ww
# done