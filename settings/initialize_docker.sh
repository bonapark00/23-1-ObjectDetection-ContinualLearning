#!/bin/bash

docker run --gpus all -it --ipc=host --net=host -v /home/vision/sw:/workspace/sw --name clad_test khs8157/iblurry:latest
