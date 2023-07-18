#!/bin/bash

# Update the package lists
apt-get update -y

# Install necessary packages
apt-get install -y git gcc g++
apt-get install unzip

apt install libgl1-mesa-glx -y
apt-get install libglib2.0-0 -y

# Install Python packages
pip install pycocotools matplotlib opencv-contrib-python 
pip install faiss
pip install faiss-cpu --no-cache
pip install h5py
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Give execute permissions to the update_torchvision.sh script
chmod +x scripts/update_torchvision.sh

# Run another shell script
./scripts/update_torchvision.sh
