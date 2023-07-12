#!/bin/bash

# Update the package lists
apt-get update -y

# Install necessary packages
apt-get install -y git gcc

# Install Python packages
pip install pycocotools matplotlib opencv-contrib-python 
pip install faiss
pip install faiss-cpu --no-cache

# Give execute permissions to the update_torchvision.sh script
chmod +x scripts/update_torchvision.sh

# Run another shell script
./scripts/update_torchvision.sh
