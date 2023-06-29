#!/bin/bash

# Define the directories
src_dir="./detection"
dest_dir="/opt/conda/lib/python3.7/site-packages/torchvision/models/detection"
dest_dir2="venv/lib/python3.10/site-packages/torchvision/models/detection"

# Delete the existing destination directory
rm -rf "$dest_dir"

# Copy the source directory to the destination
cp -R "$src_dir" "$dest_dir"