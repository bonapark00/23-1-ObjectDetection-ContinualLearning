import logging.config
import os
import random
import pdb
import pickle
from collections import defaultdict
import PIL

import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.method_manager import select_method
from clad_data import get_clad_datalist
from soda import SODADataset
import transforms as T
from clad_utils import collate_fn, data_transform, get_model_instance_segmentation

def main():
    path = './model_checkpoints/model_1ep.pth'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=7)
    model.load_state_dict(torch.load(path))
    model.eval()    
    
    dataset_task1_val = SODADataset(path="../SSLAD-2D", task_id=1,
                                  split="val", transforms=data_transform)
    
    test_dataloader = torch.utils.data.DataLoader(dataset_task1_val, batch_size=4, collate_fn=collate_fn)


if __name__ == "__main__":
    main()
