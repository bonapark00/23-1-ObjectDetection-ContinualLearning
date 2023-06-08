import os
import json
from functools import lru_cache
from itertools import product
from typing import Sequence, Dict
import requests
import zipfile
import torchvision.transforms as T
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from utils.preprocess_clad import *
from typing import List, Callable, Dict, Any
import torch

def load_label_img_dic(ann_file):
    with open(ann_file, 'r') as f:
        data = json.load(f)
    data_infos=[]

    class_names = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")
    index=0

    for img_info in data['frames']:

        bboxes=[]
        labels=[]

        for label in img_info['labels']:
            bbox = label['box2d']
            bboxes.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
            labels.append(class_names.index(label['category']))
        
        data_infos.append({
            'name':img_info['name'],
            'videoName':img_info['videoName'],
            'bboxes':bboxes,
            'labels':labels
        })

      
    
    return data_infos

    



def get_sample_objects(labels):
    boxes=[]

    for label in labels:
        boxes.append(label['box2d'])
    
    class_names = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")

    target={"boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor([class_names.index(label['category']) for label in labels], dtype=torch.int64)}
    
    return target

def get_shift_datalist(data_type: str='train'):

    datalist=[]
    obj_properties=['image_id','category_id','bbox']





    
    





