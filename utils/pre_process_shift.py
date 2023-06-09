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

    



def get_sample_objects(objects):
    boxes=[]

    for bbox in objects['bbox']:
       boxes.append([bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2']])
    
    

    target={"boxes": torch.as_tensor(objects["boxes"], dtype=torch.float32),
            "labels": torch.tensor(objects["category_id"], dtype=torch.int64)}
    
    return target

def get_shift_datalist(data_type: str='train'):

      
    """
    Creates datalist, so that single data info (sample) can be enumerated as stream.
    All data from CLAD are combined inorder.
    
    Single data is formed as below
    
        e.g) {'file_name': ~.png,  
              'objects': [list of obj annotation info], 
              'task_num': 1, 
              'split': 'train'} 
    Args:
        data_type: 'train' or 'val' (should be extended further for test data)
        val_proportion: proportion of val/train
        
    Return: 
        data list: list that contains single image data.
                   data of all tasks are combined in order. Thus shouldn't be shuffled
    """


    datalist=[]
    obj_properties=['category_id','bbox']
    path="./dataset/SHIFT_dataset/discrete/images"
    data_infos=load_label_img_dic(f"{path}/{data_type}/front/det_2d.json")
    
    for data_info in data_infos:

        datalist.append(
            {
            'file_name': {f"{path}/{data_type}/front/{data_info['videoName']}/{data_info['name']}"},
            'objects':{"category_id": data_info['labels'],"bbox": data_info['bboxes']},
            'task_num':1,
            'split':data_type
            }
        )

    return datalist







    
    





