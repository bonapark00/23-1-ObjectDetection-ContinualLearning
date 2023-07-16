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
    """
    Load label and image information from json file.
    Args:
        ann_file: path of json file
    Return:
        data_infos: list of data information
    """

    with open(ann_file, 'r') as f:
        data = json.load(f)
    data_infos=[]

    class_names = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")

    for img_info in data['frames']:
        bboxes=[]
        labels=[]

        for label in img_info['labels']:
            bbox = label['box2d']
            bboxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
            labels.append(class_names.index(label['category']) + 1)
        
        if(len(bboxes)==0):
            # If there is no object in the image, skip the image
            continue

        else:
            data_infos.append({
            'name':img_info['name'],
            'videoName':img_info['videoName'],
            'bboxes':bboxes,
            'labels':labels,
            'attributes':img_info['attributes'],
            })

    return data_infos


def get_sample_objects(objects):
    boxes=[]

    for bbox in objects['bbox']:
       boxes.append([bbox[0],bbox[1],bbox[2],bbox[3]])
    
    # area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
    # breakpoint()

    

    target={"boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(objects["category_id"], dtype=torch.int64)
            }
    
    return target

def get_shift_datalist(domain_dict, task_num, data_type: str='train', root='./dataset'):
      
    """
    Creates datalist, so that single data info (sample) can be enumerated as stream.
    All data from SHIFT are combined inorder.

    domain_dict: dictionary that contains domain information
        {
            'weather_coarse': 'rainy',
            'timeofday_coarse': 'night',
            'weather_fine': 'heavy rain',
            'timeofday_fine': 'night',
            'view': 'front',
            'town': '10HD',
            'sun_altitude_angle': '-2',
            'cloudiness': '100.0',
            'precipitation': '100.0',
            'precipitation_deposits': '100.0',
            'wind_intensity': '100.0',
            'sun_azimuth_angle': '0.0',
            'fog_density': '7.0',
            'fog_distance': '0.75',
            'wetness': '0.0',
            'fog_falloff': '0.1'
        }
    
    task_num: ID of task corresponding to domain_dict
    
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


    datalist = []
    obj_properties = ['category_id','bbox']
    path = os.path.join(root, "SHIFT_dataset/discrete/images")
    # path = "./dataset/SHIFT_dataset/discrete/images"
    data_infos = load_label_img_dic(f"{path}/{data_type}/front/det_2d.json")
    class_list = []

    if domain_dict is not None:
        for data_info in data_infos:
            # Filter out data that doesn't belong to the domain
            if not all([data_info['attributes'][key] == domain_dict[key] for key in domain_dict.keys()]):
                continue
            
            datalist.append(
                {
                'file_name': f"{path}/{data_type}/front/{data_info['videoName']}/{data_info['name']}",
                'img_info':data_info['attributes'],
                'objects':{"category_id": data_info['labels'],"bbox": data_info['bboxes']},
                'task_num':task_num,
                'split':data_type,
                }
            )
    else:
        for data_info in data_infos:
            datalist.append(
                {
                'file_name': f"{path}/{data_type}/front/{data_info['videoName']}/{data_info['name']}",
                'img_info':data_info['attributes'],
                'objects':{"category_id": data_info['labels'],"bbox": data_info['bboxes']},
                'task_num':task_num,
                'split':data_type,
                }
            )
    return datalist

def collate_fn(batch):
    # Collate function for dataloader
    return tuple(zip(*batch))






    
    





