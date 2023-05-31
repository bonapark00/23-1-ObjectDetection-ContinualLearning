import json
import os
from functools import lru_cache
from itertools import product
from typing import Sequence, Dict
import requests
import zipfile
import torchvision.transforms as T
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from clad_meta import SODA_DOMAINS
import PIL
import numpy as np
import torch
import time
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CladDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.images = []
        self.targets = []
        
        self.data_convert(self.dataset)
        
    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        image =  self.images[idx]
        target = {"boxes": self.targets[idx]['boxes'],
                "labels": self.targets[idx]['labels']}
        
        return image, target

    def data_convert(self, dataset):
        for sample in dataset:
            img_path = os.path.join("dataset", "SSLAD-2D",'labeled',sample['split'],sample['file_name'])
            image = PIL.Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)     

            
            boxes = []
            for bbox in sample['objects']['bbox']:
            # Convert from x, y, h, w to x0, y0, x1, y1
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                
            target =\
                {"boxes": torch.as_tensor(boxes, dtype=torch.float32), 
                "labels": torch.as_tensor(sample['objects']['category_id'],dtype=torch.int64), 
                "image_id": torch.as_tensor(sample['objects']['image_id'], dtype=torch.int64),
                "area": torch.as_tensor(sample['objects']['area']), 
                "iscrowd": torch.as_tensor(sample['objects']['iscrowd'], dtype=torch.int64)}
            self.images.append(image)
            self.targets.append(target)
        
@lru_cache(4)
def load_obj_img_dic(annot_file: str):
    with open(annot_file, "r") as f:
        annot_json = json.load(f)

    obj_dic = {obj["id"]: obj for obj in annot_json["annotations"]}
    img_dic = {img["id"]: img for img in annot_json["images"]}

    img_dic, obj_dic = check_if_time_date_included(img_dic, obj_dic, annot_file)
    correct_data(img_dic, obj_dic)

    return obj_dic, img_dic


def create_domain_dicts(domains: Sequence[str]) -> Dict:
    """
    Creates dictionaries for the products of all values of the given domains.
    """
    try:
        domain_values = product(*[SODA_DOMAINS[domain] for domain in domains])
    except KeyError:
        raise KeyError(f'Unkown keys, keys should be in {list(SODA_DOMAINS.keys())}')

    domain_dicts = []
    for dv in domain_values:
        domain_dicts.append({domain: value for domain, value in zip(domains, dv)})

    return domain_dicts


def correct_data(img_dic, obj_dic):
    """
    Not all data is in the correct format in the original datasets, this will correct it to
    the correct timestamps.
    """

    for key in img_dic:
        img_dic[key]['time'] = img_dic[key]['time'].strip()
        img_dic[key]['date'] = img_dic[key]['date'].strip()

        if img_dic[key]["time"] == '145960':
            img_dic[key]["time"] = '150000'
        elif img_dic[key]["time"] == '102360':
            img_dic[key]["time"] = '102400'
        elif img_dic[key]["time"] == '1221831':
            img_dic[key]["time"] = '122131'
        elif img_dic[key]["time"] == '1420214':
            img_dic[key]["time"] = '142021'
        elif img_dic[key]["time"] == '10:00':
            img_dic[key]["time"] = '100000'
        elif img_dic[key]["time"] == '13:00':
            img_dic[key]["time"] = '130000'
        elif img_dic[key]["time"] == ' 13:00':
            img_dic[key]["time"] = '130000'
        elif img_dic[key]["time"] == '12:00':
            img_dic[key]["time"] = '120000'
        elif img_dic[key]["time"] == '1111523':
            img_dic[key]["time"] = '111152'

        if img_dic[key]["date"] == "201810152":
            img_dic[key]["date"] = "20181015"
        elif img_dic[key]["date"] == "50181015":
            img_dic[key]["date"] = "20181015"


def check_if_time_date_included(img_dict, obj_dict, annot_file):
    """
    The original annotations don't include the timestamps, so test if they are detected and download correct
    annotations if necessary.
    """

    ex_img = next(iter(img_dict.values()))

    if 'time' not in ex_img:
        print('Timestamps missing, downloading annotations with timestamps')
        directory = annot_file.rsplit('/', 1)[0]
        file_path = os.path.join(directory, 'time_stamps.zip')

        r = requests.get("https://cloud.esat.kuleuven.be/index.php/s/SBQ8HPXFJFxrbN7/download/time_annotations.zip")
        with open(file_path, 'wb') as f:
            f.write(r.content)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory)

        os.remove(file_path)

        with open(annot_file, "r") as f:
            annot_json = json.load(f)

        obj_dict = {obj["id"]: obj for obj in annot_json["annotations"]}
        img_dict = {img["id"]: img for img in annot_json["images"]}

        return img_dict, obj_dict

    else:
        return img_dict, obj_dict


def squarify_bbox(bbox: Sequence[int]) -> Sequence[int]:
    """
    Rescales a XYWH bbox such that it is a square and padded with 5 pixels on each side. Longest side of the bbox
    is used as the side of the square.
    """
    # TODO: check whether this doesn't sometimes give weird bbox if W >> H or vice versa
    # TODO: maybe this should be gray/white padded rather than extra image?

    x, y, w, h = bbox
    padding = 5

    size = w if w > h else h
    size = size // 2
    size += padding
    if size > 539:
        size = 539  # Size of images is 1920 * 1080: size can be max 539

    cx, cy = x + w // 2, y + h // 2
    xb, xe, yb, ye = cx - size, cx + size, cy - size, cy + size

    if xb < 0:
        xe += -1 * xb
        xb = 0
    if xe >= 1920:
        xb -= (xe - 1919)
        xe = 1919
    if yb < 0:
        ye += -1 * yb
        yb = 0
    if ye >= 1080:
        yb -= (ye - 1079)
        ye = 1079
    return yb, ye, xb, xe

data_transform = transforms.Compose([
        transforms.ToTensor()
])

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def apply_nms(prediction, threshold):
    # torchvision returns the indices of the boxes to keep
    keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], threshold)
    
    final_prediction = prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_and_save(image_tensor, target_dict, save_path='output.png'):
    boxes = target_dict['boxes']
    labels = target_dict['labels']
    
    # Convert tensor to PIL Image
    image = T.ToPILImage()(image_tensor)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch for each bounding box
    for i, box in enumerate(boxes):
        box = box.detach().numpy()  # Convert tensor to numpy array
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=0.4, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Put label text
        label = labels[i].item()
        plt.text(box[0], box[1], label, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

    # Save the figure
    plt.savefig(save_path)
