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
from configuration.clad_meta import SODA_DOMAINS
from configuration.clad_meta import CLADD_TRAIN_VAL_DOMAINS
from utils.preprocess_clad import *
from typing import List, Callable, Dict, Any
import torch
import random

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


def create_match_dict_fn_img(match_dict: Dict[Any, Any]):
    """
    Creates a method that returns true if the image specified by the img_id
    is in the specified domain of the given match_dict.
    
    Args:
        match_dict: dictionary that should match the objects
    
    Return:
        match_fn: a function that evaluates to true if the object is from the given date
    """

    def match_fn(img_id: int, img_dic: Dict[str, Dict]) -> bool:
        img_annot = img_dic[img_id]
        for key, value in match_dict.items():
            if isinstance(value, List):
                if img_annot[key] not in value:
                    return False
            else:
                if img_annot[key] != value:
                    return False
        else:
            return True

    return match_fn


def remove_empty_images(img_ids: list, obj_dict):
    """  
    [Required because torchvision models can't handle empty lists for bbox in targets
    """
    
    non_empty_images = set()
    for obj in obj_dict.values():
        non_empty_images.add(obj["image_id"])
    
    img_ids = [img_id for img_id in img_ids if img_id in non_empty_images]
    return img_ids


def get_matching_detection_info(annot_file: str, match_fn: Callable):
    """
    Creates CLAD task info set according to match_fn
    
    Args:
        annot_file: annotation file from root
        match_fn: A function that takes a sample, the obj and img dicts and return T/F if a sample should be
                     in the dataset
    Return: 
        info set of a single task
    """
    # print(f'Creating info set for {annot_file}')
    obj_dict, img_dic = load_obj_img_dic(annot_file)
    img_ids = [image for image in img_dic if match_fn(image, img_dic)]
    img_ids = remove_empty_images(img_ids, obj_dict)
    
    return {'img_ids': img_ids, 'annot_file': annot_file}


def get_clad_trainval(root='./dataset', val_proportion= 0.1):
    """
    Selects images which satisfies CLAD domains, and creates info sets of CLAD
    For instance train info set is formed as below
    
        e.g) train info set = [ {'img_ids': [100,124,142], 'annot_file': } ... ]
        
    Args:
        root: root path to the dataset
        val_proportion: proportion of val/train
        
    Return: 
        train info set, val info set
    """
    train_info = []
    val_info = []
    
    # Split from SODA10M, not CLAD-D
    splits = ['train', 'val', 'val', 'val']
    match_fns = [create_match_dict_fn_img(train_domain) for train_domain in CLADD_TRAIN_VAL_DOMAINS]
    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations')
    trainval_info = [get_matching_detection_info(annot_file=os.path.join(annot_file, f'instance_{split}.json'),
                                                  match_fn = match_fn) for
                                                  match_fn, split in zip(match_fns, splits)]

    # Split trainval_info to train_info + val_info
    for index,item in enumerate(trainval_info):
        
        all_img_ids = item['img_ids']
        annot_file = item['annot_file']
        task_num = index+1
        cut_off = int((1.0 - val_proportion) * len(all_img_ids))
        
        train_img_ids = all_img_ids[:cut_off]
        val_img_ids = all_img_ids[cut_off: ]
        
        train_info.append({'img_ids': train_img_ids, 'annot_file': annot_file, 'task_num': task_num, 'split': splits[index]})
        val_info.append({'img_ids': val_img_ids, 'annot_file': annot_file, 'task_num': task_num, 'split': splits[index]})
    
    
    return train_info, val_info

def get_sample_objects(objects: dict):
            '''
            save objects from a single data.
            modify bbox and return target, which can directly put into model. 
            '''
            boxes = []
            for bbox in objects['bbox']:
                  # Convert from x, y, h, w to x0, y0, x1, y1
                  boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                  
            # Targets should all be tensors
            target = \
                  {"boxes": torch.as_tensor(boxes, dtype=torch.float32), 
                   "labels": torch.as_tensor(objects['category_id'],dtype=torch.int64), 
                   "image_id": torch.as_tensor(objects['image_id'], dtype=torch.int64),
                   "area": torch.as_tensor(objects['area']), 
                   "iscrowd": torch.as_tensor(objects['iscrowd'], dtype=torch.int64)}

            return target

        
def get_clad_datalist(data_type='train', val_proportion = 0.1, dataset_root='./dataset'):
    
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
    
    datalist = []
    train_info, val_info = get_clad_trainval(dataset_root, val_proportion=val_proportion)
    selected_info = train_info if data_type == 'train' else val_info
    
    obj_properties = ['image_id','category_id','bbox', 'area', 'id', 'truncated', 'occluded', 'iscrowd']
    empty_obj_properties = dict([(obj_prop, []) for obj_prop in obj_properties])
    
    for item in selected_info:
        img_ids, annot_file, task_num, split = item['img_ids'], item['annot_file'], item['task_num'], item['split']
        obj_container = dict.fromkeys(img_ids)
        obj_dict, img_dict = load_obj_img_dic(annot_file)

        for obj in obj_dict.values():
            if obj['image_id'] in img_ids:
                #the first object from img_ids appears 
                if obj_container[obj['image_id']] == None:
                    file_name = img_dict[obj['image_id']]['file_name']
                    obj_container[obj['image_id']] = \
                        {'file_name': file_name, 
                         'img_info': img_dict[int(obj['image_id'])],
                         'objects': empty_obj_properties,
                         'task_num': task_num, 
                         'split': split }
                    
                    #clear dictionary for later items
                    obj_container[obj['image_id']]['objects']['image_id'] = obj['image_id']
                    empty_obj_properties = dict([(obj_prop, []) for obj_prop in obj_properties])
                    
                for item in obj_properties[1:]:
                    obj_container[obj['image_id']]['objects'][item].append(obj[item])
        
        datalist.extend(obj_container.values())
        
    return datalist


def visualize_bbox_ssls(image, boxes, labels, ssls, save_path=None):
    """
    Visualize bounding boxes on the image.

    Args:
        image (torch.Tensor): The image tensor in shape [C, H, W].
        boxes (torch.Tensor): The bounding boxes tensor in shape [N, 4].
        labels (torch.Tensor): The labels tensor in shape [N].
        label_map (dict, optional): A mapping from label indices to label names. 
            Default is None, in which case labels are converted to string as is.
        save_path (str, optional): The path to save the visualized image. 
            If None, the image will be displayed without saving. Default is None.
        ssls: torch.Tensor: The ssls tensor in shape [2000, 4].
    """
    # Convert the image tensor to numpy array
    # Also, convert it from [C, H, W] to [H, W, C] and normalize if necessary
    image = image.permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]

    # Create a new figure and a subplot (this is needed to add patches - bounding boxes)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Get image shape
    H, W, C = image.shape

    # Iterate over all boxes
    for i in range(boxes.shape[0]):
        # Create a rectangle patch
        x1, y1, x2, y2 = boxes[i]
        box = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.5, edgecolor='r', facecolor='none')
        
        # Add the rectangle patch (bounding box) to the subplot
        ax.add_patch(box)

        # Get label
        label = str(labels[i].item())

        # Put text (label)
        plt.text(x1, y1, label, color='white')
    
    # Convert torch tensor to numpy array
    ssls = ssls.numpy()
    color = [random.randint(0, 255) for j in range(0, 3)]
    for (x, y, x2, y2) in ssls:
        # draw the region proposal bounding box on the image
        box = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=0.5, edgecolor='b', facecolor='none')
        ax.add_patch(box)

        # Check if the box is out of image boundary
        if x < 0 or y < 0 or x2 > W or y2 > H:
            print("Warning: SSL box out of image boundary!")
            print("H, W, C: ", H, W, C)
            print("Image shape: ", image.shape)
            print("SSL box: ", (x, y, x2, y2))


    # Save the image or display it
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()