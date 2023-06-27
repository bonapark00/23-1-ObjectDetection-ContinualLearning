import torch
import numpy as np
from configuration import config
from torchvision import transforms
from torch.utils.data import random_split
from collections import defaultdict
from tqdm import tqdm
import os
import PIL
import logging
from utils.preprocess_clad import get_clad_datalist, collate_fn
from utils.data_loader_clad import SODADataset
from utils.method_manager import select_method
from torch.utils import tensorboard
import torchvision
import cv2
import random 
from fast_rcnn.fast_rcnn import fastrcnn_resnet50_fpn


# #selective search
# def ssl(image_path):
#     image = cv2.imread(image_path)
#     ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#     ss.setBaseImage(image)
#     ss.switchToSelectiveSearchQuality()
#     rects = ss.process()
#     rects = rects[:2000]

#     return rects
def get_sample_objects(objects: dict):
    '''
    save objects from a single data.
    modify bbox and return target, which can directly put into model. 
    '''
    boxes = []
    for bbox in objects['bbox']:
            # Convert from x, y, h, w to x0, y0, x1, y1
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Targets should all be tensors
    target = \
            {"boxes": torch.as_tensor(boxes, dtype=torch.float32).to(device), 
            "labels": torch.as_tensor(objects['category_id'],dtype=torch.int64).to(device)}

    return target

def main():
    cur_train_datalist = get_clad_datalist('train')
    
    train_task = [
        cur_train_datalist[0:4470],
        cur_train_datalist[4470:5799],
        cur_train_datalist[5799:7278],
        cur_train_datalist[7278:7802]
    ]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fastrcnn_resnet50_fpn(num_classes=7).to(device)

    batch_images = ['HT_TRAIN_00009_SH_000.jpg', 'HT_TRAIN_000013_SH_000.jpg', 'HT_TRAIN_000019_SH_000.jpg', 'HT_TRAIN_000021_SH_000.jpg']

    images = []
    targets = []
    ssl_proposals = []

    count = 0 
    for data in train_task[0]:
        img_name = data['file_name']
        if img_name in batch_images:
            img_path = os.path.join("dataset/SSLAD-2D/labeled",data['split'],img_name)
            image = PIL.Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image).to(device)
            tar = get_sample_objects(data['objects'])
            proposals = np.load(os.path.join('precomputed_proposals', img_name[:-4] + '.npy'), allow_pickle=True)
            proposals = {'boxes':torch.as_tensor(proposals, dtype=torch.float32).to(device)}

            images.append(image)
            ssl_proposals.append(proposals)
            targets.append(tar)
        
        count += 1

        if count == 22:
            break

    # for img_name in batch_images:
    model.eval() 
    model.roi_heads.generate_soft_proposals = True
    _  = model(images, targets, ssl_proposals)
    
    pl_te = model.proposals_logits
    breakpoint()

    model.train()
    output = model(images, targets, ssl_proposals, pl_te['proposals'])
    pl_st = model.proposals_logits

    output2 = model(images, targets, ssl_proposals)
    breakpoint()
    

 

if __name__ == "__main__":
    main()
