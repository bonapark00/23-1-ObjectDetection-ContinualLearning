import logging.config
import os
import random
import pickle
from collections import defaultdict
import time
import torchvision
import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from soda import SODADataset
from clad_data import *
from clad_memory import *

def get_sample_objects( objects: dict):
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

def main():
    cur_train_datalist = get_clad_datalist('train')
    train_images=[] 
    train_targets=[]

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=7, for_distillation=True).to(device)

    random_proposal_list = [torch.rand(512,4).to(device) for i in range(2)]




    for idx, sample in enumerate(cur_train_datalist):
      
        img_name = sample['file_name']
        img_path = os.path.join("dataset",'SSLAD-2D','labeled',sample['split'],img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        image = transforms.ToTensor()(image).to(device)
        target = get_sample_objects(sample['objects'])
        target = {'boxes':target['boxes'].to(device), 'labels':target['labels'].to(device)}

        train_images.append(image)
        train_targets.append(target)

        if idx == 2:
            break
    
    #model, distillation_losses

    losses, proposals_logits, z_logits  = model(train_images, train_targets)
    
    aaa = losses 
    bbb = losses
    
    #Generalized RCNN, D
    breakpoint()
        







    '''
    start1 = time.time()
    cur_train_datalist = get_clad_datalist(data_type = 'train')
    print(f'{len(cur_train_datalist)} images in total \n')
    end1= time.time()
    
    exposed_classes = [0]
    iteration = 2
    memory = CladMemoryDataset(dataset='SSLAD-2D', device=None)
    
    for i, data in enumerate(cur_train_datalist):
        if not all(item in list(set(data['objects']['category_id']))for item in exposed_classes):
            exposed_classes = list(set(list(set(data['objects']['category_id'])) + exposed_classes))
            memory.add_new_class(exposed_classes)
            memory.replace_sample(data)
            
            for k in range(iteration):  
                
                    batch = memory.get_batch(batch_size=1)
                    if i <= 10:
                        print('_________________________________')
                        print(batch)
                        print()
                        print(memory.obj_cls_list)
                        print(memory.obj_cls_count)
                        print(memory.obj_cls_train_cnt)
                        print('_________________________________')
                        print(f'{i} th data, {k} th iter')
                        print('_________________________________')
                        print()
        
        if i == 11:
            break
    end2 = time.time()
    print(f'{end1-start1} elapsed for data preparation \n')
 '''
if __name__ == "__main__":
    main()
