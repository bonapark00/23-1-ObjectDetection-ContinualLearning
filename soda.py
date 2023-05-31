import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from clad_data import get_clad_datalist


class SODADataset(Dataset):
    def __init__(self, path="./dataset/SSLAD-2D", task_id=1, split="train", transforms=None):
        self.split = split
        self.task_id = task_id
        self.root = path
        self.transforms = transforms
        self.img_paths = []
        self.objects = []
        
        self.organize_paths(self.split, self.task_id)
        
    def organize_paths(self, split, task_id):
        train_num = [0, 4470, 5799, 7278, 7802]
        val_num = [0, 497, 645, 810, 869]
        split_num = train_num if split =='train' else val_num
        
        total_data = get_clad_datalist(data_type=split)
        target_data = total_data if self.task_id == None else total_data[split_num[task_id-1]:split_num[task_id]]
        
        for item in target_data:
            self.img_paths.append(item['file_name'])
            for i in range(len(item['objects']['bbox'])):
                box = item['objects']['bbox'][i]
                item['objects']['bbox'][i] = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            self.objects.append(item['objects'])
    

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        boxes, labels = [], []
        
        if self.split == 'train':
            if self.task_id == 1:
                task_path = 'train'
            elif self.task_id == None:
                task_path = 'train' if idx < 4470 else 'val'
            else:
                task_path = 'val'
        else: 
            if self.task_id == 1:
                task_path = 'train'
            elif self.task_id == None:
                task_path = 'train' if idx < 497 else 'val'
            else:
                task_path = 'val'
    
        img_path = f"{self.root}/labeled/{task_path}/{self.img_paths[idx]}"
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        target = {}
        boxes = torch.tensor(self.objects[idx]['bbox'], dtype=torch.float32)
        labels = torch.tensor(self.objects[idx]['category_id'])
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(self.objects[idx]['image_id'])
        target["area"] = torch.tensor(self.objects[idx]['area'])
        target["iscrowd"] = torch.zeros((len(self.objects[idx]['bbox']),), dtype=torch.int64)

        return img, target
