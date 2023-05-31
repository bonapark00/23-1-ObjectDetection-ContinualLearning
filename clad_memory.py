import logging.config
import os
from typing import List
import time
import PIL
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from time import perf_counter
from operator import itemgetter
from utils.data_loader import MemoryDataset
from clad_utils import visualize_and_save

logger = logging.getLogger()


#Incomming input eg: {'img_id': 3, 'annot_file': ~, 'task_num': 2}
class CladMemoryDataset(MemoryDataset):
      def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=False, save_test=None, keep_history=False):
      
        '''
        transform_on_gpu = True -> augmentation and normalize. need dataset info (mean, std, ...)
        '''
        self.datalist = [] 
        self.images = []
        self.objects = [] 
        self.dataset = dataset #SSLAD-2D
        
        self.img_weather_count = {'Clear': 0, 'Overcast': 0, 'Rainy': 0}
        self.img_location_count = {'Citystreet': 0, 'Countryroad': 0, 'Highway': 0}
        self.img_period_count = {'Daytime': 0, 'Night': 0}
        self.img_city_count = {'Shenzhen': 0, 'Shanghai': 0, 'Guangzhou': 0}
        
        self.obj_cls_list = [1] # [1,4] exposed class list(숫자 리스트) cat1,4 classes were exposed
        self.obj_cls_count = np.zeros(np.max(self.obj_cls_list), dtype=int) #[2,0,0,4]  2 objects in cat1, 4 objects of cat 4, in total
        self.obj_cls_train_cnt = np.zeros(np.max(self.obj_cls_list),dtype=int) #[1,0,0,2] 1 time for cat1 obj, 2 times for cat4 are trained
        self.others_loss_decrease = np.array([]) 
        self.previous_idx = np.array([], dtype=int)
        self.device = device
        
        #self.test_transform = test_transform
        self.data_dir = data_dir
        self.keep_history = keep_history
        
      def __len__(self):
        return len(self.images)
  
                    
      def __getitem__(self, idx):
            if torch.is_tensor(idx):
                  idx = idx.value()
                  
            image = self.images[idx]
            if self.transform:
                  image = self.transform(image)

            target = self.objects[idx]
            target = {'boxes':target['boxes'], 'labels':target['labels']}
            return image, target
      
        
      def add_new_class(self, obj_cls_list):
        '''
        when appeard new class, check whether to extend obj_cls_count
        if it is, extend new spaces for new classes('category_id') 
        '''
        self.obj_cls_list = obj_cls_list 
        
        if np.max(obj_cls_list) > len(self.obj_cls_count):
            extend_length = np.max(obj_cls_list)-len(self.obj_cls_count)
            self.obj_cls_count = np.pad(self.obj_cls_count, (0,extend_length), constant_values=(0)).flatten()
            self.obj_cls_train_cnt = np.pad(self.obj_cls_train_cnt, (0,extend_length), constant_values=(0)).flatten()
      

      def get_sample_objects(self, objects: dict):
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
            
      def replace_sample(self, sample, idx=None):
            '''
            idx = None -> No sample to remove. just add data to memory at behind
            idx != None -> memory is full, replace it
            
            sample -> {'file_name', 'img_info', 'objects': {'image_id':int, 'category_id','bbox', 'area', 'id', 'truncated', 'occluded','iscrowd'}
            
            should be included: store image information, object, task, ... info
            first, need to classify object information
            '''
            #Appear new sample. whether just added or replaced with unimportant sample
            
            #insert info of new sample
            #img
            img_info = sample['img_info']
            self.img_weather_count[img_info['weather']] +=1
            self.img_location_count[img_info['location']] +=1
            self.img_period_count[img_info['period']] +=1
            self.img_city_count[img_info['city']] +=1
            #objects
            obj_cls_info = np.array(sample['objects']['category_id'])
            obj_cls_id = np.bincount(obj_cls_info)[1:] #from 1 to max. [3,3] -> [0,0,2]
            obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten()
            self.obj_cls_count += obj_cls_id #numpy sum
            
            #get image, target from new sample
            try:
                  img_name = sample['file_name']
            except KeyError:
                  img_name = sample['filepath']
            
            if self.data_dir is None:
                  img_path = os.path.join("dataset", self.dataset,'labeled',sample['split'],img_name)
            else:
                  img_path = os.path.join(self.data_dir,sample['split'],img_name)
            image = PIL.Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)

            '''
            if self.transform_on_gpu:
                  image = self.transform_cpu(image)'''
            
            target = self.get_sample_objects(sample['objects'])
            
            
            #append new sample at behind(last index)
            if idx is None: 
                  self.datalist.append(sample)
                  self.images.append(image)
                  self.objects.append(target)
                  
                  #check before apply
                  '''
                  if self.save_test == 'gpu':
                        self.device_img.append(self.test_transform(image).to(self.device).unsqueeze(0))
                  elif self.save_test == 'cpu':
                        self.device_img.append(self.test_transform(image).unsqueeze(0))
                  '''
                      
                  #TODO: update class importance for the new sample
                  '''
                  if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                        self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
                  else:
                        self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]]))
                  '''      
                  
            else: 
                  #remove info of trash sample
                  discard_sample = self.datalist[idx]
                  dis_sample_img = discard_sample['img_info']
                  dis_sample_obj = discard_sample['objects']

                  #remove img info
                  self.img_weather_count[ dis_sample_img['weather']] -= 1
                  self.img_location_count[ dis_sample_img['location']] -= 1
                  self.img_period_count[ dis_sample_img['period']] -= 1
                  self.img_city_count[ dis_sample_img['city']] -=1
                  
                  #remove objects info
                  obj_cls_info = np.array(dis_sample_obj['category_id'])
                  obj_cls_id = np.bincount(obj_cls_info)[1:] #from 1 to max. [3,3] -> [0,0,2]
                  obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten() #rezie to whole array size
                  self.obj_cls_count -= obj_cls_id
                  
                  #replace idx with new sample
                  self.datalist[idx] = sample
                  self.images[idx] = image
                  self.objects[idx] = target
                  
                  #check before apply
                  '''
                  if self.save_test == 'gpu':
                        self.device_img[idx] = self.test_transform(image).to(self.device).unsqueeze(0)
                  elif self.save_test == 'cpu':
                        self.device_img[idx] = self.test_transform(image).unsqueeze(0)
                  '''
                  
                  #TODO: update class importance for the new sample 
                  '''
                  if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                        self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease)
                  else:
                        self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]])'''
      
      def get_weight(self):
            pass 
  
      @torch.no_grad()
      def get_batch(self, batch_size, use_weight=False, concat_idx=[], transform=None):
            if use_weight:
                  print('use_weight is not available')
                  #weight = self.get_weight()
                  #indices = np.random.choice(range(len(self.images)), size=batch_size, p=weight/np.sum(weight), replace=False)
            else:
                  rand_idx_num = batch_size - len(concat_idx)
                  rand_pool=np.array([k for k in range(len(self.images)) if k not in concat_idx])
                  indices = np.hstack((np.random.choice(rand_pool, size=rand_idx_num, replace=False), np.array(concat_idx))).astype('int32')

            
            '''
            <transformation, not revised yet)>
            for i in indices:
                  if transform is None:
                        if self.transform_on_gpu:
                          images.append(self.transform_gpu(self.images[i].to(self.device)))
                        else:
                          images.append(self.transform(self.images[i]))
            else:
                if self.transform_on_gpu:
                    images.append(transform(self.images[i].to(self.device)))
                else:
                    images.append(transform(self.images[i]))'''
                    
            images = [self.images[idx] for idx in indices]
            boxes = [self.objects[idx]['boxes'] for idx in indices]
            labels = [self.objects[idx]['labels'] for idx in indices]
            
            for obj in np.array(self.objects)[indices]:
                  
                  obj_cls_id = np.bincount(np.array(obj['labels'].tolist()))[1:]
                  #breakpoint()
                  if len(self.obj_cls_train_cnt) > len(obj_cls_id):
                    obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_train_cnt)-len(obj_cls_id)), constant_values=(0)).flatten()
                  
                  self.obj_cls_train_cnt += obj_cls_id
                  
            if self.keep_history:
                  #total history of indices selected for batch
                  self.previous_idx = np.append(self.previous_idx, indices) 
            
            return {'images': images, 'boxes': boxes, 'labels': labels}
       
            
            

#Incomming input eg: {'img_id': 3, 'annot_file': ~, 'task_num': 2}

class CladDistillationMemory(MemoryDataset):
      def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=False, save_test=None, keep_history=False):
      
        '''
        transform_on_gpu = True -> augmentation and normalize. need dataset info (mean, std, ...)
        '''
        self.datalist = [] 
        self.images = []
        self.objects = [] 
        self.proposals = [] #(512, 4) total 512 of region proposals
        self.class_logits = []    #class_logits: (512, 7), 
        self.box_regression = []  #box_logits: (512, 28)
        
       
        self.dataset = dataset #SSLAD-2D
        
        self.img_weather_count = {'Clear': 0, 'Overcast': 0, 'Rainy': 0}
        self.img_location_count = {'Citystreet': 0, 'Countryroad': 0, 'Highway': 0}
        self.img_period_count = {'Daytime': 0, 'Night': 0}
        self.img_city_count = {'Shenzhen': 0, 'Shanghai': 0, 'Guangzhou': 0}
        
        self.obj_cls_list = [1] # [1,4] exposed class list(숫자 리스트) cat1,4 classes were exposed
        self.obj_cls_count = np.zeros(np.max(self.obj_cls_list), dtype=int) #[2,0,0,4]  2 objects in cat1, 4 objects of cat 4, in total
        self.obj_cls_train_cnt = np.zeros(np.max(self.obj_cls_list),dtype=int) #[1,0,0,2] 1 time for cat1 obj, 2 times for cat4 are trained
        self.others_loss_decrease = np.array([]) 
        self.previous_idx = np.array([], dtype=int)
        self.device = device
        
        #self.test_transform = test_transform
        self.data_dir = data_dir
        self.keep_history = keep_history
        
      def __len__(self):
        return len(self.images)
  
                    
      def __getitem__(self, idx):
            if torch.is_tensor(idx):
                  idx = idx.value()
                  
            image = self.images[idx]
            if self.transform:
                  image = self.transform(image)

            target = self.objects[idx]
            target = {'boxes':target['boxes'], 'labels':target['labels']}
            return image, target
      
        
      def add_new_class(self, obj_cls_list):
        '''
        when appeard new class, check whether to extend obj_cls_count
        if it is, extend new spaces for new classes('category_id') 
        '''
        self.obj_cls_list = obj_cls_list 
        
        if np.max(obj_cls_list) > len(self.obj_cls_count):
            extend_length = np.max(obj_cls_list)-len(self.obj_cls_count)
            self.obj_cls_count = np.pad(self.obj_cls_count, (0,extend_length), constant_values=(0)).flatten()
            self.obj_cls_train_cnt = np.pad(self.obj_cls_train_cnt, (0,extend_length), constant_values=(0)).flatten()
      

      def get_sample_objects(self, objects: dict):
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
            
      def replace_sample(self, sample, logit, idx=None):
            '''
            idx = None -> No sample to remove. just add data to memory at behind
            idx != None -> memory is full, replace it
            
            sample -> {'file_name', 'img_info', 'objects': {'image_id':int, 'category_id','bbox', 'area', 'id', 'truncated', 'occluded','iscrowd'}
            
            should be included: store image information, object, task, ... info
            first, need to classify object information
            '''
            #Appear new sample. whether just added or replaced with unimportant sample
            
            #insert info of new sample
            #img
            img_info = sample['img_info']
            self.img_weather_count[img_info['weather']] +=1
            self.img_location_count[img_info['location']] +=1
            self.img_period_count[img_info['period']] +=1
            self.img_city_count[img_info['city']] +=1
            #objects
            obj_cls_info = np.array(sample['objects']['category_id'])
            obj_cls_id = np.bincount(obj_cls_info)[1:] #from 1 to max. [3,3] -> [0,0,2]
            obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten()
            self.obj_cls_count += obj_cls_id #numpy sum
            
            #get image, target from new sample
            try:
                  img_name = sample['file_name']
            except KeyError:
                  img_name = sample['filepath']
            
            if self.data_dir is None:
                  img_path = os.path.join("dataset", self.dataset,'labeled',sample['split'],img_name)
            else:
                  img_path = os.path.join(self.data_dir,sample['split'],img_name)
            image = PIL.Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)

            '''
            if self.transform_on_gpu:
                  image = self.transform_cpu(image)'''
            
            target = self.get_sample_objects(sample['objects'])
            
            
            #append new sample at behind(last index)
            if idx is None: 
                  self.datalist.append(sample)
                  self.images.append(image)
                  self.objects.append(target)
                  self.proposals.append(logit['proposals'])
                  self.class_logits.append(logit['class_logits'])
                  self.box_regression.append(logit['box_regression'])

                  #check before apply
                  '''
                  if self.save_test == 'gpu':
                        self.device_img.append(self.test_transform(image).to(self.device).unsqueeze(0))
                  elif self.save_test == 'cpu':
                        self.device_img.append(self.test_transform(image).unsqueeze(0))
                  '''
                      
                  #TODO: update class importance for the new sample
                  '''
                  if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                        self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
                  else:
                        self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]]))
                  '''      
                  
            else: 
                  #remove info of trash sample
                  discard_sample = self.datalist[idx]
                  dis_sample_img = discard_sample['img_info']
                  dis_sample_obj = discard_sample['objects']

                  #remove img info
                  self.img_weather_count[ dis_sample_img['weather']] -= 1
                  self.img_location_count[ dis_sample_img['location']] -= 1
                  self.img_period_count[ dis_sample_img['period']] -= 1
                  self.img_city_count[ dis_sample_img['city']] -=1
                  
                  #remove objects info
                  obj_cls_info = np.array(dis_sample_obj['category_id'])
                  obj_cls_id = np.bincount(obj_cls_info)[1:] #from 1 to max. [3,3] -> [0,0,2]
                  obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten() #rezie to whole array size
                  self.obj_cls_count -= obj_cls_id
                  
                  #replace idx with new sample
                  self.datalist[idx] = sample
                  self.images[idx] = image
                  self.objects[idx] = target
                  self.proposals[idx] = logit['proposals']
                  self.class_logits[idx] = logit['class_logits']
                  self.box_regression[idx] = logit['box_regression']
                  #check before apply
      
      def get_weight(self):
            pass 
  
      @torch.no_grad()
      def get_batch(self, batch_size, use_weight=False, transform=None):
            if use_weight:
                  print('use_weight is not available')
                  #weight = self.get_weight()
                  #indices = np.random.choice(range(len(self.images)), size=batch_size, p=weight/np.sum(weight), replace=False)
            else:
                  indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)

                    
            images = [self.images[idx] for idx in indices]
            boxes = [self.objects[idx]['boxes'] for idx in indices]
            labels = [self.objects[idx]['labels'] for idx in indices]
            proposals = [self.proposals[idx] for idx in indices]
            class_logits = [self.class_logits[idx] for idx in indices]
            box_regression = [self.box_regression[idx] for idx in indices]
            
            for obj in np.array(self.objects)[indices]:
                  
                  obj_cls_id = np.bincount(np.array(obj['labels'].tolist()))[1:]
                  #breakpoint()
                  if len(self.obj_cls_train_cnt) > len(obj_cls_id):
                    obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_train_cnt)-len(obj_cls_id)), constant_values=(0)).flatten()
                  
                  self.obj_cls_train_cnt += obj_cls_id
                  
            if self.keep_history:
                  #total history of indices selected for batch
                  self.previous_idx = np.append(self.previous_idx, indices) 
            
            return {'images': images, 'boxes': boxes, 'labels': labels, 'proposals': proposals,\
                  'class_logits': class_logits, 'box_regression': box_regression}
