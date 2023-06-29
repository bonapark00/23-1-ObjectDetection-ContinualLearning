import logging.config
import os
import PIL
import numpy as np
import torch
from torchvision import transforms
from utils.data_loader import MemoryDataset
from torch.utils.data import Dataset
from PIL import Image
from utils.preprocess_clad import get_clad_datalist, get_sample_objects

logger = logging.getLogger()

# Incomming input eg: {'img_id': 3, 'annot_file': ~, 'task_num': 2}

class CladStreamDataset(Dataset):
	def __init__(self, datalist, dataset, transform, cls_list, data_dir=None, device=None, transform_on_gpu=False):
		self.images = []
		self.labels = []
		self.objects = []
		self.dataset = 'SSLAD-2D'
		self.transform = transform
		self.cls_list = cls_list
		self.data_dir = data_dir
		self.device = device
		self.transform_on_gpu = transform_on_gpu
		self.transform_gpu = transform

		# print("#"* 50)
		# for data in datalist:
		#      print(data['file_name'])
		for data in datalist:
			try:
				img_name = data['file_name']
			except KeyError:
				img_name = data['filepath']
			
			if self.data_dir is None:
				img_path = os.path.join("dataset", self.dataset, 'labeled', data['split'], img_name)
			else:
				img_path = os.path.join(self.data_dir, data['split'], img_name)

			# print("Image path:", img_path)
			image = PIL.Image.open(img_path).convert('RGB')
			image = transforms.ToTensor()(image)
			target = get_sample_objects(data['objects'])

			# Append new samples
			self.images.append(image)
			self.objects.append(target)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		# Not used
		sample = None
		return sample
	
	@torch.no_grad()
	def get_data(self):
		images = [self.images[idx] for idx in range(len(self.images))]
		boxes = [self.objects[idx]['boxes'] for idx in range(len(self.images))]
		labels = [self.objects[idx]['labels'] for idx in range(len(self.images))]
		
		return {'images': images, 'boxes': boxes, 'labels': labels}

class CladMemoryDataset(MemoryDataset):
	def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
				 data_dir=None, transform_on_gpu=False, save_test=None, keep_history=False):
	  
		'''
		transform_on_gpu = True -> augmentation and normalize. need dataset info (mean, std, ...)
		'''
		self.datalist = []
		self.images = []
		self.objects = [] 
		self.dataset = 'SSLAD-2D'
		
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

	def show_memory_state(self):
		print("#" * 50)
		print("Samples in CladMemoryDataset")
		print("Total number of samples:", len(self.datalist))
		print("sample image name...")
		for data in self.datalist:
			img_name = data['file_name']
			print(img_name, " ")
		print("#" * 50)
		   
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
				breakpoint()
				img_path = os.path.join("dataset", self.dataset,'labeled',sample['split'],img_name)
		else:
				img_path = os.path.join(self.data_dir, sample['split'],img_name)
		image = PIL.Image.open(img_path).convert('RGB')
		image = transforms.ToTensor()(image)

		'''
		if self.transform_on_gpu:
				image = self.transform_cpu(image)'''
		
		target = get_sample_objects(sample['objects'])
		
		
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
	def get_batch(self, batch_size, use_weight=False, transform=None):
		rand_pool=np.array([k for k in range(len(self.images))])
		indices = np.random.choice(rand_pool, size=batch_size, replace=False).astype('int32')
					
		images = [self.images[idx] for idx in indices]
		boxes = [self.objects[idx]['boxes'] for idx in indices]
		labels = [self.objects[idx]['labels'] for idx in indices]
			
		for obj in np.array(self.objects)[indices]:   
			obj_cls_id = np.bincount(np.array(obj['labels'].tolist()))[1:]
			if len(self.obj_cls_train_cnt) > len(obj_cls_id):
				obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_train_cnt) - len(obj_cls_id)), 
									constant_values=(0)).flatten()
					
			self.obj_cls_train_cnt += obj_cls_id
					
			if self.keep_history:
					#total history of indices selected for batch
					self.previous_idx = np.append(self.previous_idx, indices)
				
		# breakpoint()
		return {'images': images, 'boxes': boxes, 'labels': labels}
	
	def get_two_batches(self, batch_size, test_transform):
		indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)

		images = [self.images[idx] for idx in indices]
		boxes = [self.objects[idx]['boxes'] for idx in indices]
		labels = [self.objects[idx]['labels'] for idx in indices]
		data_1 = {'images': images, 'boxes': boxes, 'labels': labels}

		images = [self.images[idx] for idx in indices]
		boxes = [self.objects[idx]['boxes'] for idx in indices]
		labels = [self.objects[idx]['labels'] for idx in indices]
		data_2 = {'images': images, 'boxes': boxes, 'labels': labels}
		
		return data_1, data_2	
			

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
		# breakpoint()
		
		self.dataset = 'SSLAD-2D'
		
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
		# breakpoint()
		self.obj_cls_list = obj_cls_list 
		
		if np.max(obj_cls_list) > len(self.obj_cls_count):
			extend_length = np.max(obj_cls_list)-len(self.obj_cls_count)
			self.obj_cls_count = np.pad(self.obj_cls_count, (0,extend_length), constant_values=(0)).flatten()
			self.obj_cls_train_cnt = np.pad(self.obj_cls_train_cnt, (0,extend_length), constant_values=(0)).flatten()
	  
			
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
		
		target = get_sample_objects(sample['objects'])
		
		
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
				# weight = self.get_weight()
				# indices = np.random.choice(range(len(self.images)), size=batch_size, p=weight/np.sum(weight), replace=False)
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
	  

class SODADataset(Dataset):
	"""
		Dataset for SODA
		Given task_id and split, initialize a Pytorch Dataset object
		Currently only used for joint training and evaluation

		Args:
			path (string): Path to the dataset folder. Default is "./dataset/SSLAD-2D"
			task_ids (list): List of task ids. Default is [1]
			split (string): Split of the dataset. Default is "train"
			transforms (callable, optional): Optional transform to be applied on a sample
	"""

	def __init__(self, path="./dataset/SSLAD-2D", task_ids=[1], split="train", transforms=None):
		self.split = split
		self.task_ids = task_ids
		self.root = path
		self.transforms = transforms
		self.img_paths = []
		self.objects = []
		self.organize_paths(self.split, self.task_ids)
		
	def organize_paths(self, split, task_ids):
		train_num = [0, 4470, 5799, 7278, 7802]
		val_num = [0, 497, 645, 810, 869]
		split_num = train_num if split =='train' else val_num
		
		# Get total_data using split
		total_data = get_clad_datalist(data_type=split)
		
		# Get target_data using task_ids, first sort task_ids
		# This is because we want to get data from task 1, 2, 3, 4 in order
		print("Target task ids: ", task_ids)
		task_ids.sort()
		target_data = []
		for task_id in task_ids:
			start_idx = split_num[task_id-1]; end_idx = split_num[task_id]
			target_data += total_data[start_idx:end_idx]
		
		# Get img_paths and objects
		for item in target_data:
			self.img_paths.append(item['file_name'])
			for i in range(len(item['objects']['bbox'])):
				box = item['objects']['bbox'][i]
				item['objects']['bbox'][i] = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
			self.objects.append(item['objects'])

	def __len__(self):
		return len(self.img_paths)
	
	def __getitem__(self, idx):
		# Task 1 is always from task_path = 'train'
		# Assume task 1 always appears first in task_ids (sorted)
		# Split is train and task 1 (idx < 4470) -> task_path = 'train'
		# Split is val and task 1 (idx < 497) -> task_path = 'train'
		# Else -> task_path = 'val'
		boxes, labels = [], []
		if self.split == 'train':
			if 1 in self.task_ids and idx < 4470: # 4470 is the number of images in task 1
				task_path = 'train'
			else:
				task_path = 'val'
		else:
			if 1 in self.task_ids and idx < 497: # 497 is the number of images in task 1
				task_path = 'train'
			else:
				task_path = 'val'
	
		img_path = f"{self.root}/labeled/{task_path}/{self.img_paths[idx]}"
		img = Image.open(img_path).convert("RGB")
		if self.transforms is not None:
			img = self.transforms(img)

		target = {}
		boxes = torch.tensor(self.objects[idx]['bbox'], dtype=torch.float32)
		labels = torch.tensor(self.objects[idx]['category_id'])
		target["img_path"] = img_path
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = torch.tensor(self.objects[idx]['image_id'])
		target["area"] = torch.tensor(self.objects[idx]['area'])
		target["iscrowd"] = torch.zeros((len(self.objects[idx]['bbox']),), dtype=torch.int64)

		return img, target
