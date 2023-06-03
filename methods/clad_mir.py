import logging
import sys
import random
import copy
import torchvision
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from methods.er_baseline import ER
from utils.data_loader_clad import CladMemoryDataset
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLAD_MIR(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory_size = 150 #kwargs["memory_size"]
        self.batch_size = 4

        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.n_classes = n_classes
        self.memory = CladMemoryDataset(dataset='SSLAD-2D', device=None)
        self.imp_update_period = kwargs['imp_update_period']
        
        self.current_trained_images = []
        self.exposed_classes = []
        self.exposed_tasks = []
        
        
        self.count_log = 0
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=n_classes).to(self.device)
        self.params =[p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=0.0001, weight_decay=0.0003)
        self.task_num = 0
        self.writer = SummaryWriter("tensorboard")
        self.replay_method = 'er'
        self.ismir = True
        self.mir_cands = 20
        self.er_num = 2
        self.current_batch = []
    

    def online_step(self, sample, sample_num, n_worker):
        """Updates the model based on new data samples. If the sample's class is new, 
        it is added to the model's class set. The memory is updated with the new sample, 
        and the model is trained if the number of updates meets a certain threshold.

        Args:
            sample
            sample_num (int): Sample count for all tasks
            n_worker (int): Number of worker, default zero
        """
        
        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)
            
        self.update_memory(sample)
        self.num_updates += self.online_iter
        if sample['task_num'] != self.task_num:
            self.writer.close()
        
            num = self.er_num if self.replay_method == 'er' else 0            
            self.writer = SummaryWriter(f"tensorboard/task{self.task_num+1}{self.replay_method}{num}_memory")
        
        self.task_num = sample['task_num']

        if len(self.current_batch) == self.er_num:
            train_loss = self.online_train(sample, self.batch_size, n_worker,
                                                    iterations=int(self.num_updates),
                                                    stream_batch_size=self.er_num)
            self.num_updates -= int(self.num_updates)
            self.current_batch.clear()
    
    
    def online_train(self, sample, batch_size, n_worker, iterations, stream_batch_size):
        """
        
        """
        self.model.train()
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            breakpoint()
            stream_data = self.memory.get_batch(concat_idx=self.current_batch, batch_size=batch_size)
            # self.count_log += memory_batch_size
            
            stream_images = [img.to(self.device) for img in stream_data['images']]
            stream_targets = []
            for i in range(len(stream_images)):
                d = {}
                d['boxes'] = stream_data['boxes'][i].to(self.device)
                d['labels'] = stream_data['labels'][i].to(self.device)
                stream_targets.append(d)
        
            loss_dict = self.model(stream_images, stream_targets) 
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            losses.backward()

            # Getting the gradients after a forward and backward pass
            grads = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grads[name] = param.grad.data

            if len(self.memory) - stream_batch_size > 0:
                memory_batch_size = min(len(self.memory) - stream_batch_size, batch_size - stream_batch_size)
                self.count_log += memory_batch_size

                lr = self.optimizer.param_groups[0]['lr']
                new_model = copy.deepcopy(self.model)
                for name, param in new_model.named_parameters():
                    if param.requires_grad:
                        param.data = param.data - lr * grads[name]

                memory_stream_concat_size = min(self.mir_cands + stream_batch_size, len(self.memory))
                memory_stream_concat = self.memory.get_batch(memory_stream_concat_size,
                                                             concat_idx=self.current_batch)
                # No longer need streams -> Get only the randomly choosen batch
                memory_cands = {k:v[:memory_stream_concat_size - stream_batch_size] for k, v in 
                                memory_stream_concat.items()}
                images = [img.to(self.device) for img in memory_cands['images']]
                targets = []
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = memory_cands['boxes'][i].to(self.device)
                    d['labels'] = memory_cands['labels'][i].to(self.device)
                    targets.append(d)
                

                scores = torch.zeros(len(images))
                with torch.no_grad():
                    for i in range(len(images)):
                        # Calculate loss of each sample
                        pre_loss_dict = self.model([images[i]], [targets[i]])
                        post_loss_dict = new_model([images[i]], [targets[i]])
                        pre_loss = sum(loss for loss in pre_loss_dict.values())
                        post_loss = sum(loss for loss in post_loss_dict.values())
                        score = post_loss - pre_loss
                        scores[i] = score.item()
                        # print(f"Image {i}, target {i} loss difference: {score}")
                
                selected_samples = torch.argsort(scores, descending=True)[:memory_batch_size]
                selected_images = [images[idx.item()] for idx in selected_samples]
                selected_targets = [targets[idx.item()] for idx in selected_samples]
                
                final_images = stream_images + selected_images
                final_targets = stream_targets + selected_targets

                self.optimizer.zero_grad()
                loss_dict = self.model(final_images, final_targets)
                losses = sum(loss for loss in loss_dict.values())

                if self.count_log % 10 == 0:
                    logging.info(f"Step {self.count_log}, Current Loss: {losses}")
                self.writer.add_scalar("Loss/train", losses, self.count_log)
                losses.backward()
                self.optimizer.step()

                total_loss += losses.item()
                num_data += len(images)


                
        
    def update_memory(self, sample):
        self.samplewise_importance_memory(sample)
        
        
    def add_new_class(self, class_name):
        """Called when a new class of data is encountered. It extends the model's final layer 
        to account for the new class, and updates the optimizer and memory accordingly.
        
        1. Modify the final layer to handle the new class
        2. Assign the previous weight to the new layer
        3. Modify the optimizer while preserving previous optimizer states

        Args:
            class_name (str): The name of the new class to be added
        """
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        # Updates the loss of the model based on the sample data.
        pass
        
    def samplewise_importance_memory(self, sample):
        # Updates the memory of the model based on the importance of the samples.
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
            
            if len(self.current_batch) < self.er_num:
                self.current_batch.append(target_idx)
                
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
            
            if len(self.current_batch) < self.er_num:
                self.current_batch.append(len(self.memory)- 1)
        
    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        # Adjusts the learning rate of the optimizer based on the learning history.
        pass
            
def collate_fn(batch):
    return tuple(zip(*batch))