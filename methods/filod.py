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
from scipy.stats import ttest_ind

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset, MemoryDataset
from clad_memory import CladMemoryDataset
from clad_utils import CladDataset, visualize_and_save, data_transform

from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score
from engine import evaluate
from soda import SODADataset


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class FILOD_DJ(ER):
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
        if kwargs["sched_name"] == 'default':
            self.sched_name = 'adaptive_lr'
        
        self.current_trained_images = []
        self.exposed_classes = []
        self.exposed_tasks = []
        
        
        # Adaptive LR variables
        self.lr_step = kwargs["lr_step"]
        self.lr_length = kwargs["lr_length"]
        self.lr_period = kwargs["lr_period"]
        self.prev_loss = None
        self.lr_is_high = True
        self.high_lr = self.lr
        self.low_lr = self.lr_step * self.lr
        self.high_lr_loss = []
        self.low_lr_loss = []
        self.current_lr = self.lr
        self.count_log = 0
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=n_classes).to(self.device)
        self.params =[p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=0.0001, weight_decay=0.0003)
        self.task_num = 0
        self.writer = SummaryWriter("tensorboard")
        self.replay_method = kwargs['replay_method'] #base, er
        self.seed_num = kwargs['seed_num']
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
            
        # update_memory 호출 -> samplewise_importance_memory 호출 -> 여기에서 memory.replace_sample 호출
        # self.memory.replace_sample(sample)
        self.update_memory(sample)
        self.num_updates += self.online_iter
        
            
        if sample['task_num'] != self.task_num:
            self.writer.close()
        
            num = self.er_num if self.replay_method == 'er' else 0            
            self.writer = SummaryWriter(f"tensorboard/task{self.task_num+1}{self.replay_method}{num}_memory")
        
        self.task_num = sample['task_num']
        
        if self.num_updates >=1:
            if self.replay_method == 'er':
                if len(self.current_batch) == self.er_num:
                    self.num_updates = (self.num_updates//self.er_num)
                    for i in range(self.er_num):
                        train_loss = self.online_train(sample, self.batch_size, n_worker, 
                                            iterations=int(self.num_updates))
                        print(f"Train_loss: {train_loss}")
                      
                    self.num_updates -= int(self.num_updates)
                    self.current_batch.clear()  
            else:
                train_loss = self.online_train(sample, self.batch_size, n_worker, 
                                           iterations=int(self.num_updates))
                print(f"Train_loss: {train_loss}")
                self.num_updates -= int(self.num_updates)
                #self.scheduler.step()
    
    def online_train(self, sample, batch_size, n_worker, iterations=1):
        """Trains the model using both memory data and new data.

        Args:
            sample (_type_): _description_
            batch_size (_type_): _description_
            n_worker (_type_): _description_
            iterations (int, optional): _description_. Defaults to 1.
            stream_batch_size (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if len(self.memory) > 0 and batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size)
        
        for i in range(iterations):
            self.model.train()
            
            if len(self.memory) > 0 and batch_size > 0:
                
                memory_data = self.memory.get_batch(memory_batch_size, concat_idx=self.current_batch)
                self.count_log += memory_batch_size
                
                self.current_trained_images = list(set(self.current_trained_images + memory_data['images']))
                print("Current trained images:", len(self.current_trained_images))
                
                images = [img.to(self.device) for img in memory_data['images']]
                targets = []
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = memory_data['boxes'][i].to(self.device)
                    d['labels'] = memory_data['labels'][i].to(self.device)
                    targets.append(d)
         
                loss_dict = self.model(images, targets) 
                losses = sum(loss for loss in loss_dict.values())

                
                #report loss
                if self.count_log % 10 == 0:
                     logging.info(f"Step {self.count_log}, Current Loss: {losses}")
                self.writer.add_scalar("Loss/train", losses, self.count_log)
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                total_loss += losses.item()
                num_data += len(images)
                
        return total_loss / iterations
                
        
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
            
            if self.replay_method == 'er' and len(self.current_batch) < self.er_num:
                self.current_batch.append(target_idx)
                
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
            
            if self.replay_method == 'er' and len(self.current_batch) < self.er_num:
                self.current_batch.append(len(self.memory)- 1)
        
    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        # Adjusts the learning rate of the optimizer based on the learning history.
        pass
    
    def online_evaluate_clad(self, test_list, batch_size, n_worker):
        test_dataset = CladDataset(test_list)
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
            collate_fn=collate_fn
        )
        eval_dict = self.evaluation(test_loader)
        return eval_dict

    def evaluation(self, test_loader, criterion=None):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images = [img.to(self.device) for img in data[0]]
                targets = []
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = data[1][i]['boxes'].to(self.device)
                    d['labels'] = data[1][i]['labels'].to(self.device)
                    targets.append(d)
                outputs = self.model(images)
                
                for i, output in enumerate(outputs):
                    # For each image in batched outputs
                    # Get the predicted boxes and scores
                    boxes_pred = output['boxes']
                    scores = output['scores']
                    labels_pred = output['labels']

                    boxes_gt = targets[i]['boxes']
                    labels_gt = targets[i]['labels']
                    
                    # Apply the detection threshold
                    mask = scores >= 0.18
                    boxes_pred = boxes_pred[mask]
                    labels_pred = labels_pred[mask]
                    
                    for cls in range(self.n_classes):
                        # Get the ground-truth and prediction data for this class
                        gt_mask = labels_gt == cls
                        boxes_gt_cls = boxes_gt[gt_mask]
                        pred_mask = labels_pred == cls
                        boxes_pred_cls = boxes_pred[pred_mask]
                        
                        # Calculate IoU
                        iou = box_iou(boxes_pred_cls, boxes_gt_cls)
                        breakpoint()
                        # Calculate the prediction and ground-truth labels based on IoU threshold
                        iou_mask = iou > 0.5
                        
                        # preds_c: whether predicted box matches any of the gt boxes
                        # gts_c: whether gt-box matches any of the predicted boxes
                        preds_c = iou_mask.any(dim=1).float().cpu().numpy()
                        gts_c = iou_mask.any(dim=0).float().cpu().numpy()
                        breakpoint()
                        # Calculate AP
                        AP = average_precision_score(gts_c, preds_c)
                        # print(f"Class {cls}: AP = {AP}")
                        breakpoint()
                pass
            
def collate_fn(batch):
    return tuple(zip(*batch))


def calc_accuracy(boxes_gt, labels_gt, boxes_pred, labels_pred):
    """
    Summary: Calculate appropriate metric for object detection, given bounding box and label information
    Args:
        boxes_gt (torch.Size([n_boxes, 4])): Ground-truth bounding boxes
        labels_gt (torch.Size([n_boxes])): Ground-truth labels corresponding to each box
        boxes_pred (torch.Size([n_boxes, 4])): Predicted bounding boxes
        labels_pred (torch.Size([n_boxes])): Predicted labels corresponding to each box
    """

    # Initialize the total number of correct predictions
    total_correct = 0

    # Loop over all the ground truth boxes
    for idx, (box_gt, label_gt) in enumerate(zip(boxes_gt, labels_gt)):

        # Find the predicted box with the highest IoU with the current ground truth box
        ious = [get_iou(box_gt, box_pred) for box_pred in boxes_pred]
        best_pred_idx = np.argmax(ious)

        # Check if the label of the best predicted box matches the ground truth label
        if labels_pred[best_pred_idx] == label_gt:
            total_correct += 1

    # Calculate the accuracy
    accuracy = total_correct / len(boxes_gt)

    return accuracy


def get_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tensor): A bounding box in format (x1, y1, x2, y2)
        box2 (tensor): The other bounding box, in the same format
    """

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the union
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou
