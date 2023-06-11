import logging
import torchvision
import os 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from methods.clad_er import CLAD_ER
from utils.data_loader_clad import CladDistillationMemory, CladStreamDataset, CladMemoryDataset
import PIL
from torchvision import transforms
import copy
import torch
import gc

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLAD_FILOD(CLAD_ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        print("CLAD_FILOD Initialized!")
        self.memory_size = kwargs["memory_size"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = 8
        #kwargs["temp_batchsize"]
        
        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.n_classes = n_classes
        self.memory = CladMemoryDataset(dataset='SSLAD-2D', device=None)

        # Exposed classes
        self.current_trained_images = []
        self.exposed_classes = []
        self.exposed_tasks = []
        
        #FILOD Model
        self.model_teacher = None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=n_classes, for_distillation=True, generate_soft_proposals=False).to(self.device)
        self.params =[p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=0.0001, weight_decay=0.0003)
        self.n_classes = n_classes

        self.seed_num = kwargs['seed_num']  #only used for tensorboard
        self.temp_batch = []             #batch for stream
        self.task_changed = 0            #num of task changed
    
    def online_step(self, sample, sample_num, n_worker):
        """Updates the model based on new data samples. If the sample's class is new, 
        it is added to the model's class set. The memory is updated with the new sample, 
        and the model is trained if the number of updates meets a certain threshold.

        Args:
            sample
            sample_num (int): Sample count for all tasks
            n_worker (int): Number of worker, default zero
        """

        # Gives information about seen images / total passed images
        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)
            
        #Need to change as function
        if sample['task_num'] != self.task_num:
            self.writer.close()     
            self.writer = SummaryWriter(f"tensorboard/{self.tensorboard_pth}")
            self.task_changed +=1

            #Switch teacher as task changed
            if self.task_changed > 1:
                self.model_teacher = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=self.n_classes, for_distillation=True, generate_soft_proposals=True).to(self.device)
                self.model_teacher.load_state_dict(self.model.state_dict())
                self.model_teacher.eval()
                self.model_teacher.roi_heads.generate_soft_proposals = True

        self.task_num = sample['task_num']   
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        
        if len(self.temp_batch) == self.temp_batchsize:
            print(self.temp_batchsize)
            #make ready for direct training (doesn't go into memory before training)
            train_loss = self.online_train(self.temp_batch, self.batch_size, n_worker, 
                                           iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize
                                            )
            print(f"Train_loss: {train_loss}\n")
            for sample in self.temp_batch:
                self.update_memory(sample)

            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
             
              
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=2,):
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
        sample_dataset = CladStreamDataset(sample, dataset="SSLAD-2D", transform=None, cls_list=None)
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        self.count_log += stream_batch_size + memory_batch_size

        self.model.train()
        
        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size) if memory_batch_size >0 else None

            if memory_data:
                self.current_trained_images = list(set(self.current_trained_images + memory_data['images']))
                print("Current trained images:", len(self.current_trained_images), ", not included stream batch")

            images_stream = []; images_memory = []
            targets_stream = []; targets_memory = []
            
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                images_stream = [img.to(self.device) for img in stream_data['images']]
                for i in range(len(images_stream)):
                    d = {}
                    d['boxes'] = stream_data['boxes'][i].to(self.device)
                    d['labels'] = stream_data['labels'][i].to(self.device)
                    targets_stream.append(d)

            if memory_batch_size > 0:
                #concat data from memory
                images_memory = [img.to(self.device) for img in memory_data['images']]
                for i in range(len(memory_data['images'])):
                    d = {}
                    d['boxes'] = memory_data['boxes'][i].to(self.device)
                    d['labels'] = memory_data['labels'][i].to(self.device)
                    targets_memory.append(d)

                # Concat stream data and memory data
                images = images_stream + images_memory
                targets = targets_stream + targets_memory

                # Calculate distillation loss
                if self.model_teacher:
                    
                    __ , proposals_logits_te = self.model_teacher(images, targets)
                    
                    losses_st, __, z_logits = self.model(images, targets, proposals_logits_te['proposals'])

                    backbone_te, rpn_te= self.model_teacher.backbone_output, self.model_teacher.rpn_output
                    backbone_st, rpn_st= self.model.backbone_output, self.model.rpn_output
                    
                    #fast rcnn loss
                    losses_st['loss_classifier'] = torch.mean(losses_st['loss_classifier'])
                    losses_st['loss_box_reg'] = torch.mean(losses_st['loss_box_reg'][0])
                    faster_rcnn_losses = sum(loss for loss in losses_st.values())

                    #backbone loss
                    feature_distillation_losses = self.calculate_feature_distillation_loss(backbone_te, backbone_st)

                    #rpn loss
                    rpn_distillation_losses = self.calculate_rpn_distillation_loss(rpn_te, rpn_st, bbox_threshold=0.1)

                    #roi head loss
                    roi_distillation_losses = self.calculate_roi_distillation_loss(proposals_logits_te, z_logits, targets)

                    #distillation loss
                    distillation_losses = roi_distillation_losses + rpn_distillation_losses + feature_distillation_losses
                    distillation_losses = distillation_losses.clone().detach()
                    if i == 1:
                        print(f"{faster_rcnn_losses}, roi:{roi_distillation_losses}, rpn:{rpn_distillation_losses} , backbone:{feature_distillation_losses}")

                    loss = faster_rcnn_losses + distillation_losses

                #while first task (do not have any teacher model)
                else:
                    losses, proposals_logits, _ = self.model(images, targets)
                    losses['loss_classifier'] = torch.mean(losses['loss_classifier'])
                    losses['loss_box_reg'] = torch.mean(losses['loss_box_reg'][0])
                
                    loss = sum(loss for loss in losses.values())
            else:
                losses, proposals_logits, _ = self.model(images_stream, targets_stream)
                losses['loss_classifier'] = torch.mean(losses['loss_classifier'])
                losses['loss_box_reg'] = torch.mean(losses['loss_box_reg'][0])
                
                loss = sum(loss for loss in losses.values())

        #report loss
        if self.count_log % 10 == 0:
                logging.info(f"Step {self.count_log}, Current Loss: {loss}")
        self.writer.add_scalar("Loss/train", loss, self.count_log)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()

        return (total_loss/iterations)
        
    def update_memory(self, sample):
        # Updates the memory of the model based on the importance of the samples.
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
                
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
        
        
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

    def calculate_feature_distillation_loss(self,backbone_te, backbone_st):
        final_feature_distillation_loss = []
        if len(backbone_te) == len(backbone_st):
            for i in range(len(backbone_te)):
                te_feature = backbone_te[i]
                st_feature = backbone_st[i]
                
                te_feature_avg = torch.mean(te_feature)
                st_feature_avg = torch.mean(st_feature)
                normalized_te_feature = te_feature - te_feature_avg  # normalize features
                normalized_st_feature = st_feature - st_feature_avg
                feature_difference = normalized_te_feature - normalized_st_feature
                feature_size = feature_difference.size()
                filter = torch.zeros(feature_size).to('cuda')
                feature_distillation_loss = torch.max(feature_difference, filter)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory

            final_feature_distillation_loss = sum(final_feature_distillation_loss)
        else:
            raise ValueError("Number of source features must equal to number of target features")

        return final_feature_distillation_loss

    def calculate_rpn_distillation_loss(self, rpn_te, rpn_st, bbox_threshold=0.1):

        rpn_objectness_te, rpn_bbox_regression_te = rpn_te
        rpn_objectness_st, rpn_bbox_regression_st = rpn_st

        # calculate rpn classification loss
        num_te_rpn_objectness = len(rpn_objectness_te)
        num_st_rpn_objectness = len(rpn_objectness_st)
        final_rpn_cls_distillation_loss = []
        objectness_difference = []

        if num_te_rpn_objectness == num_st_rpn_objectness:
            for i in range(num_st_rpn_objectness):
                
                current_te_rpn_objectness = rpn_objectness_te[i] 
                current_st_rpn_objectness = rpn_objectness_st[i]
                rpn_objectness_difference = current_te_rpn_objectness - current_st_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_te_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                
                '''
                current_te_rpn_objectness = rpn_objectness_te[i] 
                current_st_rpn_objectness = rpn_objectness_st[i]
                avrage_te_rpn_objectness = torch.mean(current_te_rpn_objectness)
                average_st_rpn_objectness = torch.mean(current_st_rpn_objectness)
                normalized_source_rpn_objectness = current_te_rpn_objectness - avrage_te_rpn_objectness
                normalized_target_rpn_objectness = current_st_rpn_objectness - average_st_rpn_objectness
                rpn_objectness_difference = normalized_source_rpn_objectness - normalized_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_te_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))'''
    
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
                
        else:
            raise ValueError("Wrong rpn objectness output")
        final_rpn_cls_distillation_loss = sum(final_rpn_cls_distillation_loss)/num_te_rpn_objectness

        # calculate rpn bounding box regression loss
        num_te_rpn_bbox = len(rpn_bbox_regression_te)
        num_st_rpn_bbox = len(rpn_bbox_regression_st)
        final_rpn_bbs_distillation_loss = []
        l2_loss = nn.MSELoss(size_average=False, reduce=False)

        if num_te_rpn_bbox == num_st_rpn_bbox:
            for i in range(num_st_rpn_bbox):
                current_te_rpn_bbox = rpn_bbox_regression_te[i]
                current_st_rpn_bbox = rpn_bbox_regression_st[i]
                current_objectness_difference = objectness_difference[i]
                [N, A, H, W] = current_objectness_difference.size()  # second dimention contains location shifting information for each anchor
                current_objectness_difference = permute_and_flatten(current_objectness_difference, N, A, 1, H, W)
                current_te_rpn_bbox = permute_and_flatten(current_te_rpn_bbox, N, A, 4, H, W)
                current_st_rpn_bbox = permute_and_flatten(current_st_rpn_bbox, N, A, 4, H, W)
                current_objectness_mask = current_objectness_difference.clone()
                current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
                current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
                masked_te_rpn_bbox = current_te_rpn_bbox * current_objectness_mask
                masked_st_rpn_bbox = current_st_rpn_bbox * current_objectness_mask
        
                current_bbox_distillation_loss = l2_loss(masked_te_rpn_bbox, masked_st_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
        else:
            raise ValueError('Wrong RPN bounding box regression output')
        final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss)/num_te_rpn_bbox

        final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
        print(f'rpn cls:{final_rpn_cls_distillation_loss} bbx:{final_rpn_bbs_distillation_loss}')
        final_rpn_loss.to('cuda')

        return final_rpn_loss


    def calculate_roi_distillation_loss(self, proposals_logits_te, z_logits, targets):

        #per batch
        cls_logit_te = proposals_logits_te['class_logits']
        cls_logit_st = z_logits['class_logits']
        
        bbox_reg_te = proposals_logits_te['box_regression']
        bbox_reg_st = z_logits['box_regression']
        
        batch_size = len(cls_logit_st)
        total_anchor_num = cls_logit_st[0].size()[0] * batch_size
        
        total_object_num = 0
        for target in targets:
            total_object_num += len(target['labels'])
        
        roi_distillation_losses = 0
        if len(cls_logit_te) == len(cls_logit_st):
            l2_loss = nn.MSELoss()
            for cls_te, cls_st, bb_te, bb_st in zip(cls_logit_te, cls_logit_st, bbox_reg_te, bbox_reg_st):

                    cls_te_avg = torch.mean(cls_st) 
                    cls_st_avg = torch.mean(cls_te)
                    normalized_cls_te = cls_te - cls_te_avg  # normalize features
                    normalized_cls_st = cls_st - cls_st_avg 
                    
                    roi_distillation_losses +=torch.add(l2_loss(normalized_cls_te, normalized_cls_st),l2_loss(bb_te, bb_st)) 
        
        roi_distillation_losses = roi_distillation_losses / (total_anchor_num+total_object_num)
        
        return roi_distillation_losses


    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        # Adjusts the learning rate of the optimizer based on the learning history.
        pass

    
def permute_and_flatten(layer, N, A, C, H, W):
        layer = layer.view(N, -1, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2)
        layer = layer.reshape(N, -1, C)
        return layer