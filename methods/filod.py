import logging
import numpy as np
import torch
import torch.nn as nn
from methods.er import ER
import torch
from utils.train_utils import select_stream, select_model

logger = logging.getLogger()

class FILOD(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, writer, **kwargs)
        logging.info("FILOD Initialized")
        
        # FILOD Model
        self.model_teacher = None

        # Num of task changed
        self.task_changed = 0
    
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
            
        # Need to change as function
        if sample['task_num'] != self.task_num:
            self.task_changed += 1

            # Switch teacher as task changed
            if self.task_changed > 1:
                self.model_teacher = select_model(mode=self.mode, num_classes=self.n_classes).to(self.device)
                self.model_teacher.load_state_dict(self.model.state_dict()) # Copy weights from student to teacher
                self.model_teacher.eval()
                self.model_teacher.roi_heads.generate_soft_proposals = True

        self.task_num = sample['task_num']   
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        
        if len(self.temp_batch) == self.temp_batchsize:
            # print(self.temp_batchsize)
            # Make ready for direct training (doesn't go into memory before training)
            return_losses = self.online_train(self.temp_batch, self.batch_size, n_worker, 
                                           iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize
                                            )
            self.report_training(sample_num, return_losses, self.writer)

            for sample in self.temp_batch:
                self.update_memory(sample)

            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
             
              
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
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
        total_loss, num_data = 0.0, 0.0
        stream_classname = select_stream(dataset=self.dataset)
        sample_dataset = stream_classname(sample, dataset=self.dataset, transform=None, cls_list=None)
        
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        self.count_log += (stream_batch_size + memory_batch_size)
        
        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size) if memory_batch_size > 0 else None
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
                    with torch.no_grad():
                        _ = self.model_teacher(images, targets)
                    proposals_logits_te = self.model_teacher.proposals_logits
                    
                    losses_st = self.model(images, targets, proposals_logits_te['proposals'])
                    student_logits = self.model.student_logits

                    backbone_te, rpn_te= self.model_teacher.backbone_output, self.model_teacher.rpn_output
                    backbone_st, rpn_st= self.model.backbone_output, self.model.rpn_output
                    
                    # Fast rcnn loss
                    faster_rcnn_losses = sum(loss for loss in losses_st.values()) * 120.0

                    # Backbone loss
                    feature_distillation_losses = self.calculate_feature_distillation_loss(backbone_te, backbone_st) * 5.0

                    # RPN loss
                    rpn_distillation_losses = self.calculate_rpn_distillation_loss(rpn_te, rpn_st, bbox_threshold=0.1) * 2.0

                    # ROI head loss
                    roi_distillation_losses = self.calculate_roi_distillation_loss(proposals_logits_te, student_logits, targets) * 100.0

                    # Distillation loss
                    distillation_losses = roi_distillation_losses + rpn_distillation_losses + feature_distillation_losses
                    distillation_losses = distillation_losses.clone().detach()
                    # if i == 1:
                    #     logging.info(f"{faster_rcnn_losses}, roi:{roi_distillation_losses}, rpn:{rpn_distillation_losses}, \
                    #         backbone:{feature_distillation_losses}")

                    loss = faster_rcnn_losses + distillation_losses
                    # loss = faster_rcnn_losses

                # While first task (do not have any teacher model)
                else:
                    losses = self.model(images, targets)
                    loss = sum(loss for loss in losses.values())
            else:
                losses = self.model(images_stream, targets_stream)
                loss = sum(loss for loss in losses.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return_losses = {
            'loss': total_loss / iterations,
            'distillation_loss': distillation_losses.item() if self.model_teacher else 0.0,
            'faster_rcnn_loss': faster_rcnn_losses.item() if self.model_teacher else 0.0,
            'roi_distillation_loss': roi_distillation_losses.item() if self.model_teacher else 0.0,
            'rpn_distillation_loss': rpn_distillation_losses.item() if self.model_teacher else 0.0,
            'feature_distillation_loss': feature_distillation_losses.item() if self.model_teacher else 0.0,
        }
        return return_losses
    
    def report_training(self, sample_num, losses, writer, log_interval=10):
        """Reports the training progress to the console.
        
        Args:
            sample_num (int): The number of samples that have been trained on.
            losses (dict): The losses from the training iteration.
            writer (SummaryWriter): The Tensorboard summary writer.
            log_interval (int): The number of iterations between each console log.
        """
        # Tensorboard logging for each loss
        writer.add_scalar(f"Train/Loss/Total", losses['loss'], sample_num)
        writer.add_scalar(f"Train/Loss/Distillation", losses['distillation_loss'], sample_num)
        writer.add_scalar(f"Train/Loss/FasterRCNN", losses['faster_rcnn_loss'], sample_num)
        writer.add_scalar(f"Train/Loss/ROIDistillation", losses['roi_distillation_loss'], sample_num)
        writer.add_scalar(f"Train/Loss/RPNDistillation", losses['rpn_distillation_loss'], sample_num)
        writer.add_scalar(f"Train/Loss/FeatureDistillation", losses['feature_distillation_loss'], sample_num)

        if sample_num % log_interval == 0:
            logger.info(
                f"Train | Sample # {sample_num} | Loss {losses['loss']:.4f}"
            )

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
        # print(f'rpn cls:{final_rpn_cls_distillation_loss} bbx:{final_rpn_bbs_distillation_loss}')
        final_rpn_loss.to('cuda')

        return final_rpn_loss


    def calculate_roi_distillation_loss(self, proposals_logits_te, student_logits, targets):

        #per batch
        cls_logit_te = proposals_logits_te['class_logits']
        cls_logit_st = student_logits['class_logits']
        
        bbox_reg_te = proposals_logits_te['box_regression']
        bbox_reg_st = student_logits['box_regression']
        
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
    
def permute_and_flatten(layer, N, A, C, H, W):
        layer = layer.view(N, -1, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2)
        layer = layer.reshape(N, -1, C)
        return layer