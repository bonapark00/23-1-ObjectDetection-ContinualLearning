import logging
import numpy as np
import torch
import torch.nn as nn
from methods.er import ER
import torch
from utils.train_utils import select_stream, select_model
import copy
from collections import OrderedDict

logger = logging.getLogger()

###########################################################
# NOTE:                                                   #
# filod.py uses 'er' in memory.                           #
# filod_random_ema.py uses 'random' memory as baseline.py #
# also, this model uses ema to update teacher model       #                         
###########################################################


class FILOD_RANDOM_EMA(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, writer, **kwargs)
        logging.info("FILOD_RANDOM_EMA method is used")
        
        # filod_ema teacher model 
        self.model_teacher = copy.deepcopy(self.model)
        self.model_teacher.eval()
        self.model_teacher.roi_heads.generate_soft_proposals = True
        
        # filod_ema hyperparameters init
        self.sdp_mean = 10000
        self.sdp_varcoeff = 0.75
        assert 0.5 - 1 / self.sdp_mean < self.sdp_varcoeff < 1 - 1 / self.sdp_mean
        self.ema_ratio = (1 - np.sqrt(2 * self.sdp_varcoeff - 1 + 2 / self.sdp_mean)) / (self.sdp_mean - 1 - self.sdp_mean * self.sdp_varcoeff)
        
        #ema teacher update
        self.ema_update = 0
        
        
    def online_step(self, sample, sample_num, n_worker):

        # Gives information about seen images / total passed images
        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)

        self.task_num = sample['task_num']   
        self.update_memory(sample)
        
        # Train the model using data from the memory
        train_loss = self.online_train(self.batch_size, n_worker, iterations=int(self.online_iter))
        self.report_training(sample_num, train_loss, self.writer)


    def online_train(self, batch_size, n_worker, iterations=1):

        total_loss, num_data = 0.0, 0.0
        memory_batch_size = min(batch_size, len(self.memory))
        assert memory_batch_size != 0, "Memory is empty, or batch_size is zero."

        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size)
            images = [img.to(self.device) for img in memory_data['images']]
            targets = []
            
            for i in range(len(memory_data['images'])):
                d = {}
                d['boxes'] = memory_data['boxes'][i].to(self.device)
                d['labels'] = memory_data['labels'][i].to(self.device)
                targets.append(d)

            with torch.no_grad():
                _ = self.model_teacher(images, targets)
            proposals_logits_te = self.model_teacher.proposals_logits

            losses_st = self.model(images, targets, proposals_logits_te['proposals'])
            student_logits = self.model.student_logits
            backbone_te, rpn_te= self.model_teacher.backbone_output, self.model_teacher.rpn_output
            backbone_st, rpn_st= self.model.backbone_output, self.model.rpn_output

            # Fast rcnn loss
            faster_rcnn_losses = sum(loss for loss in losses_st.values())

            # Backbone loss
            feature_distillation_losses = self.calculate_feature_distillation_loss(backbone_te, backbone_st)

            # RPN loss
            rpn_distillation_losses = self.calculate_rpn_distillation_loss(rpn_te, rpn_st, bbox_threshold=0.1)

            # ROI head loss
            roi_distillation_losses = self.calculate_roi_distillation_loss(proposals_logits_te, student_logits, targets)

            # Distillation loss
            distillation_losses = roi_distillation_losses + rpn_distillation_losses + feature_distillation_losses
            loss = faster_rcnn_losses + distillation_losses
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            self.update_ema_model(num_updates=1.0)
            
        return_losses = {
            'loss': total_loss / iterations,
            'distillation_loss': distillation_losses.item() if self.model_teacher else 0.0,
            'faster_rcnn_loss': faster_rcnn_losses.item() if self.model_teacher else 0.0,
            'roi_distillation_loss': roi_distillation_losses.item() if self.model_teacher else 0.0,
            'rpn_distillation_loss': rpn_distillation_losses.item() if self.model_teacher else 0.0,
            'feature_distillation_loss': feature_distillation_losses.item() if self.model_teacher else 0.0,
        }
        return return_losses
    
    @torch.no_grad()
    def update_ema_model(self, num_updates=1.0):
        ema_inv_ratio = (1 - self.ema_ratio) ** num_updates
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.model_teacher.named_parameters())
        assert model_params.keys() == ema_params.keys()
        self.ema_update += 1
        
        for name, param in model_params.items():
            ema_params[name].sub_((1. - ema_inv_ratio) * (ema_params[name] - param))
    
    
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