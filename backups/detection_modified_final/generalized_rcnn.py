"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
        self.backbone_output = None
        self.rpn_output = None
        
        self.student_logits = []
        self.proposals_logits = []

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        #type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses
        else:
            return detections

    def forward(self, images, targets=None, teacher_proposals=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
              
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features, feature_list = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        self.backbone_output = feature_list  #list of backbone features for distillation

        proposals, scores, proposal_losses, rpn_output = self.rpn(images, features, targets)
        self.rpn_output = rpn_output #tuple of rpn output for distillation
        #(concatnate, proposals:scores) for distillation (proposals: region proposal, scores: objectness), in a batch_size
        concatnated_proposals =[torch.cat((proposal, score.reshape(-1,1)), dim=1) for proposal, score in zip(proposals, scores)] 
    
        detections, detector_losses, proposals_logits = self.roi_heads(features, concatnated_proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        self.proposals_logits = proposals_logits 

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        #s_logits are generated if there are distill_proposals. 
        #s_logits are returned classification, regression logits
        if teacher_proposals:
            #use same rpn feature with the original model, but proposals are switched to distill_proposals.
            #distill_mode returns z_logits which are new logits according to distill_proposals.
            _, _, s_logits = self.roi_heads(features, teacher_proposals, images.image_sizes, targets, distill_mode = True)
            del s_logits['proposals']
            self.student_logits = s_logits

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)