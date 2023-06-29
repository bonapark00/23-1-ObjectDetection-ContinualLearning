"""
Modeified framwork for fast_rcnn
this model is only used in ilod
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    However it is only used in constructing fast rcnn 

    Args:
        backbone (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.roi_heads = roi_heads
        self.proposals_logits = []

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None, ssl_proposals=None, teacher_proposals=None):
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
                
        if not ssl_proposals:
            raise ValueError("proposals should be passed always.")

        original_image_sizes: List[Tuple[int, int]] = []

        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
      
        #resize images, and modify proposals into list
        #if proposals is not a list of dict, convert it to a list of dict (to be consistent with the output of transform)
        # if not isinstance(ssl_proposals[0], dict):
        #     for i in range(len(ssl_proposals)):
        #         ssl_proposals[i] = {'boxes': ssl_proposals[i]}

        original_images = images
        _, targets = self.transform(original_images, targets)
        images, raw_proposals = self.transform(original_images, ssl_proposals)

        proposals = []
        for i in range(len(raw_proposals)):
            proposal = raw_proposals[i]['boxes']
            proposals.append(proposal)
    
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

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

       
        detections, detector_losses, proposals_logits = self.roi_heads(features, proposals, images.image_sizes, targets, bool(teacher_proposals), teacher_proposals)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        self.proposals_logits = proposals_logits

        losses = {}
        losses.update(detector_losses)
        return self.eager_outputs(losses, detections)