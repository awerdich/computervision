""" Performance metrics for computer vision models """

import numpy as np
import torch
from torchvision import ops
from computervision.imageproc import xywh2xyxy

class DetectionMetrics:
    def __init__(self):
        pass

    @staticmethod
    def compute_iou(bbox_1: list, bbox_2: list, bbox_format: str, method: str = 'np') -> float:
        assert method in ['np', 'pt'], 'method should be either "np" or "pt"'
        assert bbox_format in ['xyxy', 'xywh'], 'bbox_format should be either "xyxy" or "xywh"'
        iou = None
        if bbox_format == 'xywh':
            bbox_1, bbox_2 = xywh2xyxy(bbox_1), xywh2xyxy(bbox_2)
        if method == 'np':
            ix1 = np.maximum(bbox_1[0], bbox_2[0])
            iy1 = np.maximum(bbox_1[1], bbox_2[1])
            ix2 = np.minimum(bbox_1[2], bbox_2[2])
            iy2 = np.minimum(bbox_1[3], bbox_2[3])
            # Intersection height and width.
            i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
            i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
            area_of_intersection = i_height * i_width
            # Ground Truth dimensions.
            gt_height = bbox_1[3] - bbox_1[1] + 1
            gt_width = bbox_1[2] - bbox_1[0] + 1
            # Prediction dimensions.
            pd_height = bbox_2[3] - bbox_2[1] + 1
            pd_width = bbox_2[2] - bbox_2[0] + 1
            area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
            iou = area_of_intersection / area_of_union
        elif method == 'pt':
            bbox_tensors = [torch.tensor([bbox_1], dtype=torch.float),
                            torch.tensor([bbox_2], dtype=torch.float)]
            iou = ops.box_iou(bbox_tensors[0], bbox_tensors[1]).item()
        return iou