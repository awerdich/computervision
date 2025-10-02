""" Inference methods for computer vision """

import numpy as np
import torch
import logging
from computervision.datasets import get_gpu_info
from computervision.imageproc import xyxy2xywh, clipxywh
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

logger = logging.getLogger(name=__name__)

class DETR:
    """DETR inference class"""
    def __init__(self, device=None, checkpoint_path=None):
        if device is None:
            device, device_str = get_gpu_info()
        self.device = device
        if checkpoint_path is None:
            raise ValueError('checkpoint_path must be provided')
        self.checkpoint_path = checkpoint_path
        self.model, self.processor = self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        """Load DETR model"""
        model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint_path, device_map=self.device)
        processor = RTDetrImageProcessor.from_pretrained(checkpoint_path)
        return model, processor

    def predict(self,image, threshold):
        """Predict bounding boxes and labels for an image using DETR model"""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_outputs = self.model(**inputs)
        output = self.processor.\
            post_process_object_detection(model_outputs,
                                          target_sizes=torch.tensor([image.shape[:2]]),
                                          threshold=threshold)
        output_boxes = output[0].get('boxes').cpu().numpy()
        if len(output_boxes) > 0:
            bboxes = [clipxywh(xyxy2xywh(list(box)),
                               xlim=(0, image.shape[1]),
                               ylim=(0, image.shape[0]),
                               decimals=0) for box in output_boxes]
            areas = [int(box[2] * box[3]) for box in bboxes]
            categories = list(output[0].get('labels').cpu().numpy())
            scores = output[0].get('scores').cpu().numpy()
            categories = [int(c) for c in categories]
            scores = [np.around(s, decimals=4) for s in scores]
            predictions = {'bboxes': bboxes, 'areas': areas, 'categories': categories, 'scores': scores}
        else:
            logger.warning(f'No predictions for threshold {threshold}.')
            predictions = None
        return predictions