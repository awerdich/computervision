""" PyTorch datsets for object detection and image classification """

# Imports
import os
import numpy as np
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
import albumentations as alb
from computervision.imageproc import ImageData, is_image
from computervision.transformations import DETRansform

logger = logging.getLogger(__name__)

# GPU checks
def get_gpu_info(device_str: str = None):

    if device_str is None:
        is_cuda = torch.cuda.is_available()
        print(f'CUDA available: {is_cuda}')
        print(f'Number of GPUs found:  {torch.cuda.device_count()}')
        if is_cuda:
            print(f'Current device ID: {torch.cuda.current_device()}')
            print(f'GPU device name:   {torch.cuda.get_device_name(0)}')
            print(f'PyTorch version:   {torch.__version__}')
            print(f'CUDA version:      {torch.version.cuda}')
            print(f'CUDNN version:     {torch.backends.cudnn.version()}')
            device_str = 'cuda:0'
            torch.cuda.empty_cache()
        else:
            device_str = 'cpu'

    info_msg = f'Device for model training/inference: {device_str}'
    device = torch.device(device_str)
    logger.info(info_msg)

    return device, device_str


class DTRdataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 image_processor,
                 image_dir: str,
                 file_name_col: str,
                 label_id_col: str,
                 bbox_col: str,
                 transforms: list = None):

        self.data = data
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.file_name_col = file_name_col
        self.label_id_col = label_id_col
        self.bbox_col = bbox_col
        self.transforms = transforms
        if transforms is None:
            self.transforms = [alb.NoOp()]
        self.file_list = [os.path.join(image_dir, file) for file in list(data[file_name_col].unique())]
        assert self.validate()
        self.bbox_format = {'format': 'coco',
                            'label_fields': ['tooth_position'],
                            'clip': True}

    def validate(self):
        """ Making sure all images can be read """
        validated = np.sum([is_image(file) for file in self.file_list])
        output = False
        try:
            assert np.sum(validated) == len(self.file_list)
        except AssertionError:
            logger.warning(f'Could not validate all images: loaded {validated} / {len(self.file_list)} images.')
        else:
            logger.info(f'Validated {validated} images.')
            output = True
        return output

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        idx %= self.__len__()
        file = self.file_list[idx]
        file_name = os.path.basename(file)
        image = ImageData().load_image(file)
        # Convert to RGB
        if len(image.shape) == 2:
            image = ImageData().np2color(image)
        bboxes = self.data.loc[self.data[self.file_name_col] == file_name, self.bbox_col].tolist()
        labels = self.data.loc[self.data[self.file_name_col] == file_name, self.label_id_col].tolist()

        # Apply image transform
        detr = DETRansform(bbox_format=self.bbox_format, transforms=self.transforms)
        transformed_im, transformed_annotations = detr. \
            format_transform(image=image, image_id=idx, bboxes=bboxes, labels=labels)

        # Apply the image processor to the augmentation transform
        processed = self.image_processor(images=transformed_im,
                                         annotations=transformed_annotations,
                                         return_tensors='pt')

        # The processor returns lists for "pixel_values" and labels
        # But we need only one image and the annotations for that image
        output = {k: v[0] for k, v in processed.items()}

        return output