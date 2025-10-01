""" Methods for transforming images and bounding boxes """
import numpy as np
import logging
import albumentations as alb
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AugmentationTransform:
    im_width: int
    im_height: int

    def get_transforms(self, name: str) -> list:
        if name == "train_14_23":
            # Transformation for training on quadrants 14/23
            crop_transforms = [alb.RandomCropFromBorders(crop_left=0.3,
                                                         crop_right=0.3,
                                                         crop_top=0.5,
                                                         crop_bottom=0.5, p=1.0),
                               alb.CenterCrop(height=self.im_height,
                                              width=self.im_width,
                                              pad_if_needed=True, p=1.0)]

            image_transforms = [alb.Affine(scale=(0.8, 1.2), rotate=1, p=0.5),
                                alb.RandomBrightnessContrast(p=0.5),
                                alb.Sharpen(p=0.5),
                                alb.CLAHE(p=0.5)]

        elif name == "val":
            crop_transforms = [alb.NoOp(p=1)]
            image_transforms = [alb.AutoContrast(p=1), alb.CLAHE(p=1)]

        elif name == "test_set":
            # Transformations for creating the test/validation sets
            crop_transforms = [alb.RandomCropFromBorders(crop_left=0.3,
                                                         crop_right=0.3,
                                                         crop_top=0.5,
                                                         crop_bottom=0.5, p=1.0)]
            image_transforms = [alb.RandomBrightnessContrast(p=1.0)]

        else:
            logger.error('Transformation "{}" not implemented'.format(name))
            print('Transformation "{}" not implemented'.format(name))
            crop_transforms = [alb.NoOp(p=1)]
            image_transforms = [alb.NoOp(p=1)]

        transforms = crop_transforms + image_transforms

        return transforms


class DETRansform:
    """
    Class to handle transformations and formatting for object detection tasks.

    This class is primarily designed for transforming image data and bounding boxes to
    a format suitable for machine learning models like the RT-DETR model. It supports
    applying a set of transformations to images, as well as handling bounding boxes and
    label fields according to a specified format. Additionally, it produces annotated
    inputs required by specific the RT-DETR object detection model.

    Attributes:
        transformations: List of transformations to be applied to the images.
                         If not provided, a default transformation is applied.
        bbox_format: Dictionary specifying the format of bounding boxes and
                     related label fields. It includes configurations such as
                     the format type, label fields, whether to clip bounding
                     boxes, and a minimum area threshold.

    Methods:
        transform(image, bboxes: list, label_fields: list):
            Applies transformations to the input image, bounding boxes,
            and label fields. Maintains consistency with the specified
            bounding box format and label fields. Returns the transformed
            image, bounding boxes, and updated related fields.

        format_transform(image, image_id, bboxes: list, labels: list):
            Formats the transformed outputs to be compatible with the RT-DETR
            model. Converts bounding boxes and labels to the required format,
            ensures input consistency, and generates annotations suitable
            for the model.
    """
    def __init__(self, transforms: list = None, bbox_format: dict = None):
        self.transforms = transforms
        self.bbox_format = bbox_format
        if transforms is None:
            self.transforms = [alb.NoOp()]
        if bbox_format is None:
            self.bbox_format = {'format': 'coco',
                                'label_fields': ['quadrants', 'positions'],
                                'clip': True,
                                'min_area': 1000}

    def transform(self, image, bboxes: list, label_fields: list):

        # The label_fields should be one list for each field
        try:
            assert len(label_fields) == len(self.bbox_format.get('label_fields'))
            labels = dict(zip(self.bbox_format.get('label_fields'), label_fields))
        except Exception as e:
            print(
                f'The argument "label_fields" must be a list of lists with labels: {self.bbox_format.get("label_fields")}')

        # The bboxes variable must be a list: convert to (N x 4) numpy array
        assert isinstance(bboxes, list)
        bbox_array = np.array(bboxes).reshape(len(bboxes), 4)

        # Set up the transformation
        transformation = alb.Compose(self.transforms, bbox_params=alb.BboxParams(**self.bbox_format))
        transformed = transformation(image=image, bboxes=bbox_array, **labels)

        # Create the output
        output = {'image': transformed['image'], 'bboxes': list(transformed['bboxes'].astype(int))}
        output.update({field: transformed[field] for field in self.bbox_format.get('label_fields')})

        return output

    def format_transform(self, image, image_id, bboxes: list, labels: list):
        """ This method produces the formatted input for the RT-DETR model. """

        # Consistency checks
        assert len(bboxes) == len(labels), 'We need as many labels as bounding boxes: len(bboxes) == len(labels)!'
        assert len(self.bbox_format.get('label_fields')) == 1, 'We can only use one set of labels.'
        assert self.bbox_format.get(
            'format') == 'coco', f'Bounding box format must be "coco", but is: {self.bbox_format.get("format")}!'
        assert isinstance(image_id, int), 'Image ID must be of type int.'
        assert all(isinstance(l, int) for l in labels), 'All labels must be class IDs (int).'

        # Transform the image
        output = self.transform(image=image, bboxes=bboxes, label_fields=[labels])
        output_image = output['image']
        output_bboxes = output['bboxes']
        output_labels = output.get(self.bbox_format.get('label_fields')[0])

        # Annotations for the model using the transformed image, bounding boxes and labels
        annotation_list = []
        # This list can only contain data if the transformed output contains at leaset one bounding box
        if len(output_labels) > 0:
            for bbox, label in zip(output_bboxes, output_labels):
                assert len(bbox) == 4, f'Incompatible bounding box: {bbox}'
                annotation = {'image_id': image_id,
                              'category_id': int(label),
                              'bbox': list(bbox),
                              'iscrowd': 0,
                              'area': bbox[2] * bbox[3]}
                annotation_list.append(annotation)
        output_annotations = {'image_id': image_id, 'annotations': annotation_list}
        return output_image, output_annotations