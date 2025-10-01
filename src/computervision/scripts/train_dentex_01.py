""" Train the Dentex model """

import sys
import os
import json
import numpy as np
import pandas as pd
import logging
import datetime
from pathlib import Path

# Set the option to display all columns
pd.set_option('display.max_columns', None)

# Hugging Face Library
import torch
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from transformers import TrainingArguments, Trainer

import computervision
from computervision.imageproc import is_image
from computervision.datasets import DETRdataset, get_gpu_info
from computervision.transformations import AugmentationTransform
from computervision.mapeval import MAPEvaluator

#%% Data files and directories
data_dir = os.environ.get('DATA')
if data_dir is None:
    raise ValueError("DATA environment variable must be set")
train_image_dir = os.path.join(data_dir, 'dentex_detection_250928')
val_image_dir = os.path.join(train_image_dir,'test')
model_dir = os.path.join(data_dir, 'model')

Path(model_dir).mkdir(parents=True, exist_ok=True)
train_annotation_file_name = 'train_quadrant_enumeration_dset.parquet'
train_annotation_file = os.path.join(train_image_dir, train_annotation_file_name)
val_annotation_file_name = 'train_quadrant_enumeration_test_set.parquet'
val_annotation_file = os.path.join(val_image_dir, val_annotation_file_name)

# Column names for the annotation files
tooth_pos_col = 'ada'
file_name_col = 'file_name'
bbox_col = 'bbox'
dset_col = 'dset'

#%% Model and training parameters
# Training and model parameters
model_version = 1
device_number = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)
device, device_str = get_gpu_info(device_number=device_number)
date_str = datetime.date.today().strftime('%y%m%d')
model_name = f'rtdetr_{date_str}_{str(model_version).zfill(2)}'
print(f'Model name: {model_name}')

# Image transformations for training and validation
im_width, im_height = 640, 640
# Augmentations
train_quadrants = [14, 23]
val_quadrants = [14, 23]
train_transform_name = 'train_14_23'
val_transform_name = 'val'
aug = AugmentationTransform(im_width=im_width, im_height=im_height)
train_transforms = aug.get_transforms(name=train_transform_name)
val_transforms = aug.get_transforms(name=val_transform_name)

# Important information about the model that we want to save
model_info = {'model_version': model_version,
              'device_number': device_number,
              'project_version': computervision.__version__,
              'model_name': model_name,
              'train_image_dir': train_image_dir,
              'val_image_dir': val_image_dir,
              'model_dir': model_dir,
              'im_width': im_width,
              'im_height': im_height,
              'hf_checkpoint': 'PekingU/rtdetr_v2_r101vd',
              'training_checkpoint': 'PekingU/rtdetr_v2_r101vd',
              'train_quadrants': train_quadrants,
              'val_quadrants': val_quadrants,
              'train_transform_name': train_transform_name,
              'val_transform_name': val_transform_name}

# Specific arguments for the Trainer. 48
# See: https://huggingface.co/docs/transformers/en/main_classes/trainer#trainer
training_args = {'output_dir': os.path.join(model_dir, model_name),
                 'num_train_epochs': 250,
                 'max_grad_norm': 0.1,
                 'learning_rate': 5e-5,
                 'warmup_steps': 300,
                 'per_device_train_batch_size': 48,
                 'dataloader_num_workers': 8,
                 'metric_for_best_model': 'eval_map',
                 'greater_is_better': True,
                 'load_best_model_at_end': True,
                 'eval_strategy': 'epoch',
                 'save_strategy': 'epoch',
                 'save_total_limit': 5,
                 'remove_unused_columns': False,
                 'eval_do_concat_batches': False}

# We want to maintain the aspect ratio of the images
# So, we resize the image first and then pad it
processor_params = {'do_resize': True,
                    'size': {'max_height': im_height,
                             'max_width': im_width},
                    'do_pad': True,
                    'pad_size': {'height': im_height,
                                 'width': im_width}}

# Bounding box format for the annotations
bbox_format = {'format': 'coco',
               'label_fields': ['tooth_position'],
               'clip': True,
               'min_area': 10000}

#%% Verify image data
train_df = pd.read_parquet(train_annotation_file)
train_df = train_df.loc[
    (train_df[dset_col] == 'train') &
    (train_df['quadrants'].isin(train_quadrants))].astype({'ada': int})

# Filter the validation images and quadrants and take only the first augmentation
val_df = pd.read_parquet(val_annotation_file)
val_df = val_df.loc[
    (val_df[dset_col] == 'val') &
    (val_df['quadrants'].isin(val_quadrants)) &
    (val_df['transformation'] == 0)].astype({'ada': int})

# Check the images on disk
train_file_list = list(train_df[file_name_col].unique())
train_checked = np.sum([is_image(os.path.join(train_image_dir, file)) for file in train_file_list])
print(f'Images in training data:         {len(train_file_list)}')
print(f'Files checked in training data:  {train_checked}')
print(f'Annotations in training data:    {train_df.shape[0]}')

print()

val_file_list = list(val_df[file_name_col].unique())
val_checked = np.sum([is_image(os.path.join(val_image_dir, file)) for file in val_file_list])
print(f'Images in validation data:       {len(val_file_list)}')
print(f'Files checked in val data:       {val_checked}')
print(f'Annotations in validation data:  {val_df.shape[0]}')

# Create the label ids (tooth position, but starting from 0)
# The model needs label ids, not labels. So we need to add a label id column
label_name_list = sorted(list(train_df[tooth_pos_col].unique()))
id2label = dict(zip(range(len(label_name_list)), label_name_list))
id2label = {int(label_id): str(label_name) for label_id, label_name in id2label.items()}
label2id = {str(label_name): int(label_id) for label_id, label_name in id2label.items()}

train_df = train_df.assign(label=train_df[tooth_pos_col].apply(lambda name: label2id.get(str(name))))
val_df = val_df.assign(label=val_df[tooth_pos_col].apply(lambda name: label2id.get(str(name))))

#%% Create the logger

parameters = {'model_info': model_info,
              'id2label': id2label,
              'training_args': training_args,
              'processor_params': processor_params,
              'bbox_format': bbox_format}

json_file = os.path.join(model_dir, f'{model_name}.json')
with open(json_file, 'w') as f:
    json.dump(parameters, f, indent=4) # indent for pretty-printing

# Set up the logger
log_file_name = f'{model_name}.log'
log_file = os.path.join(model_dir, log_file_name)
dtfmt = '%y%m%d-%H:%M'
logfmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'

logging.basicConfig(filename=log_file,
                    filemode='w',
                    level=logging.INFO,
                    format=logfmt,
                    datefmt=dtfmt,
                    force=True)

logger = logging.getLogger(name=__name__)

#%% Model and image processor

model_checkpoint = model_info.get('hf_checkpoint')
processor = RTDetrImageProcessor.\
    from_pretrained(model_checkpoint, **processor_params)

# Load model from a pretrained checkpoint
training_checkpoint = model_info.get('training_checkpoint')
model = RTDetrV2ForObjectDetection.\
    from_pretrained(training_checkpoint,
                    id2label=id2label,
                    label2id=label2id,
                    anchor_image_size=None,
                    ignore_mismatched_sizes=True)

#%% Datasets
train_dataset = DETRdataset(data=train_df,
                            image_processor=processor,
                            image_dir=train_image_dir,
                            file_name_col=file_name_col,
                            label_id_col='label',
                            bbox_col=bbox_col,
                            transforms=train_transforms)

val_dataset = DETRdataset(data=val_df,
                          image_processor=processor,
                          image_dir=val_image_dir,
                          file_name_col=file_name_col,
                          label_id_col='label',
                          bbox_col=bbox_col,
                          transforms=val_transforms)

#%% Training
def collate_fn(batch):
    """
    Collates a batch of data samples into a single dictionary for model input.
    """
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

# Set the evaluation metrics
eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.5, id2label=id2label)
training_arguments = TrainingArguments(**training_args)

trainer = Trainer(model=model,
                  args=training_arguments,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset,
                  processing_class=processor,
                  data_collator=collate_fn,
                  compute_metrics=eval_compute_metrics_fn)

trainer.train()

