
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

import math
#level to cm conversion
def level_to_cm(level_ref,level):
    floor = int(math.floor(level))
    print(floor)
    ceil = int(math.ceil(level))
    print(ceil)
    percent = level - floor

    level_cm = (level_ref[ceil][1]- level_ref[ceil][0])*percent + level_ref[floor][1]
    return level_cm


#exp2
#with trimmed mean
import scipy.stats
def compute_accuracy_exp2(image_ids):
    labels = []
    preds = []
    labels_cm = []
    preds_cm = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_level_id =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]

        #to check if no detections
        #or bbox length/class/level_ids is zero
        if (r['level_ids'].size ==0):
            r['level_ids'] = np.append(r['level_ids'],0)

        pred_level_id = r['level_ids']
        #trim mean
        gt_level = scipy.stats.trim_mean(gt_level_id,0.2)
        pred_level = scipy.stats.trim_mean(pred_level_id,0.2)

        gt_level_cm = level_to_cm(level_ref, gt_level)
        pred_level_cm = level_to_cm(level_ref, pred_level)
        #print(gt_level)
        #print(pred_level)
        labels.append(gt_level)
        preds.append(pred_level)
        labels_cm.append(gt_level_cm)
        preds_cm.append(pred_level_cm)

    return labels, preds, labels_cm, preds_cm

#without trimmed mean
def compute_accuracy_without(image_ids):
    labels = []
    preds = []
    labels_cm = []
    preds_cm = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_level_id =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]

        #to check if no detections
        #or bbox length/class/level_ids is zero
        if (r['level_ids'].size ==0):
            r['level_ids'] = np.append(r['level_ids'],0)

        pred_level_id = r['level_ids']
        #trim mean
        gt_level = np.mean(gt_level_id)
        pred_level = np.mean(pred_level_id)
        gt_level_cm = level_to_cm(level_ref, gt_level)
        pred_level_cm = level_to_cm(level_ref, pred_level)
        print(gt_level)
        print(pred_level)
        labels.append(gt_level)
        preds.append(pred_level)
        labels_cm.append(gt_level_cm)
        preds_cm.append(pred_level_cm)
    return labels, preds, labels_cm, preds_cm


%matplotlib inline



# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to Shapes trained weights
SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")

# Run one of the code blocks

# Shapes toy dataset
# import shapes
# config = shapes.ShapesConfig()

# MS COCO Dataset
import coco_fold2         # TODO :change
config = coco_fold2.CocoConfig()        # TODO :change
#COCO_DIR = "/home/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco"  # TODO: enter value here
COCO_DIR = "/scratch/pchaudha/ranking_loss/mask_exp/exp_1/k/k2"         # TODO :change

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Build validation dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco_fold2.CocoDataset()
    dataset.load_coco(COCO_DIR, "test", year=2017)

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}\nLevels:{}".format(len(dataset.image_ids), dataset.class_names, dataset.level_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set weights file path
if config.NAME == "shapes":
    weights_path = SHAPES_MODEL_PATH
elif config.NAME == "coco":
    weights_path = COCO_MODEL_PATH
# Or, uncomment to load the last model you trained
weights_path = model.find_last()[1]

# Load weights

#weights_path = "/home/pchaudha/mask-rcnn/Mask_RCNN-master/logs/coco20180723T1007/mask_rcnn_coco_0160.h5"
#weights_path = "/scratch/pchaudha/mask-rcnn/Mask_RCNN-master/logs/coco20180723T1007/mask_rcnn_coco_0160.h5"
weights_path = "/scratch/pchaudha/ranking_loss/mask_exp/exp_1/logs/fold2/mask_rcnn_coco_0160.h5"
model.load_weights(weights_path, by_name=True)
print("Loading weights ", weights_path)
print("model loaded")

#level table for interpolation

level_ref = np.array([[0.0,0.0],
         [0.0,1.0],
         [1.0,10.0],
         [10.0, 21.25],
         [21.25,42.5],
         [42.5,63.75],
         [63.75,85.0],
         [85.0,106.25],
         [106.25,127.25],
         [127.25,148.75],
         [148.75,170.0]])

print("with Trimmed mean")

labels, preds, labels_cm, preds_cm = compute_accuracy_exp2(dataset.image_ids)

n = len(labels)
labels = np.array(labels)
#print(labels)
#print(preds)
preds = np.array(preds)
diff = preds - labels
#print(diff)
diff2 = np.array(diff)
diff2 = abs(diff2)
print(diff2)
print("RMSE loss in level")
print(np.mean(diff2))


labels_cm = np.array(labels_cm)
preds_cm = np.array(preds_cm)
d = preds_cm - labels_cm

d2 = np.array(d)
d2 = abs(d2)
print("RMSE loss in cms")
print(np.mean(d2))


print("without trimmed mean")

labels, preds, labels_cm, preds_cm = compute_accuracy_without(dataset.image_ids)

n = len(labels)
labels = np.array(labels)
print(labels)
print(preds)
preds = np.array(preds)
diff = preds - labels
#print(diff)
diff2 = np.array(diff)
diff2 = abs(diff2)
print(diff2)
print("RMSE loss in cms")
print(np.mean(diff2))

labels_cm = np.array(labels_cm)
preds_cm = np.array(preds_cm)
d = preds_cm - labels_cm

d2 = np.array(d)
d2 = abs(d2)
print("RMSE loss in level")
print(np.mean(d2))

print("Done")
