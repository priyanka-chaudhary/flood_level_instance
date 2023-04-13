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
import coco
config = coco.CocoConfig()
COCO_DIR = "/home/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco"  # TODO: enter value here

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
    #dataset = shapes.ShapesDataset()
    #dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    pass
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
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
weights_path = "/home/pchaudha/mask-rcnn/Mask_RCNN-master/logs/coco20180424T1345/mask_rcnn_coco_0160.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
print("model loaded")

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_level_id =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))
print(gt_class_id)
print(gt_level_id)

import skimage

name = str(info["id"])
# Load image
#name = "Flood_7131.jpg"
#path = "/home/pchaudha/Images/After cleaning/dataset/FloodImages_newname/"
#path = "/home/pchaudha/LabelledImages/test/FloodImages__dataset11/img/"

#name = "ChristianPhilippeMichel.jpg"
#path = "/home/pchaudha/Images/After cleaning/Extra/"

#t_path = path+name
#image = skimage.io.imread(t_path)
#print(image.shape)
# If grayscale. Convert to RGB for consistency.
#if image.ndim != 3:
#    image = skimage.color.gray2rgb(image)

# Run object detection
results = model.detect([image], verbose=1)

# Display results
#ax = get_ax(1)
rows = 1
cols = 1
size = 16
_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
r = results[0]
print(r['class_ids'])
print(r['scores'])
print(r['level_ids'])
print(r['level_scores'])
visualize.display_instances(name,image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['level_ids'], dataset.level_names, r['level_scores'], r['scores'], ax=ax,
                            title="Predictions")
#log("gt_class_id", gt_class_id)
#log("gt_bbox", gt_bbox)
#log("gt_mask", gt_mask)

AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id,
                                          r['rois'], r['class_ids'], r['scores'])

AP2, precisions2, recalls2, overlaps2 = utils.compute_ap(gt_bbox, gt_level_id, r['rois'], r['level_ids'], r['level_scores'])


AP4, precisions4, recalls4,overlaps4 = utils.compute_ap_level(gt_bbox, gt_class_id,
               r['rois'], r['class_ids'], r['scores'],
                gt_level_id, r['level_ids'], r['level_scores'])

visualize.plot_precision_recall(AP, precisions, recalls)

visualize.plot_precision_recall(AP2, precisions2, recalls2)

visualize.plot_precision_recall(AP4, precisions4, recalls4)

# Draw precision-recall curve
AP3, precisions3, recalls3, overlaps3 = utils.compute_ap_condensed(gt_bbox, gt_level_id, r['rois'], r['level_ids'], r['level_scores'])
visualize.plot_precision_recall(AP3, precisions3, recalls3)

# Grid of ground truth objects and their predictions
visualize.plot_overlaps_l(gt_class_id,gt_level_id, r['class_ids'] ,r['level_ids'], r['level_scores'],
                        overlaps, dataset.level_names, dataset.class_names)

image_ids = dataset.image_ids
APs = []
for image_id in image_ids:
    # Load image
    image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_level_id =\
        modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
    # Run object detection
    results = model.detect([image], verbose=0)
    # Compute AP
    r = results[0]
    AP, precisions, recalls, overlaps =\
            utils.compute_ap_condensed(gt_bbox, gt_level_id, r['rois'], r['level_ids'], r['level_scores'])
    APs.append(AP)
print("mAP @ IoU=50: ", np.mean(APs))




#level precision recall curve
AP2, precisions2, recalls2, overlaps2 = utils.compute_ap(gt_bbox, gt_level_id, r['rois'], r['level_ids'], r['level_scores'])
utils.compute_confusionMatrix(gt_bbox, gt_level_id,r['rois'], r['level_ids'], r['level_scores'])


visualize.plot_precision_recall(AP, precisions, recalls)

# Grid of ground truth objects and their predictions
visualize.plot_overlaps_l(gt_level_id, r['level_ids'], r['level_scores'],
                        overlaps, dataset.level_names)


