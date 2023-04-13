import json
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import skimage
from skimage.transform import resize
import model as modellib
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_path, labels, transforms = None):
        'Initialization'
        #self.image_ids = image_ids
        self.labels = labels
        self.img_path = img_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_path)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #image_id = self.image_ids[index]
        #print("Image id: ",image_id)

        # Load data and get label
        X = Image.open(self.img_path[index])
        #print("Image channels: ", X.layers)
        if hasattr(X, 'layers'):
            if X.layers != 3:
                #print("Not 3 channels: ", X.layers)
                pass
        else:
            print("Doesnt have layers")
        if X.mode == 'CMYK':
            X = X.convert('RGB')
        y = float(self.labels[index])
        X = X.resize(512, 512)
        #print(y)

        return X, y


#level to cm table
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


# level to cm conversion
def level_to_cm(level_ref, level):
    floor = int(math.floor(level))
    #print(floor)
    ceil = int(math.ceil(level))
    #print(ceil)
    percent = level - floor

    level_cm = (level_ref[ceil][1] - level_ref[ceil][0]) * percent + level_ref[floor][1]
    return level_cm

params = {'batch_size': 1,
              'shuffle': False}

#load test data from json file
test_img_path = []
test_labels = []

# check file is json
# with open('imageList_test.json') as f:
with open('/scratch/pchaudha/ranking_loss/k/k5/gt_test.json') as f:
    train = json.load(f)

for key, value in train.items():
    # add image path to load it later
    test_img_path.append(str(key))

    # add image label
    test_labels.append(value)
    print(key, value)

#test_generator = DataGenerator(img_path=test_img_path, labels=test_labels, **params)

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# MS COCO Dataset
import coco_fold5
config = coco_fold5.CocoConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
MODEL_DIR = "/scratch/pchaudha/ranking_loss/mask_exp/exp_1/logs/"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = "/scratch2/logs/rank/fold5/coco20191014T1801/mask_rcnn_coco_0160.h5"
model.load_weights(weights_path, by_name=True)
print("Loading weights ", weights_path)
print("model loaded")


# without trimmed mean
def compute_accuracy_without(img_path, img_label):
    labels = []
    preds = []
    labels_cm = []
    preds_cm = []

    for i, val in enumerate(img_path):
        print(val)
        f.write(val)
        f.write("\n")
        # Load image
        image = skimage.io.imread(val)
        image = resize(image, [512, 512])

        gt_level = img_label[i]

        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]

        # to check if no detections
        # or bbox length/class/level_ids is zero
        if (r['level_ids'].size == 0):
            r['level_ids'] = np.append(r['level_ids'], 0)

        pred_level_id = r['level_ids']
        print("pred_level_id: ", pred_level_id)
        f.write("pred_level_id: ")
        f.write(str(pred_level_id))
        f.write("\n")

        # remove all zero entries
        pred_level_without_zeroes = [i for i in pred_level_id if i != 0]

        # if list is empty
        if len(pred_level_without_zeroes) == 0:
            pred_level_without_zeroes.append(0)

        print("pred_level_without_zeroes: ", pred_level_without_zeroes)
        print("ground truth: ", gt_level)

        f.write("pred_level_without_zeroes: ")
        f.write(str(pred_level_without_zeroes))
        f.write("\n")
        f.write("ground truth: ")
        f.write(str(gt_level))
        f.write("\n")

        # take mean
        pred_level = np.mean(pred_level_without_zeroes)
        print("pred after mean: ", pred_level)

        f.write("pred after mean: ")
        f.write(str(pred_level))
        f.write("\n")

        gt_level_cm = level_to_cm(level_ref, gt_level)
        pred_level_cm = level_to_cm(level_ref, pred_level)
        labels.append(gt_level)
        preds.append(pred_level)
        labels_cm.append(gt_level_cm)
        preds_cm.append(pred_level_cm)

    return labels, preds, labels_cm, preds_cm

#write output to file
f = open('fold5_output.txt', 'w')
labels, preds, labels_cm, preds_cm = compute_accuracy_without(test_img_path, test_labels)

n = len(labels)
labels = np.array(labels)
print("gt of all images: ")
print(labels)

f.write("gt of all images: ")
f.write(str(labels))
f.write("\n")

print("pred of all images: ")
print(preds)

f.write("pred of all images: ")
f.write(str(preds))
f.write("\n")

preds = np.array(preds)
diff = preds - labels
# print(diff)
diff2 = np.array(diff)
diff2 = abs(diff2)
#print(diff2)
print("RMSE loss in cms: ")
diff2_res = np.mean(diff2)
print(diff2_res)

f.write("RMSE loss in cms: ")
f.write(str(diff2_res))
f.write("\n")

labels_cm = np.array(labels_cm)
preds_cm = np.array(preds_cm)
d = preds_cm - labels_cm

d2 = np.array(d)
d2 = abs(d2)
print("RMSE loss in level : ")
d2_res = np.mean(d2)
print(d2_res)


f.write("RMSE loss in level: ")
f.write(str(d2_res))
f.write("\n")
f.close()

print("Done")