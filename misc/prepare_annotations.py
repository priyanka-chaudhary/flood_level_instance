"""
For changing annotations file.
-Adding flood dataset images
-Adding level_id for every annotation
-Extracting level_id info from groundtruth images of flood dataset
-Setting level_id to "no level" for every coco dataset annotation

Written by: Priyanka Chaudhary
Some parts taken from pycocotool functions
"""

import os
import time
import datetime
import json
import shutil
import numpy as np
import operator

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError:
        return False
    return True



if __name__ == '__main__':

##############################################################################
## Flood Dataset images
##############################################################################

    #load all annotation files: train, val, test
    train_file = "/scratch/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco/annotations/instances_train2017.json"
    val_file = "/scratch/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco/annotations/instances_val2017.json"
    test_file = "/scratch/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco/annotations/instances_test2017.json"

    with open(train_file) as f:
        train = json.load(f)

    with open(val_file) as f:
        val = json.load(f)

    with open(test_file) as f:
        test = json.load(f)

    ann_test = test['annotations']
    ann_val = val['annotations']
    ann_train = train['annotations']
    img_info = test['images']

    id_name_dict = {}
    name_id_dict = {}
    img_info_dict = {}
    for p in img_info:
        id = p['id']
        file_name = p['file_name']
        if file_name.startswith("Flood"):
            id_name_dict[file_name] = id
            name_id_dict[id] = file_name
            file_name = file_name[:-4]
            #change the id of flood images from number to Flood_x
            p['id'] = file_name
            img_info_dict[file_name] = p

    img_info = val['images']
    for p in img_info:
        id = p['id']
        file_name = p['file_name']
        if file_name.startswith("Flood"):
            id_name_dict[file_name] = id
            name_id_dict[id] = file_name
            file_name = file_name[:-4]
            # change the id of flood images from number to Flood_x
            p['id'] = file_name
            img_info_dict[file_name] = p

    img_info = train['images']
    for p in img_info:
        id = p['id']
        file_name = p['file_name']
        if file_name.startswith("Flood"):
            id_name_dict[file_name] = id
            name_id_dict[id] = file_name
            file_name = file_name[:-4]
            # change the id of flood images from number to Flood_x
            p['id'] = file_name
            img_info_dict[file_name] = p


    #write flood img_info dict
    # with open('/scratch/pchaudha/ranking_loss/mask_exp/flood_img_info.json', 'w') as f:
    #     json.dump(img_info_dict, f, indent=4)


    #to copy images from their source folder to new folder
    #which will contain both flood images and mapillary images
    # src_dir = "/scratch2/Images/After cleaning/dataset/FloodImages_newname/"
    # des_dir = "/scratch/pchaudha/ranking_loss/mask_exp/mask_rcnn/images/"
    # for name, value in id_name_dict.items():
    #     if name == 'Flood_535.jpg':
    #         pass
    #     else:
    #         image_path = src_dir + str(name)
    #         shutil.copy(image_path, des_dir)

    ann_dict = {}
    for i in ann_test:
        img_id = i['image_id']
        file_name = name_id_dict[img_id]
        if file_name.startswith("Flood"):
            file_name = file_name[:-4]
            ann_dict.setdefault(file_name, [])
            ann_dict[file_name].append(i)

    for i in ann_val:
        img_id = i['image_id']
        if img_id in name_id_dict:
            file_name = name_id_dict[img_id]
        else:
            continue
        if file_name.startswith("Flood"):
            file_name = file_name[:-4]
            ann_dict.setdefault(file_name, [])
            ann_dict[file_name].append(i)

    for i in ann_train:
        img_id = i['image_id']
        if img_id in name_id_dict:
            file_name = name_id_dict[img_id]
        else:
            continue
        if file_name.startswith("Flood"):
            file_name = file_name[:-4]
            ann_dict.setdefault(file_name, [])
            ann_dict[file_name].append(i)


    for key, values in ann_dict.items():
        for i in values:
            i['image_id'] = str(key)

    #save the ann_dic
    # with open('flood_dict.json', 'w') as fp:
    #     json.dump(ann_dict, fp, indent=4)


    ann_flood_changed = []
    #to change the image_id in the annotations to file_name
    #something like flood_x.jpg
    for key, value in ann_dict.items():
        img_ann = []
        for p in value:
            key = key[:-4]
            p['image_id'] = str(key)
            img_ann.append(p)
        ann_flood_changed.append()



print("Done")

