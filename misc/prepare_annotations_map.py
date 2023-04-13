
###################################################################################
##Now we get masks from Mapillary Dataset
###################################################################################

import os
import time
import datetime
import json
import numpy as np
import operator
from PIL import Image, ImagePalette # For indexed images
import matplotlib # For Matlab's color maps

from pycocotools import coco
from pycocotools import mask

ID = 1

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError:
        return False
    return True


def addCat(annotation_file):
    if not annotation_file == None:
        print('loading annotations into memory...')
        tic = time.time()
        with open(annotation_file) as f:
            dataset = json.load(f)
        # x = dataset['categories']
        print(dataset['categories'][0])

        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        entry1 = {'id': 91, 'name': 'flood', 'supercategory': 'flood'}
        entry2 = {'id': 92, 'name': 'house', 'supercategory': 'house'}

        # level entries
        #entry3 = {'id': 1, 'name': 'no level', 'supercategory': 'level'}
        entry4 = {'id': 0, 'name': 'level0', 'supercategory': 'level'}
        entry5 = {'id': 1, 'name': 'level1', 'supercategory': 'level'}
        entry6 = {'id': 2, 'name': 'level2', 'supercategory': 'level'}
        entry7 = {'id': 3, 'name': 'level3', 'supercategory': 'level'}
        entry8 = {'id': 4, 'name': 'level4', 'supercategory': 'level'}
        entry9 = {'id': 5, 'name': 'level5', 'supercategory': 'level'}
        entry10 = {'id': 6, 'name': 'level6', 'supercategory': 'level'}
        entry11 = {'id': 7, 'name': 'level7', 'supercategory': 'level'}
        entry12 = {'id': 8, 'name': 'level8', 'supercategory': 'level'}
        entry13 = {'id': 9, 'name': 'level9', 'supercategory': 'level'}
        entry14 = {'id': 10, 'name': 'level10', 'supercategory': 'level'}

        ##Adding additional categories to val file
        with open(annotation_file) as f:
            data = json.load(f)

        data['categories'].append(entry1)
        data['categories'].append(entry2)

        data["level_categories"] = []

        #data['level_categories'].append(entry3)
        data['level_categories'].append(entry4)
        data['level_categories'].append(entry5)
        data['level_categories'].append(entry6)
        data['level_categories'].append(entry7)
        data['level_categories'].append(entry8)
        data['level_categories'].append(entry9)
        data['level_categories'].append(entry10)
        data['level_categories'].append(entry11)
        data['level_categories'].append(entry12)
        data['level_categories'].append(entry13)
        data['level_categories'].append(entry14)

        with open(annotation_file, 'w') as f:
            json.dump(data, f)

        # check file is json
        with open(annotation_file) as f:
            text = json.load(f)
        strJson = json.dumps(text)

        print(annotation_file)
        print(is_json(strJson))


def updateLicense(annotation_file):
    entry = {'url': "not important", 'name': "Flood Dataset Images", 'id': 9}

    with open(annotation_file) as f:
        data = json.load(f)

    data['licenses'].append(entry)

    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    # check file is json
    with open(annotation_file) as f:
        text = json.load(f)
    strJson = json.dumps(text)
    print(annotation_file)
    print(is_json(strJson))


def updateCocoAnnotations(annotation_file):
    with open(annotation_file) as f:
        data = json.load(f)

    for ann in data['annotations']:
        ann['level_id'] = 1
        print(ann['id'])

    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    # check file is json
    with open(annotation_file) as f:
        text = json.load(f)
    strJson = json.dumps(text)
    print(annotation_file)
    print(is_json(strJson))


def getImageList(imageList, loc):
    ##Making list from directory path

    # os.chdir(os.path.dirname("/home/pchaudha/LabelledImages/allImages/Flood_101.png"))
    os.chdir(os.path.dirname(loc))
    print(os.getcwd())
    filenames = os.listdir(os.getcwd())
    filenames = list(filenames)
    filenames.sort()
    print(len(filenames))

    if not os.path.exists(imageList):
        open(imageList, 'w').close()
    f = open(imageList, 'w')
    for item in filenames:
        f.write("%s\n" % item)
    f.close()


def segmentationToCocoMask(labelMap, labelId):
    '''
    Encodes a segmentation mask using the Mask API.
    :param labelMap: [h x w] segmentation map that indicates the label of each pixel
    :param labelId: the label from labelMap that will be encoded
    :return: Rs - the encoded label mask for label 'labelId'
    '''
    labelMask = labelMap == labelId
    labelMask = np.expand_dims(labelMask, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)
    # print(labelMask)
    Rs = mask.encode(labelMask)
    assert len(Rs) == 1
    Rs = Rs[0]

    return Rs


def getCatId(labelId):
    # print(labelId)

    # if else to select the proper category id
    if (labelId == 19):
        #person
        idTemp = 1
    elif (labelId == 20):
        # person
        idTemp = 1
    elif (labelId == 21):
        # person
        idTemp = 1
    elif (labelId == 22):
        # person
        idTemp = 1
    elif (labelId == 52):
        # bicycle
        idTemp = 2
    elif (labelId == 54):
        # bus
        idTemp = 6
    elif (labelId == 55):
        # car
        idTemp = 3
    elif (labelId == 17):
        # building
        idTemp = 92
    else:
        pass

    return idTemp


def segmentationToCocoResult(labelMap, imgId, instance_ids_array, annotation_file, stuffStartId=1):
    '''
    Convert a segmentation map to COCO stuff segmentation result format.
    :param labelMap: [h x w] segmentation map that indicates the label of each pixel
    :param imgId: the id of the COCO image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Get stuff labels
    shape = labelMap.shape

    # to remove later
    unique, counts = np.unique(labelMap, return_counts=True)
    print(dict(zip(unique, counts)))

    if len(shape) != 2:
        raise Exception(('Error: Image has %d instead of 2 channels! Most likely you '
                         'provided an RGB image instead of an indexed image (with or without color palette).') % len(
            shape))
    [h, w] = shape
    assert h > 0 and w > 0
    labelsAll = np.unique(labelMap)

    # print(labelsStuff)

    # To get level_id
    # level_dict = imageOper(labelMap, labelsAll)

    # remove the level ids from the list as we dont want annotations for them
    # level = labelsAll[np.logical_and(labelsAll>40,labelsAll<52)]
    # print(level)
    # remove level id from labelsAll array
    # labelsAll = np.setdiff1d(labelsAll, level)

    labelsStuff = [i for i in labelsAll if i >= stuffStartId]

    # Add stuff annotations
    anns = []
    for labelId in labelsStuff:
        instance_id_temp = np.copy(instance_ids_array)
        #make the array values 255 where the label is not labelID
        #then we can extract masks of various instances of the object
        instance_id_temp[labelMap != labelId] = 255
        temp = np.unique(labelMap)

        unique_1 = np.unique(instance_id_temp, return_counts=False)
        # print(dict(zip(unique_1, counts_1)))

        #remove 255 from there as its a background class
        unique_1 = unique_1[:-1]
        for instance_id in unique_1:

            # Get cat Id for labelId
            category_id = getCatId(labelId)
            # print(labelId)

            # Create mask and encode it
            Rs = segmentationToCocoMask(instance_id_temp, instance_id)
            # print("before dict")
            # print(Rs['counts'])
            # print("Rs2 after dict")
            # Rs2 = dict(Rs)
            # print(Rs2)
            # print("Rs decoded")
            Rs['counts'] = Rs['counts'].decode('ascii')

            # Get bbox from encoded mask
            bb = mask.toBbox(Rs)
            bb = bb.astype(float)
            bb = bb.tolist()
            # print(bb)

            # Get area from the encoded mask
            ar = mask.area(Rs)
            # print("area")
            ar = ar.astype(float)

            # Get unique for the mask
            global ID
            identity = ID
            ID = ID + 1

            # Create annotation data and add it to the list
            anndata = {}
            anndata['iscrowd'] = int(0)
            anndata['image_id'] = str(imgId)
            anndata['bbox'] = bb
            anndata['id'] = identity
            anndata['category_id'] = int(category_id)
            anndata['segmentation'] = Rs
            anndata['area'] = ar
            anndata['level_id'] = int(0)

            print(anndata)
            '''
            ##Add annotations to the file
            with open(annotation_file) as f:
                data = json.load(f)
            data['annotations'].append(anndata)
            with open(annotation_file, 'w') as f:
                json.dump(data,f)
            '''

            # test for coco.py in pycocotools
            # m = coco.annToMask(anndata)
            # print(m)

            anns.append(anndata)
    return anns


def replaceBg(labelMap, ImgID):
    obj_list = [19, 20, 21, 22, 52, 54, 55, 17]
    name = 'mask/' + str(ImgID) + '.png'
    for i in range(0, 66):
        if i not in obj_list:
            labelMap[labelMap == i] = 0
        else:
            pass

    print("Label map")
    img = Image.fromarray(labelMap, mode='L')
    img.save(name)
    #img.show()
    return labelMap


def pngToCocoResult(pngPath, imgId, instance_ids_array, annotation_file, stuffStartId=1):
    '''
    Reads an indexed .png file with a label map from disk and converts it to COCO result format.
    :param pngPath: the path of the .png file
    :param imgId: the COCO id of the image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Read indexed .png file from disk
    im = Image.open(pngPath)
    labelMap = np.array(im)

    # Replace all  category ids which are not needed
    # with 0s
    labelMap = replaceBg(labelMap, imgId)
    # print(labelMap.shape)

    # Convert label map to COCO result format
    anns = segmentationToCocoResult(labelMap, imgId, instance_ids_array, annotation_file, stuffStartId)
    return anns

if __name__ == '__main__':

    mapillary_ids = []
    with open('/scratch/pchaudha/ranking_loss/k/k1/gt_val_reg.json') as f:
        val_map = json.load(f)

    for key, value in val_map.items():
        # add image path to load it later
        if "FloodImages_" in key:
            continue
        else:
            key = key.replace('.jpg', ' ')
            key = key.replace('mapillary-vistas-dataset_public_v1.1/training/images/', ' ')
            split = key.split()
            key = split[1]
        mapillary_ids.append(str(key))

    with open('/scratch/pchaudha/ranking_loss/k/k1/gt_train_reg.json') as f:
        train_map = json.load(f)

    for key, value in train_map.items():
        # add image path to load it later
        if "FloodImages_" in key:
            continue
        else:
            key = key.replace('.jpg', ' ')
            key = key.replace('mapillary-vistas-dataset_public_v1.1/training/images/', ' ')
            split = key.split()
            key = split[1]
        mapillary_ids.append(str(key))


    with open('/scratch/pchaudha/ranking_loss/k/k1/gt_test.json') as f:
        test_map = json.load(f)

    for key, value in test_map.items():
        # add image path to load it later
        if "FloodImages_" in key:
            continue
        else:
            key = key.replace('.jpg', ' ')
            key = key.replace('mapillary-vistas-dataset_public_v1.1/training/images/', ' ')
            split = key.split()
            key = split[1]
        mapillary_ids.append(str(key))

    #to copy images from their source folder to new folder
    #which will contain both flood images and mapillary images
    import shutil
    src_dir = "/scratch2/mapillary-vistas-dataset_public_v1.1/training/images/"
    des_dir = "/scratch/pchaudha/ranking_loss/mask_exp/mask_rcnn/images/"
    for name in mapillary_ids:
        image_path = src_dir + str(name) + '.jpg'
        shutil.copy(image_path, des_dir)


    #get info from panoptic.json file
    with open('/scratch2/mapillary-vistas-dataset_public_v1.1/training/panoptic/panoptic_2018.json') as f:
        panoptic = json.load(f)

    image_prefix = "/scratch2/mapillary-vistas-dataset_public_v1.1/training/images/"
    label_prefix = "/scratch2/mapillary-vistas-dataset_public_v1.1/training/labels/"
    instance_prefix = "/scratch2/mapillary-vistas-dataset_public_v1.1/training/instances/"
    map_annotation_file = "/scratch/pchaudha/ranking_loss/mask_exp/mapillary.json"

    map_dict = {}
    ann_pan = panoptic['annotations']
    for j in ann_pan:
        temp_id = j['image_id']
        if temp_id in mapillary_ids:
            map_dict[temp_id] = j['segments_info']
        else:
            continue

    # image_id = '-4VHR0_p6lfAJ3Id2RXozg'
    # path = label_prefix+ str(image_id) + '.png'
    # pngToCocoResult(pngPath=path, imgId=key, annotation_file=map_annotation_file)

    annotations_dict = {}
    ann_final = []
    img_info = []
    for key, value in map_dict.items():
        path = label_prefix + str(key) + '.png'

        # load instance image for instance label ids
        instance_path = instance_prefix + str(key) + '.png'
        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)
        instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

        #for image info part in annotation file
        [ht,wd] = instance_array.shape
        im_data = {}
        im_data['id'] = str(key)
        im_data['date_captured'] = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        im_data['license'] = int(10)
        im_data['height'] = ht
        im_data['flickr_url'] = 'No url'
        im_data['width'] = wd
        im_data['file_name'] = str(key)+'.jpg'
        im_data['coco_url'] = 'Not in coco dataset'

        img_info.append(im_data)


        ann = pngToCocoResult(pngPath=path, imgId=key,instance_ids_array=instance_ids_array , annotation_file=map_annotation_file)
        for p in ann:
            ann_final.append(p)
        annotations_dict[key] = ann

        print()

    #save mapillary dict with annotations with image_id as key
    # with open('mapillary_dict.json', 'w') as fp:
    #     json.dump(annotations_dict, fp, indent=4)


    with open('/scratch/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco/annotations/instances_test2017.json') as config_file:
        config = json.load(config_file)
    # in this example we are only interested in the labels
    licenses = config['licenses']
    categories = config['categories']
    level_cat = config['level_categories']
    entry1 = {'url': 'https://www.mapillary.com/dataset/vistas', 'description': 'Mapillary instance data', 'contributor': 'Mapillary', 'year': 2019, 'date_created':'02 Oct 2019', 'version': 3.0 }
    entry2 = {'url':'not important', 'name': 'Mapillary images', 'id': int(10)}
    licenses.append(entry2)

    data = {}
    data['annotations'] = ann_final
    data['licenses'] = licenses
    data['categories'] = categories
    data['level_categories'] = level_cat
    data['info'] = entry1
    data['images'] = img_info

    # with open(map_annotation_file, 'w') as f:
    #     json.dump(data, f)


print("Done")

