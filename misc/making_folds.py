
import json

#now we will find the annotations for the image_ids in mapillary
#or flood annotation file
#load mapillary annotation file
with open('/scratch/pchaudha/ranking_loss/mask_exp/mapillary_dict.json') as f:
    mapi_dict = json.load(f)

#load the flood annotation file
with open('/scratch/pchaudha/ranking_loss/mask_exp/flood_dict.json') as f:
    flood_dict = json.load(f)

#the ids in the annotations(flood) have repetition
#so assign new ids altogether
#load the flood_dict.json file

# new_id = 50000
# for key, value in flood_dict.items():
#     for i in value:
#         i['id'] = new_id
#         new_id = new_id + 1
#
# #add new flood_dict file with new ids
# with open('/scratch/pchaudha/ranking_loss/mask_exp/flood_dict_new.json', 'w') as f1:
#     json.dump(flood_dict, f1)


with open('/scratch/pchaudha/ranking_loss/mask_exp/mapillary.json') as config_file:
    config = json.load(config_file)
licenses = config['licenses']
categories = config['categories']
level_cat = config['level_categories']
info_map = config['info']
img_info_map = config['images']

#get image info from the whole mapillary.json
#and from flood.json
with open('/scratch/pchaudha/ranking_loss/mask_exp/flood_img_info.json') as config_file:
    flood_img_info = json.load(config_file)

#########################################################################################
############################### Test File ###############################################
#########################################################################################

#first collect the image_ids from different train, val and test files
test_img_ids = []
with open('/scratch/pchaudha/ranking_loss/k/k5/gt_test.json') as f:
    test = json.load(f)

for key, value in test.items():
    # add image path to load it later
    key = key.replace('.jpg', '')
    if 'FloodImages_newname/' in key:
        key = key.replace('FloodImages_newname/', ' ')
        key = key.split()[2]
    elif 'training/images/' in key:
        key = key.replace('training/images/', ' ')
        key = key.split()[1]
    test_img_ids.append(str(key))


test_ann = []
#now find the image ids from various lists in the annotation file
for img in test_img_ids:
    if img in mapi_dict:
        ann = mapi_dict[img]
    elif img in flood_dict:
        ann = flood_dict[img]
    else:
        print("Not possible")
        print(img)

    for i in ann:
        test_ann.append(i)


test_img_info = []
for img in test_img_ids:
    if img in mapi_dict:
        for q in img_info_map:
            #id_map = img_info_map['id']
            if str(q['id']) == img:
                test_img_info.append(q)
    elif img in flood_img_info:
        info = flood_img_info[img]
        test_img_info.append(info)


data = {}
data['annotations'] = test_ann
data['licenses'] = licenses
data['categories'] = categories
data['level_categories'] = level_cat
data['info'] = info_map
data['images'] = test_img_info

with open('/scratch/pchaudha/ranking_loss/mask_exp/k_new/k5/gt_test_mask.json', 'w') as f1:
    json.dump(data, f1)

#########################################################################################
############################### Train File ##############################################
#########################################################################################

train_img_ids = []
with open('/scratch/pchaudha/ranking_loss/k/k5/gt_train_reg.json') as f:
    train = json.load(f)

for key, value in train.items():
    # add image path to load it later
    key = key.replace('.jpg', '')
    if 'FloodImages_newname/' in key:
        key = key.replace('FloodImages_newname/', ' ')
        key = key.split()[2]
    elif 'training/images/' in key:
        key = key.replace('training/images/', ' ')
        key = key.split()[1]
    train_img_ids.append(str(key))

train_ann = []
#now find the image ids from various lists in the annotation file
for img in train_img_ids:
    if img in mapi_dict:
        ann = mapi_dict[img]
    elif img in flood_dict:
        ann = flood_dict[img]
    else:
        print("Not possible")
        print(img)

    for i in ann:
        train_ann.append(i)


train_img_info = []
for img in train_img_ids:
    if img in mapi_dict:
        for q in img_info_map:
            #id_map = img_info_map['id']
            if str(q['id']) == img:
                train_img_info.append(q)
    elif img in flood_img_info:
        info = flood_img_info[img]
        train_img_info.append(info)


data_train = {}
data_train['annotations'] = train_ann
data_train['licenses'] = licenses
data_train['categories'] = categories
data_train['level_categories'] = level_cat
data_train['info'] = info_map
data_train['images'] = train_img_info

with open('/scratch/pchaudha/ranking_loss/mask_exp/k_new/k5/gt_train_mask.json', 'w') as f2:
    json.dump(data_train, f2)

#########################################################################################
############################### Val File ################################################
#########################################################################################

val_img_ids = []
with open('/scratch/pchaudha/ranking_loss/k/k5/gt_val_reg.json') as f:
    val = json.load(f)

for key, value in val.items():
    # add image path to load it later
    key = key.replace('.jpg', '')
    if 'FloodImages_newname/' in key:
        key = key.replace('FloodImages_newname/', ' ')
        key = key.split()[2]
    elif 'training/images/' in key:
        key = key.replace('training/images/', ' ')
        key = key.split()[1]
    val_img_ids.append(str(key))

val_ann = []
#now find the image ids from various lists in the annotation file
for img in val_img_ids:
    if img in mapi_dict:
        ann = mapi_dict[img]
    elif img in flood_dict:
        ann = flood_dict[img]
    else:
        print("Not possible")
        print(img)

    for i in ann:
        val_ann.append(i)


val_img_info = []
for img in val_img_ids:
    if img in mapi_dict:
        for q in img_info_map:
            #id_map = img_info_map['id']
            if str(q['id']) == img:
                val_img_info.append(q)
    elif img in flood_img_info:
        info = flood_img_info[img]
        val_img_info.append(info)


data_val = {}
data_val['annotations'] = val_ann
data_val['licenses'] = licenses
data_val['categories'] = categories
data_val['level_categories'] = level_cat
data_val['info'] = info_map
data_val['images'] = val_img_info

with open('/scratch/pchaudha/ranking_loss/mask_exp/k_new/k5/gt_val_mask.json', 'w') as f3:
    json.dump(data_val, f3)

print("Done")