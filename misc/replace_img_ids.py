
#To replace image_ids like Flood_x to integers
#Similarly for mapillary images

import json

test_ids_mask = []

train_ids_mask = []

val_ids_mask = []

#mask files
with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k1/gt_val_mask.json') as file4:
    val_mask = json.load(file4)

with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k1/gt_train_mask.json') as file5:
    train_mask = json.load(file5)

with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k1/gt_test_mask.json') as file6:
    test_mask = json.load(file6)

with open('/scratch/pchaudha/ranking_loss/mask_exp/img_new_ids_dict.json') as f1:
    name_id_dict = json.load(f1)

#######################################################################
##########################   Val  #####################################
#######################################################################

for i in val_mask['annotations']:
    temp = i['image_id']
    if temp in name_id_dict:
        i['image_id'] = name_id_dict[temp]
        print(i)

for j in val_mask['images']:
    temp2 = j['id']
    if temp2 in name_id_dict:
        j['id'] = name_id_dict[temp2]
        print(j)

with open('/scratch/pchaudha/ranking_loss/mask_exp/k_new/k1/gt_val_mask.json', 'w') as fp:
    json.dump(val_mask, fp, indent=4)

#######################################################################
##########################  Train #####################################
#######################################################################

for i in train_mask['annotations']:
    temp = i['image_id']
    if temp in name_id_dict:
        i['image_id'] = name_id_dict[temp]
        print(i)

for j in train_mask['images']:
    temp2 = j['id']
    if temp2 in name_id_dict:
        j['id'] = name_id_dict[temp2]
        print(j)

with open('/scratch/pchaudha/ranking_loss/mask_exp/k_new/k1/gt_train_mask.json', 'w') as fp:
    json.dump(train_mask, fp, indent=4)


#######################################################################
##########################  Test  #####################################
#######################################################################

for i in test_mask['annotations']:
    temp = i['image_id']
    if temp in name_id_dict:
        i['image_id'] = name_id_dict[temp]
        print(i)

for j in test_mask['images']:
    temp2 = j['id']
    if temp2 in name_id_dict:
        j['id'] = name_id_dict[temp2]
        print(j)

with open('/scratch/pchaudha/ranking_loss/mask_exp/k_new/k1/gt_test_mask.json', 'w') as fp:
    json.dump(test_mask, fp, indent=4)


print("Done")
