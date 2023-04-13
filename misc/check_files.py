#To check if files in gt_mask and other experiments are same
#to check no mistake

import json

test_ids_rank = []
test_ids_mask = []

train_ids_rank = []
train_ids_mask = []

val_ids_rank = []
val_ids_mask = []

with open('/scratch/pchaudha/ranking_loss/k/k1/gt_val_reg.json') as f1:
    val_rank = json.load(f1)

with open('/scratch/pchaudha/ranking_loss/k/k1/gt_train_reg.json') as f2:
    train_rank = json.load(f2)

with open('/scratch/pchaudha/ranking_loss/k/k1/gt_test.json') as f3:
    test_rank = json.load(f3)

#mask files
with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k1/gt_val_mask.json') as file4:
    val_mask = json.load(file4)

with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k1/gt_train_mask.json') as file5:
    train_mask = json.load(file5)

with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k1/gt_test_mask.json') as file6:
    test_mask = json.load(file6)

#val files

for key, value in val_rank.items():
    # add image path to load it later
    key = key.replace('.jpg', '')
    if 'FloodImages_newname/' in key:
        key = key.replace('FloodImages_newname/', ' ')
        key = key.split()[2]
    elif 'training/images/' in key:
        key = key.replace('training/images/', ' ')
        key = key.split()[1]
    val_ids_rank.append(str(key))


temp = val_mask['images']
for p in temp:
    img_id = p['id']
    val_ids_mask.append(img_id)


#compare two lists if identical using set
if (set(val_ids_mask) == set(val_ids_rank)):
    print("val files equal")

#train files

for key, value in train_rank.items():
    # add image path to load it later
    key = key.replace('.jpg', '')
    if 'FloodImages_newname/' in key:
        key = key.replace('FloodImages_newname/', ' ')
        key = key.split()[2]
    elif 'training/images/' in key:
        key = key.replace('training/images/', ' ')
        key = key.split()[1]
    train_ids_rank.append(str(key))


temp = train_mask['images']
for p in temp:
    img_id = p['id']
    train_ids_mask.append(img_id)


#compare two lists if identical using set
if (set(train_ids_mask) == set(train_ids_rank)):
    print("train files equal")


#test files

for key, value in test_rank.items():
    # add image path to load it later
    key = key.replace('.jpg', '')
    if 'FloodImages_newname/' in key:
        key = key.replace('FloodImages_newname/', ' ')
        key = key.split()[2]
    elif 'training/images/' in key:
        key = key.replace('training/images/', ' ')
        key = key.split()[1]
    test_ids_rank.append(str(key))


temp = test_mask['images']
for p in temp:
    img_id = p['id']
    test_ids_mask.append(img_id)


#compare two lists if identical using set
if (set(test_ids_mask) == set(test_ids_rank)):
    print("test files equal")

#check if size in segementation of annotation file is same for all instances of the image
with open('/scratch/pchaudha/ranking_loss/mask_exp/flood_dict.json') as flood:
    flood_dict = json.load(flood)

size_dict = {}
for img, value in flood_dict.items():
    for val in value:
        size_dict.setdefault(img, [])
        s = val['segmentation']['size']
        size_dict[img].append(s)

#to check if all elements in the size list are identical
for i,j in size_dict.items():
    print(j[1:] == j[:-1])

#as the annotations are not stored properly
#i am suspecting it might be due to not having an integer
#image i. Instead having a str image id
#so to change now we make a image id for every image
with open('/scratch/pchaudha/ranking_loss/mask_exp/mapillary_dict.json') as mapi:
    mapi_dict = json.load(mapi)

list_map = list(mapi_dict.keys())
list_flood = list(flood_dict.keys())

list_com = list_flood+list_map

count = 600000
new_id_dict = {}
for pree in list_com:
    new_id_dict[pree] = count
    count = count + 1

# with open('img_new_ids_dict.json', 'w') as fp:
#     json.dump(new_id_dict, fp, indent=4)

#now we load the annotations files from /scratch/pchaudha/ranking_loss/mask_exp/k/k1
#and replace image_ids by new integer image ids as defined in img_new_ids_dict.json
# with open('/scratch/pchaudha/ranking_loss/mask_exp/mapillary_dict.json') as mapi:
#     mapi_dict = json.load(mapi)

print("DOne")
