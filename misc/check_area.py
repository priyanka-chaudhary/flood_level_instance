#getting area of bounding box zero error
#have to check if there are bounding boxes with area less than 5

import json

with open('/scratch/pchaudha/ranking_loss/mask_exp/mapillary_dict.json') as f:
    mapi_dict = json.load(f)

with open('/scratch/pchaudha/ranking_loss/mask_exp/flood_dict.json') as f:
    flood_dict = json.load(f)

for key, values in flood_dict.items():
    for i in values:
        temp = i['area']
        if temp < 10.0:
            print(key)
            print(i)

img_ids = []
ids_list = []
ann_list = []
for key, values in mapi_dict.items():
    for i in values:
        temp = i['area']
        if temp < 63.0:
            pree = i['id']
            ids_list.append(pree)
            img_ids.append(key)
            ann_list.append(i)
            print(key)
            print(i)

img_ids = list(set(img_ids))


#remove annoation ids above in ids_list + 10797 from image 'wO1V3zKD9KtnDFQp-wmM9Q'
ids_list.append(10797)

#mask files
with open('/scratch/pchaudha/ranking_loss/mask_exp/mask_rcnn/k/k5/gt_val_mask.json') as file4:
    val_mask = json.load(file4)

with open('/scratch/pchaudha/ranking_loss/mask_exp/mask_rcnn/k/k5/gt_train_mask.json') as file5:
    train_mask = json.load(file5)

with open('/scratch/pchaudha/ranking_loss/mask_exp/mask_rcnn/k/k5/gt_test_mask.json') as file6:
    test_mask = json.load(file6)

print("removing annotations")
print("===================================================================================================")
copy_ann_list = ann_list.copy()

count= 0
for i in ann_list:
    count = count+1
    #print(count)
    if i in val_mask['annotations']:
        val_mask['annotations'].remove(i)
        copy_ann_list.remove(i)
    elif i in train_mask['annotations']:
        train_mask['annotations'].remove(i)
        copy_ann_list.remove(i)
    elif i in test_mask['annotations']:
        test_mask['annotations'].remove(i)
        copy_ann_list.remove(i)
    else:
        print("Couldnt find i in val, train or test")
        print(i)

print("remaining annotations")
print(copy_ann_list)

#mask files
# with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k5/gt_val_mask.json', 'w') as f1:
#     json.dump(val_mask, f1)
#
# with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k5/gt_train_mask.json', 'w') as f2:
#     json.dump(train_mask, f2)
#
# with open('/scratch/pchaudha/ranking_loss/mask_exp/k/k5/gt_test_mask.json', 'w') as f3:
#     json.dump(test_mask, f3)

print("Done")