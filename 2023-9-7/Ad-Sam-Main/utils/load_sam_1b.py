import os 
import json
from pycocotools import mask
import numpy as np
import matplotlib.pyplot as plt
import cv2

dataset_dir = '/data/tanglv/data/sam-1b_subset/'
json_files = os.listdir(dataset_dir)
json_files = [x for x in json_files if 'json' in x]
json_files = sorted(json_files)

for file in json_files:
    dict = json.loads(open(dataset_dir+file,'r').read())
    img_id = dict['image']['image_id']
    img_path = dataset_dir + 'sa_' + str(img_id) + '.jpg'
    img = cv2.imread(img_path)
    
    mask_num = len(dict['annotations'])
    for i in range(mask_num):
        rle_encoded_mask = dict['annotations'][i]['segmentation']
        decoded_mask = mask.decode(rle_encoded_mask)
        print(decoded_mask.shape)
        cv2.imwrite('demo_mask.png', decoded_mask*255.0)
        
        x,y,w,h = dict['annotations'][i]['bbox']
        x,y,w,h = int(x),int(y),int(w),int(h)
        image_with_rectangle = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.imwrite('demo_bbox.png', image_with_rectangle)
        
        raise NameError

    


