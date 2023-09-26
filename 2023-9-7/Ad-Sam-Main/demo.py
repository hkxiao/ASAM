from glob import glob
import numpy as np
import os
import cv2

path = "/data/tanglv/data/ADE20K_2016_07_26/images/validation"
img_path_list = []

for root, dirs, fils in os.walk(path):
    img_path_list.extend(glob(root+os.sep+'*'+'seg.png'))

mx = 0
name = ''
for img_path in img_path_list:
    img = cv2.imread(img_path)
    img = img[...,0]*(1<<16) + img[...,1]*(1<<8) + img[...,2]
    unique_list = np.unique(img)
    if len(unique_list) > mx:
        mx = max(mx, len(unique_list))
        name = img_path

print(mx,name) 
