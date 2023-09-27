from glob import glob
import numpy as np
import os
import cv2

path = "/data/tanglv/data/ADE20K_2016_07_26/images/validation/i/ice_floe/ADE_val_00001439_seg.png"

img = cv2.imread(path)
img = img[...,0]*(1<<16) + img[...,1]*(1<<8) + img[...,2]
unique_list = np.unique(img)
print(img.shape)
for i in range(len(unique_list)):
    mask = np.zeros_like(img)
    mask[img==unique_list[i]] = 255.0
    print(mask.shape,mask.max(),mask.sum())
    cv2.imwrite(str(i)+'.png',mask.astype(np.uint8))


print(len(unique_list)) 
