import os
import cv2
import numpy as np

path = "/data/tanglv/data/ADE20K_2016_07_26/images/training/a/abbey/ADE_train_00000970_seg.png"

img = cv2.imread(path)

img = img[...,0]*(1<<16) + img[...,1]*(1<<8) + img[...,2]

print(img.shape)
print(len(np.unique(img)))