import os
import shutil
import cv2

dir = 'output/'

for root,dirs,files in os.walk(dir):
    for file in files:
        if file.endswith('png'):
            shutil.move(root+'/'+file,root+'/'+file.replace('png','jpg'))