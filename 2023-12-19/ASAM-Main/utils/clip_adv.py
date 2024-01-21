import os
import shutil
import cv2

dir = 'output/sa_000000-Grad/skip-ablation-01-mi-SSD-7.5-50-SAM-sam-vit_b-160-ADV-0.2-2-0.1-0.5-32.0-32.0-1.0-2/'

pair_dir = dir + 'pair'
adv_dir = dir + 'adv'

for file in os.listdir(pair_dir):
    print(pair_dir +'/'+file)
    img = cv2.imread(pair_dir +'/'+file)
    adv_img = img[:,1024*3+20*3:1024*4+20*3:] 
    
    cv2.imwrite(adv_dir+'/'+file, adv_img)
    #raise NameError