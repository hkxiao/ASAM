import cv2
import numpy as np
import os

origin_dir = 'sam-1b/sa_000138'
null_inversion_dir = 'output/sa_000138-Inversion/inv'
direct_inversion_dir = 'output/sa_000138-Direction-Inversion/inv'

len = 10

mse_null = 0
mse_direct = 0

for file in os.listdir(direct_inversion_dir):
    
    ori_img = cv2.imread(os.path.join(origin_dir, file.replace('png','jpg')))
    ori_img = cv2.resize(ori_img,(512,512))
    
    null_img = cv2.imread(os.path.join(null_inversion_dir, file))
    direct_img = cv2.imread(os.path.join(direct_inversion_dir, file))

    mse_null += np.sum((ori_img-null_img)*(ori_img-null_img))
    mse_direct += np.sum((ori_img-direct_img)*(ori_img-direct_img))

print('null mse', mse_null)
print('direct mse', mse_direct)
    