import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2, os

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
import sys
# sys.path.append("..")
import os
print(os.getcwd())
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "../sam_continue_learning/pretrained_checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

asam_checkpoint = "../sam_continue_learning/work_dirs/fine-tuning-4-dice-vit_b-11186/epoch_11.pth"
#sam.mask_decoder.load_state_dict(torch.load(asam_checkpoint), strict=False)

# generate segmeantation mask
mask_generator = SamAutomaticMaskGenerator(sam,)
# root = '/data/tanglv/xhk/ASAM/2023-9-7/Ad-Sam-Main/lecun_photo'
# for file in os.listdir(root):
#     if 'jpg' not in file: continue
    
#     image = cv2.imread(root + '/' + file)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     masks = mask_generator.generate(image)
#     sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
#     if not os.path.exists(root + '/' + file[:-4]): os.mkdir(root + '/' + file[:-4])
#     for idx,mask in enumerate(sorted_masks):
#         print(mask['segmentation'].shape, type(mask['segmentation']))
#         cv2.imwrite(os.path.join(root, file[:-4], f'segmentation_{idx}.png'), mask['segmentation'].astype(np.uint8)*255.0)

# root = '../output/lecun-Grad/Attacker-7.5-50-Definder-sam-vit_b-140-point-10-Loss-100.0-100.0-100.0-1.0-2-Perturbation-0.2-10-0.02-0.5/adv'
# result_dir = 'output/results0/'

root = '../lecun_photo'
result_dir = 'output/results_normal/'
if not os.path.exists(result_dir): 
    os.mkdir(result_dir)
    
for file in os.listdir(root):
    if 'jpg' not in file: continue
    print(file)
    #if 'png' not in file and 'jpg' not in file: continue
    image = cv2.imread(root + '/' + file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    masks = mask_generator.generate(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 
    plt.savefig(result_dir+file.replace('jpg', 'png'))
    
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    nums = len(masks)
    length = 1<<24
    
    for j, anno in enumerate(sorted_masks):
        mask = anno['segmentation'].astype(np.uint8)
        pos = int((length-1)*(j+1)/nums)  
        color = (pos%(1<<8), pos//(1<<8)%(1<<8), pos//(1<<16))
        if j == 0: control_signal = np.stack([np.zeros_like(mask)]*3,-1)
        control_signal[mask==1] = color
    
    control_signal = control_signal[:,:,::-1]
    cv2.imwrite(result_dir+file.replace('jpg', 'png').replace('.png','_vis.png'),control_signal)
    