import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam_hq_main.segment_anything import sam_model_registry_baseline, SamPredictor
import os
import json
from PIL import Image
from skimage import io
from skimage import transform

if __name__ == "__main__":
    sam_checkpoint = "sam_hq_main/pretrained_checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    #img_dir = 'temp/skip-ablation-01-mi-0.5-sam-0.01-1-2-10-Clip-0.2/adv/'
    dataset_dir = '/data/tanglv/data/sod_data/DUTS-TR'
    infos = open('/data/tanglv/data/sod_data/DUTS-TR/box.json').readlines()
    
    for i in range(len(infos)):
        info  = json.loads(infos[i])
        img_path = info['target']
        gt_path = info['source']
        box = [info['box']]

        box_torch = torch.tensor(box).cuda().to(torch.float32)
        
        img = io.imread(os.path.join(dataset_dir,img_path))
        img = torch.tensor(img.copy(), dtype=torch.float32, device=device)
        img = torch.transpose(torch.transpose(img,1,2),0,1) 
        img = torch.nn.functional.interpolate(img[None],(1024,1024),mode='bilinear')[0]
        
        print(torch.max(img), torch.min(img))
        # img_torch = torch.from_numpy(img).permute(2,0,1).contiguous().cuda().to(torch.float32)

        mask = cv2.imread(os.path.join(dataset_dir,gt_path),1).astype(np.float32)
        mask = cv2.resize(mask,(256,256)) / 255.0
        
        example = {}
        example['image'] = img
        example['box'] = box_torch
        example['original_size'] = (1024,1024)
        with torch.no_grad():
            outputs = sam([example], multimask_output=False)[0]
            low_res_logits = outputs['low_res_logits']
            mask = outputs['masks']
        
        low_mask = (low_res_logits>=0.0).to(torch.float32)
        low_mask_numpy = low_mask[0,0,...].cpu().numpy() * 255.0
        cv2.imwrite('demo.jpg', low_mask_numpy)
        
        print(torch.max(mask), mask.shape)
        mask_numpy = mask[0,0,...].cpu().numpy() * 255.0
        cv2.imwrite('demo1.jpg', mask_numpy)
        
        # masks, scores, logits = predictor.predict(
        #         box = np.array([box]),
        #         multimask_output=False,
        #     )
        
        # print(masks.shape)
        # cv2.imwrite('demo3.png',masks[0]*255)
        raise NameError
        




