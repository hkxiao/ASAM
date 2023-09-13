import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam_hq_main.segment_anything import sam_model_registry_baseline, SamPredictor
import os
import json
from PIL import Image


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
        
        img = cv2.imread(os.path.join(dataset_dir,img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(1024,1024))
        predictor.set_image(img)
        img_torch = torch.from_numpy(img.copy()).permute(2,0,1).contiguous().cuda().to(torch.float32)

        mask = cv2.imread(os.path.join(dataset_dir,gt_path),1).astype(np.float32)
        mask = cv2.resize(mask,(256,256)) / 255.0
        
        example = {}
        example['image'] = img_torch
        example['boxes'] = box_torch
        example['original_size'] = (1024,1024)
        with torch.no_grad():
            outputs = sam([example], multimask_output=False)[0]
            low_res_logits = outputs['low_res_logits']
            mask = outputs['masks']
        
        low_mask = (low_res_logits>=0.0).to(torch.float32)
        low_mask_numpy = low_mask[0,0,...].cpu().numpy() * 255.0
        cv2.imwrite('demo.png', low_mask_numpy)
        
        # print(mask.shape,low_res_logits.shape)
        # raise NameError
        mask_numpy = mask[0,0,...].cpu().numpy() * 255.0
        cv2.imwrite('demo1.png', mask_numpy)
        
        masks, scores, logits = predictor.predict(
                box = np.array([box]),
                multimask_output=False,
            )
        
        print(masks.shape)
        cv2.imwrite('demo3.png',masks[0]*255)
        raise NameError
        




