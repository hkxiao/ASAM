import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch
from process_feat import process_feat, clip_feat
from torch.nn import functional as F

class MyDataset(Dataset):
    def __init__(self,root,json_path):
        self.data = []
        self.root = root
        self.feat_root = '../sd-dino/work_dirs'
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if 'source' in item.keys():
            source_filename = item['source']
            target_filename = item['target']
            prompt = item['prompt']
            
            source = cv2.imread(self.root + '/' + source_filename)
            target = cv2.imread(self.root + '/' + target_filename)
            
            # resize 512
            source = cv2.resize(source,(512,512))
            target = cv2.resize(target,(512,512))
            
            # Do not forget that OpenCV read images in BGR order.
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0
            
            return dict(jpg=target, txt=prompt, hint=source)
        
        elif 'source_sd' in item.keys() and 'source_dino' in item.keys():
            source_sd_filename = item['source_sd']
            source_dino_filename = item['source_dino']

            target_filename = item['target']
            prompt = item['prompt']
            
            source_sd = torch.load(self.feat_root + '/' + source_sd_filename, map_location='cuda')
            source_dino = torch.load(self.feat_root + '/' + source_dino_filename, map_location='cuda')
            
            target = cv2.imread(self.root + '/' + target_filename)

            # process feat
            feat = process_feat(source_sd, source_dino, sd_target_dim=[4,4,4], dino_target_dim=12, dino_pca=True, using_sd=True, using_dino=True) #[1 C H W]
            feat = feat.flatten(-2).permute(0,2,1).unsqueeze(0) # (1,1,H*W,C)
            feat = clip_feat(feat, img_path = self.root + '/' + target_filename) #[H W C]
            feat = feat.permute(2,0,1).unsqueeze(0) #[1 C H W]
            
            # resize 512
            target = cv2.resize(target,(512,512))
            feat = F.interpolate(feat, size=(512,512), mode='bilinear', align_corners=False).permute(0,2,3,1).squeeze(0) #[H W C]
            
            # Do not forget that OpenCV read images in BGR order.
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            
            # Instance Normalize feat to [0, 1].            
            feat_min = feat.view(512*512,-1).min(dim=0,keepdim=True)[0].view(1,1,-1)
            feat_max = feat.view(512*512,-1).max(dim=0,keepdim=True)[0].view(1,1,-1)
            feat = (feat - feat_min) / (feat_max - feat_min)
            
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0
            
            return dict(jpg=target, txt=prompt, hint=feat)

