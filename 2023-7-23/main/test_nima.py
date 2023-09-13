import torch
import argparse
import numpy as np 
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import trange
import numpy
import numpy as np
import math
# scikit-image 0.16.2
# from skimage.measure import compare_ssim, compare_psnr, compare_mse
# import lpips
# from fid import get_fid
import pyiqa

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack!')
parser.add_argument('--model', type=str, default='resnet50', help='cnn')
parser.add_argument('--dataset', type=str, default='FFDF', help='cnn')
parser.add_argument('--bs', type=int, default=1, help='cnn')
parser.add_argument('--seed', type=int, default=42, help='cnn')


args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(path).convert('RGB')
            return img
    except IOError:
        print('Cannot load image ' + path)

if __name__ == '__main__':
    img1_root = 'temp/1000/original'
    img_roots = [
        'AttackResultsFinal/vit/original',
                'AttackResultsFinal/vit/inversion',
                'AttackResultsFinal/vit/ACA',
                'AttackResultsFinal/vit/NCF',
    'AttackResultsFinal/vit/SAE-vit',
                'AttackResultsFinal/vit/ADef_vit',
                'AttackResultsFinal/vit/ReColorAdv-vit',
                'AttackResultsFinal/vit/cAdv-vit-500',
    'AttackResultsFinal/vit/tadv_vit',
                'AttackResultsFinal/vit/ACE_vit',
                'AttackResultsFinal/vit/ColorFool-adv_vit']

    name = ''
    iqa_metric = pyiqa.create_metric(name, device=device)
    print(iqa_metric.lower_better)
    fid_metric = pyiqa.create_metric('fid', device=device)

    for img2_root in img_roots:
        f_list = os.listdir(img2_root)
        nnima = 0
        for i in trange(len(f_list)):
            img2 = torch.Tensor(np.array(img_loader(os.path.join(img2_root, f_list[i])).resize((224, 224)))).unsqueeze(0).to(device).permute(0, 3, 1, 2)/255.0
            score_nr = iqa_metric(img2)
            nnima += score_nr
        print('*' * 60)
        print(img2_root)
        print('name', nnima/len(f_list))



 

