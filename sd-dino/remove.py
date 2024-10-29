import torch
import os

outdir = 'output'
cnt = 0 

for file in os.listdir(outdir):
    if 'dino' not in file: continue
    
    path = os.path.join(outdir, file)
    pth = torch.load(path)
    
    if(type(pth)!=torch.Tensor): print(path)
    # print(type(pth))

print(cnt)