from diffusers import  ControlNetModel
import torch
from collections import OrderedDict

url = "../pretrained/control_v11p_sd15_canny.pth"  # can also be a local path

checkpoint = torch.load(url)
print(len(checkpoint.keys()))

url2 = '../ControlNet-main/lightning_logs/version_21/checkpoints/epoch=46-step=21948.ckpt'
checkpoint2 = torch.load(url2)["state_dict"]

new_state_dict = {}
for k,v in checkpoint2.items():
    if k in checkpoint.keys():
        new_state_dict[k] = v

print(len(new_state_dict.keys()))
torch.save(new_state_dict,"../pretrained/control_v11p_sd15_mask_sam_v2.pth")
