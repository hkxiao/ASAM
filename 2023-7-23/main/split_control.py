from diffusers import  ControlNetModel
import torch
from collections import OrderedDict

url = "./control_v11p_sd15_canny.pth"  # can also be a local path

#controlnet = ControlNetModel.from_single_file(url)

checkpoint = torch.load(url)
# for k,v in checkpoint.items():
#     print(k)
print(len(checkpoint.keys()))

url2 = '/data/tanglv/Ad-SAM/2023-7-23/ControlNet-main/lightning_logs/version_16/checkpoints/epoch=246-step=81509.ckpt'
checkpoint2 = torch.load(url2)["state_dict"]


new_state_dict = {}
for k,v in checkpoint2.items():
    if k in checkpoint.keys():
        new_state_dict[k] = v

print(len(new_state_dict.keys()))

torch.save(new_state_dict,"./control_v11p_sd15_mask.pth",)
