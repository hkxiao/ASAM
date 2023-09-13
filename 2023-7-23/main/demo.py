from diffusers import  ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
import torch
import cv2
import numpy as np
from PIL import Image

# from dict import or
device = torch.device("cuda:0")
url = "./control_v11p_sd15_mask.pth"  
controlnet = ControlNetModel.from_single_file(url).to(device)

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
ldm_stable = StableDiffusionControlNetPipeline.from_pretrained("./stable-diffusion-v1-5", use_auth_token=True,controlnet=controlnet, scheduler=scheduler).to(device)

generator = torch.manual_seed(1)

#mask = cv2.imread("/data/tanglv/data/sod_data/DUTS-TR/gts/ILSVRC2012_test_00000004.png")

mask = Image.open("/data/tanglv/data/sod_data/DUTS-TR/gts/ILSVRC2012_test_00000018.png")
print(mask.size)
out_image = ldm_stable(
    "", num_inference_steps=50, generator=generator, image=mask
).images[0]

out_image.save("demo.png")