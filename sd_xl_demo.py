import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
import time
from PIL import Image
import numpy as np
import cv2

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
controlnet = ControlNetModel.from_pretrained(
    "ckpt/diffusers---controlnet-canny-sdxl-1.0",torch_dtype=torch.float16
)
controlnet = controlnet.to("cuda")

ldm_stable = StableDiffusionXLControlNetPipeline.from_pretrained(
    "ckpt/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
)
ldm_stable = ldm_stable.to("cuda")
ldm_stable.enable_model_cpu_offload()

prompt = "a stairway leading up to a building with ivy growing on it"

# get canny image
image = Image.open('sam-1b/sa_000000/sa_1.jpg')
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
canny_image.save('canny.jpg')

controlnet_conditioning_scale=0.5

start_time = time.time()
image = ldm_stable(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image).images[0]

print('time:' ,time.time()-start_time)
image.save('control.jpg')

