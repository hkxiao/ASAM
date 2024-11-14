from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, DDIMScheduler, ControlNetModel
import torch
import json
import PIL
import textwrap
import IPython.display as display
from PIL import Image
import cv2
import numpy as np
import sys
# sys.path.append('/remote-home/tanglv/ASAM/ASAM/2023-12-19/ASAM-Main')
# from control_edit import text2image_ldm_stable_control, EmptyControl
sys.path.append('/remote-home/tanglv/ASAM')
from inversion import text2image_ldm_stable, EmptyControl

# BLIP3
# def apply_prompt_template(prompt):
#     s = (
#                 '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
#                 "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
#                 f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
#             )
#     return s 

# model_name_or_path = "../pretrained/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
# model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
# image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
# tokenizer = model.update_special_tokens(tokenizer)
# model = model.to('cuda').to(torch.float32)
# model.eval()
# tokenizer.padding_side = "left"
# tokenizer.eos_token = '<|end|>'

# image_list = []
# image_sizes = []
# fn = '/remote-home/tanglv/ASAM/sam_continue_learning/data/BIG/val/3414433756_b58a8e0cbd_o_chair_im.jpg'
# img = PIL.Image.open(fn)
# display.display(Image(filename=fn, width=300))
# image_list.append(image_processor([img], image_aspect_ratio='anyres')["pixel_values"].cuda())
# image_sizes.append(img.size)
# inputs = {
#     "pixel_values": [image_list]
# }

# query= "<image> Please describe the main content of this image"
# prompt = apply_prompt_template(query)
# language_inputs = tokenizer([prompt], return_tensors="pt")
# inputs.update(language_inputs)

# # # To cuda
# for name, value in inputs.items():
#     if isinstance(value, torch.Tensor):
#         inputs[name] = value.cuda()
        
# generated_text = model.generate(**inputs, image_size=[image_sizes],
#                                 pad_token_id=tokenizer.pad_token_id,
#                                 eos_token_id=tokenizer.eos_token_id,
#                                 temperature=0.05,
#                                 do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1,
#                                 )
# prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
# print("User: ", query)
# print("Assistant: ", textwrap.fill(prediction, width=100))
# prompt = textwrap.fill(prediction, width=100)

from lavis.models import load_model_and_preprocess
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device="cuda")
raw_image = Image.open("/remote-home/tanglv/ASAM/sam_continue_learning/data/BIG/val/3519526630_a8a3c9f06d_o_sheep_im.jpg").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to('cuda')
prompt = model.generate({"image": image})[0]

# Load SD-XL with ControlNet
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
controlnet = ControlNetModel.from_pretrained(
    "../ControlNetSDXL/work_dirs/train_output_sdxl3/checkpoint-20000/controlnet", torch_dtype=torch.float32
).cuda()

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "../pretrained/SSD-1B", controlnet=controlnet, scheduler=scheduler, torch_dtype=torch.float32).to('cuda')
pipe.mask_controlnet = controlnet

mask_control = cv2.imread("/remote-home/tanglv/ASAM/sam_continue_learning/data/BIG/val/3519526630_a8a3c9f06d_o_sheep_gt.png").astype(np.float32)
mask_control = cv2.cvtColor(mask_control, cv2.COLOR_BGR2RGB)
mask_control = cv2.resize(mask_control, (1024, 1024)) / 255.0
mask_control = torch.from_numpy(mask_control).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()

# negative_prompt = "low quality, unnatural, unrealistic, and not rich in texture"
negative_prompt = ""
img = text2image_ldm_stable(pipe, prompt=["a natural image"], negative_prompt=[negative_prompt], controller=EmptyControl(), mask_control=mask_control, mask_control_scale=1.0, \
                            aigc_model_type='SSD-1B', aigc_img_size=1024, return_type='image')
# img = text2image_ldm_stable_control(pipe, prompt=["a good girl"], controller=EmptyControl(), control_mask=mask_control)

cv2.imwrite('demo.jpg', img[0][0][:,:,::-1])
print(type(img))





 